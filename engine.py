# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher import data_prefetcher
from util import detection_metrics as det_metrics
from util import box_ops


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()

    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        samples, targets = prefetcher.next()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, args=None):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    # For custom PR/F1 metrics (e.g., OCT .pkl), we collect lightweight
    # per-image detection and ground-truth records.
    all_detections = []
    all_ground_truths = []

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        # OCT .pkl: model outputs class indices 0,1; COCO expects category_id 1,2
        if args is not None and getattr(args, 'dataset_file', None) == 'oct_pkl':
            results = [{**r, 'labels': r['labels'] + 1} for r in results]
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

        # Collect detections and ground-truths for custom PR/F1 metrics.
        # We always work in absolute xyxy pixel coordinates.
        # OCT: dataset/criterion use 0,1; PR metrics expect 1,2 (results already +1 above).
        with torch.no_grad():
            for tgt, out in zip(targets, results):
                img_id = tgt["image_id"]
                det_boxes = out["boxes"]
                det_scores = out["scores"]
                det_labels = out["labels"]

                gt_boxes_cxcywh = tgt["boxes"]
                gt_labels = tgt["labels"]
                size = tgt["size"]
                h, w = size[0].item(), size[1].item()
                scale = torch.tensor([w, h, w, h], dtype=torch.float32, device=gt_boxes_cxcywh.device)
                gt_boxes_xyxy = box_ops.box_cxcywh_to_xyxy(gt_boxes_cxcywh * scale)

                if args is not None and getattr(args, 'dataset_file', None) == 'oct_pkl':
                    gt_labels = gt_labels + 1
                all_detections.append(
                    {"image_id": img_id, "boxes": det_boxes, "scores": det_scores, "labels": det_labels}
                )
                all_ground_truths.append(
                    {"image_id": img_id, "boxes": gt_boxes_xyxy, "labels": gt_labels}
                )

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    # COCO mAP-style metrics broken out into named fields for logging.
    if coco_evaluator is not None and 'bbox' in postprocessors.keys():
        bbox_stats = coco_evaluator.coco_eval['bbox'].stats.tolist()
        # Standard COCO ordering:
        # [AP, AP50, AP75, APs, APm, APl, AR1, AR10, AR100, ARs, ARm, ARl]
        if bbox_stats and len(bbox_stats) >= 12:
            stats['coco_ap'] = float(bbox_stats[0])
            stats['coco_ap_50'] = float(bbox_stats[1])
            stats['coco_ap_75'] = float(bbox_stats[2])
            stats['coco_ap_small'] = float(bbox_stats[3])
            stats['coco_ap_medium'] = float(bbox_stats[4])
            stats['coco_ap_large'] = float(bbox_stats[5])
            stats['coco_ar_1'] = float(bbox_stats[6])
            stats['coco_ar_10'] = float(bbox_stats[7])
            stats['coco_ar_100'] = float(bbox_stats[8])
            stats['coco_ar_small'] = float(bbox_stats[9])
            stats['coco_ar_medium'] = float(bbox_stats[10])
            stats['coco_ar_large'] = float(bbox_stats[11])
        stats['coco_eval_bbox'] = bbox_stats

    if coco_evaluator is not None and 'segm' in postprocessors.keys():
        stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()

    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]

    # Custom PR/F1 metrics primarily intended for the OCT .pkl dataset.
    if args is not None and getattr(args, "dataset_file", None) == "oct_pkl":
        # OCT has 2 foreground classes: 1=fovea, 2=SCR
        class_names = {1: "fovea", 2: "SCR"}
        pr_result = det_metrics.compute_pr_f1(
            detections=all_detections,
            ground_truths=all_ground_truths,
            iou_thresh=float(getattr(args, "pr_iou_thresh", 0.5)),
            score_thresh=float(getattr(args, "pr_score_thresh", 0.5)),
            num_score_thresholds=int(getattr(args, "pr_num_thresholds", 20)),
            num_classes=2,
            class_names=class_names,
        )
        per_class = pr_result["per_class"]

        # Flatten per-class fixed and best-F1 metrics into scalar stats.
        for cls_id, cls_stats in per_class.items():
            name = cls_stats["name"]
            fixed = cls_stats["fixed"]
            best = cls_stats["best"]
            prefix_fixed = f"pr_fixed_{name}"
            prefix_best = f"pr_best_{name}"
            stats[f"{prefix_fixed}_precision"] = float(fixed["precision"])
            stats[f"{prefix_fixed}_recall"] = float(fixed["recall"])
            stats[f"{prefix_fixed}_f1"] = float(fixed["f1"])
            stats[f"{prefix_fixed}_TP"] = float(fixed["TP"])
            stats[f"{prefix_fixed}_FP"] = float(fixed["FP"])
            stats[f"{prefix_fixed}_FN"] = float(fixed["FN"])

            stats[f"{prefix_best}_f1"] = float(best["f1"])
            stats[f"{prefix_best}_precision"] = float(best["precision"])
            stats[f"{prefix_best}_recall"] = float(best["recall"])
            stats[f"{prefix_best}_thresh"] = float(best["best_thresh"])

        macro = pr_result["macro"]
        micro = pr_result["micro"]
        stats["pr_macro_precision"] = float(macro["precision"])
        stats["pr_macro_recall"] = float(macro["recall"])
        stats["pr_macro_f1"] = float(macro["f1"])
        stats["pr_micro_precision"] = float(micro["precision"])
        stats["pr_micro_recall"] = float(micro["recall"])
        stats["pr_micro_f1"] = float(micro["f1"])

        # Confusion-matrix-style summary based on fixed stats.
        per_class_fixed = {
            cls_id: det_metrics.PRStats(
                tp=int(cls_stats["fixed"]["TP"]),
                fp=int(cls_stats["fixed"]["FP"]),
                fn=int(cls_stats["fixed"]["FN"]),
            )
            for cls_id, cls_stats in per_class.items()
        }
        cm = det_metrics.build_confusion_matrix(per_class_fixed, class_names=class_names)
        # Store compact numeric view under stats.
        global_cm = cm["global"]
        stats["cm_global_TP"] = float(global_cm["TP"])
        stats["cm_global_FP"] = float(global_cm["FP"])
        stats["cm_global_FN"] = float(global_cm["FN"])

    return stats, coco_evaluator
