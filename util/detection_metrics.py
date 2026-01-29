"""
Custom detection metrics for OCT / small-class object detection.

This module provides:
  - compute_pr_f1: per-class and macro/micro precision / recall / F1
    using greedy one-to-one matching at a fixed IoU threshold, plus
    an optional scan over score thresholds to report best-F1.
  - build_confusion_matrix: simple confusion-matrix-style counts
    derived from the fixed-threshold statistics.

The goal is to keep the implementation self-contained and independent
of COCO so it can be reused for non-COCO datasets such as Nemours OCT.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


@dataclass
class PRStats:
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        if self.tp + self.fp == 0:
            return 0.0
        return float(self.tp) / float(self.tp + self.fp)

    @property
    def recall(self) -> float:
        if self.tp + self.fn == 0:
            return 0.0
        return float(self.tp) / float(self.tp + self.fn)

    @property
    def f1(self) -> float:
        p = self.precision
        r = self.recall
        if p + r == 0.0:
            return 0.0
        return 2.0 * p * r / (p + r)


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def _compute_iou_matrix(
    boxes1: np.ndarray, boxes2: np.ndarray
) -> np.ndarray:
    """
    Compute IoU between two sets of boxes (xyxy, absolute pixels).
    boxes1: [N, 4], boxes2: [M, 4]
    Returns: [N, M] IoU matrix.
    """
    if boxes1.size == 0 or boxes2.size == 0:
        return np.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=np.float32)

    x11, y11, x12, y12 = np.split(boxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(boxes2, 4, axis=1)

    xa = np.maximum(x11, x21.T)
    ya = np.maximum(y11, y21.T)
    xb = np.minimum(x12, x22.T)
    yb = np.minimum(y12, y22.T)

    inter_w = np.clip(xb - xa, a_min=0, a_max=None)
    inter_h = np.clip(yb - ya, a_min=0, a_max=None)
    inter = inter_w * inter_h

    area1 = np.clip((x12 - x11), a_min=0, a_max=None) * np.clip(
        (y12 - y11), a_min=0, a_max=None
    )
    area2 = np.clip((x22 - x21), a_min=0, a_max=None) * np.clip(
        (y22 - y21), a_min=0, a_max=None
    )

    union = area1 + area2.T - inter
    union = np.clip(union, a_min=1e-6, a_max=None)
    iou = inter / union
    return iou.astype(np.float32)


def _greedy_match_tp_fp_fn_for_class(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    iou_thresh: float,
) -> PRStats:
    """
    Greedy one-to-one matching based on IoU and score (descending).
    """
    if pred_boxes.size == 0 and gt_boxes.size == 0:
        return PRStats(tp=0, fp=0, fn=0)
    if pred_boxes.size == 0:
        return PRStats(tp=0, fp=0, fn=int(gt_boxes.shape[0]))
    if gt_boxes.size == 0:
        return PRStats(tp=0, fp=int(pred_boxes.shape[0]), fn=0)

    order = np.argsort(-pred_scores)  # high scores first
    pred_boxes = pred_boxes[order]
    pred_scores = pred_scores[order]

    ious = _compute_iou_matrix(pred_boxes, gt_boxes)
    gt_used = np.zeros((gt_boxes.shape[0],), dtype=bool)

    tp = 0
    fp = 0

    for p_idx in range(pred_boxes.shape[0]):
        iou_row = ious[p_idx]
        # best gt for this prediction
        gt_idx = int(np.argmax(iou_row))
        best_iou = float(iou_row[gt_idx])
        if best_iou >= iou_thresh and not gt_used[gt_idx]:
            tp += 1
            gt_used[gt_idx] = True
        else:
            fp += 1

    fn = int((~gt_used).sum())
    return PRStats(tp=tp, fp=fp, fn=fn)


def _aggregate_micro(pr_per_class: Dict[int, PRStats]) -> PRStats:
    tp = sum(s.tp for s in pr_per_class.values())
    fp = sum(s.fp for s in pr_per_class.values())
    fn = sum(s.fn for s in pr_per_class.values())
    return PRStats(tp=tp, fp=fp, fn=fn)


def compute_pr_f1(
    detections: Sequence[Dict[str, torch.Tensor]],
    ground_truths: Sequence[Dict[str, torch.Tensor]],
    iou_thresh: float = 0.5,
    score_thresh: float = 0.5,
    num_score_thresholds: int = 20,
    num_classes: Optional[int] = None,
    class_names: Optional[Dict[int, str]] = None,
) -> Dict[str, object]:
    """
    Compute per-class precision/recall/F1 for object detection.

    Args:
        detections: list of dicts with keys {"boxes", "scores", "labels"}
        ground_truths: list of dicts with keys {"boxes", "labels"}
        iou_thresh: IoU threshold for a match
        score_thresh: fixed score threshold for "fixed" metrics
        num_score_thresholds: number of thresholds to scan for best-F1
        num_classes: number of foreground classes (labels are 1..num_classes)
        class_names: optional mapping from class_id -> human-readable name

    Returns:
        A dict with:
          - "per_class": {cls_id: { "name": str,
                                    "fixed": {TP,FP,FN,precision,recall,f1},
                                    "best": {best_thresh,precision,recall,f1}}}
          - "macro": macro-averaged fixed metrics across classes
          - "micro": micro-averaged fixed metrics across classes
    """
    assert len(detections) == len(ground_truths)
    if num_classes is None:
        # infer from labels present (assuming 1..C)
        all_labels: List[int] = []
        for d in detections:
            if "labels" in d and d["labels"] is not None:
                all_labels.extend(_to_numpy(d["labels"]).astype(int).tolist())
        for g in ground_truths:
            if "labels" in g and g["labels"] is not None:
                all_labels.extend(_to_numpy(g["labels"]).astype(int).tolist())
        if all_labels:
            num_classes = max(all_labels)
        else:
            num_classes = 0

    # Collect boxes/scores/labels per image per class for fixed threshold.
    per_class_preds: Dict[int, List[np.ndarray]] = {c: [] for c in range(1, num_classes + 1)}
    per_class_pred_scores: Dict[int, List[np.ndarray]] = {
        c: [] for c in range(1, num_classes + 1)
    }
    per_class_gts: Dict[int, List[np.ndarray]] = {c: [] for c in range(1, num_classes + 1)}

    all_scores: List[float] = []

    for det, gt in zip(detections, ground_truths):
        d_boxes = _to_numpy(det.get("boxes", torch.zeros((0, 4))))
        d_scores = _to_numpy(det.get("scores", torch.zeros((0,))))
        d_labels = _to_numpy(det.get("labels", torch.zeros((0,), dtype=int))).astype(int)
        g_boxes = _to_numpy(gt.get("boxes", torch.zeros((0, 4))))
        g_labels = _to_numpy(gt.get("labels", torch.zeros((0,), dtype=int))).astype(int)

        # Filter predictions by fixed threshold once; for best-F1 scan we
        # will reapply per-threshold filtering.
        if d_scores.size > 0:
            # scores in [0,1]
            all_scores.extend(d_scores.tolist())

        for cls in range(1, num_classes + 1):
            # preds
            if d_boxes.size > 0:
                mask_p = d_labels == cls
                if mask_p.any():
                    per_class_preds[cls].append(d_boxes[mask_p])
                    per_class_pred_scores[cls].append(d_scores[mask_p])
            # gts
            if g_boxes.size > 0:
                mask_g = g_labels == cls
                if mask_g.any():
                    per_class_gts[cls].append(g_boxes[mask_g])

    # Concatenate per-class lists
    for cls in range(1, num_classes + 1):
        if per_class_preds[cls]:
            per_class_preds[cls] = [np.concatenate(per_class_preds[cls], axis=0)]
            per_class_pred_scores[cls] = [
                np.concatenate(per_class_pred_scores[cls], axis=0)
            ]
        if per_class_gts[cls]:
            per_class_gts[cls] = [np.concatenate(per_class_gts[cls], axis=0)]

    per_class_fixed: Dict[int, PRStats] = {}
    per_class_best: Dict[int, Dict[str, float]] = {}

    # thresholds to scan for best-F1 (if there are any scores)
    scan_thresholds: np.ndarray
    if all_scores and num_score_thresholds > 0:
        lo = float(min(all_scores))
        hi = float(max(all_scores))
        if lo == hi:
            scan_thresholds = np.array([lo], dtype=np.float32)
        else:
            scan_thresholds = np.linspace(lo, hi, num_score_thresholds, dtype=np.float32)
    else:
        scan_thresholds = np.array([], dtype=np.float32)

    for cls in range(1, num_classes + 1):
        name = class_names.get(cls, f"class_{cls}") if class_names else f"class_{cls}"
        preds_cls = (
            per_class_preds[cls][0]
            if isinstance(per_class_preds[cls], list) and per_class_preds[cls]
            else np.zeros((0, 4), dtype=np.float32)
        )
        scores_cls = (
            per_class_pred_scores[cls][0]
            if isinstance(per_class_pred_scores[cls], list) and per_class_pred_scores[cls]
            else np.zeros((0,), dtype=np.float32)
        )
        gts_cls = (
            per_class_gts[cls][0]
            if isinstance(per_class_gts[cls], list) and per_class_gts[cls]
            else np.zeros((0, 4), dtype=np.float32)
        )

        # Fixed-threshold stats
        if preds_cls.size > 0:
            mask_fixed = scores_cls >= float(score_thresh)
            preds_fixed = preds_cls[mask_fixed]
            scores_fixed = scores_cls[mask_fixed]
        else:
            preds_fixed = np.zeros((0, 4), dtype=np.float32)
            scores_fixed = np.zeros((0,), dtype=np.float32)

        fixed_stats = _greedy_match_tp_fp_fn_for_class(
            preds_fixed, scores_fixed, gts_cls, iou_thresh=iou_thresh
        )
        per_class_fixed[cls] = fixed_stats

        # Best-F1 over scan thresholds
        best_f1 = fixed_stats.f1
        best_p = fixed_stats.precision
        best_r = fixed_stats.recall
        best_t = float(score_thresh)

        for thr in scan_thresholds:
            if preds_cls.size == 0:
                stats_thr = PRStats(tp=0, fp=0, fn=int(gts_cls.shape[0]))
            else:
                mask_thr = scores_cls >= float(thr)
                preds_thr = preds_cls[mask_thr]
                scores_thr = scores_cls[mask_thr]
                stats_thr = _greedy_match_tp_fp_fn_for_class(
                    preds_thr, scores_thr, gts_cls, iou_thresh=iou_thresh
                )
            f1_thr = stats_thr.f1
            if f1_thr > best_f1:
                best_f1 = f1_thr
                best_p = stats_thr.precision
                best_r = stats_thr.recall
                best_t = float(thr)

        per_class_best[cls] = {
            "best_thresh": best_t,
            "precision": best_p,
            "recall": best_r,
            "f1": best_f1,
            "name": name,
        }

    # Macro/micro from fixed stats
    if num_classes > 0:
        macro_p = float(
            sum(s.precision for s in per_class_fixed.values()) / float(num_classes)
        )
        macro_r = float(
            sum(s.recall for s in per_class_fixed.values()) / float(num_classes)
        )
        macro_f1 = float(
            sum(s.f1 for s in per_class_fixed.values()) / float(num_classes)
        )
    else:
        macro_p = macro_r = macro_f1 = 0.0

    micro_stats = _aggregate_micro(per_class_fixed)

    per_class_out: Dict[int, Dict[str, object]] = {}
    for cls in range(1, num_classes + 1):
        stats = per_class_fixed[cls]
        name = class_names.get(cls, f"class_{cls}") if class_names else f"class_{cls}"
        per_class_out[cls] = {
            "name": name,
            "fixed": {
                "TP": int(stats.tp),
                "FP": int(stats.fp),
                "FN": int(stats.fn),
                "precision": float(stats.precision),
                "recall": float(stats.recall),
                "f1": float(stats.f1),
            },
            "best": {
                "best_thresh": float(per_class_best[cls]["best_thresh"]),
                "precision": float(per_class_best[cls]["precision"]),
                "recall": float(per_class_best[cls]["recall"]),
                "f1": float(per_class_best[cls]["f1"]),
            },
        }

    out = {
        "per_class": per_class_out,
        "macro": {
            "precision": macro_p,
            "recall": macro_r,
            "f1": macro_f1,
        },
        "micro": {
            "TP": int(micro_stats.tp),
            "FP": int(micro_stats.fp),
            "FN": int(micro_stats.fn),
            "precision": float(micro_stats.precision),
            "recall": float(micro_stats.recall),
            "f1": float(micro_stats.f1),
        },
    }
    return out


def build_confusion_matrix(
    per_class_fixed: Dict[int, PRStats],
    class_names: Optional[Dict[int, str]] = None,
) -> Dict[str, object]:
    """
    Build a simple confusion-matrix-style summary from fixed-threshold stats.

    This does NOT attempt to distinguish which wrong class a false positive
    belongs to – it only provides TP / FP / FN counts per class plus totals.
    This is sufficient for many small-class OCT use cases.
    """
    per_class_out: Dict[str, Dict[str, int]] = {}
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for cls, stats in per_class_fixed.items():
        name = class_names.get(cls, f"class_{cls}") if class_names else f"class_{cls}"
        per_class_out[name] = {
            "TP": int(stats.tp),
            "FP": int(stats.fp),
            "FN": int(stats.fn),
        }
        total_tp += stats.tp
        total_fp += stats.fp
        total_fn += stats.fn

    global_counts = {
        "TP": int(total_tp),
        "FP": int(total_fp),
        "FN": int(total_fn),
    }

    return {
        "per_class": per_class_out,
        "global": global_counts,
    }

