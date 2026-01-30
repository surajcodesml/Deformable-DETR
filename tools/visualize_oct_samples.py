import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from datasets import build_dataset
from main import get_args_parser
from models import build_model
import util.misc as utils
from util import box_ops


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _denormalize_image(t: torch.Tensor) -> np.ndarray:
    assert t.ndim == 3
    img = t.detach().cpu().numpy()
    img = (img * IMAGENET_STD[:, None, None]) + IMAGENET_MEAN[:, None, None]
    img = np.clip(img, 0.0, 1.0)
    img = (img * 255.0).astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))
    return img


def _plot_image_with_boxes(
    image: torch.Tensor,
    gt_boxes_cxcywh: torch.Tensor,
    gt_labels: torch.Tensor,
    pred_boxes_xyxy: torch.Tensor = None,
    pred_scores: torch.Tensor = None,
    pred_labels: torch.Tensor = None,
    score_thresh: float = 0.5,
    class_names=None,
    title: str = "",
    save_path: Path = None,
    vis_topk_fallback: int = 10,
):
    img_hwc = _denormalize_image(image)
    h, w, _ = img_hwc.shape

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(img_hwc)
    ax.axis("off")
    ax.set_title(title)

    # GT boxes: normalized cxcywh
    if gt_boxes_cxcywh is not None and gt_boxes_cxcywh.numel() > 0:
        scale = torch.tensor([w, h, w, h], dtype=torch.float32, device=gt_boxes_cxcywh.device)
        gt_boxes_xyxy = box_ops.box_cxcywh_to_xyxy(gt_boxes_cxcywh * scale)
        for box, label in zip(gt_boxes_xyxy, gt_labels):
            x1, y1, x2, y2 = box.tolist()
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="lime", linewidth=2)
            ax.add_patch(rect)
            cls_id = int(label.item())
            name = class_names.get(cls_id, str(cls_id)) if class_names else str(cls_id)
            ax.text(
                x1,
                y1 - 2,
                f"GT: {name}",
                fontsize=8,
                color="lime",
                bbox=dict(facecolor="black", alpha=0.5, pad=1),
            )

    # Predictions (red, score text). PostProcess returns top-100 by score; filter by score_thresh.
    # If none above threshold (common early in training), show top vis_topk_fallback so user sees the model is predicting.
    if pred_boxes_xyxy is not None and pred_boxes_xyxy.numel() > 0 and pred_scores is not None:
        keep = pred_scores >= score_thresh
        n_above = keep.sum().item()
        used_fallback = False
        if n_above == 0:
            k = min(vis_topk_fallback, pred_scores.shape[0])
            _, top_idx = pred_scores.topk(k, largest=True)
            keep = torch.zeros_like(keep)
            keep[top_idx] = True
            used_fallback = True
        pred_labels_use = pred_labels if pred_labels is not None else torch.zeros(pred_boxes_xyxy.shape[0], dtype=torch.long)
        for box, score, label in zip(
            pred_boxes_xyxy[keep],
            pred_scores[keep],
            pred_labels_use[keep],
        ):
            x1, y1, x2, y2 = box.tolist()
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="red", linewidth=2)
            ax.add_patch(rect)
            cls_id = int(label.item())
            name = class_names.get(cls_id, str(cls_id)) if class_names else str(cls_id)
            ax.text(
                x1,
                y2 + 4,
                f"{name} {float(score):.2f}",
                fontsize=8,
                color="red",
                bbox=dict(facecolor="black", alpha=0.5, pad=1),
            )
        if used_fallback:
            max_s = pred_scores.max().item()
            ax.set_title(f"{title}  [no pred ≥{score_thresh}; showing top {keep.sum().item()}, max={max_s:.2f}]")

    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def main():
    parser = get_args_parser()
    parser.add_argument("--vis_num_samples", default=10, type=int, help="Number of val images to visualize (default 10)")
    parser.add_argument("--vis_seed", default=0, type=int)
    parser.add_argument("--vis_checkpoint", default="", type=str, help="Path to checkpoint .pth (required for predictions)")
    parser.add_argument("--vis_score_thresh", default=0.3, type=float, help="Score threshold for drawing pred boxes (default 0.3)")
    parser.add_argument("--vis_topk_fallback", default=10, type=int, help="If no pred above thresh, show this many top-score boxes (default 10)")

    args = parser.parse_args()

    assert args.dataset_file == "oct_pkl", (
        "tools/visualize_oct_samples.py is designed for the OCT .pkl dataset. "
        "Please pass --dataset_file oct_pkl and the corresponding OCT arguments."
    )

    args.num_classes = 2

    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    dataset_val = build_dataset(image_set="val", args=args)

    model, criterion, postprocessors = build_model(args)
    model.to(device)
    model.eval()

    if args.vis_checkpoint:
        checkpoint = torch.load(args.vis_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)

    num_samples = min(args.vis_num_samples, len(dataset_val))
    rng = random.Random(args.vis_seed)
    indices = list(range(len(dataset_val)))
    rng.shuffle(indices)
    indices = indices[:num_samples]

    class_names = {0: "fovea", 1: "SCR"}

    output_dir = Path(args.output_dir) if args.output_dir else Path(".")
    sanity_dir = output_dir / "sanity"

    print(f"Visualizing {len(indices)} samples into {sanity_dir}")

    for idx in indices:
        image, target = dataset_val[idx]
        img_id = target["image_id"].item()
        gt_boxes = target["boxes"]
        gt_labels = target["labels"]

        pred_boxes_xyxy = None
        pred_scores = None
        pred_labels = None

        if args.vis_checkpoint:
            samples = utils.nested_tensor_from_tensor_list([image.to(device)])
            outputs = model(samples)
            orig_target_sizes = torch.as_tensor([[image.shape[1], image.shape[2]]], device=device)
            results = postprocessors["bbox"](outputs, orig_target_sizes)[0]
            pred_boxes_xyxy = results["boxes"].detach().cpu()
            pred_scores = results["scores"].detach().cpu()
            pred_labels = results["labels"].detach().cpu()

        save_path = sanity_dir / f"val_idx{idx:05d}_img{img_id:06d}.png"
        title = f"idx={idx}, image_id={img_id}"
        _plot_image_with_boxes(
            image=image,
            gt_boxes_cxcywh=gt_boxes,
            gt_labels=gt_labels,
            pred_boxes_xyxy=pred_boxes_xyxy,
            pred_scores=pred_scores,
            pred_labels=pred_labels,
            score_thresh=args.vis_score_thresh,
            class_names=class_names,
            title=title,
            save_path=save_path,
            vis_topk_fallback=args.vis_topk_fallback,
        )


if __name__ == "__main__":
    main()