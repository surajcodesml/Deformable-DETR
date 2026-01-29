#!/usr/bin/env python3
"""
Quick check: count predicted boxes above score thresholds on a small val subset.
Use this to detect "everything is object" (no-object handling broken).

Run on 20 val images and report counts at score > 0.05, 0.1, 0.3, 0.5.
If you see 100-300 boxes even at 0.3/0.5, no-object is likely broken or logits misinterpreted.

Usage (on server, same paths as training):
  python tools/check_prediction_counts.py \
    --dataset_file oct_pkl \
    --oct_pkl_root /path/to/pickle \
    --oct_train_list /path/to/train_split.txt \
    --oct_val_list /path/to/val_split.txt \
    --resume outputs/oct_pkl_slurm_<JOB_ID>/checkpoint_best.pth \
    --num_images 20
"""

import argparse
import sys
from pathlib import Path

import torch

# add repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from datasets import build_dataset
from main import get_args_parser
from models import build_model
import util.misc as utils


def main():
    parser = get_args_parser()
    parser.add_argument("--num_images", type=int, default=20, help="Number of val images to run (default 20)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.dataset_file != "oct_pkl":
        print("This script is for dataset_file=oct_pkl. Set --dataset_file oct_pkl and OCT paths.")
        sys.exit(1)

    args.num_classes = 2
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    dataset_val = build_dataset(image_set="val", args=args)
    model, criterion, postprocessors = build_model(args)
    model.to(device)
    model.eval()

    if not getattr(args, "resume", None):
        print("Warning: no --resume checkpoint; running with random weights (counts will be meaningless).")
    else:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=False)
        print(f"Loaded checkpoint: {args.resume}")

    import random
    rng = random.Random(args.seed)
    n = min(args.num_images, len(dataset_val))
    indices = rng.sample(range(len(dataset_val)), n)

    thresholds = [0.05, 0.1, 0.3, 0.5]
    counts_per_thresh = {t: [] for t in thresholds}  # per-image counts

    for idx in indices:
        image, target = dataset_val[idx]
        with torch.no_grad():
            samples = utils.nested_tensor_from_tensor_list([image.to(device)])
            outputs = model(samples)
            orig_sizes = torch.as_tensor([[image.shape[1], image.shape[2]]], device=device)
            results = postprocessors["bbox"](outputs, orig_sizes)[0]
        scores = results["scores"].cpu()
        for t in thresholds:
            counts_per_thresh[t].append((scores >= t).sum().item())

    print("\n--- Prediction count check (no-object sanity) ---")
    print(f"Dataset: oct_pkl, num_images: {n}")
    print("Count = number of predicted boxes with score > threshold (per image).")
    print("If counts are 100-300 even at 0.3/0.5, no-object handling may be broken.\n")
    for t in thresholds:
        c = counts_per_thresh[t]
        mean_c = sum(c) / len(c)
        print(f"  score > {t:.2f}:  mean={mean_c:.1f}  min={min(c)}  max={max(c)}  (per image)")
    print("\nExpected for healthy model: low mean at 0.3 and 0.5 (e.g. < 20).")
    print("(GT has typically 0-2 objects per OCT image.)")


if __name__ == "__main__":
    main()
