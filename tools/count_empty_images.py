#!/usr/bin/env python3
"""
Count % of train and val images that have 0 ground-truth boxes (empties).
Uses the same dataset build as training, so filtering (ignore label 2, etc.) is identical.

Usage (same args as training):
  python tools/count_empty_images.py \
    --dataset_file oct_pkl \
    --oct_pkl_root /path/to/pickle \
    --oct_train_list /path/to/train_split.txt \
    --oct_val_list /path/to/val_split.txt
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from datasets import build_dataset
from main import get_args_parser


def main():
    parser = get_args_parser()
    args = parser.parse_args()

    if getattr(args, "dataset_file", None) != "oct_pkl":
        print("This script is for dataset_file=oct_pkl. Set --dataset_file oct_pkl and OCT paths.")
        sys.exit(1)

    args.num_classes = 2

    train = build_dataset("train", args)
    val = build_dataset("val", args)

    def count_empty(dataset):
        n_empty = 0
        for i in range(len(dataset)):
            _, target = dataset[i]
            if target["boxes"].numel() == 0 or target["boxes"].shape[0] == 0:
                n_empty += 1
        return n_empty, len(dataset)

    n_empty_train, n_train = count_empty(train)
    n_empty_val, n_val = count_empty(val)

    pct_train = 100.0 * n_empty_train / n_train if n_train else 0.0
    pct_val = 100.0 * n_empty_val / n_val if n_val else 0.0

    print("Empty images (0 GT boxes)")
    print(f"  Train: {n_empty_train} / {n_train}  =  {pct_train:.1f}%")
    print(f"  Val:   {n_empty_val} / {n_val}  =  {pct_val:.1f}%")


if __name__ == "__main__":
    main()
