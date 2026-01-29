#!/usr/bin/env python3
"""
Helper script to create train/val/test split files for OCT .pkl dataset.

This script scans a directory of .pkl files, groups them by volume_id,
and creates split files that can be used with --oct_train_list, --oct_val_list, --oct_test_list.

Usage:
    python tools/create_oct_splits.py \
        --pkl_root /path/to/pickle/directory \
        --output_dir /path/to/output/splits \
        --train_ratio 0.7 \
        --val_ratio 0.15 \
        --test_ratio 0.15 \
        --seed 42

The script will create:
    - train_split.txt (volume IDs or filenames)
    - val_split.txt
    - test_split.txt

You can also manually edit these files or provide your own split files.
"""

import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set


def parse_volume_id(filename: str) -> str:
    """
    Parse volume_id from a pickle filename.
    
    Example: 3_L_3004.pkl -> 3_L_3 (strips last 3 chars from stem)
    """
    stem = Path(filename).stem
    if len(stem) <= 3:
        return stem
    return stem[:-3]


def scan_pkl_files(pkl_root: Path) -> Dict[str, List[str]]:
    """
    Scan directory for .pkl files and group by volume_id.
    
    Returns:
        volume_to_files: dict mapping volume_id -> sorted list of filenames
    """
    volume_to_files: Dict[str, List[str]] = defaultdict(list)
    
    for pkl_path in sorted(pkl_root.rglob("*.pkl")):
        filename = pkl_path.name
        volume_id = parse_volume_id(filename)
        volume_to_files[volume_id].append(filename)
    
    # Sort files within each volume for deterministic ordering
    for vol_id in volume_to_files:
        volume_to_files[vol_id].sort()
    
    return volume_to_files


def create_splits_by_volume(
    volume_to_files: Dict[str, List[str]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    use_filenames: bool = False,
) -> tuple[List[str], List[str], List[str]]:
    """
    Split volumes into train/val/test sets.
    
    Args:
        volume_to_files: dict mapping volume_id -> list of filenames
        train_ratio, val_ratio, test_ratio: proportions (should sum to ~1.0)
        seed: random seed for reproducibility
        use_filenames: if True, write filenames; if False, write volume_ids
    
    Returns:
        (train_entries, val_entries, test_entries)
    """
    volumes = sorted(volume_to_files.keys())
    rng = random.Random(seed)
    rng.shuffle(volumes)
    
    n_total = len(volumes)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    # test gets the remainder
    
    train_vols = volumes[:n_train]
    val_vols = volumes[n_train : n_train + n_val]
    test_vols = volumes[n_train + n_val :]
    
    print(f"Split summary:")
    print(f"  Total volumes: {n_total}")
    print(f"  Train: {len(train_vols)} volumes")
    print(f"  Val: {len(val_vols)} volumes")
    print(f"  Test: {len(test_vols)} volumes")
    
    def expand_to_entries(vol_list: List[str]) -> List[str]:
        if use_filenames:
            entries = []
            for vol_id in vol_list:
                entries.extend(volume_to_files[vol_id])
            return sorted(entries)
        else:
            return sorted(vol_list)
    
    train_entries = expand_to_entries(train_vols)
    val_entries = expand_to_entries(val_vols)
    test_entries = expand_to_entries(test_vols)
    
    print(f"\nFile counts:")
    print(f"  Train: {len(train_entries)} entries")
    print(f"  Val: {len(val_entries)} entries")
    print(f"  Test: {len(test_entries)} entries")
    
    return train_entries, val_entries, test_entries


def write_split_file(output_path: Path, entries: List[str]):
    """Write a split file (one entry per line)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for entry in entries:
            f.write(f"{entry}\n")
    print(f"  Wrote {len(entries)} entries to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create train/val/test split files for OCT .pkl dataset"
    )
    parser.add_argument(
        "--pkl_root",
        type=str,
        required=True,
        help="Root directory containing .pkl files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write split files (train_split.txt, val_split.txt, test_split.txt)",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Proportion of volumes for training (default: 0.7)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Proportion of volumes for validation (default: 0.15)",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.15,
        help="Proportion of volumes for testing (default: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--use_filenames",
        action="store_true",
        help="Write individual .pkl filenames instead of volume_ids",
    )
    
    args = parser.parse_args()
    
    pkl_root = Path(args.pkl_root)
    if not pkl_root.exists():
        raise FileNotFoundError(f"PKL root directory not found: {pkl_root}")
    
    output_dir = Path(args.output_dir)
    
    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        print(f"Warning: Ratios sum to {total_ratio:.3f}, not 1.0")
    
    print(f"Scanning .pkl files in: {pkl_root}")
    volume_to_files = scan_pkl_files(pkl_root)
    
    print(f"\nFound {len(volume_to_files)} unique volume IDs")
    print(f"Total .pkl files: {sum(len(files) for files in volume_to_files.values())}")
    
    # Create splits
    train_entries, val_entries, test_entries = create_splits_by_volume(
        volume_to_files,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed,
        use_filenames=args.use_filenames,
    )
    
    # Write split files
    print(f"\nWriting split files to: {output_dir}")
    write_split_file(output_dir / "train_split.txt", train_entries)
    write_split_file(output_dir / "val_split.txt", val_entries)
    write_split_file(output_dir / "test_split.txt", test_entries)
    
    print("\n✓ Split files created successfully!")
    print(f"\nTo use these splits, set:")
    print(f"  --oct_train_list {output_dir / 'train_split.txt'}")
    print(f"  --oct_val_list {output_dir / 'val_split.txt'}")
    if test_entries:
        print(f"  --oct_test_list {output_dir / 'test_split.txt'}")


if __name__ == "__main__":
    main()
