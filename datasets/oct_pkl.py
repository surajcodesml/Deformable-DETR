"""
OCT B-scan detection dataset backed by Nemours pickle files.

This integrates the Nemours OCT format (as used in RCNN-OCT) into
Deformable-DETR while reusing the standard COCO-style transforms and
COCO evaluator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pickle

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from pycocotools.coco import COCO

from .coco import make_coco_transforms


CLASSES: Dict[int, str] = {
    1: "fovea",
    2: "SCR",
}


def _load_pickle(path: Path) -> dict:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _extract_arrays(sample: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract image, boxes, and labels arrays from a loaded pickle sample.

    The Nemours pickles follow the conventions used in RCNN-OCT:
    - image/img: numpy array (H, W), (H, W, 1), (H, W, 3), (1, H, W), or (3, H, W)
                 values are typically float32 in [0, 1] or uint8 in [0, 255]
    - box/boxes/bboxes: numpy array (N, 4) with normalized [cx, cy, w, h] in [0, 1]
    - label/labels: numpy array (N,) with integer labels {0=fovea, 1=SCR, 2=ignore}
    """
    if not isinstance(sample, dict):
        raise TypeError(f"Expected sample dict, got {type(sample)}")

    # image
    if "image" in sample:
        image = sample["image"]
    elif "img" in sample:
        image = sample["img"]
    else:
        raise KeyError("Sample must contain 'image' or 'img' key")

    if hasattr(image, "numpy"):
        image = image.numpy()
    image_np = np.asarray(image, dtype=np.float32)

    # boxes
    if "boxes" in sample:
        boxes = sample["boxes"]
    elif "bboxes" in sample:
        boxes = sample["bboxes"]
    elif "box" in sample:
        boxes = sample["box"]
    else:
        boxes = np.zeros((0, 4), dtype=np.float32)

    if hasattr(boxes, "numpy"):
        boxes = boxes.numpy()
    boxes_np = np.asarray(boxes, dtype=np.float32)

    # labels
    if "labels" in sample:
        labels = sample["labels"]
    elif "label" in sample:
        labels = sample["label"]
    else:
        labels = np.zeros((boxes_np.shape[0],), dtype=np.int64)

    if hasattr(labels, "numpy"):
        labels = labels.numpy()
    labels_np = np.asarray(labels, dtype=np.int64)

    return image_np, boxes_np, labels_np


def _to_pil_rgb(image_np: np.ndarray) -> Image.Image:
    """
    Convert a numpy image array to a 3-channel RGB PIL.Image.
    """
    arr = image_np
    if arr.ndim == 3:
        # HWC or CHW
        if arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
            # Likely CHW -> convert to HWC
            arr = np.transpose(arr, (1, 2, 0))
    elif arr.ndim == 2:
        # (H, W) -> (H, W, 1)
        arr = arr[:, :, None]

    # Now arr is HWC or HWC-like
    if arr.dtype != np.uint8:
        # Assume float in [0, 1] or arbitrary float; clamp and scale to [0, 255]
        arr = np.clip(arr, 0.0, 1.0) * 255.0
        arr = arr.astype(np.uint8)

    if arr.shape[2] == 1:
        pil_img = Image.fromarray(arr[:, :, 0], mode="L").convert("RGB")
    elif arr.shape[2] == 3:
        pil_img = Image.fromarray(arr, mode="RGB")
    else:
        raise ValueError(f"Unexpected channel dimension in image array: {arr.shape}")
    return pil_img


def _convert_boxes_cxcywh_norm_to_xyxy_abs(
    boxes: np.ndarray, height: int, width: int
) -> np.ndarray:
    """
    Convert normalized [cx, cy, w, h] in [0, 1] to absolute [x1, y1, x2, y2] pixels.
    """
    if boxes.size == 0:
        return boxes.astype(np.float32).reshape(0, 4)

    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = (cx - w / 2.0) * width
    y1 = (cy - h / 2.0) * height
    x2 = (cx + w / 2.0) * width
    y2 = (cy + h / 2.0) * height

    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
    boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, width)
    boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, height)
    boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, width)
    boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, height)
    return boxes_xyxy


def _parse_volume_id(path: Path) -> str:
    """
    Parse a volume_id from a pickle filename.

    Example:
        3_L_3004.pkl -> stem '3_L_3004'
                      -> strip last 3 chars -> '3_L_3'
                      -> volume_id='3_L_3'
    """
    stem = path.stem
    if len(stem) <= 3:
        return stem
    return stem[:-3]


def _load_split_entries(split_path: Path) -> List[str]:
    """
    Load split definition file. Supports simple TXT or CSV with one entry per line.

    Each line may contain either:
      - a volume_id (e.g., '3_L_3'), or
      - a .pkl filename (e.g., '3_L_3004.pkl')
    """
    entries: List[str] = []
    with split_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # If CSV, take the first column
            if "," in line:
                line = line.split(",", 1)[0].strip()
            entries.append(line)
    return entries


def _index_pkl_files(root: Path) -> Tuple[Dict[str, List[Path]], Dict[str, Path]]:
    """
    Index all .pkl files under root.

    Returns:
        volume_to_files: volume_id -> sorted list of paths
        name_to_path: basename (with .pkl) -> Path
    """
    volume_to_files: Dict[str, List[Path]] = {}
    name_to_path: Dict[str, Path] = {}

    for path in sorted(root.rglob("*.pkl")):
        name_to_path[path.name] = path
        vol_id = _parse_volume_id(path)
        volume_to_files.setdefault(vol_id, []).append(path)

    # sort each volume's files to have deterministic order
    for vol_files in volume_to_files.values():
        vol_files.sort()

    return volume_to_files, name_to_path


def _expand_split(
    entries: Sequence[str],
    volume_to_files: Dict[str, List[Path]],
    name_to_path: Dict[str, Path],
    split_name: str,
) -> List[Path]:
    """
    Expand a list of volume_ids or filenames into concrete .pkl paths.
    """
    result: List[Path] = []
    for entry in entries:
        if entry.endswith(".pkl"):
            if entry not in name_to_path:
                raise FileNotFoundError(
                    f"Entry '{entry}' in split '{split_name}' not found under OCT root"
                )
            result.append(name_to_path[entry])
        else:
            # treat as volume_id
            if entry not in volume_to_files:
                raise KeyError(
                    f"Volume_id '{entry}' in split '{split_name}' has no matching .pkl files"
                )
            result.extend(volume_to_files[entry])
    return sorted(result)


def _check_volume_leakage(
    train_vols: Iterable[str],
    val_vols: Iterable[str],
    test_vols: Iterable[str],
) -> None:
    """
    Ensure that no volume_id appears in more than one split.
    """
    train_set = set(train_vols)
    val_set = set(val_vols)
    test_set = set(test_vols)

    def _nonempty_intersection(a: set, b: set) -> Optional[set]:
        inter = a.intersection(b)
        return inter if inter else None

    leak_train_val = _nonempty_intersection(train_set, val_set)
    leak_train_test = _nonempty_intersection(train_set, test_set)
    leak_val_test = _nonempty_intersection(val_set, test_set)

    if leak_train_val or leak_train_test or leak_val_test:
        msgs: List[str] = []
        if leak_train_val:
            msgs.append(f"train/val: {sorted(leak_train_val)}")
        if leak_train_test:
            msgs.append(f"train/test: {sorted(leak_train_test)}")
        if leak_val_test:
            msgs.append(f"val/test: {sorted(leak_val_test)}")
        raise RuntimeError(
            "Volume leakage detected across splits (same volume_id in multiple splits): "
            + "; ".join(msgs)
        )


def _build_splits(root: Path, args) -> Dict[str, List[Path]]:
    """
    Build train/val/test file lists from split definition files and enforce
    volume-wise separation.
    """
    volume_to_files, name_to_path = _index_pkl_files(root)

    if getattr(args, "oct_train_list", None) is None or getattr(
        args, "oct_val_list", None
    ) is None:
        raise ValueError(
            "For dataset_file=oct_pkl you must provide --oct_train_list and --oct_val_list"
        )

    train_entries = _load_split_entries(Path(args.oct_train_list))
    val_entries = _load_split_entries(Path(args.oct_val_list))
    test_entries: List[str] = []
    if getattr(args, "oct_test_list", None):
        test_entries = _load_split_entries(Path(args.oct_test_list))

    train_files = _expand_split(train_entries, volume_to_files, name_to_path, "train")
    val_files = _expand_split(val_entries, volume_to_files, name_to_path, "val")
    test_files: List[Path] = []
    if test_entries:
        test_files = _expand_split(test_entries, volume_to_files, name_to_path, "test")

    # volume sets per split for leakage checking
    train_vols = {_parse_volume_id(p) for p in train_files}
    val_vols = {_parse_volume_id(p) for p in val_files}
    test_vols = {_parse_volume_id(p) for p in test_files}
    _check_volume_leakage(train_vols, val_vols, test_vols)

    splits: Dict[str, List[Path]] = {"train": train_files, "val": val_files}
    if test_files:
        splits["test"] = test_files
    return splits


def _build_coco_api(files: Sequence[Path], image_ids: Sequence[int]) -> COCO:
    """
    Build an in-memory COCO API object from the OCT pickle files for COCO-style
    mAP evaluation.
    """
    assert len(files) == len(image_ids)

    images: List[dict] = []
    annotations: List[dict] = []

    ann_id = 1
    for img_id, path in zip(image_ids, files):
        sample = _load_pickle(path)
        image_np, boxes_np, labels_np = _extract_arrays(sample)

        # normalize shapes
        if image_np.ndim == 2:
            h, w = int(image_np.shape[0]), int(image_np.shape[1])
        elif image_np.ndim == 3:
            if image_np.shape[0] in (1, 3) and image_np.shape[2] not in (1, 3):
                # CHW
                h, w = int(image_np.shape[1]), int(image_np.shape[2])
            else:
                # HWC
                h, w = int(image_np.shape[0]), int(image_np.shape[1])
        else:
            raise ValueError(f"Unexpected image shape for {path}: {image_np.shape}")

        # filter boxes/labels (remove zeros and ignore label=2) then convert to xyxy
        if boxes_np.size == 0:
            boxes_xyxy = boxes_np.reshape(0, 4)
            labels = labels_np.reshape(0)
        else:
            # remove zero boxes
            zero_boxes = np.all(boxes_np == 0, axis=1)
            labels = labels_np.copy()
            valid_mask = ~zero_boxes
            # remove ignore label=2
            valid_mask &= labels != 2
            boxes_np = boxes_np[valid_mask]
            labels = labels[valid_mask]
            boxes_xyxy = _convert_boxes_cxcywh_norm_to_xyxy_abs(boxes_np, h, w)

        images.append(
            {
                "id": int(img_id),
                "width": w,
                "height": h,
                "file_name": str(path.name),
            }
        )

        for box, label in zip(boxes_xyxy, labels):
            # map original labels {0,1} to {1,2}
            if int(label) not in (0, 1):
                continue
            coco_label = int(label) + 1
            x1, y1, x2, y2 = box.tolist()
            w_box = float(x2 - x1)
            h_box = float(y2 - y1)
            area = max(w_box, 0.0) * max(h_box, 0.0)
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": int(img_id),
                    "category_id": coco_label,
                    "bbox": [float(x1), float(y1), float(w_box), float(h_box)],
                    "area": float(area),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    categories = [
        {"id": cid, "name": name, "supercategory": "oct"}
        for cid, name in CLASSES.items()
    ]

    coco_dict = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    coco = COCO()
    coco.dataset = coco_dict
    coco.createIndex()
    return coco


class OctPKLDataset(Dataset):
    """
    OCT B-scan dataset for Deformable-DETR.

    __getitem__ returns:
      - image: PIL.Image in RGB
      - target: dict with COCO-style fields:
            boxes: Tensor [N,4] absolute xyxy in pixels
            labels: Tensor [N] with values in {1,2}
            image_id: Tensor [1]
            area: Tensor [N]
            iscrowd: Tensor [N]
            orig_size: Tensor [2] [h, w]
            size: Tensor [2] [h, w]

    The transforms pipeline will later convert boxes to normalized cxcywh.
    """

    def __init__(
        self,
        files: Sequence[Path],
        image_ids: Sequence[int],
        transforms,
        coco_gt: Optional[COCO],
    ) -> None:
        assert len(files) == len(image_ids)
        self.files: List[Path] = list(files)
        self.ids: List[int] = list(int(i) for i in image_ids)
        self._transforms = transforms
        # Expose COCO API for get_coco_api_from_dataset
        self.coco: Optional[COCO] = coco_gt

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        sample = _load_pickle(path)
        image_np, boxes_np, labels_np = _extract_arrays(sample)

        # determine image size
        if image_np.ndim == 2:
            h, w = int(image_np.shape[0]), int(image_np.shape[1])
        elif image_np.ndim == 3:
            if image_np.shape[0] in (1, 3) and image_np.shape[2] not in (1, 3):
                h, w = int(image_np.shape[1]), int(image_np.shape[2])
            else:
                h, w = int(image_np.shape[0]), int(image_np.shape[1])
        else:
            raise ValueError(f"Unexpected image shape for {path}: {image_np.shape}")

        # filter invalid / ignore boxes
        if boxes_np.size == 0:
            boxes_xyxy = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)
        else:
            zero_boxes = np.all(boxes_np == 0, axis=1)
            labels = labels_np.copy()
            valid_mask = ~zero_boxes
            valid_mask &= labels != 2  # drop ignore label
            boxes_np = boxes_np[valid_mask]
            labels = labels[valid_mask]
            boxes_xyxy = _convert_boxes_cxcywh_norm_to_xyxy_abs(boxes_np, h, w)

        # map labels {0,1} -> {1,2}
        if labels.size > 0:
            labels = np.array([int(lbl) + 1 for lbl in labels], dtype=np.int64)
        else:
            labels = np.zeros((0,), dtype=np.int64)

        # build target tensors
        boxes_tensor = torch.as_tensor(boxes_xyxy, dtype=torch.float32)
        labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
        if boxes_tensor.numel() == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)

        if boxes_tensor.numel() > 0:
            area = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (
                boxes_tensor[:, 3] - boxes_tensor[:, 1]
            )
        else:
            area = torch.zeros((0,), dtype=torch.float32)

        image_id = torch.tensor([self.ids[idx]], dtype=torch.int64)
        iscrowd = torch.zeros((labels_tensor.shape[0],), dtype=torch.int64)
        size = torch.as_tensor([int(h), int(w)], dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
            "orig_size": size,
            "size": size,
        }

        image = _to_pil_rgb(image_np)

        if self._transforms is not None:
            image, target = self._transforms(image, target)

        return image, target


def build(image_set: str, args):
    """
    Build OCT pickle dataset for the given image_set ('train', 'val', or 'test').
    """
    root = Path(args.oct_pkl_root)
    assert root.exists(), f"provided OCT pickle root {root} does not exist"

    splits = _build_splits(root, args)
    if image_set not in splits:
        raise ValueError(f"Unknown image_set '{image_set}' for oct_pkl (have {list(splits.keys())})")

    files = splits[image_set]
    # deterministically assign image_ids based on global index within this split
    image_ids = list(range(len(files)))

    transforms = make_coco_transforms("train" if image_set == "train" else "val")

    # For COCO-style evaluation (mAP), we build a COCO API for the validation
    # (and test, if present) splits. Training split does not need COCO.
    coco_gt = None
    if image_set in ("val", "test"):
        coco_gt = _build_coco_api(files, image_ids)

    dataset = OctPKLDataset(files=files, image_ids=image_ids, transforms=transforms, coco_gt=coco_gt)
    return dataset

