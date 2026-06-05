# filename: src/data/dataset.py
# purpose:  PyTorch Dataset for SmartVision classification (NUM_CLASSES from config)
# version:  2.0

import logging
import platform
from pathlib import Path
from typing import Optional

from PIL import Image
from torch.utils.data import DataLoader, Dataset

# Windows requires num_workers=0 (no fork); Linux/Colab can use 2 workers
_NUM_WORKERS = 0 if platform.system() == "Windows" else 2

from config import (
    CLASS_TO_IDX,
    CLASSES,
    DATA_PROCESSED_DIR,
    MODEL_CONFIGS,
    RANDOM_STATE,
    TRAIN_SPLIT,
    VAL_SPLIT,
)

logger = logging.getLogger(__name__)

# Session-level cache: guarantees identical splits across multiple get_dataloaders()
# calls. Key = (str(data_root), random_state).
_split_cache: dict[tuple, dict] = {}


def create_stratified_split(
    data_root: Path,
    random_state: int = RANDOM_STATE,
) -> dict[str, list[tuple[Path, int]]]:
    """
    Build a stratified random split from the existing on-disk split directories.

    Reads ALL images from train/ + val/ + test/ subdirs, shuffles per class
    with a fixed seed, then re-splits 70/15/15. Uses round() not int() to avoid
    silently dropping images for class counts not divisible by 20.

    Integrity checks (in order):
      1. Completeness: len(assigned) == len(all_samples)  — catches round() edge cases
      2. Uniqueness:   no path appears twice              — structural guarantee
      3. Cross-split assertions (redundant by construction; kept as documentation)

    Caches by (data_root, random_state): identical splits across get_dataloaders() calls.

    Returns:
        {'train': [(Path, int), ...], 'val': [...], 'test': [...]}
    """
    cache_key = (str(data_root), random_state)
    if cache_key in _split_cache:
        logger.debug("create_stratified_split: returning cached split")
        return _split_cache[cache_key]

    import random as _random

    all_samples: list[tuple[Path, int]] = []
    for split_name in ("train", "val", "test"):
        for class_name in sorted(CLASS_TO_IDX.keys()):
            class_dir = data_root / split_name / class_name
            if not class_dir.exists():
                continue
            class_idx = CLASS_TO_IDX[class_name]
            for img_path in sorted(class_dir.glob("*.jpg")):
                all_samples.append((img_path, class_idx))

    rng = _random.Random(random_state)
    splits: dict[str, list] = {"train": [], "val": [], "test": []}
    per_class: dict[int, list] = {}
    for item in all_samples:
        per_class.setdefault(item[1], []).append(item)

    for class_idx, items in per_class.items():
        rng.shuffle(items)
        n         = len(items)
        train_end = round(n * TRAIN_SPLIT)
        val_end   = round(n * (TRAIN_SPLIT + VAL_SPLIT))
        splits["train"].extend(items[:train_end])
        splits["val"].extend(items[train_end:val_end])
        splits["test"].extend(items[val_end:])

    all_assigned   = splits["train"] + splits["val"] + splits["test"]
    assigned_paths = [str(p) for p, _ in all_assigned]
    assert len(assigned_paths) == len(all_samples), (
        f"Split dropped {len(all_samples) - len(assigned_paths)} images. "
        f"total={len(all_samples)} assigned={len(assigned_paths)}"
    )
    assert len(set(assigned_paths)) == len(assigned_paths), (
        f"Duplicate paths: {len(assigned_paths) - len(set(assigned_paths))} duplicates"
    )
    train_set = {str(p) for p, _ in splits["train"]}
    val_set   = {str(p) for p, _ in splits["val"]}
    test_set  = {str(p) for p, _ in splits["test"]}
    assert not (train_set & val_set),  f"Train/val overlap: {len(train_set & val_set)}"
    assert not (train_set & test_set), f"Train/test overlap: {len(train_set & test_set)}"
    assert not (val_set   & test_set), f"Val/test overlap: {len(val_set & test_set)}"

    logger.info(
        f"create_stratified_split: train={len(splits['train'])} "
        f"val={len(splits['val'])} test={len(splits['test'])}"
    )
    _split_cache[cache_key] = splits
    return splits


class SmartVisionDataset(Dataset):
    """
    Classification dataset: one cropped 224×224 image per sample, one class label.
    Folder structure: {root}/{split}/{class_name}/*.jpg
    """

    def __init__(
        self,
        split: str,
        transform=None,
        root: Optional[Path] = None,
        samples: Optional[list] = None,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test'; got {split!r}")
        self.split     = split
        self.transform = transform
        self.root      = Path(root) if root else DATA_PROCESSED_DIR / "classification"

        if samples is not None:
            self.samples = list(samples)   # shallow copy — (Path, int) tuples are immutable
            logger.info(
                f"SmartVisionDataset [{self.split}]: {len(self.samples)} samples (pre-split)"
            )
        else:
            self.samples: list[tuple[Path, int]] = []
            self._load_samples()

    def _load_samples(self) -> None:
        split_dir = self.root / self.split
        if not split_dir.exists():
            logger.warning(f"Split directory not found: {split_dir}")
            return

        for class_name in sorted(CLASS_TO_IDX.keys()):
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue
            class_idx = CLASS_TO_IDX[class_name]
            for img_path in sorted(class_dir.glob("*.jpg")):
                self.samples.append((img_path, class_idx))

        logger.info(f"SmartVisionDataset [{self.split}]: {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, class_idx = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, class_idx

    def class_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {cls: 0 for cls in CLASSES}
        for _, class_idx in self.samples:
            counts[CLASSES[class_idx]] += 1
        return counts


def get_dataloaders(
    model_name: str,
    train_transform,
    eval_transform,
    fast_mode: bool = True,
    use_random_split: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Returns (train_loader, val_loader, test_loader) for the given model."""
    batch_size = MODEL_CONFIGS[model_name]["batch"]

    if use_random_split:
        root       = DATA_PROCESSED_DIR / "classification"
        all_splits = create_stratified_split(root)
        train_ds   = SmartVisionDataset("train", transform=train_transform,
                                        samples=all_splits["train"])
        val_ds     = SmartVisionDataset("val",   transform=eval_transform,
                                        samples=all_splits["val"])
        test_ds    = SmartVisionDataset("test",  transform=eval_transform,
                                        samples=all_splits["test"])
    else:
        train_ds = SmartVisionDataset("train", transform=train_transform)
        val_ds   = SmartVisionDataset("val",   transform=eval_transform)
        test_ds  = SmartVisionDataset("test",  transform=eval_transform)

    if fast_mode and len(train_ds) < 50:
        logger.warning(
            f"Only {len(train_ds)} training samples found. "
            "Run 01_data_acquisition.py with FAST_MODE=True first."
        )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=_NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=_NUM_WORKERS, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=_NUM_WORKERS, pin_memory=True,
    )

    logger.info(
        f"DataLoaders [{model_name}]: "
        f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} batch={batch_size}"
    )
    return train_loader, val_loader, test_loader
