# filename: src/data/dataset.py
# purpose:  PyTorch Dataset for SmartVision 25-class classification
# version:  2.0

import logging
from pathlib import Path
from typing import Optional

from PIL import Image
from torch.utils.data import DataLoader, Dataset

from config import (
    CLASS_TO_IDX,
    CLASSES,
    DATA_PROCESSED_DIR,
    MODEL_CONFIGS,
)

logger = logging.getLogger(__name__)


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
    ) -> None:
        assert split in ("train", "val", "test"), f"Invalid split: {split}"
        self.split     = split
        self.transform = transform
        self.root      = Path(root) if root else DATA_PROCESSED_DIR / "classification"

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
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Returns (train_loader, val_loader, test_loader) for the given model."""
    batch_size = MODEL_CONFIGS[model_name]["batch"]

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
        num_workers=2, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    logger.info(
        f"DataLoaders [{model_name}]: "
        f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} batch={batch_size}"
    )
    return train_loader, val_loader, test_loader
