# filename: src/data/loader.py
# purpose:  HuggingFace COCO streaming, collection, and dataset saving pipeline
# version:  2.0

# SCHEMA NOTES (detection-datasets/coco):
#   sample["image"]                → PIL.Image (direct, not bytes dict)
#   sample["objects"]["bbox"]      → list of [x, y, w, h] (top-left + width/height, absolute pixels)
#   sample["objects"]["category"]  → list of 0-indexed HF category IDs (0-79, NOT COCO IDs 1-90)
#   No trust_remote_code needed. No HF token required for public dataset.
#   bbox format confirmed via mentor notebook: crop((x, y, x+w, y+h)) produces correct results.

import io
import logging
import os
from pathlib import Path
from typing import Optional

from PIL import Image

from config import (
    CLASSES,
    CLASS_TO_IDX,
    CHECKPOINT_FILE,
    DATA_PROCESSED_DIR,
    FAST_IMAGES_PER_CLASS,
    HF_CATEGORY_TO_CLASS_IDX,
    HF_TOKEN,
    IMAGES_PER_CLASS,
    NUM_CLASSES,
    SELECTED_CLASSES,
    TRAIN_SPLIT,
    VAL_SPLIT,
)
from src.utils.helpers import load_json, save_json

logger = logging.getLogger(__name__)


# ── COCO mapping verification ─────────────────────────────────────────────────
# Rule 28: call this as the first thing in 01_data_acquisition.py

def verify_coco_mapping() -> None:
    """Verify SELECTED_CLASSES and HF_CATEGORY_TO_CLASS_IDX are consistent."""
    assert len(SELECTED_CLASSES) == NUM_CLASSES, (
        f"SELECTED_CLASSES has {len(SELECTED_CLASSES)} entries, expected {NUM_CLASSES}"
    )
    assert len(HF_CATEGORY_TO_CLASS_IDX) == NUM_CLASSES, (
        f"HF_CATEGORY_TO_CLASS_IDX has {len(HF_CATEGORY_TO_CLASS_IDX)} entries, expected {NUM_CLASSES}"
    )
    # Check round-trip: class_name → hf_id → class_idx → class_name
    for cls_name, hf_id in SELECTED_CLASSES.items():
        assert hf_id in HF_CATEGORY_TO_CLASS_IDX, (
            f"HF ID {hf_id} (for '{cls_name}') missing from HF_CATEGORY_TO_CLASS_IDX"
        )
        class_idx = HF_CATEGORY_TO_CLASS_IDX[hf_id]
        assert CLASSES[class_idx] == cls_name, (
            f"Round-trip mismatch: '{cls_name}' → hf_id={hf_id} → idx={class_idx} → '{CLASSES[class_idx]}'"
        )
    # No duplicate HF IDs
    hf_ids = list(SELECTED_CLASSES.values())
    assert len(hf_ids) == len(set(hf_ids)), "Duplicate HF category IDs in SELECTED_CLASSES"
    logger.info("✅ COCO mapping verified: 25 classes, no duplicates")
    print("✅ COCO mapping verified (25 classes, HF 0-indexed IDs, no duplicates)")


# ── Checkpoint / resume ───────────────────────────────────────────────────────

def load_checkpoint() -> dict[str, int]:
    if CHECKPOINT_FILE.exists():
        data = load_json(CHECKPOINT_FILE)
        total = sum(data.values())
        logger.info(f"Resumed from checkpoint: {total} images already collected")
        return data
    return {cls: 0 for cls in CLASSES}


def save_checkpoint(progress: dict[str, int]) -> None:
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    save_json(progress, CHECKPOINT_FILE)


# ── YOLO bbox conversion ──────────────────────────────────────────────────────
# bbox in HF dataset: [x, y, w, h] — top-left corner + width/height (absolute pixels)
# Confirmed by mentor notebook: crop((x, y, x+w, y+h)) produces correct object crops

def bbox_to_yolo(
    x: float, y: float, w: float, h: float,
    img_w: int, img_h: int,
    class_idx: int,
) -> list[float]:
    """Convert [x, y, w, h] absolute pixels → YOLO [class, x_center, y_center, w, h] normalized."""
    x_center = (x + w / 2.0) / img_w
    y_center = (y + h / 2.0) / img_h
    w_norm   = w / img_w
    h_norm   = h / img_h
    # Clamp to valid range
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    w_norm   = max(0.001, min(1.0, w_norm))
    h_norm   = max(0.001, min(1.0, h_norm))
    return [class_idx, x_center, y_center, w_norm, h_norm]


# ── Crop and save classification image ───────────────────────────────────────

def save_classification_crop(
    img: Image.Image,
    bbox: list[float],
    class_name: str,
    split: str,
    img_idx: int,
    classification_dir: Path,
    crop_size: int = 224,
) -> bool:
    """Crop object bbox from image, resize to 224×224, save to split/class folder."""
    x, y, w, h = bbox
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(img.width,  int(x + w))
    y2 = min(img.height, int(y + h))

    if x2 <= x1 or y2 <= y1:
        return False

    try:
        crop = img.crop((x1, y1, x2, y2))
        crop = crop.resize((crop_size, crop_size), Image.LANCZOS)
        out_dir  = classification_dir / split / class_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{class_name}_{split}_{img_idx:04d}.jpg"
        crop.save(str(out_path), "JPEG", quality=95)
        return True
    except Exception as e:
        logger.warning(f"Crop failed for {class_name} idx={img_idx}: {e}")
        return False


# ── Save detection sample ─────────────────────────────────────────────────────

def save_detection_sample(
    img: Image.Image,
    annotations: dict,
    img_id: int,
    split: str,
    detection_dir: Path,
) -> None:
    """Save full image and YOLO-format .txt label for detection task."""
    img_w, img_h = img.size
    bboxes   = annotations["bbox"]
    cat_ids  = annotations["category"]

    # Build YOLO annotation lines for our 25 classes only
    yolo_lines = []
    for bbox, cat_id in zip(bboxes, cat_ids):
        if cat_id not in HF_CATEGORY_TO_CLASS_IDX:
            continue
        class_idx = HF_CATEGORY_TO_CLASS_IDX[cat_id]
        x, y, w, h = bbox
        vals = bbox_to_yolo(x, y, w, h, img_w, img_h, class_idx)
        yolo_lines.append(
            f"{int(vals[0])} {vals[1]:.6f} {vals[2]:.6f} {vals[3]:.6f} {vals[4]:.6f}"
        )

    if not yolo_lines:
        return

    img_dir = detection_dir / "images" / split
    lbl_dir = detection_dir / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    img.save(str(img_dir / f"image_{img_id:06d}.jpg"), "JPEG", quality=95)
    (lbl_dir / f"image_{img_id:06d}.txt").write_text(
        "\n".join(yolo_lines), encoding="utf-8"
    )


# ── Split helper ──────────────────────────────────────────────────────────────

def get_split(img_idx: int, total: int) -> str:
    """Return 'train', 'val', or 'test' based on img_idx position."""
    train_end = int(total * TRAIN_SPLIT)
    val_end   = train_end + int(total * VAL_SPLIT)
    if img_idx < train_end:
        return "train"
    elif img_idx < val_end:
        return "val"
    return "test"
