# %% [markdown]
# # SmartVision AI — Section 2: Dataset Acquisition
# **Run in Google Colab (CPU is fine — no GPU needed for this step)**
#
# Based on mentor's reference notebook with these improvements:
# - 25 classes (mentor's notebook accidentally included 26 by adding 'train' vehicle)
# - Checkpoint/resume — safe to re-run if Colab disconnects
# - Proper detection train/val split (mentor dumped everything flat into one folder)
#
# Runtime: ~15-20 min for full 2500 images | ~2 min for FAST_MODE (250 images)

# %% [markdown]
# ## Step 0: Colab Setup

# %%
# Mount Google Drive for persistence
from google.colab import drive
drive.mount('/content/drive')

import sys, os
PROJECT_ROOT = '/content/drive/MyDrive/SmartVisionAI'  # adjust if different
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# Install if needed (datasets + pillow usually pre-installed on Colab)
# !pip install -q datasets pillow tqdm pydantic-settings

# %%
# ================================================================
# FAST_MODE — LOCAL variable, passed as argument to functions
# True  → 10 images/class (250 total, ~2 min)
# False → 100 images/class (2500 total, ~15-20 min)
# ================================================================
FAST_MODE = True
# ================================================================
print(f"FAST_MODE = {FAST_MODE}")
print(f"Target: {'10' if FAST_MODE else '100'} images/class × 25 classes = {'250' if FAST_MODE else '2500'} total")

# %% [markdown]
# ## Step 1: Imports and Config

# %%
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import yaml
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from config import (
    ARTIFACTS_DIR, CHECKPOINT_FILE, CLASSES, CLASS_TO_IDX,
    DATA_PROCESSED_DIR, FAST_IMAGES_PER_CLASS, HF_CATEGORY_TO_CLASS_IDX,
    HF_TOKEN, IMAGES_PER_CLASS, NUM_CLASSES, SELECTED_CLASSES,
    TRAIN_SPLIT, VAL_SPLIT,
)
from src.data.loader import (
    verify_coco_mapping, load_checkpoint, save_checkpoint,
    save_classification_crop, save_detection_sample, get_split, bbox_to_yolo,
)
from src.utils.helpers import save_json, load_json, create_hub_repo

print("✅ Imports OK")
print(f"NUM_CLASSES = {NUM_CLASSES}")
print(f"Classes: {CLASSES}")

# %% [markdown]
# ## Step 2: Verify COCO Mapping (FAIL FAST)
# Confirms HF category IDs are correct before streaming a single image.

# %%
verify_coco_mapping()

# %% [markdown]
# ## Step 3: Test HF Schema (1 Sample)
# Confirm field names before committing to full download.

# %%
print("Testing HuggingFace COCO schema with 1 sample...")
test_ds = load_dataset(
    "detection-datasets/coco",
    split="train",
    streaming=True,
    token=HF_TOKEN or None,
)
sample = next(iter(test_ds))
print(f"Keys:              {list(sample.keys())}")
print(f"image type:        {type(sample['image'])}")
print(f"image size:        {sample['image'].size}")
print(f"objects keys:      {list(sample['objects'].keys())}")
print(f"num annotations:   {len(sample['objects']['bbox'])}")
print(f"categories:        {sample['objects']['category']}")
print(f"first bbox:        {sample['objects']['bbox'][0]}")
print("✅ Schema confirmed — image=PIL, objects.bbox=[x,y,w,h], objects.category=HF 0-indexed")

# %% [markdown]
# ## Step 4: Collect Images from Stream (Checkpoint/Resume)
#
# Stores PIL images in memory per class, then saves to disk in Step 5.
# **Safe to re-run** — reads checkpoint and skips already-collected classes.

# %%
target = FAST_IMAGES_PER_CLASS if FAST_MODE else IMAGES_PER_CLASS

# Load checkpoint
progress = load_checkpoint()
completed_classes = {cls for cls, cnt in progress.items() if cnt >= target}
remaining_classes = [cls for cls in CLASSES if cls not in completed_classes]

print(f"Target per class : {target}")
print(f"Already complete : {len(completed_classes)}/25 classes")
print(f"Still needed     : {len(remaining_classes)} classes")

# In-memory storage: {class_name: [{'image': PIL, 'annotations': dict, 'idx': int}]}
class_images = {cls: [] for cls in CLASSES}

# Pre-fill already-collected counts (from checkpoint)
# Note: actual images not reloaded — only new ones collected
class_counts = dict(progress)

# %%
if remaining_classes:
    print(f"\n⏳ Streaming COCO dataset...")
    print(f"   (processes ~10k images to collect {len(remaining_classes)} classes)")
    print()

    dataset = load_dataset(
        "detection-datasets/coco",
        split="train",
        streaming=True,
        token=HF_TOKEN or None,
    )

    total_collected = sum(progress.values())
    images_processed = 0
    MAX_ITER = 60000  # safety limit

    for idx, item in enumerate(dataset):
        if images_processed >= MAX_ITER:
            print(f"⚠️  Reached safety limit of {MAX_ITER} iterations")
            break

        if not remaining_classes:
            print("🎉 All classes collected!")
            break

        images_processed += 1
        if images_processed % 2000 == 0:
            print(f"   Processed {images_processed} | Collected {total_collected}/{target * NUM_CLASSES}")

        categories = item["objects"]["category"]

        # Check if any target class is in this image (O(n) lookup via set)
        for cat_id in set(categories):  # deduplicate cat_ids per image
            if cat_id not in HF_CATEGORY_TO_CLASS_IDX:
                continue
            class_name = CLASSES[HF_CATEGORY_TO_CLASS_IDX[cat_id]]
            if class_name not in remaining_classes:
                continue
            if class_counts[class_name] >= target:
                remaining_classes.remove(class_name)
                continue

            class_images[class_name].append({
                "image":       item["image"],
                "annotations": item["objects"],
                "idx":         images_processed,
            })
            class_counts[class_name] += 1
            total_collected += 1

            if class_counts[class_name] >= target:
                remaining_classes.remove(class_name)
                print(f"   ✅ {class_name}: {class_counts[class_name]} images "
                      f"({NUM_CLASSES - len(remaining_classes)}/25 done)")

    print(f"\n📊 Stream complete: processed {images_processed} images")

else:
    print("✅ All classes already collected per checkpoint. Reload images from disk if needed.")

# %%
# Summary
print("\n=== Collection Summary ===")
for cls in CLASSES:
    count = len(class_images[cls]) + progress.get(cls, 0)
    status = "✅" if count >= target else f"⚠️  ({count}/{target})"
    print(f"  {status}  {cls}")

# %% [markdown]
# ## Step 5: Split Data (70/15/15)

# %%
print(f"Splitting data: {int(TRAIN_SPLIT*100)}% train / {int(VAL_SPLIT*100)}% val / 15% test")

train_data: dict[str, list] = {}
val_data:   dict[str, list] = {}
test_data:  dict[str, list] = {}

for class_name in CLASSES:
    items = class_images[class_name]
    if not items:
        print(f"  ⚠️  {class_name}: no new images (relies on previous checkpoint data)")
        train_data[class_name] = []
        val_data[class_name]   = []
        test_data[class_name]  = []
        continue

    n = len(items)
    train_end = int(TRAIN_SPLIT * n)
    val_end   = int((TRAIN_SPLIT + VAL_SPLIT) * n)

    train_data[class_name] = items[:train_end]
    val_data[class_name]   = items[train_end:val_end]
    test_data[class_name]  = items[val_end:]

    print(f"  {class_name:20s}: train={len(train_data[class_name])} "
          f"val={len(val_data[class_name])} test={len(test_data[class_name])}")

# %% [markdown]
# ## Step 6A: Save Classification Images (Cropped 224×224)

# %%
classification_dir = DATA_PROCESSED_DIR / "classification"
classification_stats = {"train": 0, "val": 0, "test": 0}

print("Saving classification crops (224×224)...\n")

for split_name, split_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
    print(f"  Processing {split_name.upper()}...")
    for class_name in tqdm(CLASSES, desc=f"    {split_name}", leave=False):
        items = split_data.get(class_name, [])
        class_id = SELECTED_CLASSES[class_name]

        for img_idx, item in enumerate(items):
            img         = item["image"]
            bboxes      = item["annotations"]["bbox"]
            categories  = item["annotations"]["category"]

            # Find first bbox matching this class
            for bbox, cat_id in zip(bboxes, categories):
                if cat_id == class_id:
                    # Determine global idx for filename uniqueness
                    global_idx = progress.get(class_name, 0) + img_idx
                    saved = save_classification_crop(
                        img, bbox, class_name, split_name,
                        global_idx, classification_dir,
                    )
                    if saved:
                        classification_stats[split_name] += 1
                    break

print()
print("=== Classification Dataset ===")
print(f"  Train: {classification_stats['train']} images")
print(f"  Val:   {classification_stats['val']} images")
print(f"  Test:  {classification_stats['test']} images")
print(f"  Total: {sum(classification_stats.values())} images")

# Update checkpoint
for cls in CLASSES:
    progress[cls] = class_counts.get(cls, 0)
save_checkpoint(progress)
print("\n✅ Checkpoint saved")

# %% [markdown]
# ## Step 6B: Save Detection Images (Full Images + YOLO Labels)
#
# Improvement over mentor: proper train/val split instead of flat folder.
# Uses train + val classification images (not test) for detection training.

# %%
detection_dir = DATA_PROCESSED_DIR / "detection"
det_stats     = {"train": {"images": 0, "objects": 0}, "val": {"images": 0, "objects": 0}}
img_id        = 0

print("Saving detection images + YOLO labels...\n")

for split_name, split_data in [("train", train_data), ("val", val_data)]:
    print(f"  Processing {split_name.upper()} split...")
    for class_name in tqdm(CLASSES, desc=f"    {split_name}", leave=False):
        for item in split_data.get(class_name, []):
            img         = item["image"]
            annotations = item["annotations"]

            # Count valid objects in this image
            valid_cats = [c for c in annotations["category"] if c in HF_CATEGORY_TO_CLASS_IDX]
            if not valid_cats:
                continue

            save_detection_sample(img, annotations, img_id, split_name, detection_dir)
            det_stats[split_name]["images"]  += 1
            det_stats[split_name]["objects"] += len(valid_cats)
            img_id += 1

total_det_images  = sum(s["images"]  for s in det_stats.values())
total_det_objects = sum(s["objects"] for s in det_stats.values())

print()
print("=== Detection Dataset ===")
print(f"  Train: {det_stats['train']['images']} images, {det_stats['train']['objects']} objects")
print(f"  Val:   {det_stats['val']['images']} images,   {det_stats['val']['objects']} objects")
print(f"  Total: {total_det_images} images, {total_det_objects} objects")
if total_det_images > 0:
    print(f"  Avg objects/image: {total_det_objects / total_det_images:.1f}")

# %% [markdown]
# ## Step 6C: Create data.yaml for YOLO Training

# %%
def create_yolo_data_yaml(detection_dir: Path) -> Path:
    """Generate data.yaml with absolute paths — works in Colab AND Docker."""
    config = {
        "path":  str(detection_dir.absolute()),
        "train": "images/train",
        "val":   "images/val",
        "nc":    NUM_CLASSES,
        "names": {i: cls for i, cls in enumerate(CLASSES)},
    }
    yaml_path = detection_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    return yaml_path

yaml_path = create_yolo_data_yaml(detection_dir)

# Verify it loads cleanly
with open(yaml_path) as f:
    loaded = yaml.safe_load(f)
assert loaded["nc"] == NUM_CLASSES, f"nc mismatch: {loaded['nc']} vs {NUM_CLASSES}"
assert len(loaded["names"]) == NUM_CLASSES
print(f"✅ data.yaml created → {yaml_path}")
print(f"   path: {loaded['path']}")
print(f"   nc:   {loaded['nc']}")
print(f"   names: {list(loaded['names'].values())[:5]}...")

# %% [markdown]
# ## Step 7: Save Metadata

# %%
# Count images per split per class from disk
split_counts: dict[str, dict[str, int]] = {}
for split in ["train", "val", "test"]:
    split_counts[split] = {}
    for cls in CLASSES:
        cls_dir = classification_dir / split / cls
        split_counts[split][cls] = len(list(cls_dir.glob("*.jpg"))) if cls_dir.exists() else 0

total_cls_images = sum(sum(v.values()) for v in split_counts.values())

metadata = {
    "dataset":              "COCO 2017 (25-class subset)",
    "total_images":         total_cls_images,
    "images_per_class":     target,
    "fast_mode":            FAST_MODE,
    "num_classes":          NUM_CLASSES,
    "classes":              CLASSES,
    "split_counts":         split_counts,
    "detection_stats":      det_stats,
    "detection_dir":        str(detection_dir),
    "classification_dir":   str(classification_dir),
    "data_yaml":            str(yaml_path),
}

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
save_json(metadata, DATA_PROCESSED_DIR / "metadata.json")
print(f"✅ Metadata saved → {DATA_PROCESSED_DIR / 'metadata.json'}")

# %% [markdown]
# ## Summary

# %%
print("\n" + "="*65)
print("SECTION 2 COMPLETE — Dataset Acquisition")
print("="*65)
print(f"  Classification images : {total_cls_images} ({target}/class × 25 classes)")
print(f"  Detection images      : {total_det_images} (train={det_stats['train']['images']} val={det_stats['val']['images']})")
print(f"  FAST_MODE             : {FAST_MODE}")
print(f"  Checkpoint file       : {CHECKPOINT_FILE}")
print(f"  data.yaml             : {yaml_path}")
print()
print("Next steps:")
print("  • Section 3: EDA  (notebooks/02_eda.py)")
print("  • Section 4: Preprocessing + Augmentation  (notebooks/03_preprocessing.py)")
print("  • Section 5: VGG16 training  (set MODEL='vgg16' in 04_train_classifier.py)")
