# %% [markdown]
# # SmartVision AI — Section 2: Dataset Acquisition
# **Run in terminal (CPU is fine — no GPU needed for this step)**
#
# Based on mentor's reference notebook with these improvements:
# - 25 classes (mentor's notebook accidentally included 26 by adding 'train' vehicle)
# - Checkpoint/resume — safe to re-run if Colab disconnects
# - Proper detection train/val split (mentor dumped everything flat into one folder)
#
# Runtime: ~15-20 min for full 2500 images | ~2 min for FAST_MODE (250 images)

# %% [markdown]
# ## Step 0: Environment Setup
# Run locally:  `python notebooks/01_data_acquisition.py`
# Run in Colab: upload project to Drive, set PROJECT_ROOT below, then run

# %%
import sys, os
from pathlib import Path

# Add project root to path so config.py and src/ are importable
# Works whether running as: python notebooks/01_data_acquisition.py  (terminal)
#                        or: in Colab after mounting Drive
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent  # terminal
except NameError:
    PROJECT_ROOT = Path.cwd().parent  # Colab/Jupyter — assumes notebooks/ subdir
    if not (PROJECT_ROOT / "config.py").exists():
        PROJECT_ROOT = Path.cwd()  # fallback: running from project root
sys.path.insert(0, str(PROJECT_ROOT))

# Optional Colab setup — ignored when running locally in terminal
try:
    from google.colab import drive  # type: ignore[import-untyped]
    drive.mount('/content/drive')
    COLAB_ROOT = '/content/drive/MyDrive/Smart Vision AI'
    sys.path.insert(0, COLAB_ROOT)
    os.chdir(COLAB_ROOT)
    print("Running in Colab")
except Exception:  # ImportError=not in Colab; MessageError=Drive auth failed (re-run cell after authorising)
    print("Running locally")

# Install if missing (already in requirements.txt for local; pre-installed on Colab)
# !pip install -q datasets pillow tqdm pydantic-settings

# %%
# ================================================================
# FAST_MODE — LOCAL variable, passed as argument to functions
# True  -> FAST_IMAGES_PER_CLASS (10) images/class
# False -> IMAGES_PER_CLASS (200) images/class
# ================================================================
FAST_MODE = False   # Full run: 200 images/class = 4400 total (22 classes)
# ================================================================
print(f"FAST_MODE = {FAST_MODE}")

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
    check_crop_quality, check_person_overlap,
    save_classification_crop, save_detection_sample,
    get_split, bbox_to_yolo,
)
from src.utils.helpers import save_json, load_json, create_hub_repo

print("OK Imports OK")
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
print("OK Schema confirmed — image=PIL, objects.bbox=[x,y,w,h], objects.category=HF 0-indexed")

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
print(f"Already complete : {len(completed_classes)}/{NUM_CLASSES} classes")
print(f"Still needed     : {len(remaining_classes)} classes")

# In-memory storage: {class_name: [{'image': PIL, 'annotations': dict, 'idx': int}]}
class_images = {cls: [] for cls in CLASSES}

# Pre-fill already-collected counts (from checkpoint)
# Note: actual images not reloaded — only new ones collected
class_counts = dict(progress)

# %%
if remaining_classes:
    print(f"\n Streaming COCO dataset...")
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
    MAX_ITER = 150000  # raised from 80k — 80px floor needs more streaming for small-object classes

    # Overlap threshold for chair/person co-occurrence filter.
    # 0.50 = conservative: reject when person covers >50% of chair bbox area.
    # Adjust to 0.30 in a second run if chair accuracy remains low.
    CHAIR_PERSON_THRESHOLD = 0.50

    rejection_stats = {
        "too_small":            0,
        "too_tiny":             0,
        "bad_aspect":           0,
        "accepted":             0,
        "chair_person_overlap": 0,
    }

    print(f"Config: MAX_ITER={MAX_ITER}  CHAIR_PERSON_THRESHOLD={CHAIR_PERSON_THRESHOLD}")
    print(f"Quality gates: min_crop=80px, min_area_ratio=2%")

    # One-time bbox format verification — confirms [x,y,w,h] before streaming starts.
    # Fails immediately if HF dataset changes to [x1,y1,x2,y2] format.
    _fmt_sample = next(iter(dataset))
    _fmt_bbox   = _fmt_sample["objects"]["bbox"][0]
    assert _fmt_bbox[2] < _fmt_sample["image"].width, (
        f"bbox[2]={_fmt_bbox[2]} >= image_width={_fmt_sample['image'].width}. "
        "Format appears to be [x1,y1,x2,y2], not [x,y,w,h]. "
        "Update compute_overlap_ratio and check_crop_quality."
    )
    print(f"OK Bbox format confirmed [x,y,w,h]. Sample: {_fmt_bbox}")

    for idx, item in enumerate(dataset):
        if images_processed >= MAX_ITER:
            print(f"WARNING  Reached safety limit of {MAX_ITER} iterations")
            break

        if not remaining_classes:
            print(" All classes collected!")
            break

        images_processed += 1
        if images_processed % 2000 == 0:
            print(f"   Processed {images_processed} | Collected {total_collected}/{target * NUM_CLASSES}")

        categories = item["objects"]["category"]

        # Check if any target class is in this image (O(n) lookup via set)
        bboxes   = item["objects"]["bbox"]
        cat_list = item["objects"]["category"]

        for cat_id in set(categories):  # deduplicate cat_ids per image
            if cat_id not in HF_CATEGORY_TO_CLASS_IDX:
                continue
            class_name = CLASSES[HF_CATEGORY_TO_CLASS_IDX[cat_id]]
            if class_name not in remaining_classes:
                continue
            if class_counts[class_name] >= target:
                remaining_classes.remove(class_name)
                continue

            # Find the bbox for this cat_id and apply quality gates
            target_bbox = next((b for b, c in zip(bboxes, cat_list) if c == cat_id), None)
            if target_bbox is None:
                continue
            img = item["image"]
            x, y, w, h = target_bbox
            x1 = max(0, int(x));  y1 = max(0, int(y))
            x2 = min(img.width, int(x + w));  y2 = min(img.height, int(y + h))
            quality = check_crop_quality(img, x1, y1, x2, y2)
            if quality != "ok":
                rejection_stats[quality] += 1
                continue

            # Chair-specific: reject crops where a person dominates the bbox.
            # bboxes/cat_list are full-image variables (loaded above this loop).
            # SELECTED_CLASSES["person"] = 0 is the HF 0-indexed person category ID.
            if class_name == "chair":
                if check_person_overlap(
                    target_bbox, bboxes, cat_list,
                    person_cat_id=SELECTED_CLASSES["person"],
                    threshold=CHAIR_PERSON_THRESHOLD,
                ):
                    rejection_stats["chair_person_overlap"] += 1
                    continue

            rejection_stats["accepted"] += 1

            class_images[class_name].append({
                "image":       item["image"],
                "annotations": item["objects"],
                "idx":         images_processed,
            })
            class_counts[class_name] += 1
            total_collected += 1

            if class_counts[class_name] >= target:
                remaining_classes.remove(class_name)
                print(f"   OK {class_name}: {class_counts[class_name]} images "
                      f"({NUM_CLASSES - len(remaining_classes)}/{NUM_CLASSES} done)")

    print(f"\n Stream complete: processed {images_processed} images")

    _total = sum(rejection_stats.values())
    if _total == 0:
        print("  WARNING: No crops evaluated — check streaming loop.")
    else:
        print("\n=== Crop Quality Report ===")
        for _key, _label in [
            ("accepted",             "Accepted"),
            ("too_small",            "Rejected (too_small  <80px)"),
            ("too_tiny",             "Rejected (too_tiny   <2% area)"),
            ("bad_aspect",           "Rejected (bad_aspect >5:1)"),
            ("chair_person_overlap", f"Rejected (chair/person >={CHAIR_PERSON_THRESHOLD:.2f})"),
        ]:
            _n = rejection_stats.get(_key, 0)
            print(f"  {_label:<50} {_n:5d}  ({100*_n/_total:.1f}%)")

    # Per-class count check — stricter gates hit small-object classes harder
    print(f"\n=== Per-class collection counts (target={target}) ===")
    for _cls in sorted(CLASSES, key=lambda c: class_counts.get(c, 0)):
        _count  = class_counts.get(_cls, 0)
        _status = "OK" if _count >= target * 0.90 else "WARNING <90%"
        print(f"  {_status:<14} {_cls:<20} {_count:>4}/{target}")

    _under_target = {
        _cls: class_counts.get(_cls, 0)
        for _cls in CLASSES
        if class_counts.get(_cls, 0) < target * 0.90
    }
    if _under_target:
        import logging as _logging
        _logging.getLogger(__name__).warning(
            f"{len(_under_target)} classes below 90% of target ({target}): "
            + ", ".join(f"{c}={n}" for c, n in
                        sorted(_under_target.items(), key=lambda x: x[1]))
            + ". Consider increasing MAX_ITER."
        )

else:
    print("OK All classes already collected per checkpoint. Reload images from disk if needed.")

# %%
# Summary
print("\n=== Collection Summary ===")
for cls in CLASSES:
    count = len(class_images[cls]) + progress.get(cls, 0)
    status = "OK" if count >= target else f"WARNING  ({count}/{target})"
    print(f"  {status}  {cls}")

# %% [markdown]
# ## Step 5: Split Data (70/15/15)
# Skipped if resuming from checkpoint (images already on disk from a previous run)

# %%
HAS_NEW_IMAGES = any(len(class_images[cls]) > 0 for cls in CLASSES)

if not HAS_NEW_IMAGES:
    print("Resuming: all images already on disk. Skipping split/save steps.")
    print("Skip to Step 7 (data.yaml) to regenerate config if needed.")
else:
    print(f"Splitting data: {int(TRAIN_SPLIT*100)}% train / {int(VAL_SPLIT*100)}% val / 15% test")

train_data: dict[str, list] = {cls: [] for cls in CLASSES}
val_data:   dict[str, list] = {cls: [] for cls in CLASSES}
test_data:  dict[str, list] = {cls: [] for cls in CLASSES}

if not HAS_NEW_IMAGES:
    pass  # split_data dicts stay empty — save steps will be skipped below

for class_name in CLASSES if HAS_NEW_IMAGES else []:
    items = class_images[class_name]
    if not items:
        print(f"  WARNING  {class_name}: no new images (relies on previous checkpoint data)")
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
# ## Step 6A: Save Classification Images (Cropped 224x224)

# %%
classification_dir = DATA_PROCESSED_DIR / "classification"
classification_stats = {"train": 0, "val": 0, "test": 0}

if not HAS_NEW_IMAGES:
    # Read from disk — images already saved in a previous run
    for split in ["train", "val", "test"]:
        classification_stats[split] = len(list((classification_dir / split).rglob("*.jpg")))
    print("Resume: classification images already on disk.")
else:
    print("Saving classification crops (224x224)...\n")

if HAS_NEW_IMAGES:
    for split_name, split_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        print(f"  Processing {split_name.upper()}...")
        for class_name in tqdm(CLASSES, desc=f"    {split_name}", leave=False):
            items = split_data.get(class_name, [])
            class_id = SELECTED_CLASSES[class_name]
            for img_idx, item in enumerate(items):
                img        = item["image"]
                bboxes     = item["annotations"]["bbox"]
                categories = item["annotations"]["category"]
                for bbox, cat_id in zip(bboxes, categories):
                    if cat_id == class_id:
                        global_idx = progress.get(class_name, 0) + img_idx
                        result = save_classification_crop(
                            img, bbox, class_name, split_name,
                            global_idx, classification_dir,
                        )
                        if result is True:
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
print("\nOK Checkpoint saved")

# %% [markdown]
# ## Step 6B: Save Detection Images (Full Images + YOLO Labels)
#
# Improvement over mentor: proper train/val split instead of flat folder.
# Uses train + val classification images (not test) for detection training.

# %%
detection_dir = DATA_PROCESSED_DIR / "detection"
det_stats = {"train": {"images": 0, "objects": 0}, "val": {"images": 0, "objects": 0}}
img_id    = 0

if not HAS_NEW_IMAGES:
    # Read counts from disk
    for split in ["train", "val"]:
        img_dir = detection_dir / "images" / split
        lbl_dir = detection_dir / "labels" / split
        det_stats[split]["images"] = len(list(img_dir.glob("*.jpg"))) if img_dir.exists() else 0
        det_stats[split]["objects"] = sum(
            len(f.read_text().strip().splitlines())
            for f in lbl_dir.glob("*.txt") if f.exists()
        ) if lbl_dir.exists() else 0
    print("Resume: detection images already on disk.")
else:
    print("Saving detection images + YOLO labels...\n")
    for split_name, split_data in [("train", train_data), ("val", val_data)]:
        print(f"  Processing {split_name.upper()} split...")
        for class_name in tqdm(CLASSES, desc=f"    {split_name}", leave=False):
            for item in split_data.get(class_name, []):
                img         = item["image"]
                annotations = item["annotations"]
                valid_cats  = [c for c in annotations["category"] if c in HF_CATEGORY_TO_CLASS_IDX]
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
print(f"OK data.yaml created -> {yaml_path}")
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
    "dataset":              f"COCO 2017 ({NUM_CLASSES}-class subset)",
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
print(f"OK Metadata saved -> {DATA_PROCESSED_DIR / 'metadata.json'}")

# %% [markdown]
# ## Summary

# %%
print("\n" + "="*65)
print("SECTION 2 COMPLETE — Dataset Acquisition")
print("="*65)
print(f"  Classification images : {total_cls_images} ({target}/class x {NUM_CLASSES} classes)")
print(f"  Detection images      : {total_det_images} (train={det_stats['train']['images']} val={det_stats['val']['images']})")
print(f"  FAST_MODE             : {FAST_MODE}")
print(f"  Checkpoint file       : {CHECKPOINT_FILE}")
print(f"  data.yaml             : {yaml_path}")
print()
print("Next steps:")
print("  • Section 3: EDA  (notebooks/02_eda.py)")
print("  • Section 4: Preprocessing + Augmentation  (notebooks/03_preprocessing.py)")
print("  • Section 5: VGG16 training  (set MODEL='vgg16' in 04_train_classifier.py)")
