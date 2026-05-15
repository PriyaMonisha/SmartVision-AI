# %% [markdown]
# # SmartVision AI — Section 4: Preprocessing + Augmentation
# Run locally: `python notebooks/03_preprocessing.py`
# No GPU needed.
#
# Covers:
# 1. Verify full dataset structure (2500 classification + 1925 detection)
# 2. Validate all YOLO annotations
# 3. Regenerate data.yaml with current absolute paths
# 4. Show augmentation pipeline + save sample augmented images
# 5. Confirm train/eval transforms produce correct tensor shapes

# %%
import sys
from pathlib import Path

try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent  # terminal
except NameError:
    PROJECT_ROOT = Path.cwd().parent  # Colab/Jupyter — assumes notebooks/ subdir
    if not (PROJECT_ROOT / "config.py").exists():
        PROJECT_ROOT = Path.cwd()  # fallback: running from project root
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from google.colab import drive
    drive.mount('/content/drive')
    COLAB_ROOT = '/content/drive/MyDrive/SmartVisionAI'
    sys.path.insert(0, COLAB_ROOT)
    print("Running in Colab")
except ImportError:
    print("Running locally")

# %% [markdown]
# ## Step 1: Imports

# %%
import json
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image

from config import (
    CLASSES, NUM_CLASSES, DATA_PROCESSED_DIR, DOCS_FIGURES_DIR,
    IMAGE_SIZE, IMAGES_PER_CLASS,
)
from src.data.preprocessor import (
    validate_yolo_annotations,
    create_yolo_data_yaml,
    verify_dataset_structure,
)
from src.data.augmentor import (
    TORCH_AVAILABLE, IMAGENET_MEAN, IMAGENET_STD,
    get_train_transforms, get_eval_transforms, denormalize,
)
from src.utils.helpers import save_json

FIGURES_DIR = DOCS_FIGURES_DIR
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CLASSIFICATION_DIR = DATA_PROCESSED_DIR / "classification"
DETECTION_DIR      = DATA_PROCESSED_DIR / "detection"

print(f"Torch available locally: {TORCH_AVAILABLE}")
print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"Classes: {NUM_CLASSES}")

# %% [markdown]
# ## Step 2: Verify Dataset Structure

# %%
print("Verifying dataset structure...")
stats = verify_dataset_structure()

print()
print("=== Classification ===")
for split in ["train", "val", "test"]:
    total = stats["classification"][split]["total"]
    expected = {"train": int(IMAGES_PER_CLASS * 0.70 * NUM_CLASSES),
                "val":   int(IMAGES_PER_CLASS * 0.15 * NUM_CLASSES),
                "test":  int(IMAGES_PER_CLASS * 0.15 * NUM_CLASSES)}
    status = "OK" if total > 0 else "MISSING"
    print(f"  {status}  {split:5s}: {total} images (expected ~{expected[split]})")
print(f"  Total: {stats['classification']['total']}")

print()
print("=== Detection ===")
for split in ["train", "val"]:
    d = stats["detection"][split]
    match = "OK" if d["images"] == d["labels"] else "MISMATCH"
    print(f"  {match}  {split:5s}: {d['images']} images, {d['labels']} labels")
print(f"  Total: {stats['detection']['total_images']} detection images")

print()
if stats["issues"]:
    print(f"ISSUES ({len(stats['issues'])}):")
    for issue in stats["issues"]:
        print(f"  {issue}")
else:
    print("No issues found.")

# %% [markdown]
# ## Step 3: Validate YOLO Annotations

# %%
print("Validating YOLO annotations...")
for split in ["train", "val"]:
    lbl_dir = DETECTION_DIR / "labels" / split
    validate_yolo_annotations(lbl_dir)
print("All YOLO annotations valid.")

# %% [markdown]
# ## Step 4: Regenerate data.yaml (absolute paths for current machine/Colab)

# %%
yaml_path = create_yolo_data_yaml(DETECTION_DIR)
print(f"data.yaml ready: {yaml_path}")

# %% [markdown]
# ## Step 5: Augmentation Pipeline Summary

# %%
print("Augmentation pipeline (train):")
print("  RandomHorizontalFlip(p=0.5)")
print("  RandomRotation(degrees=15)")
print("  ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05)")
print("  RandomZoomOut(side_range=(1.0, 1.3), p=0.3)")
print(f"  Resize({IMAGE_SIZE}, {IMAGE_SIZE})")
print("  ToImage() + ToDtype(float32)")
print(f"  Normalize(mean={IMAGENET_MEAN}, std={IMAGENET_STD})")
print()
print("Evaluation pipeline (val/test):")
print(f"  Resize({IMAGE_SIZE}, {IMAGE_SIZE})")
print("  ToImage() + ToDtype(float32)")
print(f"  Normalize(mean={IMAGENET_MEAN}, std={IMAGENET_STD})")

# %% [markdown]
# ## Step 6: Show Augmented Samples
# Shows original vs 4 augmented versions for one image per class (sample of 5 classes).
# Uses torch if available; falls back to PIL raw crops if not.

# %%
SAMPLE_CLASSES = ["car", "dog", "bottle", "airplane", "chair"]

if TORCH_AVAILABLE:
    train_tf = get_train_transforms(IMAGE_SIZE)
    eval_tf  = get_eval_transforms(IMAGE_SIZE)

    fig, axes = plt.subplots(len(SAMPLE_CLASSES), 5, figsize=(15, 3 * len(SAMPLE_CLASSES)))
    fig.suptitle("Augmentation Samples (col 1 = eval/original, cols 2-5 = train augmented)",
                 fontsize=11, fontweight="bold")

    for row, cls in enumerate(SAMPLE_CLASSES):
        cls_dir = CLASSIFICATION_DIR / "train" / cls
        imgs    = sorted(cls_dir.glob("*.jpg")) if cls_dir.exists() else []
        if not imgs:
            for col in range(5):
                axes[row][col].axis("off")
                axes[row][col].set_title("no image" if col == 0 else "")
            continue

        pil_img = Image.open(imgs[0]).convert("RGB")

        # Col 0: eval transform (clean, no augmentation)
        eval_tensor = eval_tf(pil_img)
        axes[row][0].imshow(denormalize(eval_tensor))
        axes[row][0].set_title(f"{cls}\n(eval)", fontsize=8)
        axes[row][0].axis("off")

        # Cols 1-4: different augmented versions
        for col in range(1, 5):
            aug_tensor = train_tf(pil_img)
            axes[row][col].imshow(denormalize(aug_tensor))
            axes[row][col].set_title(f"aug {col}", fontsize=8)
            axes[row][col].axis("off")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "preprocessing_01_augmented_samples.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: preprocessing_01_augmented_samples.png")

else:
    # No torch — show raw PIL crops
    print("torch not available locally — showing raw crops (augmentation runs in Colab/venv with GPU)")
    fig, axes = plt.subplots(1, len(SAMPLE_CLASSES), figsize=(15, 3))
    fig.suptitle("Raw Classification Crops (augmentation visualization requires torch)", fontsize=10)

    for col, cls in enumerate(SAMPLE_CLASSES):
        cls_dir = CLASSIFICATION_DIR / "train" / cls
        imgs    = sorted(cls_dir.glob("*.jpg")) if cls_dir.exists() else []
        if imgs:
            axes[col].imshow(Image.open(imgs[0]).convert("RGB"))
        axes[col].set_title(cls, fontsize=9)
        axes[col].axis("off")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "preprocessing_01_augmented_samples.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: preprocessing_01_augmented_samples.png (raw crops, no augmentation)")

# %% [markdown]
# ## Step 7: Tensor Shape Verification

# %%
if TORCH_AVAILABLE:
    import torch
    train_tf = get_train_transforms(IMAGE_SIZE)
    eval_tf  = get_eval_transforms(IMAGE_SIZE)

    # Pick one real image
    test_img_path = next((CLASSIFICATION_DIR / "train" / "car").glob("*.jpg"), None)
    if test_img_path:
        pil_img = Image.open(test_img_path).convert("RGB")
        t = train_tf(pil_img)
        e = eval_tf(pil_img)

        assert t.shape == (3, IMAGE_SIZE, IMAGE_SIZE), f"Wrong train shape: {t.shape}"
        assert e.shape == (3, IMAGE_SIZE, IMAGE_SIZE), f"Wrong eval shape: {e.shape}"
        assert t.dtype == torch.float32
        assert e.dtype == torch.float32
        print(f"Train tensor: shape={tuple(t.shape)}, dtype={t.dtype}, range=[{t.min():.2f}, {t.max():.2f}]")
        print(f"Eval tensor:  shape={tuple(e.shape)}, dtype={e.dtype}, range=[{e.min():.2f}, {e.max():.2f}]")
        print("Tensor shapes and dtypes OK")
    else:
        print("No car images found to test")
else:
    print("Skipping tensor shape check (torch not available locally)")
    print("Tensor shapes will be verified in Colab during Section 5 training.")

# %% [markdown]
# ## Step 8: Save Preprocessing Summary

# %%
summary = {
    "dataset_stats":        stats,
    "image_size":           IMAGE_SIZE,
    "imagenet_mean":        IMAGENET_MEAN,
    "imagenet_std":         IMAGENET_STD,
    "augmentations_train": [
        "RandomHorizontalFlip(p=0.5)",
        "RandomRotation(degrees=15)",
        "ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05)",
        "RandomZoomOut(side_range=(1.0,1.3), p=0.3)",
        f"Resize({IMAGE_SIZE},{IMAGE_SIZE})",
        "Normalize(ImageNet)",
    ],
    "augmentations_eval": [
        f"Resize({IMAGE_SIZE},{IMAGE_SIZE})",
        "Normalize(ImageNet)",
    ],
    "yolo_validation": "passed",
    "data_yaml":        str(yaml_path),
}
save_json(summary, DATA_PROCESSED_DIR / "preprocessing_summary.json")
print(f"Summary saved: {DATA_PROCESSED_DIR / 'preprocessing_summary.json'}")

# %%
print()
print("=" * 60)
print("SECTION 4 COMPLETE -- Preprocessing + Augmentation")
print("=" * 60)
print(f"  Classification: {stats['classification']['total']} images (2500)")
print(f"  Detection:      {stats['detection']['total_images']} images")
print(f"  YOLO labels:    validated OK")
print(f"  data.yaml:      regenerated with absolute paths")
print(f"  Transforms:     train (augmented) + eval (clean) defined")
print()
print("Next: Section 5 -- CNN Training")
print("  Set MODEL='vgg16' in notebooks/04_train_classifier.py")
print("  Run in Google Colab T4 (GPU required)")
