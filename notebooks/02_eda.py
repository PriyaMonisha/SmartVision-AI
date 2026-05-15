# %% [markdown]
# # SmartVision AI — Section 3: Exploratory Data Analysis
# **Run in Google Colab after Section 2 (dataset must exist)**
#
# Covers:
# 1. Class distribution + chi-squared balance test
# 2. Sample image grid (5×5)
# 3. Bounding box size distribution per class
# 4. Image quality scan (brightness, contrast, aspect ratio)
# 5. Objects-per-image histogram
# 6. Class co-occurrence heatmap
# 7. Class difficulty predictor
# All figures saved to docs/figures/

# %% [markdown]
# ## Setup
# Run locally:  `python notebooks/02_eda.py`
# Run in Colab: upload project to Drive, set PROJECT_ROOT below

# %%
import sys, os
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
    os.chdir(COLAB_ROOT)
    print("Running in Colab")
except Exception:  # ImportError=not in Colab; MessageError=Drive auth failed (re-run cell after authorising)
    print("Running locally")

# %%
import json
import math
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, ImageStat
from scipy.stats import chisquare
from tqdm import tqdm

warnings.filterwarnings("ignore")

from config import (
    ARTIFACTS_DIR, CLASSES, CLASS_TO_IDX, DATA_PROCESSED_DIR,
    DOCS_FIGURES_DIR, HF_CATEGORY_TO_CLASS_IDX, NUM_CLASSES, SELECTED_CLASSES,
)
from src.utils.helpers import load_json, save_json

FIGURES_DIR = DOCS_FIGURES_DIR
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
(ARTIFACTS_DIR / "eda").mkdir(parents=True, exist_ok=True)

CLASSIFICATION_DIR = DATA_PROCESSED_DIR / "classification"
DETECTION_DIR      = DATA_PROCESSED_DIR / "detection"

# Discrete color palette — never sequential for category charts (Rule 10)
PALETTE = [
    "#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00",
    "#a65628","#f781bf","#999999","#66c2a5","#fc8d62",
    "#8da0cb","#e78ac3","#a6d854","#ffd92f","#e5c494",
    "#b3b3b3","#1b9e77","#d95f02","#7570b3","#e7298a",
    "#66a61e","#e6ab02","#a6761d","#666666","#17becf",
]

print("EDA setup complete")
print(f"Classification dir: {CLASSIFICATION_DIR}")
print(f"Detection dir:      {DETECTION_DIR}")

# %% [markdown]
# ## 1. Count Images Per Split Per Class

# %%
split_counts: dict[str, dict[str, int]] = {}
for split in ["train", "val", "test"]:
    split_counts[split] = {}
    for cls in CLASSES:
        cls_dir = CLASSIFICATION_DIR / split / cls
        count = len(list(cls_dir.glob("*.jpg"))) if cls_dir.exists() else 0
        split_counts[split][cls] = count

total_per_class = {cls: sum(split_counts[s][cls] for s in ["train","val","test"]) for cls in CLASSES}
total_images    = sum(total_per_class.values())

print(f"Total images: {total_images} across {NUM_CLASSES} classes\n")
print(f"{'Class':<20} {'Train':>6} {'Val':>5} {'Test':>5} {'Total':>6}")
print("-" * 45)
for cls in CLASSES:
    t = split_counts["train"][cls]
    v = split_counts["val"][cls]
    ts = split_counts["test"][cls]
    print(f"{cls:<20} {t:>6} {v:>5} {ts:>5} {t+v+ts:>6}")

# %% [markdown]
# ## 2. Class Distribution Bar Chart + Chi-Squared Balance Test

# %%
fig, ax = plt.subplots(figsize=(14, 5))

x = np.arange(NUM_CLASSES)
totals = [total_per_class[cls] for cls in CLASSES]

bars = ax.bar(x, totals, color=PALETTE[:NUM_CLASSES], edgecolor="white", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("Total Images")
ax.set_title("SmartVision — Class Distribution (Train + Val + Test)", fontweight="bold")
ax.axhline(np.mean(totals), color="black", linestyle="--", linewidth=1, label=f"Mean = {np.mean(totals):.0f}")
ax.legend()

for bar, val in zip(bars, totals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            str(val), ha="center", va="bottom", fontsize=7)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "eda_01_class_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: eda_01_class_distribution.png")

# Chi-squared balance test
observed  = totals
expected  = [total_images / NUM_CLASSES] * NUM_CLASSES
chi2, p   = chisquare(observed, expected)
balanced  = p > 0.05
print(f"\nChi-squared balance test:")
print(f"  chi2 = {chi2:.4f}  |  p = {p:.4f}")
print(f"  {'OK BALANCED (p>0.05)' if balanced else 'WARNING  IMBALANCED (p<=0.05)'}")
if all(v == totals[0] for v in totals):
    print(f"  NOTE: All classes have exactly {totals[0]} images (FAST_MODE or perfectly balanced).")
    print(f"  Chi2=0, p=1.0 is trivially true. Run with FAST_MODE=False (100/class) for real insight.")

# %% [markdown]
# ## 3. Sample Image Grid (5×5 — one per class)

# %%
fig, axes = plt.subplots(5, 5, figsize=(14, 14))
fig.suptitle("SmartVision — Sample Images (one per class)", fontsize=14, fontweight="bold", y=1.01)

for i, cls in enumerate(CLASSES):
    ax = axes[i // 5][i % 5]
    # Pick first available image from train
    cls_dir = CLASSIFICATION_DIR / "train" / cls
    imgs    = sorted(cls_dir.glob("*.jpg")) if cls_dir.exists() else []
    if imgs:
        img = Image.open(imgs[0]).convert("RGB")
        ax.imshow(img)
    else:
        ax.set_facecolor("#eeeeee")
        ax.text(0.5, 0.5, "no image", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="gray")
    ax.set_title(cls, fontsize=8, pad=2)
    ax.axis("off")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "eda_02_sample_grid.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: eda_02_sample_grid.png")

# %% [markdown]
# ## 4. Image Quality Scan (Brightness, Contrast, Aspect Ratio)

# %%
print("Scanning image quality...")
print("  Brightness/Contrast: classification crops (train)")
print("  Aspect Ratio: original full detection images (pre-resize)\n")

# Brightness + Contrast from classification crops (train)
quality_data: list[dict] = []
for cls in tqdm(CLASSES, desc="Crop quality scan"):
    cls_dir = CLASSIFICATION_DIR / "train" / cls
    if not cls_dir.exists():
        continue
    for img_path in list(cls_dir.glob("*.jpg"))[:20]:
        try:
            img  = Image.open(img_path).convert("RGB")
            stat = ImageStat.Stat(img)
            quality_data.append({
                "class":      cls,
                "brightness": sum(stat.mean) / 3,
                "contrast":   sum(stat.stddev) / 3,
            })
        except Exception:
            pass

# Aspect Ratio from ORIGINAL full detection images (not 224x224 crops)
# Classification crops are all 224x224 (ratio=1.0) — scanning them is meaningless
aspect_data: list[float] = []
det_img_dir = DETECTION_DIR / "images" / "train"
if det_img_dir.exists():
    for img_path in list(det_img_dir.glob("*.jpg"))[:100]:
        try:
            img = Image.open(img_path)
            w, h = img.size
            aspect_data.append(w / h)
        except Exception:
            pass
print(f"  Aspect ratio scanned from {len(aspect_data)} full detection images")

df_q = pd.DataFrame(quality_data)
ar   = pd.Series(aspect_data)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Image Quality Metrics", fontweight="bold")

# Brightness
axes[0].hist(df_q["brightness"], bins=30, color="#377eb8", edgecolor="white")
axes[0].axvline(20,  color="red",    linestyle="--", label="Low (< 20)")
axes[0].axvline(235, color="orange", linestyle="--", label="High (> 235)")
axes[0].set_xlabel("Brightness"); axes[0].set_ylabel("Count")
axes[0].set_title("Brightness (Cropped Objects)"); axes[0].legend(fontsize=8)

# Contrast
axes[1].hist(df_q["contrast"], bins=30, color="#4daf4a", edgecolor="white")
axes[1].set_xlabel("Contrast (std-dev)")
axes[1].set_title("Contrast (Cropped Objects)")

# Aspect Ratio from FULL detection images — meaningful values
if len(ar) > 0:
    axes[2].hist(ar, bins=30, color="#984ea3", edgecolor="white")
    axes[2].axvline(3.0, color="red", linestyle="--", label="Extreme (> 3:1)")
    axes[2].axvline(1/3, color="red", linestyle="--", label="Extreme (< 1:3)")
    axes[2].axvline(ar.mean(), color="black", linestyle="-", linewidth=1.5,
                    label=f"Mean = {ar.mean():.2f}")
    axes[2].set_title("Aspect Ratio (Full Detection Images)")
else:
    axes[2].text(0.5, 0.5, "No detection images found", ha="center", va="center",
                 transform=axes[2].transAxes)
    axes[2].set_title("Aspect Ratio (N/A)")
axes[2].set_xlabel("Aspect Ratio (w/h)"); axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "eda_03_image_quality.png", dpi=150, bbox_inches="tight")
plt.show()

low_bright  = (df_q["brightness"] < 20).sum()
high_bright = (df_q["brightness"] > 235).sum()
extreme_ar  = ((ar > 3) | (ar < 1/3)).sum() if len(ar) > 0 else 0
print(f"Low-brightness images  : {low_bright} ({low_bright/len(df_q)*100:.1f}%)")
print(f"High-brightness images : {high_bright}")
print(f"Extreme aspect ratios  : {extreme_ar} / {len(ar)} full images")
print("Saved: eda_03_image_quality.png")

# %% [markdown]
# ## 5. Bounding Box Size Distribution (from Detection Labels)

# %%
print("Reading YOLO labels...")

bbox_data: list[dict] = []
for split in ["train", "val"]:
    lbl_dir = DETECTION_DIR / "labels" / split
    if not lbl_dir.exists():
        continue
    for lbl_file in list(lbl_dir.glob("*.txt"))[:500]:  # sample
        try:
            lines = lbl_file.read_text().strip().splitlines()
            for line in lines:
                parts = line.split()
                if len(parts) < 5:
                    continue
                class_idx = int(parts[0])
                w_norm    = float(parts[3])
                h_norm    = float(parts[4])
                area_norm = w_norm * h_norm
                if class_idx < NUM_CLASSES:
                    bbox_data.append({
                        "class":     CLASSES[class_idx],
                        "w_norm":    w_norm,
                        "h_norm":    h_norm,
                        "area_norm": area_norm,
                    })
        except Exception:
            pass

df_bbox = pd.DataFrame(bbox_data)

if not df_bbox.empty:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Bounding Box Statistics (YOLO Labels)", fontweight="bold")

    # Box area per class
    means = df_bbox.groupby("class")["area_norm"].mean().reindex(CLASSES)
    axes[0].bar(range(NUM_CLASSES), means.values, color=PALETTE[:NUM_CLASSES])
    axes[0].set_xticks(range(NUM_CLASSES))
    axes[0].set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("Mean Normalized Area"); axes[0].set_title("Mean BBox Area per Class")

    # Area distribution histogram
    axes[1].hist(df_bbox["area_norm"], bins=40, color="#ff7f00", edgecolor="white")
    axes[1].set_xlabel("Normalized BBox Area (w×h)"); axes[1].set_title("BBox Area Distribution")
    axes[1].axvline(df_bbox["area_norm"].median(), color="red", linestyle="--",
                    label=f"Median = {df_bbox['area_norm'].median():.3f}")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "eda_04_bbox_distribution.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"BBox stats from {len(df_bbox)} annotations")
    print("Saved: eda_04_bbox_distribution.png")
else:
    print("No YOLO labels found — run Section 2 first")

# %% [markdown]
# ## 6. Objects-Per-Image Histogram

# %%
if not df_bbox.empty:
    # Count objects per label file
    obj_counts: list[int] = []
    for split in ["train", "val"]:
        lbl_dir = DETECTION_DIR / "labels" / split
        if not lbl_dir.exists():
            continue
        for lbl_file in lbl_dir.glob("*.txt"):
            lines = lbl_file.read_text().strip().splitlines()
            obj_counts.append(len([l for l in lines if l.strip()]))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(obj_counts, bins=range(0, max(obj_counts)+2), color="#e41a1c",
            edgecolor="white", align="left")
    ax.set_xlabel("Objects per Image"); ax.set_ylabel("Number of Images")
    ax.set_title("Objects-per-Image Distribution (Detection Dataset)", fontweight="bold")
    ax.axvline(np.mean(obj_counts), color="black", linestyle="--",
               label=f"Mean = {np.mean(obj_counts):.1f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "eda_05_objects_per_image.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Mean objects/image: {np.mean(obj_counts):.1f}")
    print(f"Max objects/image:  {max(obj_counts)}")
    print("Saved: eda_05_objects_per_image.png")

# %% [markdown]
# ## 7. Class Co-Occurrence Heatmap

# %%
if not df_bbox.empty:
    # Build co-occurrence matrix
    cooccur = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for split in ["train", "val"]:
        lbl_dir = DETECTION_DIR / "labels" / split
        if not lbl_dir.exists():
            continue
        for lbl_file in lbl_dir.glob("*.txt"):
            lines  = lbl_file.read_text().strip().splitlines()
            idxs   = list({int(l.split()[0]) for l in lines if l.strip()
                           and int(l.split()[0]) < NUM_CLASSES})
            for i in idxs:
                for j in idxs:
                    cooccur[i][j] += 1

    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(
        cooccur, xticklabels=CLASSES, yticklabels=CLASSES,
        cmap="YlOrRd", annot=False, fmt="d", linewidths=0.3,
        ax=ax,
    )
    ax.set_title("Class Co-Occurrence (how often two classes appear in same image)",
                 fontweight="bold", pad=12)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "eda_06_cooccurrence.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: eda_06_cooccurrence.png")

# %% [markdown]
# ## 8. Class Difficulty Predictor
# Predicts which classes will be hard to classify based on visual similarity.
# Validated against actual confusion matrix after training.

# %%
# Compute mean bbox area per class (proxy for object scale)
mean_area: dict[str, float] = {}
for cls in CLASSES:
    subset = df_bbox[df_bbox["class"] == cls]["area_norm"] if not df_bbox.empty else pd.Series([], dtype=float)
    mean_area[cls] = float(subset.mean()) if len(subset) > 0 else 0.1

# Difficulty based on:
# 1. Visually similar neighbours within same supercategory (shape/size overlap)
# 2. Classes that are visually DISTINCT even in same group are NOT hard

# Manually defined visually-similar pairs (classes commonly confused)
VISUALLY_SIMILAR: dict[str, list[str]] = {
    "car":          ["truck", "bus"],
    "truck":        ["car", "bus"],
    "bus":          ["car", "truck"],
    "motorcycle":   ["bicycle"],
    "bicycle":      ["motorcycle"],
    "airplane":     [],              # unique shape — easy despite being vehicle
    "dog":          ["cat"],
    "cat":          ["dog"],
    "horse":        ["cow"],
    "cow":          ["horse"],
    "bird":         [],              # small, distinct — medium
    "elephant":     [],              # unique size/shape — easy
    "cup":          ["bowl", "bottle"],
    "bowl":         ["cup"],
    "bottle":       ["cup"],
    "pizza":        [],              # distinct food — medium
    "cake":         [],              # distinct food — medium
    "chair":        ["couch", "bench"],
    "couch":        ["chair", "bed"],
    "bench":        ["chair"],
    "bed":          ["couch"],
    "potted plant": [],
    "traffic light":[],
    "stop sign":    [],
    "person":       [],
}

difficulty: dict[str, dict] = {}
for cls in CLASSES:
    similar = VISUALLY_SIMILAR.get(cls, [])
    # Penalise further if size is similar to neighbours
    size_penalties = []
    for other in similar:
        a1 = mean_area.get(cls, 0.1)
        a2 = mean_area.get(other, 0.1)
        ratio = min(a1, a2) / max(a1, a2) if max(a1, a2) > 0 else 0
        size_penalties.append(ratio)
    avg_size_sim = np.mean(size_penalties) if size_penalties else 0

    if len(similar) >= 2 and avg_size_sim > 0.6:
        predicted = "hard"
    elif len(similar) >= 1:
        predicted = "medium"
    else:
        predicted = "easy"

    difficulty[cls] = {
        "visually_similar_to": similar,
        "avg_size_similarity":  round(float(avg_size_sim), 4),
        "predicted_difficulty": predicted,
    }

# Save
save_json(difficulty, ARTIFACTS_DIR / "eda" / "class_difficulty.json")

# Plot difficulty summary
diff_labels  = [d["predicted_difficulty"] for d in difficulty.values()]
color_map    = {"hard": "#e41a1c", "medium": "#ff7f00", "easy": "#4daf4a"}
bar_colors   = [color_map[d] for d in diff_labels]

fig, ax = plt.subplots(figsize=(14, 4))
ax.bar(range(NUM_CLASSES), [1]*NUM_CLASSES, color=bar_colors, edgecolor="white")
ax.set_xticks(range(NUM_CLASSES))
ax.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=9)
ax.set_yticks([])
ax.set_title("Predicted Classification Difficulty per Class", fontweight="bold")
patches = [mpatches.Patch(color=c, label=l) for l, c in color_map.items()]
ax.legend(handles=patches, loc="upper right")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "eda_07_class_difficulty.png", dpi=150, bbox_inches="tight")
plt.show()

hard_classes = [cls for cls, d in difficulty.items() if d["predicted_difficulty"] == "hard"]
print(f"Predicted HARD classes ({len(hard_classes)}): {hard_classes}")
print("Validate these against the confusion matrix after training.")
print("Saved: eda_07_class_difficulty.png + artifacts/eda/class_difficulty.json")

# %% [markdown]
# ## Summary

# %%
# Consolidate EDA results
eda_summary = {
    "total_images":         total_images,
    "num_classes":          NUM_CLASSES,
    "class_balance": {
        "chi2":    round(chi2, 6),
        "p_value": round(p, 6),
        "balanced": balanced,
    },
    "image_quality": {
        "low_brightness_count":  int(low_bright),
        "high_brightness_count": int(high_bright),
        "extreme_aspect_count":  int(extreme_ar),
        "mean_brightness":       round(float(df_q["brightness"].mean()), 4) if not df_q.empty else 0,
        "mean_contrast":         round(float(df_q["contrast"].mean()), 4)   if not df_q.empty else 0,
    },
    "detection": {
        "total_bbox_annotations":  int(len(df_bbox)),
        "mean_objects_per_image":  round(float(np.mean(obj_counts)), 2) if obj_counts else 0,
        "max_objects_per_image":   int(max(obj_counts)) if obj_counts else 0,
    },
    "split_counts": split_counts,
    "difficulty": difficulty,
}
save_json(eda_summary, ARTIFACTS_DIR / "eda" / "eda_summary.json")

print("\n" + "="*60)
print("SECTION 3 COMPLETE — EDA")
print("="*60)
print(f"  Total images        : {total_images}")
print(f"  Class balance       : {'OK Balanced' if balanced else 'WARNING  Imbalanced'} (chi2={chi2:.2f}, p={p:.4f})")
print(f"  Hard classes        : {hard_classes}")
print(f"  Figures saved to    : {FIGURES_DIR}")
print(f"  Artifacts saved to  : {ARTIFACTS_DIR / 'eda'}")
print(f"\nNext: Section 4 — Preprocessing + Augmentation (03_preprocessing.py)")
