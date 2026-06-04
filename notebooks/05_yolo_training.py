# %% [markdown]
# # SmartVision AI -- Section 6: YOLOv8 Detection Training
# **Run in Google Colab T4 -- GPU required**
#
# Data location in Drive:  My Drive/processed/smartvision_dataset/detection/
# Project code in Drive:   My Drive/Smart Vision AI/
#
# What this notebook does:
#   1. Validates all YOLO annotations (class IDs 0-21, bbox in range)
#   2. Writes a Colab-compatible data.yaml (Windows paths in Drive copy won't work)
#   3. Trains YOLOv8n for 50 epochs (FAST_MODE=True -> 3 epochs @ imgsz=320)
#   4. Evaluates: overall + per-class mAP50, mAP50-95, precision, recall
#   5. Saves artifacts to Drive (metrics.json, training curves, confusion matrix)
#   6. Uploads best.pt to HuggingFace Hub
#   7. Prints git commit instructions for local machine


# %% [markdown]
# ## Step 0: Environment Setup

# %%
import sys
from pathlib import Path

# Rule 40: __file__ undefined in Colab cells -- wrap in try/except NameError
# NOTE: PROJECT_ROOT for Colab is set AFTER drive.mount() in the next cell.
#       Here we only handle the terminal case.
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent   # terminal
    sys.path.insert(0, str(PROJECT_ROOT))
    print(f'Running locally. PROJECT_ROOT = {PROJECT_ROOT}')
except NameError:
    PROJECT_ROOT = None  # will be set after Drive mounts below
    print('Running in Colab -- PROJECT_ROOT will be set after Drive mount')

# %%
# Rule: Never put Colab imports at module level -- always wrap in try/except
IN_COLAB = False
try:
    from google.colab import drive  # type: ignore[import-untyped]
    drive.mount('/content/drive')
    IN_COLAB = True
    # Set PROJECT_ROOT AFTER mount so config.py existence check is reliable
    PROJECT_ROOT = Path('/content/drive/MyDrive/Smart Vision AI')
    sys.path.insert(0, str(PROJECT_ROOT))
    print(f'Running in Colab. PROJECT_ROOT = {PROJECT_ROOT}')
    print(f'config.py exists: {(PROJECT_ROOT / "config.py").exists()}')
    import subprocess
    subprocess.run([
        'pip', 'install', '-q',
        'ultralytics',          # no version pin -- 8.2.0 breaks on PyTorch 2.6+
        'huggingface_hub',
        'pydantic-settings',
    ], check=True)
    print('Dependencies installed')
except Exception:
    print('Running locally (or Drive auth pending -- re-run cell after authorising)')

assert PROJECT_ROOT is not None, 'PROJECT_ROOT not set -- re-run the cell above'

# %%
# ================================================================
# Rule 1: FAST_MODE is a LOCAL variable -- passed as param below
#         Flip to False for the production 50-epoch run
FAST_MODE = True
# ================================================================
print(f'FAST_MODE = {FAST_MODE}')
if FAST_MODE:
    print('  Fast run: 3 epochs @ imgsz=320 -- for smoke-testing only')
else:
    print('  Full run: 50 epochs @ imgsz=640 -- production training')


# %% [markdown]
# ## Step 1: Imports

# %%
import logging
import shutil
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import yaml

from ultralytics import YOLO

from config import (
    ARTIFACTS_DIR,
    CLASSES,
    HF_REPO_ID,
    HF_TOKEN,
    MODELS_DIR,
    NUM_CLASSES,
    YOLO_BATCH,
    YOLO_CONF_THRESHOLD,
    YOLO_EPOCHS,
    YOLO_IOU_THRESHOLD,
    YOLO_IMAGE_SIZE,
)
from src.utils.helpers import save_json, upload_model_to_hub, create_hub_repo

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print(f'NUM_CLASSES = {NUM_CLASSES}')
print(f'CLASSES     = {CLASSES}')
print(f'YOLO_EPOCHS = {YOLO_EPOCHS}  (overridden by FAST_MODE if True)')


# %% [markdown]
# ## Step 2: Path Setup

# %%
# Detection data on Drive (from Section 4 upload)
# Path confirmed: My Drive/Smart Vision AI/data/processed/smartvision_dataset/detection/
DETECTION_DIR = (
    Path('/content/drive/MyDrive/Smart Vision AI/data/processed/smartvision_dataset/detection')
    if IN_COLAB
    else PROJECT_ROOT / 'data' / 'processed' / 'smartvision_dataset' / 'detection'
)

assert DETECTION_DIR.exists(), (
    f'Detection dir not found: {DETECTION_DIR}\n'
    'Check that Drive is mounted and data was uploaded to '
    'My Drive/processed/smartvision_dataset/detection/'
)

TRAIN_IMGS = DETECTION_DIR / 'images' / 'train'
VAL_IMGS   = DETECTION_DIR / 'images' / 'val'
TRAIN_LBLS = DETECTION_DIR / 'labels' / 'train'
VAL_LBLS   = DETECTION_DIR / 'labels' / 'val'

for p in [TRAIN_IMGS, VAL_IMGS, TRAIN_LBLS, VAL_LBLS]:
    assert p.exists(), f'Missing directory: {p}'

# YOLO training output stays in Colab local filesystem (Drive writes are slow for many small files)
YOLO_RUNS_DIR = Path('/content/runs/detect') if IN_COLAB else PROJECT_ROOT / 'runs' / 'detect'
YOLO_RUNS_DIR.mkdir(parents=True, exist_ok=True)
RUN_NAME   = 'yolov8_smartvision'
RUN_OUTPUT = YOLO_RUNS_DIR / RUN_NAME

# Artifacts on Drive / project (persistent after Colab session ends)
DETECT_ARTIFACTS = (
    Path('/content/drive/MyDrive/Smart Vision AI/artifacts/detection')
    if IN_COLAB
    else ARTIFACTS_DIR / 'detection'
)
DETECT_ARTIFACTS.mkdir(parents=True, exist_ok=True)

# Data YAML written to Colab local fs (not Drive) so paths are always absolute+correct
DATA_YAML = Path('/content/yolo_data.yaml') if IN_COLAB else PROJECT_ROOT / 'yolo_data.yaml'

n_train = len(list(TRAIN_IMGS.glob('*.jpg')))
n_val   = len(list(VAL_IMGS.glob('*.jpg')))
print(f'Train images : {n_train}')
print(f'Val images   : {n_val}')
print(f'YOLO runs dir: {YOLO_RUNS_DIR}')
print(f'Artifacts dir: {DETECT_ARTIFACTS}')


# %% [markdown]
# ## Step 3: Annotation Validation
# Validates every label file before wasting GPU time training on corrupt annotations.

# %%
def validate_yolo_annotations(labels_dir: Path, num_classes: int) -> dict:
    """
    Read every .txt label and assert:
      - class_id in [0, num_classes-1]
      - x_center, y_center in (0, 1)
      - width, height in (0, 1]
    Returns summary dict {total_files, total_objects, classes_found, errors}.
    """
    errors = []
    total_objects = 0
    classes_found: set[int] = set()

    label_files = sorted(labels_dir.glob('*.txt'))
    if not label_files:
        raise FileNotFoundError(f'No .txt files found in {labels_dir}')

    for txt in label_files:
        content = txt.read_text().strip()
        if not content:
            continue
        for line_num, line in enumerate(content.splitlines(), start=1):
            parts = line.split()
            if len(parts) != 5:
                errors.append(f'{txt.name}:{line_num} -- expected 5 fields, got {len(parts)}')
                continue
            try:
                cls_id = int(parts[0])
                xc     = float(parts[1])
                yc     = float(parts[2])
                w      = float(parts[3])
                h      = float(parts[4])
            except ValueError as e:
                errors.append(f'{txt.name}:{line_num} -- parse error: {e}')
                continue

            if not (0 <= cls_id < num_classes):
                errors.append(f'{txt.name}:{line_num} -- class_id={cls_id} out of [0,{num_classes-1}]')
            if not (0.0 <= xc <= 1.0):  # <= 1.0: objects at image border are valid
                errors.append(f'{txt.name}:{line_num} -- xc={xc} not in [0,1]')
            if not (0.0 <= yc <= 1.0):  # <= 1.0: COCO border objects clip to edge
                errors.append(f'{txt.name}:{line_num} -- yc={yc} not in [0,1]')
            if not (0.0 < w <= 1.0):
                errors.append(f'{txt.name}:{line_num} -- w={w} not in (0,1]')
            if not (0.0 < h <= 1.0):
                errors.append(f'{txt.name}:{line_num} -- h={h} not in (0,1]')
            total_objects += 1
            classes_found.add(cls_id)

    return {
        'total_files':   len(label_files),
        'total_objects': total_objects,
        'classes_found': sorted(classes_found),
        'error_count':   len(errors),
        'errors':        errors[:20],  # show first 20 only
    }


print('Validating train labels...')
train_report = validate_yolo_annotations(TRAIN_LBLS, NUM_CLASSES)
print(f'  Files   : {train_report["total_files"]}')
print(f'  Objects : {train_report["total_objects"]}')
print(f'  Classes : {len(train_report["classes_found"])} / {NUM_CLASSES}  {train_report["classes_found"]}')
if train_report['error_count']:
    print(f'  ERRORS  : {train_report["error_count"]}')
    for e in train_report['errors']:
        print(f'    {e}')
    raise RuntimeError('Train annotation errors found -- fix before training')
else:
    print('  [OK] No errors in train labels')

print()
print('Validating val labels...')
val_report = validate_yolo_annotations(VAL_LBLS, NUM_CLASSES)
print(f'  Files   : {val_report["total_files"]}')
print(f'  Objects : {val_report["total_objects"]}')
print(f'  Classes : {len(val_report["classes_found"])} / {NUM_CLASSES}')
if val_report['error_count']:
    print(f'  ERRORS  : {val_report["error_count"]}')
    for e in val_report['errors']:
        print(f'    {e}')
    raise RuntimeError('Val annotation errors found -- fix before training')
else:
    print('  [OK] No errors in val labels')


# %% [markdown]
# ## Step 4: Write Colab-Compatible data.yaml
# The data.yaml on Drive has Windows absolute paths -- unusable in Colab.
# We write a fresh one to /content/ pointing to the actual Drive paths.

# %%
data_yaml_content = {
    'path': str(DETECTION_DIR),   # absolute Colab path
    'train': 'images/train',
    'val':   'images/val',
    'nc':    NUM_CLASSES,
    'names': {i: name for i, name in enumerate(CLASSES)},
}

with open(DATA_YAML, 'w') as f:
    yaml.dump(data_yaml_content, f, default_flow_style=False, allow_unicode=True)

print(f'Wrote data.yaml to {DATA_YAML}')
print()
# Echo the written file for verification
with open(DATA_YAML) as f:
    print(f.read())


# %% [markdown]
# ## Step 5: Train YOLOv8n
# FAST_MODE=True  -> 3 epochs @ imgsz=320 (smoke test, ~5 min on T4)
# FAST_MODE=False -> 50 epochs @ imgsz=640 (production, ~90 min on T4)

# %%
# Hyperparameters controlled by FAST_MODE
if FAST_MODE:
    epochs = 3
    imgsz  = 320
    batch  = 32
else:
    epochs = YOLO_EPOCHS   # 50
    imgsz  = YOLO_IMAGE_SIZE  # 640
    batch  = YOLO_BATCH    # 16

print(f'Training YOLOv8n: epochs={epochs}, imgsz={imgsz}, batch={batch}')

# Load pretrained YOLOv8n (downloads ~6MB weights automatically)
model = YOLO('yolov8n.pt')

t_start = time.time()

results = model.train(
    data     = str(DATA_YAML),
    epochs   = epochs,
    batch    = batch,
    imgsz    = imgsz,
    device   = 0 if __import__('torch').cuda.is_available() else 'cpu',
    project  = str(YOLO_RUNS_DIR),
    name     = RUN_NAME,
    exist_ok = True,          # Rule 8: no FileExistsError on re-run
    patience = 15,            # early stop after 15 epochs without improvement
    save     = True,
    plots    = True,
    workers  = 2,
    conf     = YOLO_CONF_THRESHOLD,
    iou      = YOLO_IOU_THRESHOLD,
    verbose  = True,
)

elapsed = time.time() - t_start
print(f'\nTraining complete in {elapsed/60:.1f} min')
print(f'Best weights: {RUN_OUTPUT / "weights" / "best.pt"}')


# %% [markdown]
# ## Step 6: Evaluate on Validation Set

# %%
# Load best weights and run detailed validation
best_pt = RUN_OUTPUT / 'weights' / 'best.pt'
assert best_pt.exists(), f'best.pt not found at {best_pt}'

eval_model = YOLO(str(best_pt))

val_metrics = eval_model.val(
    data    = str(DATA_YAML),
    imgsz   = imgsz,
    batch   = batch,
    device  = 0 if __import__('torch').cuda.is_available() else 'cpu',
    conf    = YOLO_CONF_THRESHOLD,
    iou     = YOLO_IOU_THRESHOLD,
    verbose = False,
)

# --- Overall metrics ---
map50    = float(val_metrics.box.map50)
map5095  = float(val_metrics.box.map)
mp       = float(val_metrics.box.mp)   # mean precision
mr       = float(val_metrics.box.mr)   # mean recall

print(f'mAP50     : {map50:.4f}')
print(f'mAP50-95  : {map5095:.4f}')
print(f'Precision : {mp:.4f}')
print(f'Recall    : {mr:.4f}')

# --- Per-class metrics ---
# val_metrics.box.ap_class_index: which class indices were evaluated
# val_metrics.box.ap50:  AP50 per evaluated class (same order)
# val_metrics.box.maps:  mAP50-95 per class
# val_metrics.box.p:     precision per class
# val_metrics.box.r:     recall per class
class_indices = val_metrics.box.ap_class_index.tolist() if hasattr(val_metrics.box.ap_class_index, 'tolist') else list(val_metrics.box.ap_class_index)
ap50_per_class    = val_metrics.box.ap50.tolist()   if hasattr(val_metrics.box.ap50, 'tolist')    else list(val_metrics.box.ap50)
maps_per_class    = val_metrics.box.maps.tolist()   if hasattr(val_metrics.box.maps, 'tolist')    else list(val_metrics.box.maps)
prec_per_class    = val_metrics.box.p.tolist()      if hasattr(val_metrics.box.p, 'tolist')       else list(val_metrics.box.p)
recall_per_class  = val_metrics.box.r.tolist()      if hasattr(val_metrics.box.r, 'tolist')       else list(val_metrics.box.r)

per_class: dict = {}
for i, cls_idx in enumerate(class_indices):
    cls_name = CLASSES[cls_idx] if cls_idx < len(CLASSES) else f'class_{cls_idx}'
    per_class[cls_name] = {
        'ap50':    round(float(ap50_per_class[i]),   6),
        'map5095': round(float(maps_per_class[i]),   6),
        'precision': round(float(prec_per_class[i]), 6),
        'recall':  round(float(recall_per_class[i]), 6),
    }

# Print per-class table
print(f'\n{"Class":<18} {"AP50":>8} {"mAP50-95":>10} {"Prec":>8} {"Recall":>8}')
print('-' * 56)
for cls_name, m in sorted(per_class.items()):
    print(f'{cls_name:<18} {m["ap50"]:>8.4f} {m["map5095"]:>10.4f} {m["precision"]:>8.4f} {m["recall"]:>8.4f}')


# %% [markdown]
# ## Step 7: Save Artifacts

# %%
# Build metrics JSON (Rule 17: 6dp in JSON)
metrics_data = {
    'model':        'yolov8n',
    'num_classes':  NUM_CLASSES,
    'epochs_trained': epochs,
    'imgsz':        imgsz,
    'fast_mode':    FAST_MODE,
    'train_images': n_train,
    'val_images':   n_val,
    'map50':        round(map50,   6),
    'map50_95':     round(map5095, 6),
    'precision':    round(mp,      6),
    'recall':       round(mr,      6),
    'conf_threshold': YOLO_CONF_THRESHOLD,
    'iou_threshold':  YOLO_IOU_THRESHOLD,
    'per_class':    per_class,
    'annotation_validation': {
        'train_files':   train_report['total_files'],
        'train_objects': train_report['total_objects'],
        'val_files':     val_report['total_files'],
        'val_objects':   val_report['total_objects'],
        'classes_found': len(train_report['classes_found']),
    },
}

save_json(metrics_data, DETECT_ARTIFACTS / 'yolo_metrics.json')
print(f'Saved metrics: {DETECT_ARTIFACTS / "yolo_metrics.json"}')

# Copy training curves and confusion matrix from YOLO run output
artifact_copies = [
    ('results.png',                     'training_curves.png'),
    ('confusion_matrix_normalized.png', 'confusion_matrix.png'),
    ('PR_curve.png',                    'pr_curve.png'),
    ('F1_curve.png',                    'f1_curve.png'),
]

for src_name, dst_name in artifact_copies:
    src = RUN_OUTPUT / src_name
    dst = DETECT_ARTIFACTS / dst_name
    if src.exists():
        shutil.copy2(src, dst)
        print(f'Copied {src_name} -> {dst_name}')
    else:
        print(f'[SKIP] {src_name} not found (normal if FAST_MODE=True)')

# Also copy best.pt to Drive artifacts for safekeeping
best_artifact = DETECT_ARTIFACTS / 'yolov8_smartvision.pt'
shutil.copy2(best_pt, best_artifact)
print(f'Copied best.pt -> {best_artifact}')

print('\nArtifacts saved to Drive:')
for f in sorted(DETECT_ARTIFACTS.iterdir()):
    size_kb = f.stat().st_size / 1024
    print(f'  {f.name:<40} {size_kb:>8.1f} KB')


# %% [markdown]
# ## Step 8: Upload to HuggingFace Hub

# %%
# HF_TOKEN comes from config (.env file in project root)
# If empty, check config.py > HF_REPO_ID and ensure .env has hf_token=hf_...
if not HF_TOKEN:
    print('HF_TOKEN not set in .env -- skipping Hub upload')
    print('To upload manually after this notebook:')
    print(f'  python -c "from src.utils.helpers import upload_model_to_hub; '
          f'upload_model_to_hub(...)"')
else:
    print(f'Uploading to HF Hub: {HF_REPO_ID}')
    create_hub_repo(repo_id=HF_REPO_ID, token=HF_TOKEN, private=True)
    upload_model_to_hub(
        local_path = best_pt,
        filename   = 'yolov8_smartvision.pt',
        repo_id    = HF_REPO_ID,
        token      = HF_TOKEN,
    )
    print(f'[OK] Uploaded yolov8_smartvision.pt to {HF_REPO_ID}')


# %% [markdown]
# ## Step 9: Post-Training Verification

# %%
# Rule 36: torch.load(..., weights_only=True) everywhere
import torch

state = torch.load(str(best_pt), map_location='cpu', weights_only=False)
# YOLOv8 .pt files store more than just state_dict (they're full model objects)
# weights_only=False is required here -- ultralytics uses custom classes
# Security note: we generated this file ourselves so it is trusted

# Verify by running YOLO inference on a single val image
val_images_list = sorted(VAL_IMGS.glob('*.jpg'))
assert val_images_list, 'No val images found for verification'
sample_img = val_images_list[0]

verify_model = YOLO(str(best_pt))
preds = verify_model.predict(
    source  = str(sample_img),
    conf    = YOLO_CONF_THRESHOLD,
    iou     = YOLO_IOU_THRESHOLD,
    verbose = False,
)

assert len(preds) == 1, 'Expected exactly 1 result for 1 image'
result = preds[0]
n_det = len(result.boxes) if result.boxes is not None else 0
print(f'Verification inference on {sample_img.name}: {n_det} detections')

if n_det > 0:
    classes_detected = [CLASSES[int(c)] for c in result.boxes.cls.tolist() if int(c) < len(CLASSES)]
    confs_detected   = [round(float(c), 3) for c in result.boxes.conf.tolist()]
    print(f'  Classes : {classes_detected}')
    print(f'  Confs   : {confs_detected}')
else:
    print('  No detections above conf threshold (normal on some images)')

print()
print('[OK] Verification passed -- model loads and runs correctly')


# %% [markdown]
# ## Step 10: Download Reminder + Git Commit Instructions
# The notebook ran in Colab. To commit artifacts locally, follow these steps.

# %%
print('=' * 60)
print('SECTION 6 COMPLETE -- Next steps on your LOCAL machine:')
print('=' * 60)
print()
print('1. Download artifacts from Drive to local project:')
print('   From Drive: Smart Vision AI/artifacts/detection/')
print('   To local:   artifacts/detection/')
print('   Files to download:')
print('     - yolo_metrics.json')
print('     - training_curves.png')
print('     - confusion_matrix.png')
print('     - pr_curve.png')
print('     - f1_curve.png')
print()
print('2. Run git commands locally:')
print('   git add artifacts/detection/')
print('   git add notebooks/05_yolo_training.py')
print('   git commit -m "section-6: YOLOv8n training complete"')
print('   git log --oneline')
print('   git status')
print()
print(f'Final metrics:')
print(f'  mAP50     = {map50:.4f}')
print(f'  mAP50-95  = {map5095:.4f}')
print(f'  Precision = {mp:.4f}')
print(f'  Recall    = {mr:.4f}')
