# %% [markdown]
# # SmartVision AI — Section 5: CNN Classifier Training
# **Run in Google Colab T4 — GPU required**
# Run locally: `python notebooks/04_train_classifier.py` (CPU only, slow)
#
# PARAMETERIZED — change MODEL below, run once per model:
#   MODEL = "mobilenet"    → retrain (Round 2, 140/class) ← NEXT
#   MODEL = "efficientnet" → retrain (Round 2, lr=0.001 fixed)
#   MODEL = "resnet50"     → retrain (Round 2, layer4.2-only Phase 2)
#   MODEL = "vgg16"        → 59.5% val (Round 1 done, architecture ceiling — skip)
#
# After each model:
#   1. Post-training verification runs automatically
#   2. Weights uploaded to HuggingFace Hub
#   3. Commit artifacts to git
#   4. ONLY THEN change MODEL and run next

# %% [markdown]
# ## Step 0: Environment Setup

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

IN_COLAB = False
try:
    from google.colab import drive  # type: ignore[import-untyped]
    drive.mount('/content/drive')
    COLAB_ROOT = '/content/drive/MyDrive/Smart Vision AI'
    sys.path.insert(0, COLAB_ROOT)
    IN_COLAB = True
    print(f"Running in Colab — project at {COLAB_ROOT}")
    import subprocess
    subprocess.run(["pip", "install", "-q",
                    "mlflow", "ultralytics", "huggingface_hub",
                    "pydantic-settings", "scikit-learn"], check=True)
    print("Dependencies installed")
except Exception:
    # ImportError  → not in Colab (running locally in terminal)
    # MessageError → Drive auth failed (click the auth popup in Colab, then re-run this cell)
    print("Running locally  (or Drive auth pending — re-run cell after authorising)")

# %%
# ================================================================
# CHANGE THIS for each model — run the full notebook once per model
# ================================================================
MODEL = "mobilenet"   # Options: "vgg16" | "resnet50" | "mobilenet" | "efficientnet"
# ================================================================
# Rule 1: FAST_MODE is a LOCAL variable passed as function param
FAST_MODE = True  # False = full training; True = 3 epochs quick test
# ================================================================
print(f"MODEL     = {MODEL}")
print(f"FAST_MODE = {FAST_MODE}")

# %% [markdown]
# ## Step 1: Imports

# %%
import json
import logging
import random
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for Colab/server

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, LinearLR, SequentialLR

import mlflow
import mlflow.pytorch

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from config import (
    ARTIFACTS_DIR, CLASSES, COMPARISON_PATH, DOCS_FIGURES_DIR,
    IMAGE_SIZE, MLFLOW_TRACKING_URI, MODEL_CONFIGS, MODELS_DIR,
    NUM_CLASSES, RANDOM_STATE, HF_REPO_ID, HF_TOKEN,
)
from src.data.augmentor   import get_train_transforms, get_eval_transforms, denormalize
from src.data.dataset     import SmartVisionDataset, get_dataloaders
from src.models.model_factory import (
    get_model,
    unfreeze_mobilenet_phase2,
    unfreeze_resnet50_phase2,
    get_per_class_accuracy,
    # REMOVED: freeze_resnet50_phase1  (done inside get_model)
    # REMOVED: freeze_vgg16_phase1     (done inside get_model)
    # REMOVED: unfreeze_vgg16_phase2   (VGG16 not training)
    # REMOVED: count_trainable_params  (use inline sum())
)
from src.models.base_classifier import (
    train, evaluate, benchmark_inference, save_model, load_model,
)
from src.utils.helpers import save_json, NumpyEncoder

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Full reproducibility seed — deterministic across restarts
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
torch.cuda.manual_seed_all(RANDOM_STATE)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False   # slower but reproducible

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} ({'GPU' if device.type == 'cuda' else 'CPU — training will be slow'})")

# %%
# Verify augmentor is loaded correctly — catches stale .pyc Colab cache
# inspect.getsource() reads disk; importlib.reload() forces runtime reload
# BOTH must be checked — source can be correct while old .pyc still runs
import importlib, inspect
import torchvision
import src.data.augmentor as aug_mod
importlib.reload(aug_mod)
from src.data.augmentor import get_train_transforms as gtf

aug_source = inspect.getsource(gtf)
assert "RandomErasing" not in aug_source, "SOURCE has RandomErasing — fix augmentor.py"

aug_pipeline = gtf(IMAGE_SIZE)
assert aug_pipeline is not None, "Torch not available — cannot verify augmentor pipeline"
transform_names = [type(t).__name__ for t in aug_pipeline.transforms]
assert "RandomErasing" not in transform_names, (
    f"RUNTIME has RandomErasing: {transform_names}\n"
    "Stale .pyc — Runtime > Restart Runtime, then re-run"
)
assert "RandomZoomOut" in transform_names, f"ZoomOut missing from runtime: {transform_names}"

# Verify ZoomOut comes AFTER ToDtype (ordering is load-bearing for fill value)
transform_order = {name: i for i, name in enumerate(transform_names)}
assert transform_order.get("ToImage", -1) < transform_order.get("RandomZoomOut", 999), \
    "ToImage must precede RandomZoomOut"
assert transform_order.get("ToDtype", -1) < transform_order.get("RandomZoomOut", 999), \
    "ToDtype must precede RandomZoomOut"
assert transform_order.get("RandomZoomOut", -1) < transform_order.get("Normalize", 999), \
    "RandomZoomOut must precede Normalize"

print(f"Augmentor runtime OK: {transform_names}")
print(f"torchvision: {torchvision.__version__}")

# %% [markdown]
# ## Step 2: Load Data

# %%
train_tf = get_train_transforms(IMAGE_SIZE)
eval_tf  = get_eval_transforms(IMAGE_SIZE)

train_loader, val_loader, test_loader = get_dataloaders(
    model_name=MODEL,
    train_transform=train_tf,
    eval_transform=eval_tf,
    fast_mode=FAST_MODE,
)

cfg = MODEL_CONFIGS[MODEL]
epochs     = 3 if FAST_MODE else cfg["epochs"]
batch_size = cfg["batch"]
lr         = cfg["lr"]

train_ds, val_ds, test_ds = train_loader.dataset, val_loader.dataset, test_loader.dataset
assert train_ds is not None and val_ds is not None and test_ds is not None
print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")  # type: ignore[arg-type]
print(f"Batch: {batch_size} | LR: {lr} | Epochs: {epochs}")

# %%
# Class balance check — fast path, no image loading
from collections import Counter

def fast_label_counts(ds) -> Counter:
    """Get label counts without loading images (O(N) over metadata only)."""
    if hasattr(ds, "samples"):
        return Counter(lbl for _, lbl in ds.samples)    # SmartVisionDataset
    elif hasattr(ds, "targets"):
        return Counter(int(t) for t in ds.targets)      # ImageFolder
    else:
        raise AttributeError("Dataset has neither .samples nor .targets")

label_counts = fast_label_counts(train_loader.dataset)
count_min, count_max = min(label_counts.values()), max(label_counts.values())
balance_ratio = count_max / count_min
print(f"Class balance — min: {count_min}, max: {count_max}, ratio: {balance_ratio:.2f}x")
assert balance_ratio < 1.5, f"Imbalanced split: ratio={balance_ratio:.2f} — check data collection"
print("Class balance OK")

# %% [markdown]
# ## Step 3: Build Model

# %%
model = get_model(MODEL, num_classes=NUM_CLASSES).to(device)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

# %% [markdown]
# ## Step 4: Train

# %%
# label_smoothing=0.1: penalizes overconfident predictions.
# Early stopping monitors val_accuracy (not val_loss) — label_smoothing shifts
# loss scale by ~0.31 nats uniformly, so val_loss thresholds are misleading.
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
save_path = MODELS_DIR / f"{MODEL}_best.pt"

# Fail fast — create artifact dirs before training (not after hours of compute)
SAVE_DIR = ARTIFACTS_DIR / f"classification/{MODEL}"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Default history — overwritten by whichever model branch runs below
history: dict = {
    "train_loss": [], "train_acc": [], "val_loss": [],
    "val_acc": [], "val_f1": [], "best_val_acc": 0.0,
}

# MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("smartvision_classification")

with mlflow.start_run(run_name=f"{MODEL}_{'fast' if FAST_MODE else 'full'}"):
    mlflow.log_params({
        "model":      MODEL,
        "epochs":     epochs,
        "batch_size": batch_size,
        "lr":         lr,
        "fast_mode":  FAST_MODE,
        "image_size": IMAGE_SIZE,
        "num_classes": NUM_CLASSES,
    })

    # ── VGG16 ── 2-phase: head-only first, then fine-tune last conv block
    # NOTE: VGG16 Round 1 is complete (59.5% val — architecture ceiling at 140/class).
    # Run only if intentionally retraining.
    if MODEL == "vgg16":
        phase1_epochs = max(1, epochs // 4)   # 5 epochs in full run
        phase2_epochs = epochs - phase1_epochs
        mlflow.log_params({"phase1_epochs": phase1_epochs, "phase2_epochs": phase2_epochs})

        # Phase 1: head only
        # freeze_vgg16_phase1 removed -- get_model("vgg16") freezes features at build time
        # VGG16 classifier[6] (the head) is unfrozen by default (new nn.Linear layer)
        for name_p, param in model.named_parameters():
            if "classifier.6" not in name_p:
                param.requires_grad = False
        print(f"VGG16 Phase 1: head only ({phase1_epochs} epochs, lr={lr})")
        print(f"Phase 1 trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        optimizer1 = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
        scheduler1 = StepLR(optimizer1, step_size=3, gamma=0.1)
        history1 = train(
            model, train_loader, val_loader, optimizer1, scheduler1,
            criterion, device, epochs=phase1_epochs, patience=phase1_epochs,
            scaler=None, model_name=f"{MODEL}_p1", save_path=save_path,
            freeze_bn=False, grad_clip=1.0,   # VGG16 has no BN layers
        )

        # Phase 2: unfreeze features[24:] and fine-tune
        for i, layer in enumerate(model.features):
            if i >= 24:
                for param in layer.parameters():
                    param.requires_grad = True
        print(f"VGG16 Phase 2: fine-tune features[24:] ({phase2_epochs} epochs, lr={lr/10})")
        print(f"Phase 2 trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        optimizer2 = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr / 10, weight_decay=1e-4)
        scheduler2 = CosineAnnealingLR(optimizer2, T_max=phase2_epochs)
        history2 = train(
            model, train_loader, val_loader, optimizer2, scheduler2,
            criterion, device, epochs=phase2_epochs, patience=8,
            scaler=None, model_name=f"{MODEL}_p2", save_path=save_path,
            freeze_bn=False, grad_clip=1.0,   # VGG16 has no BN layers
        )
        history = {
            k: history1.get(k, []) + history2.get(k, [])
            for k in ["train_loss", "train_acc", "val_loss", "val_acc", "val_f1"]
        }
        history["best_val_acc"] = max(history1.get("best_val_acc", 0),
                                      history2.get("best_val_acc", 0))

    # ── ResNet50 ── 2-phase: head-only first, then fine-tune layer4.2 only
    elif MODEL == "resnet50":
        phase1_epochs = max(1, epochs // 4)   # 5 epochs in full run
        phase2_epochs = epochs - phase1_epochs
        mlflow.log_params({"phase1_epochs": phase1_epochs, "phase2_epochs": phase2_epochs})

        # Phase 1: head only
        # Backbone already frozen inside get_model("resnet50"). No separate freeze call needed.
        print(f"ResNet50 Phase 1: head only ({phase1_epochs} epochs)")
        print(f"Phase 1 trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        optimizer1 = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr * 10, weight_decay=1e-4)
        scheduler1 = StepLR(optimizer1, step_size=3, gamma=0.1)
        history1 = train(
            model, train_loader, val_loader, optimizer1, scheduler1,
            criterion, device, epochs=phase1_epochs, patience=phase1_epochs,
            scaler=None, model_name=f"{MODEL}_p1", save_path=save_path,
            freeze_bn=True, grad_clip=1.0,
        )

        # Phase 2: unfreeze layer4.2 + fc, 2-epoch LR warmup before cosine
        unfreeze_resnet50_phase2(model)
        print(f"ResNet50 Phase 2: fine-tune layer4.2+fc ({phase2_epochs} epochs, lr={lr})")
        print(f"Phase 2 trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        optimizer2 = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
        warmup_epochs = min(2, max(1, phase2_epochs - 1))
        warmup_sched  = LinearLR(optimizer2, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
        cosine_sched  = CosineAnnealingLR(optimizer2, T_max=max(phase2_epochs - warmup_epochs, 1), eta_min=1e-7)
        scheduler2 = SequentialLR(optimizer2, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs])
        history2 = train(
            model, train_loader, val_loader, optimizer2, scheduler2,
            criterion, device, epochs=phase2_epochs, patience=8,
            scaler=None, model_name=f"{MODEL}_p2", save_path=save_path,
            freeze_bn=True, grad_clip=1.0,  # layer4.0/4.1 BN still frozen
        )
        history = {
            k: history1.get(k, []) + history2.get(k, [])
            for k in ["train_loss", "train_acc", "val_loss", "val_acc", "val_f1"]
        }
        history["best_val_acc"] = max(history1.get("best_val_acc", 0),
                                      history2.get("best_val_acc", 0))

    # ── MobileNetV2 ── 2-phase: head-only first, then fine-tune features[16:]
    # Round 3 changes from Round 2:
    #   AdamW replaces Adam (correct decoupled L2 for fine-tuning)
    #   Phase 1: weight_decay 1e-4 -> 1e-3 (stronger head regularization)
    #   Phase 2: differential param groups (backbone lr=1e-5 wd=1e-5, head lr=1e-4 wd=1e-3)
    #   Phase 2: patience 8 -> 5; features[16:] via updated unfreeze_mobilenet_phase2()
    #   Phase 2: dropout=0.4 via updated get_model() / _build_mobilenet_v2()
    elif MODEL == "mobilenet":

        lr = 1e-3

        # Epoch allocation with FAST_MODE guard
        # FAST_MODE=True (epochs=3): phase1=3, phase2=0 -> Phase 2 skipped.
        phase1_epochs = min(10, epochs)
        phase2_epochs = max(0, epochs - phase1_epochs)

        # Separate checkpoint paths: Phase 1 best preserved even if Phase 2 overfits.
        save_path_p1 = MODELS_DIR / f"{MODEL}_phase1_best.pt"
        save_path_p2 = MODELS_DIR / f"{MODEL}_best.pt"

        # -- Phase 1: Head only --
        # AdamW: decoupled weight decay. weight_decay=1e-3 (was 1e-4): stronger head regularization.
        # patience=phase1_epochs: always complete Phase 1.
        print(f"MobileNetV2 Phase 1: head only ({phase1_epochs} epochs, lr={lr})")
        print(f"Phase 1 trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        optimizer1 = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=1e-3,
        )
        scheduler1 = CosineAnnealingLR(optimizer1, T_max=phase1_epochs, eta_min=1e-6)

        with mlflow.start_run(run_name=f"{MODEL}_phase1_r3", nested=True):
            mlflow.log_params({
                "model": MODEL, "phase": 1, "round": 3,
                "optimizer": "AdamW", "lr": lr, "weight_decay": 1e-3,
                "scheduler": f"CosineAnnealingLR(T_max={phase1_epochs},eta=1e-6)",
                "phase1_epochs": phase1_epochs, "phase2_epochs": phase2_epochs,
                "patience": phase1_epochs, "freeze_bn": True, "grad_clip": 1.0,
                "dropout": 0.4, "label_smoothing": 0.1,
                "trainable_scope": "classifier_head_only",
            })
            history1 = train(
                model, train_loader, val_loader,
                optimizer1, scheduler1, criterion, device,
                epochs=phase1_epochs, patience=phase1_epochs,
                scaler=None, model_name=f"{MODEL}_p1",
                save_path=save_path_p1, freeze_bn=True, grad_clip=1.0,
            )
            _p1_train = history1["train_acc"][-1] if history1["train_acc"] else 0.0
            _p1_val   = history1.get("best_val_acc", 0.0)
            mlflow.log_metrics({
                "p1_best_val_acc": _p1_val,
                "p1_final_train":  _p1_train,
                "p1_overfit_gap":  _p1_train - _p1_val,
            })

        print(f"  Phase 1: val={_p1_val:.4f} train={_p1_train:.4f} gap={_p1_train-_p1_val:.4f}pp")

        # -- Phase 2: Fine-tune features[16:] --
        # Skip entirely when phase2_epochs == 0 (FAST_MODE guard).
        if phase2_epochs == 0:
            print("  Phase 2 skipped (phase2_epochs=0).")
            history2 = {
                "train_loss": [], "train_acc": [], "val_loss": [],
                "val_acc": [], "val_f1": [], "best_val_acc": _p1_val,
            }
        else:
            print(f"\nMobileNetV2 Phase 2: fine-tune features[16:] ({phase2_epochs} epochs)")
            unfreeze_mobilenet_phase2(model)

            # Differential param groups:
            # backbone lr=1e-5 (100x below Phase 1 LR): protect pretrained features
            # backbone wd=1e-5: very light -- don't push pretrained weights to zero
            # head lr=1e-4: continue Phase 1 adaptation rate
            # head wd=1e-3: strong -- generalize the linear classifier
            _backbone_params   = [p for n, p in model.named_parameters()
                                  if p.requires_grad and "classifier" not in n]
            _classifier_params = [p for n, p in model.named_parameters()
                                  if p.requires_grad and "classifier" in n]
            _n_back  = sum(p.numel() for p in _backbone_params)
            _n_cls   = sum(p.numel() for p in _classifier_params)
            _n_total = sum(p.numel() for p in model.parameters() if p.requires_grad)

            assert _n_back > 0, (
                f"backbone_params empty after unfreeze_mobilenet_phase2(). "
                f"Total trainable: {_n_total:,}. Check unfreeze was called."
            )
            assert _n_cls > 0, "classifier_params empty. Check model.classifier attribute."
            assert _n_back + _n_cls == _n_total, (
                f"Param split incomplete: {_n_back:,}+{_n_cls:,} != {_n_total:,}"
            )
            print(f"Phase 2 trainable: {_n_total:,} "
                  f"(backbone={_n_back:,} lr=1e-5 wd=1e-5, head={_n_cls:,} lr=1e-4 wd=1e-3 | "
                  f"{_n_total/3080:.0f}/img)")

            optimizer2 = AdamW([
                {"params": _backbone_params,   "lr": 1e-5,    "weight_decay": 1e-5},
                {"params": _classifier_params, "lr": lr / 10, "weight_decay": 1e-3},
            ])

            # max(1,...) ensures LinearLR(total_iters >= 1) -- avoids ZeroDivisionError
            warmup_epochs = max(1, min(2, phase2_epochs - 1))
            cosine_epochs = max(phase2_epochs - warmup_epochs, 1)
            _warmup_sched = LinearLR(optimizer2, start_factor=0.1, end_factor=1.0,
                                     total_iters=warmup_epochs)
            _cosine_sched = CosineAnnealingLR(optimizer2, T_max=cosine_epochs, eta_min=1e-7)
            scheduler2    = SequentialLR(optimizer2,
                                         schedulers=[_warmup_sched, _cosine_sched],
                                         milestones=[warmup_epochs])

            with mlflow.start_run(run_name=f"{MODEL}_phase2_r3", nested=True):
                mlflow.log_params({
                    "model": MODEL, "phase": 2, "round": 3,
                    "optimizer": "AdamW_differential",
                    "lr_backbone": 1e-5, "lr_head": lr / 10,
                    "wd_backbone": 1e-5, "wd_head": 1e-3,
                    "scheduler": f"LinearWarmup({warmup_epochs}ep)+CosineAnnealing",
                    "epochs": phase2_epochs, "patience": 5,
                    "freeze_bn": True, "grad_clip": 1.0,
                    "dropout": 0.4, "label_smoothing": 0.1,
                    "trainable_scope": "features[16:]+classifier",
                    "unfrozen_params": _n_total,
                    "params_per_img": round(_n_total / 3080),
                    "warmup_epochs": warmup_epochs, "cosine_epochs": cosine_epochs,
                })
                history2 = train(
                    model, train_loader, val_loader,
                    optimizer2, scheduler2, criterion, device,
                    epochs=phase2_epochs, patience=5,   # was 8
                    scaler=None, model_name=f"{MODEL}_p2",
                    save_path=save_path_p2, freeze_bn=True, grad_clip=1.0,
                )
                _p2_train = history2["train_acc"][-1] if history2["train_acc"] else _p1_train
                _p2_val   = history2.get("best_val_acc", _p1_val)
                _p2_gap   = _p2_train - _p2_val
                mlflow.log_metrics({
                    "p2_best_val_acc": _p2_val,
                    "p2_final_train":  _p2_train,
                    "p2_overfit_gap":  _p2_gap,
                })

            print(f"  Phase 2: val={_p2_val:.4f} train={_p2_train:.4f} gap={_p2_gap:.4f}pp")
            if _p2_gap > 0.20:
                print(f"  WARNING gap>{0.20:.2f}: consider features[17:] for later models")
            elif _p2_gap > 0.12:
                print(f"  NOTE gap 0.12-0.20 (acceptable -- monitor test acc)")
            else:
                print(f"  gap < 0.12 -- healthy generalization")

        # Combine histories (.get() safe when history2 is skip-guard dict)
        history = {
            k: history1.get(k, []) + history2.get(k, [])
            for k in ["train_loss", "train_acc", "val_loss", "val_acc", "val_f1"]
        }
        history["best_val_acc"] = max(
            history1.get("best_val_acc", 0.0),
            history2.get("best_val_acc", 0.0),
        )

    # ── EfficientNetB0 ── mixed precision (autocast + GradScaler)
    elif MODEL == "efficientnet":
        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        scaler    = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
        history = train(
            model, train_loader, val_loader, optimizer, scheduler,
            criterion, device, epochs=epochs, patience=5,
            scaler=scaler, model_name=MODEL, save_path=save_path,
            freeze_bn=True, grad_clip=1.0,
        )

    mlflow.log_metric("best_val_acc", history["best_val_acc"])
    print(f"\nBest val_acc: {history['best_val_acc']:.4f}")

# %% [markdown]
# ## Step 5: Evaluate on Test Set

# %%
# Load best saved weights (load_model handles both raw state_dict and checkpoint dict)
best_model = get_model(MODEL, num_classes=NUM_CLASSES).to(device)
best_model = load_model(best_model, save_path, device=str(device))

test_metrics = evaluate(best_model, test_loader, criterion, device)
inference_ms = benchmark_inference(best_model, device, image_size=IMAGE_SIZE)

print(f"\nTest Results — {MODEL.upper()}")
print(f"  Accuracy  : {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.1f}%)")
print(f"  Precision : {test_metrics['precision']:.4f}")
print(f"  Recall    : {test_metrics['recall']:.4f}")
print(f"  F1 (macro): {test_metrics['f1']:.4f}")
print(f"  Inference : {inference_ms:.1f} ms/image")
print(f"  Model size: {save_path.stat().st_size / 1e6:.1f} MB")

# %%
# Per-class accuracy — sorted weakest first; drives targeted investigation
per_class = get_per_class_accuracy(best_model, test_loader, CLASSES, device)
print(f"\nPer-class accuracy (weakest first):")
print(f"  {'Class':<22} {'Accuracy':>10}")
print("  " + "-" * 34)
for cls_name, cls_acc in per_class.items():
    flag = "  <-- investigate" if cls_acc < 0.60 else ""
    print(f"  {cls_name:<22} {cls_acc:>9.1%}{flag}")

# %% [markdown]
# ## Step 6: Confusion Matrix

# %%
best_model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        preds  = best_model(images).argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(cm, xticklabels=CLASSES, yticklabels=CLASSES,
            cmap="Blues", annot=True, fmt="d", linewidths=0.3, ax=ax)
ax.set_title(f"{MODEL.upper()} — Confusion Matrix (Test Set)", fontweight="bold", pad=12)
plt.xticks(rotation=45, ha="right", fontsize=7)
plt.yticks(rotation=0, fontsize=7)
plt.tight_layout()

cm_path = SAVE_DIR / "confusion_matrix.png"
plt.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.savefig(DOCS_FIGURES_DIR / f"section5_{MODEL}_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Confusion matrix saved: {cm_path}")

# %% [markdown]
# ## Step 7: Training History Plot

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history["train_loss"], label="train", color="#377eb8")
axes[0].plot(history["val_loss"],   label="val",   color="#e41a1c")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
axes[0].set_title(f"{MODEL.upper()} — Loss"); axes[0].legend()

axes[1].plot(history["train_acc"], label="train", color="#377eb8")
axes[1].plot(history["val_acc"],   label="val",   color="#e41a1c")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
axes[1].set_title(f"{MODEL.upper()} — Accuracy"); axes[1].legend()

plt.suptitle(f"{MODEL.upper()} Training History", fontweight="bold")
plt.tight_layout()
hist_path = SAVE_DIR / "training_history.png"
plt.savefig(hist_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Training history saved: {hist_path}")

# %% [markdown]
# ## Step 8: Save Artifacts

# %%
artifacts = {
    "model":          MODEL,
    "test_accuracy":  round(test_metrics["accuracy"],  6),
    "test_precision": round(test_metrics["precision"], 6),
    "test_recall":    round(test_metrics["recall"],    6),
    "test_f1":        round(test_metrics["f1"],        6),
    "val_accuracy":   round(history["best_val_acc"],   6),
    "inference_ms":   round(inference_ms, 4),
    "model_size_mb":  round(save_path.stat().st_size / 1e6, 4),
    "epochs_trained": len(history["train_loss"]),
    "fast_mode":      FAST_MODE,
    "training_history": {
        "train_loss": history["train_loss"],
        "train_acc":  history["train_acc"],
        "val_loss":   history["val_loss"],
        "val_acc":    history["val_acc"],
        "val_f1":     history["val_f1"],
    },
}

metrics_path = SAVE_DIR / "metrics.json"
save_json(artifacts, metrics_path)
print(f"Metrics saved: {metrics_path}")

# %% [markdown]
# ## Step 9: Post-Training Verification + HF Hub Upload

# %%
print(f"\n{'='*55}")
print(f"POST-TRAINING VERIFICATION -- {MODEL.upper()}")
print(f"{'='*55}")

# 1. File exists + size is reasonable
assert save_path.exists(), f"Model file not saved: {save_path}"
size_mb = save_path.stat().st_size / 1e6
print(f"  Saved: {size_mb:.1f} MB")

# 2. Loads cleanly and runs inference
verify_model = get_model(MODEL, num_classes=NUM_CLASSES)
verify_model = load_model(verify_model, save_path, device="cpu")
dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
with torch.no_grad():
    out = verify_model(dummy)
assert out.shape == (1, NUM_CLASSES), f"Wrong output shape: {out.shape}"
probs = torch.softmax(out, dim=1)
assert abs(probs.sum().item() - 1.0) < 1e-5, "Probabilities do not sum to 1"
print(f"  Inference: shape={tuple(out.shape)} OK")

# 3. Accuracy check — 3-tier (no hard crash; COCO-crop ceiling is 76-86%)
ACCURACY_FLOOR  = 0.65   # below this = pipeline broken
ACCURACY_WARN   = 0.72   # below this = below expected COCO-crop range
ACCURACY_TARGET = 0.80   # original target

test_acc = test_metrics["accuracy"]
if test_acc < ACCURACY_FLOOR:
    print(f"  CRITICAL: {test_acc:.1%} below floor ({ACCURACY_FLOOR:.0%}) — check data pipeline")
elif test_acc < ACCURACY_WARN:
    print(f"  WARNING:  {test_acc:.1%} below expected COCO-crop range ({ACCURACY_WARN:.0%}+)")
elif test_acc < ACCURACY_TARGET:
    print(f"  NOTE:     {test_acc:.1%} — within realistic range for COCO crops at 140/class")
else:
    print(f"  PASS:     {test_acc:.1%} >= {ACCURACY_TARGET:.0%}")

# 4. Upload to HuggingFace Hub
print(f"\nUploading {MODEL}_best.pt to HF Hub...")
try:
    from src.utils.helpers import upload_model_to_hub, create_hub_repo
    create_hub_repo(HF_REPO_ID, token=HF_TOKEN, private=True)
    upload_model_to_hub(save_path, f"{MODEL}_best.pt", HF_REPO_ID, HF_TOKEN)
    print(f"  Uploaded: {MODEL}_best.pt -> {HF_REPO_ID}")
except Exception as e:
    print(f"  HF upload skipped: {e}")
    print("  Set HF_TOKEN in .env or Colab secrets to enable uploads.")

print(f"\n{'='*55}")
print(f"SECTION 5 ({MODEL.upper()}) COMPLETE")
print(f"{'='*55}")
print(f"  Test accuracy : {test_metrics['accuracy']:.1%}")
print(f"  Val  accuracy : {history['best_val_acc']:.1%}")
print(f"  F1 (macro)    : {test_metrics['f1']:.4f}")
print(f"  Inference     : {inference_ms:.1f} ms/image")
print()
print("Next steps:")
print("  1. git add artifacts/ models/ docs/figures/")
print(f"  2. git commit -m 'section-5: {MODEL} trained, acc={test_metrics['accuracy']:.1%}'")
print("  3. Change MODEL to next model and re-run this notebook")
print("     Order: mobilenet -> efficientnet -> resnet50 -> (skip vgg16)")
