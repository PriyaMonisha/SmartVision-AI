# %% [markdown]
# # SmartVision AI — Section 5: CNN Classifier Training
# **Run in Google Colab T4 — GPU required**
# Run locally: `python notebooks/04_train_classifier.py` (CPU only, slow)
#
# PARAMETERIZED — change MODEL below, run once per model:
#   MODEL = "vgg16"        → ~20 min on T4
#   MODEL = "resnet50"     → ~30 min on T4 (2-phase training)
#   MODEL = "mobilenet"    → ~18 min on T4
#   MODEL = "efficientnet" → ~25 min on T4 (mixed precision)
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from google.colab import drive
    drive.mount('/content/drive')
    COLAB_ROOT = '/content/drive/MyDrive/SmartVisionAI'
    sys.path.insert(0, COLAB_ROOT)
    print("Running in Colab")
    # Install dependencies in Colab
    # !pip install -q ultralytics huggingface_hub pydantic-settings scikit-learn mlflow
except ImportError:
    print("Running locally")

# %%
# ================================================================
# CHANGE THIS for each model — run the full notebook once per model
# ================================================================
MODEL = "vgg16"   # Options: "vgg16" | "resnet50" | "mobilenet" | "efficientnet"
# ================================================================
# Rule 38: FAST_MODE is a LOCAL variable passed as function param
FAST_MODE = False  # False = full training; True = 3 epochs quick test
# ================================================================
print(f"MODEL     = {MODEL}")
print(f"FAST_MODE = {FAST_MODE}")

# %% [markdown]
# ## Step 1: Imports

# %%
import json
import logging
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for Colab/server

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

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
from src.models.model_factory import get_model, freeze_resnet50_phase1, unfreeze_resnet50_phase2, count_trainable_params
from src.models.base_classifier import (
    train, evaluate, benchmark_inference, save_model, load_model,
)
from src.utils.helpers import save_json, NumpyEncoder

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

torch.manual_seed(RANDOM_STATE)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} ({'GPU' if device.type == 'cuda' else 'CPU — training will be slow'})")

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

print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")
print(f"Batch: {batch_size} | LR: {lr} | Epochs: {epochs}")

# %% [markdown]
# ## Step 3: Build Model

# %%
model = get_model(MODEL, num_classes=NUM_CLASSES).to(device)
trainable = count_trainable_params(model)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

# %% [markdown]
# ## Step 4: Train

# %%
criterion = nn.CrossEntropyLoss()
save_path = MODELS_DIR / f"{MODEL}_best.pt"

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

    # ── VGG16 ── single-phase, frozen features
    if MODEL == "vgg16":
        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
        scheduler = StepLR(optimizer, step_size=7, gamma=0.1)
        scaler    = None

        print("VGG16: training classifier head only (features frozen)")
        history = train(
            model, train_loader, val_loader, optimizer, scheduler,
            criterion, device, epochs=epochs, patience=5,
            scaler=None, model_name=MODEL, save_path=save_path,
        )

    # ── ResNet50 ── 2-phase: head-only first, then fine-tune layer3+
    elif MODEL == "resnet50":
        phase1_epochs = max(1, epochs // 4)   # 5 epochs in full run
        phase2_epochs = epochs - phase1_epochs

        # Phase 1: head only
        freeze_resnet50_phase1(model)
        print(f"ResNet50 Phase 1: head only ({phase1_epochs} epochs)")
        optimizer1 = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr * 10, weight_decay=1e-4)
        scheduler1 = StepLR(optimizer1, step_size=3, gamma=0.1)
        history1 = train(
            model, train_loader, val_loader, optimizer1, scheduler1,
            criterion, device, epochs=phase1_epochs, patience=phase1_epochs,
            scaler=None, model_name=f"{MODEL}_p1", save_path=save_path,
        )

        # Phase 2: unfreeze layer3+ and fine-tune
        unfreeze_resnet50_phase2(model)
        print(f"ResNet50 Phase 2: fine-tune layer3+ ({phase2_epochs} epochs, lr={lr})")
        optimizer2 = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
        scheduler2 = CosineAnnealingLR(optimizer2, T_max=phase2_epochs)
        history2 = train(
            model, train_loader, val_loader, optimizer2, scheduler2,
            criterion, device, epochs=phase2_epochs, patience=8,
            scaler=None, model_name=f"{MODEL}_p2", save_path=save_path,
        )
        # Merge histories
        history = {
            k: history1.get(k, []) + history2.get(k, [])
            for k in ["train_loss", "train_acc", "val_loss", "val_acc", "val_f1"]
        }
        history["best_val_acc"] = max(history1.get("best_val_acc", 0),
                                      history2.get("best_val_acc", 0))

    # ── MobileNetV2 ── single-phase, frozen features
    elif MODEL == "mobilenet":
        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        history = train(
            model, train_loader, val_loader, optimizer, scheduler,
            criterion, device, epochs=epochs, patience=5,
            scaler=None, model_name=MODEL, save_path=save_path,
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
        )

    mlflow.log_metric("best_val_acc", history["best_val_acc"])
    print(f"\nBest val_acc: {history['best_val_acc']:.4f}")

# %% [markdown]
# ## Step 5: Evaluate on Test Set

# %%
# Load best saved weights
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

cm_path = ARTIFACTS_DIR / f"classification/{MODEL}/confusion_matrix.png"
cm_path.parent.mkdir(parents=True, exist_ok=True)
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
hist_path = ARTIFACTS_DIR / f"classification/{MODEL}/training_history.png"
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

metrics_path = ARTIFACTS_DIR / f"classification/{MODEL}/metrics.json"
metrics_path.parent.mkdir(parents=True, exist_ok=True)
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
verify_model = load_model(verify_model, save_path, device="cpu")  # weights_only=True inside
dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
with torch.no_grad():
    out = verify_model(dummy)
assert out.shape == (1, NUM_CLASSES), f"Wrong output shape: {out.shape}"
probs = torch.softmax(out, dim=1)
assert abs(probs.sum().item() - 1.0) < 1e-5, "Probabilities do not sum to 1"
print(f"  Inference: shape={tuple(out.shape)} OK")

# 3. Accuracy meets 80% threshold
assert test_metrics["accuracy"] >= 0.80, (
    f"ACCURACY {test_metrics['accuracy']:.1%} IS BELOW 80% THRESHOLD\n"
    f"Do NOT upload to HF Hub. Investigate and retrain."
)
print(f"  Accuracy: {test_metrics['accuracy']:.1%} >= 80% OK")

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
print("     Order: vgg16 -> resnet50 -> mobilenet -> efficientnet")
