# filename: src/models/base_classifier.py
# purpose:  Shared training/evaluation engine for all 4 CNN classifiers
# version:  2.0

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
    )
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _freeze_backbone_bn(model: "nn.Module") -> None:
    """
    Set BatchNorm layers to eval() if their affine params are frozen.

    Must be called AFTER model.train() each epoch — model.train() resets all
    layers to training mode first; we then selectively revert frozen BN layers
    back to eval().

    Why module.weight.requires_grad:
      When a backbone block is frozen (param.requires_grad=False), its BN's
      affine weight inherits requires_grad=False. This is the reliable proxy
      for "is this BN inside a frozen block?"

    Why not module.parameters():
      running_mean and running_var are buffers, not parameters — they never
      appear in .parameters(). They update whenever module.training==True,
      regardless of requires_grad. Checking requires_grad alone is insufficient.

    Why affine=False BN (weight is None) is always frozen:
      No learnable params at all — either frozen or has no learning signal.
      Either way, updating running stats from current mini-batches corrupts
      pretrained statistics.

    Handles mixed-freeze correctly:
      e.g. MobileNetV2 features[0:13] frozen, features[14:] unfrozen:
      BN in features[0:13] get eval(), BN in features[14:] stays train().
    """
    for module in model.modules():
        if not isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            continue
        if module.weight is None:
            module.eval()                     # affine=False: always freeze stats
        elif not module.weight.requires_grad:
            module.eval()                     # affine frozen: parent block is frozen


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    scaler=None,
    freeze_bn: bool = False,
    grad_clip: Optional[float] = None,
) -> dict:
    """Run one training epoch.

    freeze_bn: call _freeze_backbone_bn() after model.train(). Use True whenever
               any backbone layers are frozen. Per-module check handles partial
               unfreeze (e.g., features[14:] only) correctly.
    grad_clip: max norm for gradient clipping. 1.0 for both phases. Do not use
               0.5 — too aggressive, suppresses signal in newly unfrozen layers.
    """
    model.train()
    # CRITICAL: call AFTER model.train() — model.train() resets everything to
    # training mode first; we then selectively revert frozen BN layers to eval().
    if freeze_bn:
        _freeze_backbone_bn(model)

    total_loss = 0.0
    correct    = 0
    total      = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss    = criterion(outputs, labels)
            scaler.scale(loss).backward()

            if grad_clip is not None:
                # unscale_ MUST precede clip_grad_norm_ in AMP path —
                # gradients are stored scaled; unscale converts to true grads first
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad),
                    max_norm=grad_clip,
                )

            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad),
                    max_norm=grad_clip,
                )

            optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds       = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)

    return {
        "loss":     total_loss / total,
        "accuracy": correct / total,
    }


def evaluate(
    model,
    loader,
    criterion,
    device,
    num_classes: int = 25,
) -> dict:
    """Evaluate model. Returns loss, accuracy, precision, recall, F1 (macro)."""
    model.eval()
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            loss    = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    n = len(all_labels)
    return {
        "loss":      total_loss / n,
        "accuracy":  accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "recall":    recall_score(all_labels, all_preds,    average="macro", zero_division=0),
        "f1":        f1_score(all_labels, all_preds,        average="macro", zero_division=0),
    }


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    criterion,
    device,
    epochs: int,
    patience: int = 5,
    scaler=None,
    model_name: str = "model",
    save_path: Optional[Path] = None,
    freeze_bn: bool = False,
    grad_clip: Optional[float] = None,
) -> dict:
    """Full training loop with early stopping (monitors val_accuracy).

    Monitors val_accuracy, not val_loss. Rationale: label_smoothing=0.1 inflates
    loss by ~0.31 nats uniformly — loss trends correctly but absolute values are
    misleading for early stopping. val_accuracy is scale-invariant.

    Reloads best checkpoint before returning so caller always gets best model,
    not last epoch.
    """
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [], "val_f1": [],
    }
    best_val_acc   = 0.0
    patience_count = 0

    # Resolve checkpoint path ONCE before the loop — avoid reassigning inside loop
    checkpoint_path = None
    if save_path is not None:
        checkpoint_path = Path(save_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            scaler=scaler, freeze_bn=freeze_bn, grad_clip=grad_clip,
        )
        val_metrics = evaluate(model, val_loader, criterion, device)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics["loss"])
            else:
                scheduler.step()

        history["train_loss"].append(round(train_metrics["loss"],     6))
        history["train_acc"].append( round(train_metrics["accuracy"], 6))
        history["val_loss"].append(  round(val_metrics["loss"],       6))
        history["val_acc"].append(   round(val_metrics["accuracy"],   6))
        history["val_f1"].append(    round(val_metrics["f1"],         6))

        elapsed = time.time() - t0
        logger.info(
            f"[{model_name}] Epoch {epoch:3d}/{epochs} | "
            f"train_acc={train_metrics['accuracy']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | {elapsed:.1f}s"
        )
        print(
            f"Epoch {epoch:3d}/{epochs}  "
            f"train_acc={train_metrics['accuracy']:.4f}  "
            f"val_acc={val_metrics['accuracy']:.4f}  "
            f"val_f1={val_metrics['f1']:.4f}  ({elapsed:.0f}s)"
        )

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            patience_count = 0
            if checkpoint_path is not None:
                # Cast to Python native types — numpy.float64 (from accuracy_score)
                # is rejected by weights_only=True during reload
                torch.save({
                    "epoch":            int(epoch),
                    "model_state_dict": model.state_dict(),
                    "val_acc":          float(best_val_acc),
                    "val_loss":         float(val_metrics["loss"]),
                    "val_f1":           float(val_metrics["f1"]),
                }, checkpoint_path)
                logger.info(f"  Saved best: {checkpoint_path.name} (val_acc={best_val_acc:.4f})")
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"Early stopping at epoch {epoch} (patience={patience})")
                break

    # Reload best checkpoint so caller receives best model, not last epoch
    if checkpoint_path is not None and checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Reloaded best checkpoint: val_acc={ckpt['val_acc']:.4f}")

    history["best_val_acc"] = best_val_acc
    return history


# ── Inference helpers ─────────────────────────────────────────────────────────

def save_model(model, path: Path) -> None:
    """Save state_dict only — never full model pickle (Rule 5)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(model, path: Path, device: str = "cpu"):
    """Load weights with weights_only=True (Rule 36). Handles both raw state_dict
    and checkpoint dict (keys: model_state_dict, epoch, val_acc, ...) formats."""
    state_dict = torch.load(path, map_location=device, weights_only=True)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model.eval()
    return model


def benchmark_inference(model, device, image_size: int = 224, n: int = 100, warmup: int = 10) -> float:
    """Returns mean inference time in milliseconds over n runs after warmup.

    CUDA ops are async — synchronize before/after each timing window so that
    time.perf_counter() measures actual compute, not just kernel scheduling.
    """
    model.eval()
    dummy = torch.randn(1, 3, image_size, image_size).to(device)
    is_cuda = str(device).startswith("cuda")

    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)
        if is_cuda:
            torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(n):
            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(dummy)
            if is_cuda:
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    return float(np.mean(times))
