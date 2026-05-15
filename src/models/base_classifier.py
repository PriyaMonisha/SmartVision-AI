# filename: src/models/base_classifier.py
# purpose:  Shared training/evaluation engine for all 4 CNN classifiers
# version:  1.0

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


# ── Training ──────────────────────────────────────────────────────────────────

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    scaler=None,
) -> dict:
    """Run one training epoch. Returns loss and accuracy for the epoch."""
    model.train()
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
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
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
    """Evaluate model on a DataLoader. Returns loss, accuracy, precision, recall, F1."""
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
    metrics = {
        "loss":      total_loss / n,
        "accuracy":  accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "recall":    recall_score(all_labels, all_preds,    average="macro", zero_division=0),
        "f1":        f1_score(all_labels, all_preds,        average="macro", zero_division=0),
    }
    return metrics


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
) -> dict:
    """
    Full training loop with early stopping.
    Saves best weights (state_dict) when val accuracy improves.
    Returns training history dict.
    """
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_val_acc   = 0.0
    patience_count = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_metrics   = evaluate(model, val_loader, criterion, device)

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
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} | "
            f"{elapsed:.1f}s"
        )
        print(
            f"Epoch {epoch:3d}/{epochs}  "
            f"train_acc={train_metrics['accuracy']:.4f}  "
            f"val_acc={val_metrics['accuracy']:.4f}  "
            f"val_f1={val_metrics['f1']:.4f}  ({elapsed:.0f}s)"
        )

        # Save best
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            patience_count = 0
            if save_path is not None:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), save_path)  # state_dict only — Rule 5
                logger.info(f"  Saved best: {save_path.name} (val_acc={best_val_acc:.4f})")
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    history["best_val_acc"] = best_val_acc
    return history


# ── Inference helpers ─────────────────────────────────────────────────────────

def save_model(model, path: Path) -> None:
    """Save state_dict only — never full model pickle (Rule 5)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(model, path: Path, device: str = "cpu"):
    """Load state_dict with weights_only=True (Rule 36)."""
    state_dict = torch.load(
        path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def benchmark_inference(model, device, image_size: int = 224, n: int = 100, warmup: int = 10) -> float:
    """Returns mean inference time in milliseconds over n runs after warmup."""
    model.eval()
    dummy = torch.randn(1, 3, image_size, image_size).to(device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
    times = []
    with torch.no_grad():
        for _ in range(n):
            t0 = time.perf_counter()
            _ = model(dummy)
            times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times))
