# src/models/model_factory.py
# Model construction and phase-based unfreeze strategies for SmartVision.
#
# Public API (import these in the notebook):
#   get_model(name, num_classes, dropout)       -> nn.Module
#   unfreeze_mobilenet_phase2(model)            -> None
#   unfreeze_efficientnet_phase2(model, block)  -> None
#   unfreeze_resnet50_phase2(model)             -> None
#   get_per_class_accuracy(...)                 -> dict[str, float]
#
# Internal only (_build_*, _log_trainable) -- do NOT import in notebook.
#
# Removed from API (now internal or eliminated):
#   freeze_resnet50_phase1  -> done inside _build_resnet50
#   freeze_vgg16_phase1     -> done inside _build_vgg16
#   unfreeze_vgg16_phase2   -> VGG16 not retraining (59.5% ceiling)
#   count_trainable_params  -> use inline sum() or _log_trainable()
#
# Round 3 changes:
#   MobileNetV2: dropout default 0.2 -> 0.4 (full classifier replaced)
#   unfreeze_mobilenet_phase2: features[14:] slice -> enumerate idx >= 16
#   _build_resnet50: freeze ALL params first, then replace fc
#
# Verified param counts for features[16:]:
#   features[16]: InvertedResidual(in=160,out=160,t=6)
#     expand   Conv2d(160,960,k=1)+BN(960) = 155,520
#     dw       Conv2d(960,960,k=3,g=960)+BN =  10,560
#     project  Conv2d(960,160,k=1)+BN(160) = 153,920
#     subtotal                              = 320,000
#     [Intermediate audit claimed 164K by omitting expand conv -- wrong]
#   features[17]: ~474,000 params
#   features[18]: ~412,000 params
#   classifier:    ~28,000 params
#   TOTAL: ~1,234,000 / 3,080 = ~401 params/img
#
# Param budget:
#   > 500/img: high overfit risk (Round 2 at 552)
#   200-500/img: acceptable for 140-200 samples/class
#   < 200/img: may under-adapt for fine-grained discrimination

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torchvision.models as models

logger = logging.getLogger(__name__)

_TRAIN_SIZE: int = 3080


def get_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    dropout: float | None = None,
) -> nn.Module:
    """
    Build a model with frozen backbone and trainable classifier head.

    Single entry point -- notebook must NOT call _build_* directly.

    Args:
        model_name:  "mobilenet" | "efficientnet" | "resnet50" | "vgg16"
        num_classes: Output classes (22 for SmartVision).
        pretrained:  True (default) = load ImageNet weights.
                     False = random init, no download. Use for CPU latency
                     benchmarking where weight values do not affect timing.
        dropout:     Override default. None = model-specific default:
                       mobilenet=0.4, efficientnet=0.3, resnet50=0.3, vgg16=0.5

    Returns:
        nn.Module (on CPU). Call .to(device) after construction.
        Backbone params: requires_grad=False.
        Classifier params: requires_grad=True.
        ResNet50 exception: ALL params frozen first, then fc replaced.
        New fc.* layers have requires_grad=True by PyTorch default.

    Raises:
        ValueError: unknown model_name.
    """
    _builders = {
        "mobilenet":    _build_mobilenet_v2,
        "efficientnet": _build_efficientnet_b0,
        "resnet50":     _build_resnet50,
        "vgg16":        _build_vgg16,
    }
    if model_name not in _builders:
        raise ValueError(
            f"Unknown model: {model_name!r}. Valid: {sorted(_builders.keys())}"
        )
    kwargs: dict = {"num_classes": num_classes, "pretrained": pretrained}
    if dropout is not None:
        kwargs["dropout"] = dropout
    return _builders[model_name](**kwargs)


def unfreeze_mobilenet_phase2(model: nn.Module) -> None:
    """
    Unfreeze MobileNetV2 features[16:] for Phase 2 fine-tuning.

    Call after Phase 1 completes, before constructing Phase 2 optimizer.
    Model must have been built with get_model("mobilenet").

    Round 3: was `for param in model.features[14:].parameters()` (slice).
    Now uses enumerate with explicit idx >= 16 (unambiguous boundary).

    Verified composition of features[16:]:
      features[16]: InvertedResidual(in=160,out=160,t=6) ~320K params
        expand Conv2d(160,960)+BN: 155,520
        dw     Conv2d(960,960,g=960)+BN: 10,560
        proj   Conv2d(960,160)+BN: 153,920
        [Note: 164K was an erroneous estimate -- 320K is correct]
      features[17]: ~474K params
      features[18]: ~412K params
      classifier:    ~28K params
      TOTAL: ~1.23M / 3,080 = ~401 params/img

    freeze_bn=True remains correct after partial unfreeze:
      _freeze_backbone_bn (base_classifier.py) checks each BN individually:
        features[0:15] BN: requires_grad=False -> eval() (stats locked) OK
        features[16:18] BN: requires_grad=True  -> train() (stats update) OK
    """
    for idx, block in enumerate(model.features):
        if idx >= 16:
            for param in block.parameters():
                param.requires_grad = True
    _log_trainable(model, "MobileNetV2 Phase 2 (features[16:])")


def unfreeze_efficientnet_phase2(model: nn.Module, from_block: int = 7) -> None:
    """
    Unfreeze EfficientNetB0 features[from_block:] for Phase 2 fine-tuning.

    EfficientNetB0 feature block layout (9 blocks, indices 0-8):
      [0] stem Conv2d (3→32)
      [1] MBConv1 ×1  (16ch)
      [2] MBConv6 ×2  (24ch)
      [3] MBConv6 ×2  (40ch)
      [4] MBConv6 ×3  (80ch)
      [5] MBConv6 ×3  (112ch)
      [6] MBConv6 ×4  (192ch)  ← ~2.5M params
      [7] MBConv6 ×1  (320ch)  ← default start (~1.5M params)
      [8] top Conv2d  (1280ch) ← ~413K params

    Default from_block=7 → ~1.9M backbone params / 3080 train ≈ 617 params/img.
    Exceeds model_factory "safe" threshold (500/img) but AdamW wd=1e-5 + dropout +
    3-epoch warmup + grad_clip compensate. Use from_block=6 ONLY after increasing
    wd_backbone to 1e-4 AND confirming val loss still plateaued at from_block=7.
    """
    for idx, block in enumerate(model.features):
        if idx >= from_block:
            for param in block.parameters():
                param.requires_grad = True
    _log_trainable(model, f"EfficientNetB0 Phase 2 (features[{from_block}:])")


def unfreeze_resnet50_phase2(model: nn.Module) -> None:
    """
    Unfreeze ResNet50 layer4.2 and fc for Phase 2 fine-tuning.

    BUG-1 fix:
      "fc.0.weight".split(".")[:2] joined = "fc.0" (not "fc")
      Prefix set {"layer4.2","fc"} never matches "fc.0" or "fc.1" ->
      ALL fc params silently stayed frozen in original code.
      Fix: name.startswith("fc.") matches fc.0.*, fc.1.*, etc.

    Why layer4.2 only:
      Full layer4: ~15.7M / 3,080 = 5,097/img (very high overfit risk)
      layer4.2:     ~4.5M / 3,080 = 1,461/img (acceptable)
    """
    for name_p, param in model.named_parameters():
        prefix = ".".join(name_p.split(".")[:2])
        if prefix == "layer4.2" or name_p.startswith("fc."):
            param.requires_grad = True
    _log_trainable(model, "ResNet50 Phase 2 (layer4.2 + fc)")


def get_per_class_accuracy(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    class_names: list[str],
    device: torch.device,
) -> dict[str, float]:
    """Per-class accuracy sorted ascending (weakest first). Forward passes only."""
    model.eval()
    correct: dict[str, int] = {n: 0 for n in class_names}
    total:   dict[str, int] = {n: 0 for n in class_names}
    with torch.no_grad():
        for images, labels in loader:
            images  = images.to(device, non_blocking=True)
            labels  = labels.to(device, non_blocking=True)
            preds   = model(images).argmax(dim=1)
            for label, pred in zip(labels.cpu(), preds.cpu()):
                name = class_names[label.item()]
                total[name]   += 1
                correct[name] += int(label.item() == pred.item())
    return dict(sorted(
        {n: correct[n] / total[n] for n in class_names if total[n] > 0}.items(),
        key=lambda kv: kv[1],
    ))


def _build_mobilenet_v2(num_classes: int, dropout: float = 0.4, pretrained: bool = True) -> nn.Module:
    """
    MobileNetV2: all features frozen, new classifier head.
    dropout=0.4 (Round 3, up from pretrained default 0.2).
    Full Sequential replacement -- classifier[1]-only replacement silently
    kept Dropout(0.2) from the pretrained classifier[0].
    pretrained=False: random init, no weight download (for CPU benchmarking).
    """
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    for param in model.features.parameters():
        param.requires_grad = False
    in_features = model.classifier[1].in_features   # read BEFORE replacement
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    _log_trainable(model, "MobileNetV2 Phase 1")
    return model


def _build_efficientnet_b0(num_classes: int, dropout: float = 0.3, pretrained: bool = True) -> nn.Module:
    """
    EfficientNetB0: features frozen, new classifier head. Head-only training.
    dropout NOT raised: no Phase 2 unfreeze, structurally lower overfit risk.
    Historical: lr=0.0001 caused non-convergence (RCA-2). Use lr=0.001.
    pretrained=False: random init, no weight download (for CPU benchmarking).
    """
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    for param in model.features.parameters():
        param.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    _log_trainable(model, "EfficientNetB0")
    return model


def _build_resnet50(num_classes: int, dropout: float = 0.3, pretrained: bool = True) -> nn.Module:
    """
    ResNet50: ALL params frozen, then new fc head (Phase 1 ready).
    Round 3 correction: freeze ALL (incl. original fc), then replace fc.
    New fc.* layers created fresh with requires_grad=True by default.
    Unambiguous: after construction, only fc.0.* and fc.1.* are trainable.
    pretrained=False: random init, no weight download (for CPU benchmarking).
    """
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)
    for param in model.parameters():                 # freeze ALL
        param.requires_grad = False
    in_features = model.fc.in_features              # 2048
    model.fc = nn.Sequential(                       # new: requires_grad=True
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    _log_trainable(model, "ResNet50 Phase 1")
    return model


def _build_vgg16(num_classes: int, dropout: float = 0.5, pretrained: bool = True) -> nn.Module:
    """
    VGG16: SKIPPED (59.5% ceiling). Included so get_model("vgg16") doesn't crash.
    VGG16 has no BatchNorm -> freeze_bn=False at all train() call sites.
    pretrained=False: random init, no weight download (for CPU benchmarking).
    """
    weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.vgg16(weights=weights)
    for param in model.features.parameters():
        param.requires_grad = False
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)
    _log_trainable(model, "VGG16 (SKIPPED -- 59.5% ceiling)")
    return model


def _log_trainable(model: nn.Module, label: str) -> None:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct       = 100.0 * trainable / total if total > 0 else 0.0
    msg = (f"{label}: {trainable:,} / {total:,} trainable "
           f"({pct:.1f}%) | {trainable / _TRAIN_SIZE:.0f} params/img")
    logger.info(msg)
    print(f"  {msg}")
