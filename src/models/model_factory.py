# filename: src/models/model_factory.py
# purpose:  Builds all 4 pretrained CNN models with NUM_CLASSES-class head
# version:  2.0

import logging

from config import NUM_CLASSES

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torchvision.models as models
    from torchvision.models import (
        VGG16_Weights,
        ResNet50_Weights,
        MobileNet_V2_Weights,
        EfficientNet_B0_Weights,
    )
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def get_model(name: str, num_classes: int = NUM_CLASSES):
    """
    Returns a pretrained model with custom head for num_classes.

    VGG16        — freeze ALL params on load; caller does 2-phase unfreeze
                   Phase 1: head only (classifier[6])
                   Phase 2: features[24:] + head (last conv block)
    ResNet50     — keep fc frozen initially; caller does 2-phase unfreeze
    MobileNetV2  — freeze features, replace classifier[-1]
    EfficientNetB0 — freeze features, replace classifier[-1]

    All models expect input: (B, 3, 224, 224), ImageNet-normalized.
    """
    assert TORCH_AVAILABLE, "torch/torchvision not installed"
    name = name.lower().strip()

    if name == "vgg16":
        model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        # Freeze ALL parameters first (features + classifier[0-5])
        for param in model.parameters():
            param.requires_grad = False
        # Replace classifier[6] — new layer defaults to requires_grad=True
        # Result: only ~102K params trainable (the 4096→25 head), not 118M
        model.classifier[6] = nn.Linear(4096, num_classes)
        logger.info(f"VGG16 loaded (all frozen, {num_classes}-class head only)")

    elif name == "resnet50":
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # Replace the FC layer — caller freezes/unfreezes layers for 2-phase training
        model.fc = nn.Linear(2048, num_classes)
        logger.info(f"ResNet50 loaded ({num_classes}-class head, unfreeze managed by caller)")

    elif name in ("mobilenet", "mobilenetv2"):
        model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        # Freeze all features — only train classifier
        for param in model.features.parameters():
            param.requires_grad = False
        # Replace the final FC layer (1280 → num_classes)
        model.classifier[1] = nn.Linear(1280, num_classes)
        logger.info(f"MobileNetV2 loaded (frozen features, {num_classes}-class head)")

    elif name in ("efficientnet", "efficientnetb0"):
        model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        # Freeze all features — only train classifier
        for param in model.features.parameters():
            param.requires_grad = False
        # Replace the final FC layer (1280 → num_classes)
        model.classifier[1] = nn.Linear(1280, num_classes)
        logger.info(f"EfficientNetB0 loaded (frozen features, {num_classes}-class head)")

    else:
        raise ValueError(f"Unknown model: '{name}'. Choose: vgg16, resnet50, mobilenet, efficientnet")

    return model


def freeze_vgg16_phase1(model) -> None:
    """Phase 1: ensure all params frozen except classifier[6] (the head)."""
    for name_p, param in model.named_parameters():
        if "classifier.6" not in name_p:
            param.requires_grad = False


def unfreeze_vgg16_phase2(model) -> None:
    """Phase 2: unfreeze features[24:] (last conv block) for fine-tuning.

    Layers unlocked:
        features[24] Conv2d(512,512)  features[25] ReLU
        features[26] Conv2d(512,512)  features[27] ReLU
        features[28] Conv2d(512,512)  features[29] ReLU
        features[30] MaxPool2d
    classifier[6] stays trainable from phase 1.
    """
    for i, layer in enumerate(model.features):
        if i >= 24:
            for param in layer.parameters():
                param.requires_grad = True


def unfreeze_mobilenet_phase2(model) -> None:
    """Phase 2: unfreeze features[14:] (last 3 inverted residuals + final conv).

    Layers unlocked:
        features[14] InvertedResidual(160)
        features[15] InvertedResidual(160)
        features[16] InvertedResidual(160)
        features[17] InvertedResidual(320)
        features[18] Conv2d(320->1280)
    Total new trainable: ~2.3M params.
    With 3,475 training images: 2,330,000 / 3,475 = 670 params/image — safe.
    """
    for param in model.features[14:].parameters():
        param.requires_grad = True


def freeze_resnet50_phase1(model) -> None:
    """Phase 1: freeze everything except the FC head."""
    for name_p, param in model.named_parameters():
        if "fc" not in name_p:
            param.requires_grad = False


def unfreeze_resnet50_phase2(model) -> None:
    """Phase 2: unfreeze only layer4.2 (last bottleneck) + fc.

    Bug fixed: ".".join("fc.weight".split(".")[:2]) = "fc.weight" ≠ "fc" — the
    old check `prefix in {"fc"}` silently missed ALL fc params, leaving the head
    frozen during Phase 2. Fix: use name_p.startswith("fc.") for fc params.

    layer4 full:   ~15.7M / 3,080 = 5,097 params/img — very high overfit risk
    layer4.2 only: ~4.5M  / 3,080 = 1,461 params/img — acceptable
    """
    for name_p, param in model.named_parameters():
        parts  = name_p.split(".")
        prefix = ".".join(parts[:2])        # "layer4.2" for "layer4.2.conv1.weight"
        if prefix == "layer4.2" or name_p.startswith("fc."):
            param.requires_grad = True

    unfrozen = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"ResNet50 Phase 2: {unfrozen:,} unfrozen params (layer4.2 + fc)")


def count_trainable_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_per_class_accuracy(model, loader, class_names: list, device) -> dict:
    """Per-class accuracy on a DataLoader. Returns dict sorted ascending (weakest first).

    Aggregate accuracy can hide 95% on easy classes and 40% on hard ones.
    Per-class breakdown drives targeted fixes.
    """
    model.eval()
    class_correct = {name: 0 for name in class_names}
    class_total   = {name: 0 for name in class_names}

    with torch.no_grad():
        for images, labels in loader:
            images  = images.to(device, non_blocking=True)
            outputs = model(images)
            preds   = outputs.argmax(dim=1)
            for label, pred in zip(labels.cpu(), preds.cpu()):
                name = class_names[label.item()]
                class_total[name]   += 1
                class_correct[name] += int(label.item() == pred.item())

    per_class = {
        name: class_correct[name] / class_total[name]
        for name in class_names
        if class_total[name] > 0
    }
    return dict(sorted(per_class.items(), key=lambda kv: kv[1]))
