# filename: src/models/model_factory.py
# purpose:  Builds all 4 pretrained CNN models with 25-class head
# version:  1.0

import logging

logger = logging.getLogger(__name__)

try:
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


def get_model(name: str, num_classes: int = 25):
    """
    Returns a pretrained model with custom head for num_classes.

    VGG16        — freeze all features, replace classifier[-1]
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


def freeze_resnet50_phase1(model) -> None:
    """Phase 1: freeze everything except the FC head."""
    for name_p, param in model.named_parameters():
        if "fc" not in name_p:
            param.requires_grad = False


def unfreeze_resnet50_phase2(model) -> None:
    """Phase 2: unfreeze layer3, layer4, and fc for fine-tuning."""
    for name_p, param in model.named_parameters():
        if any(x in name_p for x in ["layer3", "layer4", "fc"]):
            param.requires_grad = True


def count_trainable_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
