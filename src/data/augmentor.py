# filename: src/data/augmentor.py
# purpose:  torchvision.transforms.v2 augmentation pipelines for SmartVision CNN training
# version:  2.0

try:
    import torch
    import torchvision.transforms.v2 as T
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 224):
    """
    Training augmentation pipeline.

    Pipeline order is load-bearing:
      Geometric/color ops first (PIL — cheap, no precision loss)
      ToImage → uint8 tensor, ToDtype → float32 [0,1]
      RandomZoomOut receives float32 tensor — works on all torchvision versions
        fill=IMAGENET_MEAN: after Normalize, padded pixels = (mean-mean)/std = 0.0 exactly
        fill=0.5 would give asymmetric artifacts: R:+0.065, G:+0.196, B:+0.418 after norm
      Resize after ZoomOut (single resize pass)
      Normalize last
    """
    if not TORCH_AVAILABLE:
        return None

    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        T.ToImage(),                              # PIL → uint8 CHW tensor
        T.ToDtype(torch.float32, scale=True),     # [0,255] → [0.0,1.0]
        T.RandomZoomOut(
            fill=IMAGENET_MEAN,                   # list fill — per-channel; normalizes to 0.0 after Normalize
            side_range=(1.0, 1.3),
            p=0.3,
        ),
        T.Resize((image_size, image_size)),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms(image_size: int = 224):
    """Evaluation pipeline — deterministic, matches inference-time preprocessing."""
    if not TORCH_AVAILABLE:
        return None

    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def denormalize(tensor):
    """Reverse ImageNet normalization for visualization. Returns HWC uint8 numpy."""
    if not TORCH_AVAILABLE:
        return None
    import numpy as np
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img  = tensor.clone().cpu() * std + mean
    img  = img.clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
