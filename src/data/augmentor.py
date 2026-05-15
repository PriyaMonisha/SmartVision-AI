# filename: src/data/augmentor.py
# purpose:  torchvision.transforms.v2 augmentation pipelines for SmartVision CNN training
# version:  1.0

# torch/torchvision wrapped at module level — not available in local venv (fbgemm.dll)
# Functions return None gracefully when torch is absent; notebooks guard with TORCH_AVAILABLE.
try:
    import torchvision.transforms.v2 as T
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ImageNet normalization — required for ALL torchvision pretrained models (Rule 9)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 224):
    """
    Augmentation pipeline for training split.
    Applies spatial + colour augmentations then normalises with ImageNet stats.
    Compatible with SmartVisionDataset which returns PIL images.
    """
    if not TORCH_AVAILABLE:
        return None

    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        T.RandomZoomOut(fill=0, side_range=(1.0, 1.3), p=0.3),
        T.Resize((image_size, image_size)),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms(image_size: int = 224):
    """
    Evaluation pipeline for val and test splits.
    No augmentation — resize + normalize only.
    Must be identical at training-time and inference-time to avoid train/serve skew.
    """
    if not TORCH_AVAILABLE:
        return None

    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def denormalize(tensor):
    """
    Reverse ImageNet normalization for visualization.
    tensor: CHW float32 torch.Tensor
    Returns: HWC uint8 numpy array (0-255) suitable for plt.imshow
    """
    if not TORCH_AVAILABLE:
        return None
    import numpy as np

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img  = tensor.clone().cpu() * std + mean
    img  = img.clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
