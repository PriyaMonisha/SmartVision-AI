# src/data/augmentor.py
# Augmentation pipelines for SmartVision CNN training.
#
# Pipeline design principles:
#
#   PIL-space first (indices 0-4, before ToImage at index 5):
#     Geometric/color transforms on PIL input. PIL handles uint8 arithmetic
#     and boundary conditions correctly. Cost is CPU-bound but negligible
#     at 224x224 images.
#
#   Float tensor space (indices 7-9, after ToDtype at index 6):
#     RandomZoomOut requires float32 input with fill values in [0.0, 1.0].
#     fill=IMAGENET_MEAN: after Normalize, padded pixels become exactly 0.0.
#       Proof: (IMAGENET_MEAN - IMAGENET_MEAN) / IMAGENET_STD = [0, 0, 0]
#     fill=0.0: (0 - mean) / std = [-2.12, -2.04, -1.80] after Normalize (wrong)
#     fill=0.5: gives [+0.065, +0.196, +0.418] per channel (wrong, colored border)
#
#   RandomPerspective border artifact (accepted tradeoff):
#     PIL RandomPerspective default fill=0 -> black border in uint8.
#     After full pipeline: (0/255 - mean) / std ~= [-2.12, -2.04, -1.80].
#     At distortion_scale=0.2, p=0.3: border <5% of pixels on 30% of samples.
#     Net gradient impact is small. Mitigation if needed: move after ToDtype
#     and set fill=IMAGENET_MEAN.
#
#   Why RandomGrayscale(p=0.05):
#     COCO household items (cup/bowl/bottle) are confused via memorized color
#     patterns from specific training images. 5% grayscale forces color-agnostic
#     feature learning occasionally without degrading vehicle/animal color signal.
#
#   Why RandomPerspective (not RandomErasing):
#     COCO ground-truth boxes reflect real camera perspective variation.
#     Perspective augmentation models this directly.
#     RandomErasing destroys local texture patches -- the primary discriminator
#     for fine-grained categories. Do not add it.
#
#   Round 3 changes from Round 2:
#     RandomRotation: 15 -> 20 degrees
#     ColorJitter brightness/contrast: 0.2 -> 0.3
#     ColorJitter saturation: 0.1 -> 0.2
#     RandomGrayscale(p=0.05): ADDED (PIL-space)
#     RandomPerspective(distortion_scale=0.2, p=0.3): ADDED (PIL-space)
#     RandomZoomOut side_range: (1.0, 1.3) -> (1.0, 1.4)

import torch
import torchvision.transforms.v2 as T

IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_STD:  list[float] = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 224) -> T.Compose:
    """
    Training augmentation pipeline -- 10 transforms.

    Ordered execution (order is load-bearing):
      [0]  RandomHorizontalFlip(p=0.5)                PIL geometric
      [1]  RandomRotation(degrees=20)                 PIL geometric
      [2]  ColorJitter(0.3, 0.3, 0.2, 0.05)          PIL color
      [3]  RandomGrayscale(p=0.05)                    PIL color      NEW Round 3
      [4]  RandomPerspective(scale=0.2, p=0.3)        PIL geometric  NEW Round 3
      [5]  ToImage()                                  PIL -> uint8 CHW tensor
      [6]  ToDtype(torch.float32, scale=True)          uint8 -> float32 [0, 1]
      [7]  RandomZoomOut(fill=IMAGENET_MEAN, ...)     float32 tensor
      [8]  Resize((image_size, image_size))
      [9]  Normalize(IMAGENET_MEAN, IMAGENET_STD)

    Hard ordering constraints:
      [3], [4] < [5]: PIL ops must run before ToImage conversion
      [6]      < [7]: ZoomOut fill is float32 [0, 1] space
      [9]     = last: Normalize must be final transform
    """
    return T.Compose([
        # PIL-space geometric
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=20),

        # PIL-space color
        T.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.05,
        ),
        T.RandomGrayscale(p=0.05),

        # PIL-space geometric (perspective)
        # fill=0 default: black border in uint8 -> ~[-2.12,-2.04,-1.80] after Normalize.
        # Accepted: borders are small (<5% of pixels) and infrequent (p=0.3).
        T.RandomPerspective(distortion_scale=0.2, p=0.3),

        # Convert to float32 tensor
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),

        # Float32 tensor augmentation
        # fill=IMAGENET_MEAN: padded pixels normalize to 0.0 per channel.
        T.RandomZoomOut(
            fill=IMAGENET_MEAN,
            side_range=(1.0, 1.4),
            p=0.3,
        ),

        # Finalize (single resize after all spatial augmentation)
        T.Resize((image_size, image_size), antialias=True),

        # Normalize last
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms(image_size: int = 224) -> T.Compose:
    """
    Evaluation/test pipeline -- fully deterministic.
    Matches inference-time preprocessing exactly.
    Contains zero stochastic transforms.
    """
    return T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Resize((image_size, image_size), antialias=True),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def denormalize(tensor: torch.Tensor) -> "np.ndarray":
    """Reverse ImageNet normalization. Returns HWC uint8 numpy array [0, 255]."""
    import numpy as np
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD,  dtype=torch.float32).view(3, 1, 1)
    img  = tensor.clone().cpu() * std + mean
    return (img.clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
