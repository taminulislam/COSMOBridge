"""Image loading, preprocessing, and augmentation for molecular images."""

import numpy as np
from PIL import Image
from pathlib import Path

try:
    import torch
    from torchvision import transforms
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def get_train_transforms(img_size: int = 224, config: dict = None):
    """Get training image transforms with augmentation."""
    aug_config = (config or {}).get("image", {}).get("augmentation", {})

    transform_list = [
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5 if aug_config.get("horizontal_flip", True) else 0),
        transforms.RandomVerticalFlip(p=0.5 if aug_config.get("vertical_flip", True) else 0),
        transforms.RandomRotation(aug_config.get("random_rotation", 30)),
    ]

    cj = aug_config.get("color_jitter", {})
    if cj:
        transform_list.append(transforms.ColorJitter(
            brightness=cj.get("brightness", 0.2),
            contrast=cj.get("contrast", 0.2),
            saturation=cj.get("saturation", 0.2),
            hue=cj.get("hue", 0.1),
        ))

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config.get("image", {}).get("normalize_mean", [0.485, 0.456, 0.406]) if config else [0.485, 0.456, 0.406],
            std=config.get("image", {}).get("normalize_std", [0.229, 0.224, 0.225]) if config else [0.229, 0.224, 0.225],
        ),
    ])

    return transforms.Compose(transform_list)


def get_eval_transforms(img_size: int = 224, config: dict = None):
    """Get evaluation image transforms (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config.get("image", {}).get("normalize_mean", [0.485, 0.456, 0.406]) if config else [0.485, 0.456, 0.406],
            std=config.get("image", {}).get("normalize_std", [0.229, 0.224, 0.225]) if config else [0.229, 0.224, 0.225],
        ),
    ])


def load_image(path: str, transform=None) -> "torch.Tensor":
    """Load an image and apply transforms.

    Returns a tensor of shape (3, H, W).
    If path is None or file doesn't exist, returns a zero tensor.
    """
    if path is None or not Path(path).exists():
        # Return placeholder zero tensor
        size = 224
        if transform:
            # Try to infer size from transform
            size = 224
        return torch.zeros(3, size, size)

    img = Image.open(path).convert("RGB")

    if transform:
        img = transform(img)
    else:
        img = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])(img)

    return img


def load_and_cache_images(image_paths: dict, transform=None) -> dict:
    """Load and cache a dict of image paths.

    Args:
        image_paths: dict mapping image_type -> file_path
        transform: torchvision transform to apply

    Returns:
        dict mapping image_type -> tensor
    """
    cached = {}
    for img_type, path in image_paths.items():
        cached[img_type] = load_image(path, transform)
    return cached
