"""
PyTorch Dataset and DataLoader factory for coffee bean classification.

Supports both roast-level and defect-detection datasets.
Implements data augmentation: random rotations, flips, contrast adjustments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ── ImageNet normalization stats ────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(img_size: int = 224) -> transforms.Compose:
    """Training augmentations: flips, rotations, color jitter, affine."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=30),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
        ),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.05,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms(img_size: int = 224) -> transforms.Compose:
    """Validation / test transforms: resize, center-crop, normalize."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class CoffeeBeanDataset(Dataset):
    """
    Image classification dataset that loads from class-organized directories.

    Expected structure:
        data_dir/
          ClassName1/
            img001.jpg
            img002.jpg
          ClassName2/
            ...
    """

    def __init__(
        self,
        data_dir: str | Path,
        transform: Optional[transforms.Compose] = None,
        class_names: Optional[list[str]] = None,
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform

        # Discover classes from subdirectory names
        if class_names:
            self.class_names = sorted(class_names)
        else:
            self.class_names = sorted(
                d.name for d in self.data_dir.iterdir() if d.is_dir()
            )

        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}

        # Collect all image paths and labels
        self.samples: list[tuple[Path, int]] = []
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        for class_name in self.class_names:
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                continue
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in valid_exts:
                    self.samples.append(
                        (img_path, self.class_to_idx[class_name])
                    )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_dataloaders(
    data_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: int = 224,
    class_names: Optional[list[str]] = None,
) -> dict[str, DataLoader]:
    """
    Create train, val, and test DataLoaders from a split directory.

    Args:
        data_dir: Root directory containing train/, val/, test/ subdirs.
        batch_size: Batch size for all loaders.
        num_workers: Number of parallel data loading workers.
        img_size: Image size for transforms.
        class_names: Optional explicit class name list.

    Returns:
        Dict with keys 'train', 'val', 'test' mapping to DataLoaders.
    """
    data_dir = Path(data_dir)

    loaders = {}
    for split, is_train in [("train", True), ("val", False), ("test", False)]:
        split_dir = data_dir / split
        if not split_dir.exists():
            print(f"⚠️  Split directory not found: {split_dir}")
            continue

        transform = get_train_transforms(img_size) if is_train else get_eval_transforms(img_size)

        dataset = CoffeeBeanDataset(
            data_dir=split_dir,
            transform=transform,
            class_names=class_names,
        )

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=is_train,
        )

        print(f"  {split}: {len(dataset)} images, {len(loaders[split])} batches")

    return loaders
