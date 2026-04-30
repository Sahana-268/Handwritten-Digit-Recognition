from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def build_transforms(augment: bool = True) -> tuple[transforms.Compose, transforms.Compose]:
    train_steps: list[torch.nn.Module] = []
    if augment:
        train_steps.append(
            transforms.RandomAffine(
                degrees=10,
                translate=(0.08, 0.08),
                scale=(0.90, 1.10),
                shear=5,
            )
        )
    train_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ]
    )
    return transforms.Compose(train_steps), eval_transform


def create_data_loaders(
    data_dir: str | Path,
    batch_size: int,
    val_split: float = 0.10,
    num_workers: int = 2,
    augment: bool = True,
    seed: int = 42,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test loaders for MNIST."""

    if not 0.0 < val_split < 1.0:
        raise ValueError("val_split must be between 0 and 1.")

    data_path = Path(data_dir)
    train_transform, eval_transform = build_transforms(augment=augment)

    train_full = datasets.MNIST(
        root=data_path,
        train=True,
        transform=train_transform,
        download=True,
    )
    val_full = datasets.MNIST(
        root=data_path,
        train=True,
        transform=eval_transform,
        download=True,
    )
    test_set = datasets.MNIST(
        root=data_path,
        train=False,
        transform=eval_transform,
        download=True,
    )

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(train_full), generator=generator).tolist()
    val_size = int(len(indices) * val_split)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_set = Subset(train_full, train_indices)
    val_set = Subset(val_full, val_indices)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader

