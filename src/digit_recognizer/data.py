from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Standard MNIST normalization values
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
DIGIT_LABELS = tuple(str(index) for index in range(10))
LETTER_LABELS = tuple(chr(ord("A") + index) for index in range(26))
BALANCED_LABELS = (*DIGIT_LABELS, *LETTER_LABELS, "a", "b", "d", "e", "f", "g", "h", "n", "q", "r", "t")


def labels_for_dataset(dataset_name: str) -> tuple[str, ...]:
    if dataset_name == "mnist":
        return DIGIT_LABELS
    if dataset_name == "emnist-letters":
        return LETTER_LABELS
    if dataset_name == "emnist-balanced":
        return BALANCED_LABELS
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def create_data_loaders(
    data_dir: str | Path,
    dataset_name: str = "mnist",
    batch_size: int = 32,
    val_split: float = 0.1,
    num_workers: int = 0,
    augment: bool = True,
    download: bool = True,
    seed: int = 42,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test data loaders for character datasets.
    
    Args:
        data_dir: Path to the data directory
        batch_size: Batch size for data loaders
        val_split: Fraction of training data to use for validation (0.0-1.0)
        num_workers: Number of worker processes for data loading
        augment: Whether to apply data augmentation to training data
        seed: Random seed for reproducibility
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_dir = Path(data_dir)
    
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    
    # Define transformations
    orientation_transform = []
    if dataset_name in {"emnist-letters", "emnist-balanced"}:
        orientation_transform = [
            transforms.Lambda(lambda image: transforms.functional.hflip(transforms.functional.rotate(image, -90))),
        ]

    if augment:
        train_transform = transforms.Compose([
            *orientation_transform,
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ])
    else:
        train_transform = transforms.Compose([
            *orientation_transform,
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ])
    
    eval_transform = transforms.Compose([
        *orientation_transform,
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    ])
    
    if dataset_name == "mnist":
        train_dataset = datasets.MNIST(
            root=data_dir,
            train=True,
            download=download,
            transform=train_transform,
        )
        test_dataset = datasets.MNIST(
            root=data_dir,
            train=False,
            download=download,
            transform=eval_transform,
        )
    elif dataset_name in {"emnist-letters", "emnist-balanced"}:
        split = "letters" if dataset_name == "emnist-letters" else "balanced"
        target_transform = (lambda label: int(label) - 1) if dataset_name == "emnist-letters" else None
        train_dataset = datasets.EMNIST(
            root=data_dir,
            split=split,
            train=True,
            download=download,
            transform=train_transform,
            target_transform=target_transform,
        )
        test_dataset = datasets.EMNIST(
            root=data_dir,
            split=split,
            train=False,
            download=download,
            transform=eval_transform,
            target_transform=target_transform,
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Split training data into train and validation
    val_size = int(len(train_dataset) * val_split)
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    return train_loader, val_loader, test_loader
