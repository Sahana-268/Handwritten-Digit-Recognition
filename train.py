from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from digit_recognizer.data import MNIST_MEAN, MNIST_STD, create_data_loaders, labels_for_dataset
from digit_recognizer.model import DigitCNN
from digit_recognizer.utils import (
    accuracy_from_logits,
    resolve_device,
    save_confusion_matrix,
    save_json,
    set_seed,
)


def serializable_args(args: argparse.Namespace) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in vars(args).items():
        payload[key] = str(value) if isinstance(value, Path) else value
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CNN on MNIST digits.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="MNIST data directory.")
    parser.add_argument(
        "--dataset",
        choices=["mnist", "emnist-letters", "emnist-balanced"],
        default="mnist",
        help="Dataset to train on. Use emnist-balanced for digits plus letters.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"), help="Model output directory.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Maximum learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--val-split", type=float, default=0.10, help="Fraction of training data for validation.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker processes.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps.")
    parser.add_argument("--no-augment", action="store_true", help="Disable training augmentation.")
    parser.add_argument("--no-download", action="store_true", help="Disable dataset download.")
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()

        correct, seen = accuracy_from_logits(logits, labels)
        total_loss += float(loss.item()) * seen
        total_correct += correct
        total_seen += seen

    return {
        "loss": total_loss / total_seen,
        "accuracy": total_correct / total_seen,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> tuple[dict[str, float], np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)
        predictions = logits.argmax(dim=1)

        correct, seen = accuracy_from_logits(logits, labels)
        total_loss += float(loss.item()) * seen
        total_correct += correct
        total_seen += seen

        for actual, predicted in zip(labels.cpu().numpy(), predictions.cpu().numpy()):
            matrix[int(actual), int(predicted)] += 1

    return {
        "loss": total_loss / total_seen,
        "accuracy": total_correct / total_seen,
    }, matrix


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=args.data_dir,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        augment=not args.no_augment,
        download=not args.no_download,
        seed=args.seed,
        pin_memory=device.type == "cuda",
    )

    class_labels = labels_for_dataset(args.dataset)
    model = DigitCNN(num_classes=len(class_labels)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.15,
        anneal_strategy="cos",
    )

    best_val_accuracy = 0.0
    best_state = copy.deepcopy(model.state_dict())
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        val_metrics, _ = evaluate(model, val_loader, criterion, device, len(class_labels))

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(epoch_metrics)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_metrics['loss']:.4f}, acc {train_metrics['accuracy']:.4f} | "
            f"val loss {val_metrics['loss']:.4f}, acc {val_metrics['accuracy']:.4f}"
        )

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_state = copy.deepcopy(model.state_dict())
            checkpoint_path = args.output_dir / "best_model.pt"
            torch.save(
                {
                    "model_state_dict": best_state,
                    "model_name": "DigitCNN",
                    "dataset": args.dataset,
                    "class_labels": list(class_labels),
                    "num_classes": len(class_labels),
                    "epoch": epoch,
                    "val_accuracy": best_val_accuracy,
                    "mnist_mean": MNIST_MEAN,
                    "mnist_std": MNIST_STD,
                    "args": serializable_args(args),
                },
                checkpoint_path,
            )
            print(f"Saved new best checkpoint to {checkpoint_path}")

    model.load_state_dict(best_state)
    test_metrics, confusion_matrix = evaluate(model, test_loader, criterion, device, len(class_labels))

    metrics_payload = {
        "best_validation_accuracy": best_val_accuracy,
        "test_accuracy": test_metrics["accuracy"],
        "test_loss": test_metrics["loss"],
        "history": history,
    }
    save_json(metrics_payload, args.output_dir / "metrics.json")
    save_confusion_matrix(confusion_matrix, args.output_dir / "confusion_matrix.csv", class_labels)

    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Saved metrics to {args.output_dir / 'metrics.json'}")
    print(f"Saved confusion matrix to {args.output_dir / 'confusion_matrix.csv'}")


if __name__ == "__main__":
    main()
