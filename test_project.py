from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from digit_recognizer.data import MNIST_MEAN, MNIST_STD
from digit_recognizer.model import DigitCNN
from digit_recognizer.preprocess import preprocess_digit_image


def main() -> None:
    checkpoint_path = PROJECT_ROOT / "artifacts" / "best_model.pt"
    metrics_path = PROJECT_ROOT / "artifacts" / "metrics.json"
    sample_path = PROJECT_ROOT / "artifacts" / "sample_digit.png"

    print("Handwritten Digit Recognition self-test")
    print("=" * 43)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing trained model: {checkpoint_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

    print("[OK] Trained model found")
    print("[OK] Metrics file found")

    device = torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = DigitCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"[OK] Checkpoint loaded: epoch {checkpoint['epoch']}, validation accuracy {checkpoint['val_accuracy']:.4f}")

    raw_dataset = datasets.MNIST(root=PROJECT_ROOT / "data", train=False, download=False)
    if not sample_path.exists():
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        image, _ = raw_dataset[0]
        image.save(sample_path)

    image, expected_label = raw_dataset[0]
    image.save(sample_path)
    image_tensor = preprocess_digit_image(sample_path).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(image_tensor), dim=1).squeeze(0)
        predicted = int(probabilities.argmax().item())
        confidence = float(probabilities[predicted].item())

    print(f"[OK] Sample prediction: expected {expected_label}, predicted {predicted}, confidence {confidence:.4f}")
    if predicted != expected_label:
        raise RuntimeError("Sample prediction failed.")

    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ]
    )
    test_dataset = datasets.MNIST(root=PROJECT_ROOT / "data", train=False, download=False, transform=eval_transform)
    quick_loader = DataLoader(Subset(test_dataset, range(1000)), batch_size=256)

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in quick_loader:
            predictions = model(images.to(device)).argmax(dim=1)
            correct += int((predictions.cpu() == labels).sum().item())
            total += int(labels.numel())

    accuracy = correct / total
    print(f"[OK] Quick MNIST accuracy: {correct}/{total} = {accuracy:.4f}")
    if accuracy < 0.98:
        raise RuntimeError("Quick accuracy check is lower than expected.")

    print("=" * 43)
    print("PROJECT IS WORKING PROPERLY")


if __name__ == "__main__":
    main()

