from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from digit_recognizer.model import DigitCNN
from digit_recognizer.preprocess import preprocess_digit_image, save_preprocessed_preview
from digit_recognizer.utils import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a handwritten digit image.")
    parser.add_argument("image", type=Path, help="Path to a digit image.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/best_model.pt"),
        help="Trained checkpoint path.",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of probabilities to print.")
    parser.add_argument("--no-auto-invert", action="store_true", help="Disable automatic background inversion.")
    parser.add_argument("--preview", type=Path, help="Optional path to save the 28x28 preprocessed image.")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = DigitCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image_tensor = preprocess_digit_image(
        args.image,
        auto_invert=not args.no_auto_invert,
    ).to(device)

    if args.preview:
        save_preprocessed_preview(image_tensor.cpu(), args.preview)

    logits = model(image_tensor)
    probabilities = torch.softmax(logits, dim=1).squeeze(0)
    top_k = min(max(args.top_k, 1), 10)
    values, indices = torch.topk(probabilities, k=top_k)

    print(f"Predicted digit: {int(indices[0])}")
    print("Top probabilities:")
    for probability, digit in zip(values, indices):
        print(f"  {int(digit)}: {float(probability):.4f}")


if __name__ == "__main__":
    main()
