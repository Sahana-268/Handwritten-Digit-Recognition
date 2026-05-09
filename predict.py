from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from digit_recognizer.data import DIGIT_LABELS
from digit_recognizer.model import DigitCNN
from digit_recognizer.preprocess import extract_character_images, preprocess_digit_image, preprocess_digit_pil, save_preprocessed_preview
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
    parser.add_argument("--word", action="store_true", help="Split the image into characters and predict left to right.")
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
    class_labels = list(checkpoint.get("class_labels", DIGIT_LABELS))
    model = DigitCNN(num_classes=len(class_labels)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if args.word:
        characters = extract_character_images(
            Image.open(args.image),
            auto_invert=not args.no_auto_invert,
        )
        predictions: list[str] = []
        for character in characters:
            tensor = preprocess_digit_pil(character, auto_invert=False).to(device)
            probabilities = torch.softmax(model(tensor), dim=1).squeeze(0)
            predictions.append(class_labels[int(probabilities.argmax().item())])
        print(f"Predicted text: {''.join(predictions)}")
        print(f"Characters found: {len(predictions)}")
        return

    image_tensor = preprocess_digit_image(
        args.image,
        auto_invert=not args.no_auto_invert,
    ).to(device)

    if args.preview:
        save_preprocessed_preview(image_tensor.cpu(), args.preview)

    logits = model(image_tensor)
    probabilities = torch.softmax(logits, dim=1).squeeze(0)
    top_k = min(max(args.top_k, 1), len(class_labels))
    values, indices = torch.topk(probabilities, k=top_k)

    print(f"Predicted label: {class_labels[int(indices[0])]}")
    print("Top probabilities:")
    for probability, index in zip(values, indices):
        print(f"  {class_labels[int(index)]}: {float(probability):.4f}")


if __name__ == "__main__":
    main()
