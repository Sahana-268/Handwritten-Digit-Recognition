from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .data import MNIST_MEAN, MNIST_STD


def preprocess_digit_image(
    image_path: str | Path,
    auto_invert: bool = True,
    canvas_size: int = 28,
    digit_size: int = 20,
) -> torch.Tensor:
    """Convert a custom digit image into a normalized MNIST-like tensor."""

    image = Image.open(image_path).convert("L")
    return preprocess_digit_pil(
        image,
        auto_invert=auto_invert,
        canvas_size=canvas_size,
        digit_size=digit_size,
    )


def preprocess_digit_pil(
    image: Image.Image,
    auto_invert: bool = True,
    canvas_size: int = 28,
    digit_size: int = 20,
) -> torch.Tensor:
    """Convert a PIL digit image into a normalized MNIST-like tensor."""

    image = image.convert("L")
    array = np.asarray(image, dtype=np.float32) / 255.0

    if auto_invert and _background_is_light(array):
        array = 1.0 - array

    array = _crop_foreground(array)
    image = Image.fromarray(np.uint8(np.clip(array, 0.0, 1.0) * 255), mode="L")
    image.thumbnail((digit_size, digit_size), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (canvas_size, canvas_size), 0)
    x = (canvas_size - image.width) // 2
    y = (canvas_size - image.height) // 2
    canvas.paste(image, (x, y))

    tensor = torch.from_numpy(np.asarray(canvas, dtype=np.float32) / 255.0)
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    tensor = (tensor - MNIST_MEAN) / MNIST_STD
    return tensor


def save_preprocessed_preview(tensor: torch.Tensor, output_path: str | Path) -> None:
    array = tensor.detach().cpu().squeeze().numpy()
    array = (array * MNIST_STD) + MNIST_MEAN
    array = np.clip(array, 0.0, 1.0)
    image = Image.fromarray(np.uint8(array * 255), mode="L")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def _background_is_light(array: np.ndarray) -> bool:
    border = np.concatenate(
        [
            array[0, :],
            array[-1, :],
            array[:, 0],
            array[:, -1],
        ]
    )
    return float(np.median(border)) > 0.5


def _crop_foreground(array: np.ndarray, threshold: float = 0.10) -> np.ndarray:
    mask = array > threshold
    if not np.any(mask):
        return array

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    top, bottom = rows[0], rows[-1]
    left, right = cols[0], cols[-1]

    cropped = array[top : bottom + 1, left : right + 1]
    height, width = cropped.shape
    side = max(height, width)
    padded = np.zeros((side, side), dtype=np.float32)
    y = (side - height) // 2
    x = (side - width) // 2
    padded[y : y + height, x : x + width] = cropped
    return padded
