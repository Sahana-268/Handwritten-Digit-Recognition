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


def extract_character_images(
    image: Image.Image,
    auto_invert: bool = True,
    min_component_area: int = 20,
    gap_padding: int = 6,
) -> list[Image.Image]:
    """Split a drawing or uploaded image into left-to-right character crops."""

    gray = image.convert("L")
    array = np.asarray(gray, dtype=np.float32) / 255.0
    if auto_invert and _background_is_light(array):
        array = 1.0 - array

    mask = array > 0.10
    if not np.any(mask):
        return []

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    top, bottom = rows[0], rows[-1]
    left, right = cols[0], cols[-1]
    mask = mask[top : bottom + 1, left : right + 1]
    source = array[top : bottom + 1, left : right + 1]

    boxes = _connected_component_boxes(mask, min_component_area)
    if not boxes:
        boxes = _projection_boxes(mask)

    crops: list[Image.Image] = []
    height, width = source.shape
    for box_left, box_top, box_right, box_bottom in boxes:
        padded_left = max(0, box_left - gap_padding)
        padded_top = max(0, box_top - gap_padding)
        padded_right = min(width - 1, box_right + gap_padding)
        padded_bottom = min(height - 1, box_bottom + gap_padding)
        crop = source[padded_top : padded_bottom + 1, padded_left : padded_right + 1]
        crop_image = Image.fromarray(np.uint8(np.clip(crop, 0.0, 1.0) * 255), mode="L")
        crops.append(crop_image)

    return crops


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


def _connected_component_boxes(mask: np.ndarray, min_area: int) -> list[tuple[int, int, int, int]]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    boxes: list[tuple[int, int, int, int]] = []

    for start_y, start_x in zip(*np.where(mask & ~visited)):
        if visited[start_y, start_x]:
            continue

        stack = [(int(start_x), int(start_y))]
        visited[start_y, start_x] = True
        min_x = max_x = int(start_x)
        min_y = max_y = int(start_y)
        area = 0

        while stack:
            x, y = stack.pop()
            area += 1
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)

            for next_x in range(max(0, x - 1), min(width, x + 2)):
                for next_y in range(max(0, y - 1), min(height, y + 2)):
                    if visited[next_y, next_x] or not mask[next_y, next_x]:
                        continue
                    visited[next_y, next_x] = True
                    stack.append((next_x, next_y))

        if area >= min_area:
            boxes.append((min_x, min_y, max_x, max_y))

    boxes.sort(key=lambda box: box[0])
    return _merge_overlapping_boxes(boxes)


def _projection_boxes(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    columns = mask.any(axis=0)
    boxes: list[tuple[int, int, int, int]] = []
    start: int | None = None

    for index, active in enumerate(columns):
        if active and start is None:
            start = index
        elif not active and start is not None:
            boxes.append(_box_for_column_range(mask, start, index - 1))
            start = None

    if start is not None:
        boxes.append(_box_for_column_range(mask, start, len(columns) - 1))

    return boxes


def _box_for_column_range(mask: np.ndarray, left: int, right: int) -> tuple[int, int, int, int]:
    submask = mask[:, left : right + 1]
    rows = np.where(submask.any(axis=1))[0]
    return (left, int(rows[0]), right, int(rows[-1]))


def _merge_overlapping_boxes(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    if not boxes:
        return boxes

    merged = [boxes[0]]
    for left, top, right, bottom in boxes[1:]:
        prev_left, prev_top, prev_right, prev_bottom = merged[-1]
        horizontal_overlap = left <= prev_right + 2
        if horizontal_overlap:
            merged[-1] = (
                min(prev_left, left),
                min(prev_top, top),
                max(prev_right, right),
                max(prev_bottom, bottom),
            )
        else:
            merged.append((left, top, right, bottom))
    return merged
