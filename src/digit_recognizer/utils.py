from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def resolve_device(choice: str = "auto") -> torch.device:
    choice = choice.lower()
    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(choice)


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> tuple[int, int]:
    predictions = logits.argmax(dim=1)
    correct = (predictions == labels).sum().item()
    return int(correct), int(labels.size(0))


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def save_confusion_matrix(matrix: np.ndarray, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = ",".join(["actual\\predicted", *[str(i) for i in range(matrix.shape[1])]])
    rows = [header]
    for digit, row in enumerate(matrix):
        rows.append(",".join([str(digit), *[str(int(value)) for value in row]]))
    output_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

