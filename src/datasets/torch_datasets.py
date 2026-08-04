"""PyTorch datasets backed by deterministic split JSON files."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
for path in (SRC_DIR, CURRENT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dataset_registry import PROJECT_ROOT
from utils.io import load_json


class SplitJsonImageDataset(Dataset):
    """Image dataset that reads samples from a split JSON file."""

    def __init__(
        self,
        split_file: str | Path,
        split: str,
        transform: Any | None = None,
        project_root: str | Path = PROJECT_ROOT,
    ) -> None:
        self.split_file = Path(split_file)
        self.split = split
        self.transform = transform
        self.project_root = Path(project_root)

        if not self.split_file.exists():
            raise FileNotFoundError(f"Split file not found: {self.split_file}")

        metadata = load_json(self.split_file)
        if split not in metadata.get("splits", {}):
            available = ", ".join(sorted(metadata.get("splits", {}).keys()))
            raise ValueError(f"Split '{split}' not found in {self.split_file}. Available: {available}")

        self.metadata = metadata
        self.samples = metadata["splits"][split]
        self.classes = metadata["classes"]
        self.dataset_name = metadata["dataset_name"]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image_path = self.project_root / sample["path"]
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)

        return image, int(sample["label_id"])
