"""Utilities for class-per-directory image datasets."""

from __future__ import annotations

from collections import Counter
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def get_class_dirs(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")
    return sorted([path for path in root.iterdir() if path.is_dir()], key=lambda path: path.name.lower())


def collect_imagefolder_samples(root: Path, project_root: Path) -> tuple[list[dict], list[str]]:
    class_dirs = get_class_dirs(root)
    classes = [class_dir.name for class_dir in class_dirs]
    class_to_id = {class_name: index for index, class_name in enumerate(classes)}
    samples: list[dict] = []

    for class_dir in class_dirs:
        class_name = class_dir.name
        for image_path in sorted(class_dir.rglob("*"), key=lambda path: str(path).lower()):
            if not is_image_file(image_path):
                continue
            samples.append(
                {
                    "path": image_path.relative_to(project_root).as_posix(),
                    "label": class_name,
                    "label_id": class_to_id[class_name],
                }
            )

    return samples, classes


def count_by_class(samples: list[dict]) -> Counter:
    return Counter(sample["label"] for sample in samples)
