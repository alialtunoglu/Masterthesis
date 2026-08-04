"""Utilities for Plant Pathology 2021 metadata."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

try:
    from .imagefolder_utils import IMAGE_EXTENSIONS
except ImportError:
    from imagefolder_utils import IMAGE_EXTENSIONS


REQUIRED_COLUMNS = {"image", "labels"}


def load_train_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    dataframe = pd.read_csv(csv_path)
    missing_columns = REQUIRED_COLUMNS.difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")

    dataframe["image"] = dataframe["image"].astype(str).str.strip()
    dataframe["labels"] = dataframe["labels"].fillna("").astype(str).str.strip()
    return dataframe


def split_label_tokens(label_value: str) -> list[str]:
    return [token for token in label_value.split() if token]


def add_label_flags(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.copy()
    dataframe["label_tokens"] = dataframe["labels"].apply(split_label_tokens)
    dataframe["num_labels"] = dataframe["label_tokens"].apply(len)
    dataframe["is_single_label"] = dataframe["num_labels"] == 1
    return dataframe


def list_image_files(image_dir: Path) -> set[str]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    return {
        path.name
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def get_single_label_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    flagged = add_label_flags(dataframe)
    return flagged[flagged["is_single_label"]].copy()


def collect_plantpathology_samples(csv_path: Path, image_dir: Path, project_root: Path) -> tuple[list[dict], list[str], list[str]]:
    dataframe = get_single_label_dataframe(load_train_csv(csv_path))
    available_images = list_image_files(image_dir)
    dataframe["label"] = dataframe["label_tokens"].apply(lambda tokens: tokens[0])

    classes = sorted(dataframe["label"].unique().tolist())
    class_to_id = {class_name: index for index, class_name in enumerate(classes)}
    samples: list[dict] = []
    missing_images: list[str] = []

    for row in dataframe.sort_values(["label", "image"]).itertuples(index=False):
        image_name = row.image
        if image_name not in available_images:
            missing_images.append(image_name)
            continue

        image_path = image_dir / image_name
        label = row.label
        samples.append(
            {
                "path": image_path.relative_to(project_root).as_posix(),
                "label": label,
                "label_id": class_to_id[label],
            }
        )

    return samples, classes, sorted(missing_images)


def summarize_labels(dataframe: pd.DataFrame) -> dict:
    flagged = add_label_flags(dataframe)
    single_label = flagged[flagged["is_single_label"]].copy()
    single_label["label"] = single_label["label_tokens"].apply(lambda tokens: tokens[0] if tokens else "")

    return {
        "total_csv_rows": int(len(flagged)),
        "single_label_count": int(flagged["is_single_label"].sum()),
        "multi_label_count": int((flagged["num_labels"] > 1).sum()),
        "empty_label_count": int((flagged["num_labels"] == 0).sum()),
        "single_label_class_counts": Counter(single_label["label"].tolist()),
        "unique_labels": sorted({token for tokens in flagged["label_tokens"] for token in tokens}),
    }
