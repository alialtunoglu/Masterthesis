"""Load result files for the Streamlit dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ui_utils import get_project_root


def load_csv_if_exists(path: str | Path) -> pd.DataFrame | None:
    """Load a CSV file when it exists."""
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = get_project_root() / candidate
    if not candidate.exists():
        return None
    try:
        return pd.read_csv(candidate)
    except pd.errors.EmptyDataError:
        return None


def safe_read_json(path: str | Path) -> dict[str, Any] | None:
    """Read a JSON object safely."""
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = get_project_root() / candidate
    if not candidate.exists():
        return None
    try:
        with candidate.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def load_dataset_summary() -> pd.DataFrame | None:
    """Load dataset analysis summary."""
    return load_csv_if_exists("results/dataset_analysis/dataset_summary.csv")


def load_baseline_results() -> pd.DataFrame | None:
    """Load cumulative baseline results."""
    return load_csv_if_exists("results/baseline/baseline_results.csv")


def load_split_summaries() -> pd.DataFrame:
    """Load compact summaries for all split JSON files."""
    split_dir = get_project_root() / "splits"
    rows: list[dict[str, Any]] = []
    for split_file in sorted(split_dir.glob("*.json")):
        data = safe_read_json(split_file)
        if not data:
            continue
        splits = data.get("splits", {})
        rows.append(
            {
                "file": split_file.name,
                "dataset_name": data.get("dataset_name"),
                "total_samples": data.get("total_samples"),
                "num_classes": data.get("num_classes"),
                "train": len(splits.get("train", [])),
                "val": len(splits.get("val", [])),
                "test": len(splits.get("test", [])),
            }
        )
    return pd.DataFrame(rows)
