"""Load result files for the Streamlit dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ui_utils import get_project_root


def _default_mlflow_client(tracking_uri: str):
    from mlflow.tracking import MlflowClient

    return MlflowClient(tracking_uri=tracking_uri)


def mlflow_experiment_overview(
    tracking_uri: str, client_factory: Any = None
) -> list[dict[str, Any]]:
    """List the experiments in the tracking store with their run counts.

    Read live rather than hardcoded so the page cannot drift out of date, and
    degrade to an empty list when the store is missing or unreadable.
    """
    factory = client_factory or _default_mlflow_client
    try:
        client = factory(tracking_uri)
        experiments = client.search_experiments()
    except Exception:
        return []
    rows = []
    for experiment in experiments:
        try:
            runs = len(client.search_runs([experiment.experiment_id], max_results=1000))
        except Exception:
            runs = 0
        if experiment.name == "Default" and not runs:
            continue
        rows.append({"experiment": experiment.name, "runs": runs})
    return sorted(rows, key=lambda row: row["experiment"])


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
