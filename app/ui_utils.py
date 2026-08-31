"""Shared helpers for the local Streamlit dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parents[1]


def config_selection_mismatches(
    defaults: dict[str, Any], dataset: str, model: str
) -> list[str]:
    """Describe where the pickers disagree with the selected config file.

    The launcher pages seed dataset and model from the config but let both be
    changed afterwards, so the command can carry a --config that contradicts
    its own --dataset.
    """
    fields = (("dataset_name", "Dataset", dataset), ("model_name", "Model", model))
    return [
        f"{title} config'de '{defaults[key]}' ama '{chosen}' seçili."
        for key, title, chosen in fields
        if defaults.get(key) and defaults[key] != chosen
    ]


def format_metric(value: Any) -> str:
    """Format a metric value for display."""
    if pd.isna(value):
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def file_exists(path: str | Path) -> bool:
    """Return whether a project-relative or absolute path exists."""
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = get_project_root() / candidate
    return candidate.exists()


def display_path_status(path: str | Path) -> None:
    """Display a file path with an existence indicator."""
    import streamlit as st

    candidate = Path(path)
    resolved = candidate if candidate.is_absolute() else get_project_root() / candidate
    if resolved.exists():
        st.success(f"Found: {candidate}")
    else:
        st.warning(f"Missing: {candidate}")


def show_dataframe_or_warning(df: pd.DataFrame | None, message: str) -> None:
    """Show a DataFrame if it has rows, otherwise show a warning."""
    import streamlit as st

    if df is None or df.empty:
        st.warning(message)
    else:
        st.dataframe(df, width="stretch")
