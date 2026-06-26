"""Consistent experiment and run naming helpers."""

from __future__ import annotations


def _normalize(value: object) -> str:
    return str(value).strip().lower().replace(" ", "_")


def build_run_name(dataset_name: str, model_name: str, seed: int, extra: str | None = None) -> str:
    """Build a stable MLflow run name."""
    parts = ["baseline", _normalize(dataset_name), _normalize(model_name), f"seed{seed}"]
    if extra:
        parts.append(_normalize(extra))
    return "__".join(parts)


def build_experiment_name(stage: str) -> str:
    """Build a project-level MLflow experiment name."""
    stage_name = str(stage).strip().replace("_", " ").title().replace(" ", "")
    return f"MasterThesis-{stage_name}"
