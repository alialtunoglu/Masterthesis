"""Consistent experiment and run naming helpers."""

from __future__ import annotations

import uuid


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


def build_unique_run_name(stage: str, seed: int, device: str, dry_run: bool) -> str:
    """Keep new artifact paths short enough for Windows recovery snapshots."""
    prefix = {"baseline": "bl", "teacher_cnn": "tc", "teacher_vision_transformer": "tv"}[stage]
    return f"{prefix}__s{seed}__{device}__{'dry' if dry_run else 'full'}__{uuid.uuid4().hex[:12]}"
