"""Thin MLflow wrapper used by training and evaluation scripts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


DEFAULT_EXPERIMENT_NAME = "MasterThesis-Baseline"


def _import_mlflow():
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError(
            "MLflow is required for experiment tracking. Install project dependencies with "
            "`pip install -r requirements.txt` and rerun the command."
        ) from exc
    return mlflow


def setup_mlflow(experiment_name: str = DEFAULT_EXPERIMENT_NAME, tracking_uri: str | None = None) -> None:
    """Configure MLflow tracking and select/create the experiment."""
    mlflow = _import_mlflow()
    if tracking_uri is None:
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        tracking_uri = Path("mlruns").resolve().as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def start_run(run_name: str | None = None):
    """Start and return an MLflow run."""
    mlflow = _import_mlflow()
    return mlflow.start_run(run_name=run_name)


def log_params(params: dict[str, Any]) -> None:
    """Log non-null parameters to MLflow."""
    mlflow = _import_mlflow()
    clean_params = {key: value for key, value in params.items() if value is not None}
    if clean_params:
        mlflow.log_params(clean_params)


def log_metrics(metrics: dict[str, Any], step: int | None = None) -> None:
    """Log numeric metrics to MLflow."""
    mlflow = _import_mlflow()
    clean_metrics = {}
    for key, value in metrics.items():
        if value is None:
            continue
        try:
            clean_metrics[key] = float(value)
        except (TypeError, ValueError):
            continue
    if clean_metrics:
        mlflow.log_metrics(clean_metrics, step=step)


def log_artifact(path: str | Path) -> None:
    """Log a single artifact file."""
    mlflow = _import_mlflow()
    mlflow.log_artifact(str(path))


def log_artifacts(path: str | Path) -> None:
    """Log all artifacts under a directory."""
    mlflow = _import_mlflow()
    mlflow.log_artifacts(str(path))


def log_text(text: str, artifact_file: str) -> None:
    """Log text content as an MLflow artifact."""
    mlflow = _import_mlflow()
    mlflow.log_text(text, artifact_file)


def end_run() -> None:
    """End the active MLflow run if one exists."""
    mlflow = _import_mlflow()
    if mlflow.active_run() is not None:
        mlflow.end_run()
