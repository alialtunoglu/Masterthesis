"""Train a baseline student model from a deterministic split JSON file."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
for path in (SRC_DIR, CURRENT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from datasets.dataloaders import create_dataloaders
from datasets.dataset_registry import DATASETS, PROJECT_ROOT, SPLITS_DIR
from evaluation.metrics import compute_per_class_metrics
from evaluation.reports import save_confusion_matrix_artifacts, save_learning_curves
from models.model_factory import SUPPORTED_MODELS, create_model
from models.model_info import get_model_summary_dict
from tracking.experiment_naming import build_experiment_name, build_run_name
from tracking.mlflow_tracker import (
    end_run,
    log_artifact,
    log_metrics,
    log_params,
    log_text,
    setup_mlflow,
    start_run,
)
from training.trainer import fit
from utils.io import ensure_dir, load_json, save_json
from utils.seed import set_seed


RESULTS_DIR = PROJECT_ROOT / "results" / "baseline"
BASELINE_RUNS_DIR = RESULTS_DIR / "runs"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

DEFAULT_CONFIG: dict[str, Any] = {
    "stage": "baseline",
    "dataset_name": None,
    "model_name": None,
    "seed": 42,
    "image_size": 224,
    "batch_size": 32,
    "epochs": 5,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "pretrained": False,
    "optimizer": "adamw",
    "scheduler": "none",
    "num_workers": 0,
    "device": "auto",
    "dry_run": False,
    "max_train_batches": None,
    "max_val_batches": None,
    "max_test_batches": None,
    "experiment_name": None,
    "tracking_uri": None,
}

BASELINE_RESULT_COLUMNS = [
    "run_name",
    "mlflow_run_id",
    "dataset_name",
    "model_name",
    "model_type",
    "seed",
    "split_file",
    "num_classes",
    "total_train_samples",
    "total_val_samples",
    "total_test_samples",
    "image_size",
    "batch_size",
    "epochs",
    "optimizer",
    "learning_rate",
    "weight_decay",
    "pretrained",
    "device",
    "best_epoch",
    "best_val_accuracy",
    "best_val_macro_f1",
    "test_accuracy",
    "test_macro_precision",
    "test_macro_recall",
    "test_macro_f1",
    "test_weighted_precision",
    "test_weighted_recall",
    "test_weighted_f1",
    "params",
    "model_size_mb",
    "checkpoint_path",
    "history_path",
    "per_class_metrics_path",
    "confusion_matrix_csv_path",
    "confusion_matrix_png_path",
    "learning_curves_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a baseline student training experiment.")
    parser.add_argument("--config", default=None, help="Path to a baseline JSON config file.")
    parser.add_argument("--dataset", choices=sorted(DATASETS.keys()), default=None)
    parser.add_argument("--model", choices=sorted(SUPPORTED_MODELS), default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--pretrained", dest="pretrained", action="store_true", default=None)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--max-test-batches", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--tracking-uri", default=None)
    return parser.parse_args()


def normalize_config_keys(config: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(config)
    if "dataset" in normalized and "dataset_name" not in normalized:
        normalized["dataset_name"] = normalized.pop("dataset")
    if "model" in normalized and "model_name" not in normalized:
        normalized["model_name"] = normalized.pop("model")
    if "lr" in normalized and "learning_rate" not in normalized:
        normalized["learning_rate"] = normalized.pop("lr")
    return normalized


def load_config_file(config_path: str | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    loaded = load_json(path)
    if not isinstance(loaded, dict):
        raise ValueError(f"Config file must contain a JSON object: {path}")
    return normalize_config_keys(loaded)


def cli_overrides(args: argparse.Namespace) -> dict[str, Any]:
    values = {
        "dataset_name": args.dataset,
        "model_name": args.model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "pretrained": args.pretrained,
        "device": args.device,
        "max_train_batches": args.max_train_batches,
        "max_val_batches": args.max_val_batches,
        "max_test_batches": args.max_test_batches,
        "dry_run": args.dry_run,
        "experiment_name": args.experiment_name,
        "tracking_uri": args.tracking_uri,
    }
    return {key: value for key, value in values.items() if value is not None}


def build_final_config(args: argparse.Namespace) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    config.update(load_config_file(args.config))
    config.update(cli_overrides(args))
    config["stage"] = config.get("stage") or "baseline"
    config["optimizer"] = str(config.get("optimizer") or "adamw").lower()
    config["scheduler"] = str(config.get("scheduler") or "none").lower()

    if config["dataset_name"] not in DATASETS:
        supported = ", ".join(sorted(DATASETS))
        raise ValueError(f"Unsupported or missing dataset_name. Supported datasets: {supported}")
    if config["model_name"] not in SUPPORTED_MODELS:
        supported = ", ".join(sorted(SUPPORTED_MODELS))
        raise ValueError(f"Unsupported or missing model_name. Supported models: {supported}")
    if config["optimizer"] != "adamw":
        raise ValueError("Only optimizer='adamw' is supported in the baseline pipeline for now.")
    if config["scheduler"] != "none":
        raise ValueError("Only scheduler='none' is supported in the baseline pipeline for now.")

    return config


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(device_arg)


def resolve_split_file(dataset_key: str) -> Path:
    config = DATASETS[dataset_key]
    split_file = SPLITS_DIR / config.split_filename
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    return split_file


def append_summary_row(row: dict[str, Any], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    row_dataframe = pd.DataFrame([row])
    if output_path.exists():
        existing = pd.read_csv(output_path)
        output = pd.concat([existing, row_dataframe], ignore_index=True)
    else:
        output = row_dataframe

    extra_columns = [column for column in output.columns if column not in BASELINE_RESULT_COLUMNS]
    output = output.reindex(columns=BASELINE_RESULT_COLUMNS + extra_columns)
    output.to_csv(output_path, index=False)


def prefixed(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def run_dry_check(model, train_loader, device: torch.device) -> dict[str, float]:
    model.eval()
    inputs, targets = next(iter(train_loader))
    inputs = inputs.to(device)
    with torch.no_grad():
        outputs = model(inputs)
    return {
        "dry_run_batch_size": float(targets.size(0)),
        "dry_run_output_classes": float(outputs.shape[1]),
    }


def build_artifact_paths(run_name: str, dataset_name: str, model_name: str) -> dict[str, Path]:
    checkpoint_dir = CHECKPOINTS_DIR / dataset_name / model_name
    run_dir = BASELINE_RUNS_DIR / dataset_name / model_name / run_name
    return {
        "run_dir": run_dir,
        "config": run_dir / "config.json",
        "history": run_dir / "history.csv",
        "per_class_metrics": run_dir / "per_class_metrics.csv",
        "confusion_matrix_csv": run_dir / "confusion_matrix.csv",
        "confusion_matrix_png": run_dir / "confusion_matrix.png",
        "learning_curves": run_dir / "learning_curves.png",
        "summary": RESULTS_DIR / "baseline_results.csv",
        "checkpoint": checkpoint_dir / f"{run_name}_best.pt",
    }


def build_run_name_extra(config: dict[str, Any], device: torch.device) -> str:
    run_mode = "dry_run" if bool(config["dry_run"]) else "full"
    return f"{device.type}__{run_mode}__epochs{int(config['epochs'])}"


def add_runtime_fields(
    config: dict[str, Any],
    split_file: Path,
    class_names: list[str],
    train_loader,
    val_loader,
    test_loader,
    device: torch.device,
    model_summary: dict[str, float | int],
    paths: dict[str, Path],
) -> dict[str, Any]:
    final_config = dict(config)
    final_config.update(
        {
            "model_type": "baseline_student",
            "num_classes": len(class_names),
            "split_file": relative(split_file),
            "device": str(device),
            "total_train_samples": len(train_loader.dataset),
            "total_val_samples": len(val_loader.dataset),
            "total_test_samples": len(test_loader.dataset),
            "checkpoint_path": relative(paths["checkpoint"]),
            "history_path": relative(paths["history"]),
            "per_class_metrics_path": relative(paths["per_class_metrics"]),
            "confusion_matrix_csv_path": relative(paths["confusion_matrix_csv"]),
            "confusion_matrix_png_path": relative(paths["confusion_matrix_png"]),
            "learning_curves_path": relative(paths["learning_curves"]),
        }
    )
    final_config.update(model_summary)
    return final_config


def write_training_artifacts(
    training_output: dict[str, Any],
    class_names: list[str],
    paths: dict[str, Path],
) -> None:
    ensure_dir(paths["run_dir"])
    history_dataframe = pd.DataFrame(training_output["history"])
    history_dataframe.to_csv(paths["history"], index=False)

    per_class_dataframe = compute_per_class_metrics(
        training_output["test_y_true"],
        training_output["test_y_pred"],
        class_names,
    )
    per_class_dataframe.to_csv(paths["per_class_metrics"], index=False)

    save_confusion_matrix_artifacts(
        training_output["test_y_true"],
        training_output["test_y_pred"],
        class_names,
        paths["confusion_matrix_csv"],
        paths["confusion_matrix_png"],
    )
    save_learning_curves(training_output["history"], paths["learning_curves"])


def build_summary_row(
    config: dict[str, Any],
    run_name: str,
    mlflow_run_id: str,
    training_output: dict[str, Any],
) -> dict[str, Any]:
    best_val_metrics = training_output["best_val_metrics"]
    test_metrics = training_output["test_metrics"]
    return {
        **config,
        "run_name": run_name,
        "mlflow_run_id": mlflow_run_id,
        "best_epoch": training_output["best_epoch"],
        "best_val_accuracy": best_val_metrics.get("accuracy"),
        "best_val_macro_f1": best_val_metrics.get("macro_f1"),
        "test_accuracy": test_metrics.get("accuracy"),
        "test_macro_precision": test_metrics.get("macro_precision"),
        "test_macro_recall": test_metrics.get("macro_recall"),
        "test_macro_f1": test_metrics.get("macro_f1"),
        "test_weighted_precision": test_metrics.get("weighted_precision"),
        "test_weighted_recall": test_metrics.get("weighted_recall"),
        "test_weighted_f1": test_metrics.get("weighted_f1"),
    }


def main() -> None:
    args = parse_args()
    config = build_final_config(args)
    set_seed(int(config["seed"]))
    ensure_dir(RESULTS_DIR)
    ensure_dir(BASELINE_RUNS_DIR)
    ensure_dir(CHECKPOINTS_DIR / "pretrained")
    os.environ.setdefault("TORCH_HOME", str((CHECKPOINTS_DIR / "pretrained").resolve()))

    split_file = resolve_split_file(config["dataset_name"])
    device = select_device(config["device"])
    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        split_file=split_file,
        image_size=int(config["image_size"]),
        batch_size=int(config["batch_size"]),
        num_workers=int(config["num_workers"]),
        use_augmentation=True,
    )

    model = create_model(
        config["model_name"],
        num_classes=len(class_names),
        pretrained=bool(config["pretrained"]),
    ).to(device)
    model_summary = get_model_summary_dict(model)
    run_name = build_run_name(
        config["dataset_name"],
        config["model_name"],
        int(config["seed"]),
        extra=build_run_name_extra(config, device),
    )
    paths = build_artifact_paths(run_name, config["dataset_name"], config["model_name"])
    ensure_dir(paths["run_dir"])
    final_config = add_runtime_fields(
        config=config,
        split_file=split_file,
        class_names=class_names,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        model_summary=model_summary,
        paths=paths,
    )
    save_json(final_config, paths["config"])

    experiment_name = final_config["experiment_name"] or build_experiment_name("baseline")
    setup_mlflow(experiment_name=experiment_name, tracking_uri=final_config["tracking_uri"])
    run = start_run(run_name=run_name)
    print(
        "JOB_RUN_CONTEXT "
        + json.dumps(
            {
                "run_name": run_name,
                "mlflow_run_id": run.info.run_id,
                "tracking_uri": final_config["tracking_uri"],
                "artifact_paths": {
                    name: relative(path)
                    for name, path in paths.items()
                    if name != "summary"
                },
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    try:
        log_params(final_config)
        log_artifact(paths["config"])
        log_text("\n".join(class_names), "class_names.txt")

        if bool(final_config["dry_run"]):
            dry_metrics = run_dry_check(model, train_loader, device)
            log_metrics(dry_metrics)
            print(f"Dry-run successful. MLflow run_id={run.info.run_id}")
            print(f"Forward pass output classes: {int(dry_metrics['dry_run_output_classes'])}")
            return

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(final_config["learning_rate"]),
            weight_decay=float(final_config["weight_decay"]),
        )
        training_output = fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            class_names=class_names,
            optimizer=optimizer,
            device=device,
            epochs=int(final_config["epochs"]),
            checkpoint_path=paths["checkpoint"],
            logger=lambda metrics, step: log_metrics(metrics, step=step),
            max_train_batches=final_config["max_train_batches"],
            max_val_batches=final_config["max_val_batches"],
            max_test_batches=final_config["max_test_batches"],
            checkpoint_metadata=final_config,
        )

        write_training_artifacts(training_output, class_names, paths)
        best_val_metrics = training_output["best_val_metrics"]
        test_metrics = training_output["test_metrics"]
        log_metrics(
            {
                "best_val_accuracy": best_val_metrics.get("accuracy"),
                "best_val_macro_f1": best_val_metrics.get("macro_f1"),
                "test_accuracy": test_metrics.get("accuracy"),
                "test_macro_precision": test_metrics.get("macro_precision"),
                "test_macro_recall": test_metrics.get("macro_recall"),
                "test_macro_f1": test_metrics.get("macro_f1"),
                "test_weighted_f1": test_metrics.get("weighted_f1"),
            }
        )
        log_metrics(prefixed("best_val", best_val_metrics))
        log_metrics({"test_loss": training_output["test_loss"], **prefixed("test", test_metrics)})

        summary_row = build_summary_row(final_config, run_name, run.info.run_id, training_output)
        append_summary_row(summary_row, paths["summary"])

        for artifact_key in (
            "history",
            "per_class_metrics",
            "confusion_matrix_csv",
            "confusion_matrix_png",
            "learning_curves",
            "summary",
            "checkpoint",
        ):
            if paths[artifact_key].exists():
                log_artifact(paths[artifact_key])

        print(f"Baseline run complete. MLflow run_id={run.info.run_id}")
        print(f"Best epoch: {training_output['best_epoch']}")
        print(f"Best val accuracy: {best_val_metrics.get('accuracy'):.4f}")
        print(f"Best val macro_f1: {best_val_metrics.get('macro_f1'):.4f}")
        print(f"Test accuracy: {test_metrics.get('accuracy'):.4f}")
        print(f"Test macro_f1: {test_metrics.get('macro_f1'):.4f}")
        print(f"Checkpoint: {relative(paths['checkpoint'])}")
        print(f"Results: {relative(paths['summary'])}")
    finally:
        end_run()


if __name__ == "__main__":
    main()
