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
from datasets.dataset_registry import DATASETS, PROJECT_ROOT, SPLITS_DIR, get_dataset_catalog, DatasetRef
from datasets.dataset_runtime import resolve_dataset_runtime, verify_dataset_content
from training.experiment_config import resolve_experiment_config
from evaluation.metrics import compute_per_class_metrics
from evaluation.reports import save_confusion_matrix_artifacts, save_learning_curves
from models.model_factory import STUDENT_MODELS, create_model
from models.model_catalog import STUDENT_VIT_MODELS
from tracking.experiment_taxonomy import enrich_experiment_metadata
from models.model_info import get_model_summary_dict
from tracking.experiment_naming import build_experiment_name, build_unique_run_name
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
from training.kd_config import TrainingConfig
from training.optimization import create_optimizer, create_scheduler
from utils.io import ensure_dir, load_json, save_json, upsert_result_row
from utils.seed import set_seed
from training.recovery import (add_recovery_arguments, create_recovery, recovery_settings,
                               mark_recovery, TrainingStopped)


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
    "optimizer_params": {},
    "scheduler_params": {},
    "early_stopping_patience": None,
    "monitor_metric": "val_macro_f1",
    "use_augmentation": True,
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
    "stage", "training_method", "model_role", "architecture_family",
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
    parser.add_argument("--dataset", choices=sorted(get_dataset_catalog(PROJECT_ROOT)), default=None)
    parser.add_argument("--model", choices=sorted(STUDENT_MODELS), default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--scheduler", choices=["none", "cosine"], default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
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
    add_recovery_arguments(parser)
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
        "scheduler": args.scheduler,
        "early_stopping_patience": args.early_stopping_patience,
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
    config["stage"] = "baseline"
    config["optimizer"] = str(config.get("optimizer") or "adamw").lower()
    config["scheduler"] = str(config.get("scheduler") or "none").lower()

    if config["dataset_name"] not in get_dataset_catalog(PROJECT_ROOT) and not config.get("dataset_ref"):
        supported = ", ".join(sorted(get_dataset_catalog(PROJECT_ROOT)))
        raise ValueError(f"Unsupported or missing dataset_name. Supported datasets: {supported}")
    if config["model_name"] not in STUDENT_MODELS:
        supported = ", ".join(sorted(STUDENT_MODELS))
        raise ValueError(f"Unsupported or missing model_name. Supported models: {supported}")
    if config["model_name"] in STUDENT_VIT_MODELS and int(config["image_size"]) != 224:
        raise ValueError("Student ViT models require image_size=224.")
    if config["optimizer"] != "adamw":
        raise ValueError("Only optimizer='adamw' is supported in the baseline pipeline for now.")
    if config["scheduler"] not in {"none", "cosine"}:
        raise ValueError("Baseline scheduler must be 'none' or 'cosine'.")
    for key in ("optimizer_params", "scheduler_params"):
        if not isinstance(config[key], dict):
            raise ValueError(f"{key} must be a JSON object.")
    patience = config["early_stopping_patience"]
    if patience is not None and (not isinstance(patience, int) or isinstance(patience, bool) or patience < 0):
        raise ValueError("early_stopping_patience must be a non-negative integer or null.")
    config["early_stopping_patience"] = patience or None
    if config["monitor_metric"] != "val_macro_f1":
        raise ValueError("Baseline monitor_metric must be 'val_macro_f1'.")

    return enrich_experiment_metadata(resolve_experiment_config(PROJECT_ROOT, config))


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
    upsert_result_row(row, output_path, BASELINE_RESULT_COLUMNS)


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
            "run_dir": relative(paths["run_dir"]),
            "config_path": relative(paths["config"]),
            "run_metadata_path": relative(paths["run_dir"] / "run_metadata.json"),
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

    dataset_runtime = resolve_dataset_runtime(PROJECT_ROOT, config.get("dataset_ref"), config["dataset_name"])
    if config.get("dataset_ref"):
        verify_dataset_content(PROJECT_ROOT, DatasetRef(**config["dataset_ref"]))
    split_file = Path(dataset_runtime["split_file"])
    device = select_device(config["device"])
    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        split_file=split_file,
        image_size=int(config["image_size"]),
        batch_size=int(config["batch_size"]),
        num_workers=int(config["num_workers"]),
        use_augmentation=bool(config["use_augmentation"]),
        recovery_seed=int(config["seed"]) if recovery_settings(args, config)["enabled"] else None,
        project_root=PROJECT_ROOT,
    )

    model = create_model(
        config["model_name"],
        num_classes=len(class_names),
        pretrained=bool(config["pretrained"]),
    ).to(device)
    model_summary = get_model_summary_dict(model)
    run_name = build_unique_run_name("baseline", int(config["seed"]), device.type, bool(config["dry_run"]))
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
    final_config.update({key: value for key, value in dataset_runtime.items() if key != "split_file" and value is not None})
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

    recovery = None
    status = "FINISHED"
    try:
        recovery = create_recovery(args, config, paths, split_file, PROJECT_ROOT, run.info.run_id)
        log_params(final_config)
        log_artifact(paths["config"])
        log_text("\n".join(class_names), "class_names.txt")

        if bool(final_config["dry_run"]):
            dry_metrics = run_dry_check(model, train_loader, device)
            log_metrics(dry_metrics)
            print(f"Dry-run successful. MLflow run_id={run.info.run_id}")
            print(f"Forward pass output classes: {int(dry_metrics['dry_run_output_classes'])}")
            return

        optimization = TrainingConfig(
            epochs=int(final_config["epochs"]),
            learning_rate=float(final_config["learning_rate"]),
            weight_decay=float(final_config["weight_decay"]),
            optimizer=final_config["optimizer"],
            optimizer_params=final_config["optimizer_params"],
            scheduler=final_config["scheduler"],
            scheduler_params=final_config["scheduler_params"],
        )
        optimizer = create_optimizer(model.parameters(), optimization)
        scheduler = create_scheduler(optimizer, optimization)
        training_output = fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            class_names=class_names,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epochs=int(final_config["epochs"]),
            checkpoint_path=paths["checkpoint"],
            logger=lambda metrics, step: log_metrics(metrics, step=step),
            max_train_batches=final_config["max_train_batches"],
            max_val_batches=final_config["max_val_batches"],
            max_test_batches=final_config["max_test_batches"],
            checkpoint_metadata=final_config,
            early_stopping_patience=final_config["early_stopping_patience"],
            monitor_metric=final_config["monitor_metric"],
            recovery=recovery,
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
        if summary_row.get("dataset_ref"):
            summary_row["dataset_ref"] = json.dumps(summary_row["dataset_ref"], sort_keys=True)
        summary_row.update({"training_time_seconds": recovery.elapsed_seconds(), "backup_time_seconds": recovery.backup_seconds})
        log_metrics({key: summary_row[key] for key in ("training_time_seconds", "backup_time_seconds")})
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
    except TrainingStopped:
        status = "KILLED"
        print("TRAINING_STOPPED saved at epoch boundary", flush=True)
    except BaseException:
        status = "FAILED"
        raise
    finally:
        mark_recovery(recovery, status)
        try:
            if (paths["run_dir"] / "run_metadata.json").is_file():
                log_artifact(paths["run_dir"] / "run_metadata.json")
        except Exception as error:
            print(f"RUN_METADATA_LOG_WARNING {error}", flush=True)
        end_run(status)


if __name__ == "__main__":
    main()
