"""Run a validated single-teacher knowledge-distillation experiment."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
for import_path in (SRC_DIR, CURRENT_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from datasets.dataloaders import create_dataloaders
from datasets.dataset_registry import DATASETS, PROJECT_ROOT, SPLITS_DIR
from evaluation.metrics import compute_per_class_metrics
from evaluation.reports import save_confusion_matrix_artifacts, save_learning_curves
from models.model_factory import create_model
from models.model_info import get_model_compute_dict, get_model_summary_dict
from tracking.mlflow_tracker import (
    end_run,
    log_artifact,
    log_metrics,
    log_params,
    log_text,
    setup_mlflow,
    start_run,
)
from training.kd_config import KDConfig, load_kd_config, merge_kd_config
from training.kd_features import FeatureCapture, default_feature_layer, infer_feature_shapes
from training.kd_strategies import create_distillation_strategy
from training.kd_trainer import fit_kd, freeze_teacher
from training.optimization import create_optimizer, create_scheduler
from utils.io import ensure_dir, save_json
from utils.seed import set_seed


RESULTS_DIR = PROJECT_ROOT / "results/knowledge_distillation"
RUNS_DIR = RESULTS_DIR / "runs"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints/knowledge_distillation"
SUMMARY_PATH = RESULTS_DIR / "kd_results.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one knowledge-distillation experiment.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--teacher-run-id")
    parser.add_argument("--teacher-checkpoint")
    parser.add_argument("--dry-run", action="store_true", default=None)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> KDConfig:
    config = load_kd_config(args.config)
    return merge_kd_config(
        config,
        {
            "experiment": {
                "teacher_run_id": args.teacher_run_id,
                "teacher_checkpoint_path": args.teacher_checkpoint,
            },
            "training": {"dry_run": args.dry_run},
        },
    )


def select_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(value)


def resolve_split_file(dataset_name: str) -> Path:
    path = SPLITS_DIR / DATASETS[dataset_name].split_filename
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    return path


def load_teacher_checkpoint(
    teacher: torch.nn.Module,
    checkpoint_path: Path,
    expected_classes: list[str],
) -> dict[str, Any]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Teacher checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_classes = list(checkpoint.get("class_names", []))
    if checkpoint_classes != list(expected_classes):
        raise ValueError("Teacher checkpoint class_names do not match the dataset class order.")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("Teacher checkpoint does not contain model_state_dict.")
    teacher.load_state_dict(state)
    return checkpoint


def build_run_name(config: KDConfig, device: torch.device) -> str:
    exp = config.experiment
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    token = uuid.uuid4().hex[:6]
    return (
        f"kd__{exp.dataset_name}__{exp.student_model_name}__{exp.teacher_model_name}__"
        f"{exp.kd_type}__seed{exp.seed}__{device.type}__{timestamp}_{token}"
    )


def build_paths(config: KDConfig, run_name: str) -> dict[str, Path]:
    exp = config.experiment
    run_dir = RUNS_DIR / exp.dataset_name / exp.kd_type / run_name
    checkpoint_dir = CHECKPOINTS_DIR / exp.dataset_name / exp.kd_type
    return {
        "run_dir": run_dir,
        "config": run_dir / "config.json",
        "history": run_dir / "history.csv",
        "per_class_metrics": run_dir / "per_class_metrics.csv",
        "confusion_matrix_csv": run_dir / "confusion_matrix.csv",
        "confusion_matrix_png": run_dir / "confusion_matrix.png",
        "learning_curves": run_dir / "learning_curves.png",
        "checkpoint": checkpoint_dir / f"{run_name}_best.pt",
        "summary": SUMMARY_PATH,
    }


def relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def model_latency_ms(model, image_size: int, device: torch.device) -> float:
    sample = torch.zeros(1, 3, image_size, image_size, device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(3):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / 10


def flatten_params(config: KDConfig) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for section, values in config.to_dict().items():
        for key, value in values.items():
            flattened[f"{section}.{key}"] = json.dumps(value) if isinstance(value, dict) else value
    return flattened


def write_artifacts(output: dict, class_names: list[str], paths: dict[str, Path]) -> None:
    pd.DataFrame(output["history"]).to_csv(paths["history"], index=False)
    compute_per_class_metrics(
        output["test_y_true"], output["test_y_pred"], class_names
    ).to_csv(paths["per_class_metrics"], index=False)
    save_confusion_matrix_artifacts(
        output["test_y_true"],
        output["test_y_pred"],
        class_names,
        paths["confusion_matrix_csv"],
        paths["confusion_matrix_png"],
    )
    save_learning_curves(output["history"], paths["learning_curves"])


def append_summary(row: dict[str, Any]) -> None:
    ensure_dir(SUMMARY_PATH.parent)
    current = pd.DataFrame([row])
    if SUMMARY_PATH.exists() and SUMMARY_PATH.stat().st_size:
        current = pd.concat([pd.read_csv(SUMMARY_PATH), current], ignore_index=True)
    current.to_csv(SUMMARY_PATH, index=False)


def main() -> None:
    config = build_config(parse_args())
    exp, train, kd = config.experiment, config.training, config.distillation
    if not exp.teacher_checkpoint_path or not exp.teacher_run_id:
        raise ValueError("A completed teacher_run_id and teacher_checkpoint_path are required.")
    set_seed(exp.seed)
    os.environ.setdefault(
        "TORCH_HOME", str((PROJECT_ROOT / "checkpoints/pretrained").resolve())
    )
    device = select_device(train.device)
    split_file = resolve_split_file(exp.dataset_name)
    loaders = create_dataloaders(
        split_file,
        train.image_size,
        train.batch_size,
        train.num_workers,
        use_augmentation=train.data_augmentation,
    )
    train_loader, val_loader, test_loader, class_names = loaders
    student = create_model(
        exp.student_model_name, len(class_names), train.student_pretrained
    ).to(device)
    teacher = create_model(exp.teacher_model_name, len(class_names), False).to(device)
    load_teacher_checkpoint(
        teacher, PROJECT_ROOT / exp.teacher_checkpoint_path, class_names
    )
    freeze_teacher(teacher)

    student_layer = kd.student_layer or default_feature_layer(exp.student_model_name)
    teacher_layer = kd.teacher_layer or default_feature_layer(exp.teacher_model_name)
    shapes = None
    if exp.kd_type != "logit_based":
        shapes = infer_feature_shapes(
            student, teacher, student_layer, teacher_layer, train.image_size, device
        )
    strategy = create_distillation_strategy(
        exp.kd_type,
        kd,
        student_channels=shapes.student[1] if shapes else None,
        teacher_channels=shapes.teacher[1] if shapes else None,
    ).to(device)
    optimizer = create_optimizer(
        [*student.parameters(), *strategy.parameters()], train
    )
    scheduler = create_scheduler(optimizer, train)

    run_name = build_run_name(config, device)
    paths = build_paths(config, run_name)
    ensure_dir(paths["run_dir"])
    student_info = {
        **get_model_summary_dict(student),
        **get_model_compute_dict(student, train.image_size),
    }
    teacher_info = {
        **get_model_summary_dict(teacher),
        **get_model_compute_dict(teacher, train.image_size),
    }
    adapter_params = sum(parameter.numel() for parameter in strategy.parameters())
    runtime = {
        **config.to_dict(),
        "run_name": run_name,
        "device": device.type,
        "split_file": relative(split_file),
        "num_classes": len(class_names),
        "student": {**student_info, "inference_latency_ms": model_latency_ms(student, train.image_size, device)},
        "teacher": {**teacher_info, "inference_latency_ms": model_latency_ms(teacher, train.image_size, device)},
        "adapter_params": adapter_params,
        "feature_shapes": {
            "student": list(shapes.student) if shapes else None,
            "teacher": list(shapes.teacher) if shapes else None,
        },
        "artifact_paths": {key: relative(value) for key, value in paths.items()},
    }
    save_json(runtime, paths["config"])

    setup_mlflow(train.experiment_name, train.tracking_uri)
    run = start_run(run_name)
    print(
        "JOB_RUN_CONTEXT "
        + json.dumps(
            {
                "run_name": run_name,
                "mlflow_run_id": run.info.run_id,
                "tracking_uri": train.tracking_uri,
                "artifact_paths": {
                    key: relative(value) for key, value in paths.items() if key != "summary"
                },
            }
        ),
        flush=True,
    )
    student_capture = None
    teacher_capture = None
    try:
        log_params(flatten_params(config))
        log_params(
            {
                "teacher_run_id": exp.teacher_run_id,
                "student_params": student_info["params"],
                "teacher_params": teacher_info["params"],
                "student_macs": student_info["macs"],
                "teacher_macs": teacher_info["macs"],
                "adapter_params": adapter_params,
            }
        )
        log_artifact(paths["config"])
        log_text("\n".join(class_names), "class_names.txt")
        if train.dry_run:
            inputs, _targets = next(iter(train_loader))
            with torch.no_grad():
                student_output = student(inputs[:1].to(device))
                teacher_output = teacher(inputs[:1].to(device))
            if student_output.shape != teacher_output.shape:
                raise ValueError("Teacher and student output dimensions do not match.")
            print(
                f"KD dry-run successful. MLflow run_id={run.info.run_id} output={tuple(student_output.shape)}",
                flush=True,
            )
            return

        if exp.kd_type != "logit_based":
            student_capture = FeatureCapture(student, student_layer)
            teacher_capture = FeatureCapture(teacher, teacher_layer)
        output = fit_kd(
            student,
            teacher,
            strategy,
            train_loader,
            val_loader,
            test_loader,
            class_names,
            optimizer,
            scheduler,
            device,
            config,
            paths["checkpoint"],
            student_capture,
            teacher_capture,
            logger=lambda metrics, step: log_metrics(metrics, step),
        )
        write_artifacts(output, class_names, paths)
        best, test = output["best_val_metrics"], output["test_metrics"]
        summary = {
            "run_name": run_name,
            "mlflow_run_id": run.info.run_id,
            "stage": "knowledge_distillation",
            "dataset_name": exp.dataset_name,
            "kd_type": exp.kd_type,
            "student_model_name": exp.student_model_name,
            "teacher_model_name": exp.teacher_model_name,
            "teacher_run_id": exp.teacher_run_id,
            "seed": exp.seed,
            "best_epoch": output["best_epoch"],
            "best_val_accuracy": best.get("accuracy"),
            "best_val_macro_f1": best.get("macro_f1"),
            "test_accuracy": test.get("accuracy"),
            "test_macro_precision": test.get("macro_precision"),
            "test_macro_recall": test.get("macro_recall"),
            "test_macro_f1": test.get("macro_f1"),
            "student_params": student_info["params"],
            "student_macs": student_info["macs"],
            "student_estimated_flops": student_info["estimated_flops"],
            "student_model_size_mb": student_info["model_size_mb"],
            "student_inference_latency_ms": runtime["student"]["inference_latency_ms"],
            "teacher_params": teacher_info["params"],
            "teacher_macs": teacher_info["macs"],
            "teacher_estimated_flops": teacher_info["estimated_flops"],
            "teacher_model_size_mb": teacher_info["model_size_mb"],
            "teacher_inference_latency_ms": runtime["teacher"]["inference_latency_ms"],
            "adapter_params": adapter_params,
            "checkpoint_path": relative(paths["checkpoint"]),
            "run_dir": relative(paths["run_dir"]),
            "config_path": relative(paths["config"]),
            "history_path": relative(paths["history"]),
            "per_class_metrics_path": relative(paths["per_class_metrics"]),
            "confusion_matrix_csv_path": relative(paths["confusion_matrix_csv"]),
            "confusion_matrix_png_path": relative(paths["confusion_matrix_png"]),
            "learning_curves_path": relative(paths["learning_curves"]),
            "tracking_uri": train.tracking_uri,
        }
        append_summary(summary)
        log_metrics(
            {
                "best_val_accuracy": best.get("accuracy"),
                "best_val_macro_f1": best.get("macro_f1"),
                "test_accuracy": test.get("accuracy"),
                "test_macro_f1": test.get("macro_f1"),
            }
        )
        for key, path in paths.items():
            if key != "run_dir" and path.exists():
                log_artifact(path)
        print(f"KD run complete. MLflow run_id={run.info.run_id}", flush=True)
    finally:
        if student_capture:
            student_capture.close()
        if teacher_capture:
            teacher_capture.close()
        end_run()


if __name__ == "__main__":
    main()
