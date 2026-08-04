"""Run one validated multi-teacher knowledge-distillation experiment."""

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
from training.kd_features import FeatureCapture, infer_feature_shapes
from training.kd_trainer import evaluate_student, freeze_teacher
from training.multi_kd_aggregation import resolve_teacher_weights
from training.multi_kd_config import (
    MultiKDConfig,
    load_multi_kd_config,
    validate_multi_kd_config,
)
from training.multi_kd_strategies import create_multi_teacher_strategy
from training.multi_kd_trainer import (
    fit_multi_teacher_kd,
    train_multi_teacher_epoch,
)
from training.optimization import create_optimizer, create_scheduler
from training.train_kd import load_teacher_checkpoint, model_latency_ms, select_device
from utils.io import ensure_dir, save_json
from utils.seed import set_seed


RESULTS_DIR = PROJECT_ROOT / "results/multi_teacher_knowledge_distillation"
RUNS_DIR = RESULTS_DIR / "runs"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints/multi_teacher_knowledge_distillation"
SUMMARY_PATH = RESULTS_DIR / "multi_kd_results.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-teacher KD.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_runtime_config(config: MultiKDConfig, dry_run: bool) -> MultiKDConfig:
    weights = resolve_teacher_weights(
        config.distillation.aggregation, config.teachers
    )
    payload = config.to_dict()
    payload["training"]["dry_run"] = dry_run or config.training.dry_run
    for teacher, weight in zip(payload["teachers"], weights):
        teacher["resolved_weight"] = weight
    resolved = load_multi_kd_config(payload)
    validate_multi_kd_config(resolved, runtime=True)
    return resolved


def build_run_name(config: MultiKDConfig, device: torch.device) -> str:
    model_tokens = {
        "mobilenet_v3_small": "mnetv3s",
        "densenet201": "dn201",
        "resnet50": "rn50",
        "regnet_y_8gf": "rgy8",
    }
    kd_tokens = {
        "logit_based": "logit",
        "feature_based": "feature",
        "relation_based": "relation",
    }
    aggregation_tokens = {
        "uniform": "uni",
        "validation_weighted": "val",
        "manual": "manual",
    }
    teachers = "+".join(
        model_tokens[teacher.model_name] for teacher in config.teachers
    )
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    token = uuid.uuid4().hex[:6]
    return (
        f"mkd__pp21__{model_tokens[config.student.model_name]}__{teachers}__"
        f"{kd_tokens[config.experiment.kd_type]}__"
        f"{aggregation_tokens[config.distillation.aggregation]}__"
        f"s{config.experiment.seed}__{device.type}__{timestamp}_{token}"
    )


def build_paths(config: MultiKDConfig, run_name: str) -> dict[str, Path]:
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


def flatten_params(config: MultiKDConfig) -> dict[str, Any]:
    payload = config.to_dict()
    return {
        "experiment": json.dumps(payload["experiment"], sort_keys=True),
        "student": json.dumps(payload["student"], sort_keys=True),
        "teachers": json.dumps(payload["teachers"], sort_keys=True),
        "training": json.dumps(payload["training"], sort_keys=True),
        "distillation": json.dumps(payload["distillation"], sort_keys=True),
    }


def write_artifacts(output: dict, class_names: list[str], paths: dict[str, Path]):
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
    frame = pd.DataFrame([row])
    if SUMMARY_PATH.exists() and SUMMARY_PATH.stat().st_size:
        frame = pd.concat([pd.read_csv(SUMMARY_PATH), frame], ignore_index=True)
    frame.to_csv(SUMMARY_PATH, index=False)


def main() -> None:
    args = parse_args()
    config = resolve_runtime_config(load_multi_kd_config(args.config), args.dry_run)
    exp, train = config.experiment, config.training
    set_seed(exp.seed)
    os.environ.setdefault(
        "TORCH_HOME", str((PROJECT_ROOT / "checkpoints/pretrained").resolve())
    )
    device = select_device(train.device)
    split_file = SPLITS_DIR / DATASETS[exp.dataset_name].split_filename
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        split_file,
        train.image_size,
        train.batch_size,
        train.num_workers,
        use_augmentation=train.data_augmentation,
    )
    student = create_model(
        config.student.model_name, len(class_names), train.student_pretrained
    ).to(device)
    teachers: list[torch.nn.Module] = []
    for spec in config.teachers:
        teacher = create_model(spec.model_name, len(class_names), False).to(device)
        load_teacher_checkpoint(
            teacher, PROJECT_ROOT / str(spec.checkpoint_path), class_names
        )
        freeze_teacher(teacher)
        teachers.append(teacher)

    shapes = []
    if exp.kd_type != "logit_based":
        for teacher, spec in zip(teachers, config.teachers):
            shapes.append(
                infer_feature_shapes(
                    student,
                    teacher,
                    config.student.feature_layer,
                    spec.feature_layer,
                    train.image_size,
                    device,
                )
            )
    weights = [float(teacher.resolved_weight or 0) for teacher in config.teachers]
    strategy = create_multi_teacher_strategy(
        exp.kd_type,
        config.distillation,
        [teacher.model_name for teacher in config.teachers],
        weights,
        student_channels=shapes[0].student[1] if shapes else None,
        teacher_channels=[shape.teacher[1] for shape in shapes] if shapes else None,
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
        "inference_latency_ms": model_latency_ms(student, train.image_size, device),
    }
    teacher_infos = []
    for model, spec in zip(teachers, config.teachers):
        teacher_infos.append(
            {
                "model_name": spec.model_name,
                **get_model_summary_dict(model),
                **get_model_compute_dict(model, train.image_size),
                "inference_latency_ms": model_latency_ms(
                    model, train.image_size, device
                ),
            }
        )
    runtime = {
        **config.to_dict(),
        "run_name": run_name,
        "device": device.type,
        "class_names": class_names,
        "student_profile": student_info,
        "teacher_profiles": teacher_infos,
        "adapter_params": sum(p.numel() for p in strategy.parameters()),
        "feature_shapes": [
            {"student": list(shape.student), "teacher": list(shape.teacher)}
            for shape in shapes
        ],
        "artifact_paths": {key: relative(path) for key, path in paths.items()},
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
                    key: relative(path)
                    for key, path in paths.items()
                    if key != "summary"
                },
            }
        ),
        flush=True,
    )
    student_capture = None
    teacher_captures: list[FeatureCapture] = []
    status = "FINISHED"
    try:
        log_params(flatten_params(config))
        log_artifact(paths["config"])
        log_text("\n".join(class_names), "class_names.txt")
        if exp.kd_type != "logit_based":
            student_capture = FeatureCapture(
                student, config.student.feature_layer
            )
            teacher_captures = [
                FeatureCapture(model, spec.feature_layer)
                for model, spec in zip(teachers, config.teachers)
            ]
        if train.dry_run:
            train_metrics = train_multi_teacher_epoch(
                student,
                teachers,
                [teacher.model_name for teacher in config.teachers],
                strategy,
                train_loader,
                optimizer,
                device,
                student_capture,
                teacher_captures,
                1,
            )
            validation = evaluate_student(
                student, val_loader, device, class_names, 1
            )
            log_metrics(
                {
                    **train_metrics,
                    "val_loss": validation["loss"],
                    "val_macro_f1": validation["metrics"]["macro_f1"],
                }
            )
            print(
                f"Multi-teacher KD dry-run successful. MLflow run_id={run.info.run_id}",
                flush=True,
            )
            return
        output = fit_multi_teacher_kd(
            student,
            teachers,
            [teacher.model_name for teacher in config.teachers],
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
            teacher_captures,
            logger=lambda metrics, step: log_metrics(metrics, step),
        )
        write_artifacts(output, class_names, paths)
        best, test, ensemble = (
            output["best_val_metrics"],
            output["test_metrics"],
            output["ensemble_metrics"],
        )
        summary = {
            "run_name": run_name,
            "mlflow_run_id": run.info.run_id,
            "stage": exp.stage,
            "dataset_name": exp.dataset_name,
            "kd_type": exp.kd_type,
            "aggregation": config.distillation.aggregation,
            "student_model_name": config.student.model_name,
            "teacher_models": json.dumps(
                [teacher.model_name for teacher in config.teachers]
            ),
            "teacher_run_ids": json.dumps(
                [teacher.run_id for teacher in config.teachers]
            ),
            "resolved_teacher_weights": json.dumps(weights),
            "teacher_count": len(config.teachers),
            "seed": exp.seed,
            "best_epoch": output["best_epoch"],
            "best_val_accuracy": best.get("accuracy"),
            "best_val_macro_f1": best.get("macro_f1"),
            "test_accuracy": test.get("accuracy"),
            "test_macro_f1": test.get("macro_f1"),
            "ensemble_test_accuracy": ensemble.get("accuracy"),
            "ensemble_test_macro_f1": ensemble.get("macro_f1"),
            "student_params": student_info["params"],
            "student_macs": student_info["macs"],
            "student_estimated_flops": student_info["estimated_flops"],
            "student_model_size_mb": student_info["model_size_mb"],
            "student_inference_latency_ms": student_info["inference_latency_ms"],
            "teacher_total_params": sum(info["params"] for info in teacher_infos),
            "teacher_total_macs": sum(info["macs"] for info in teacher_infos),
            "teacher_total_estimated_flops": sum(
                info["estimated_flops"] for info in teacher_infos
            ),
            "teacher_profiles": json.dumps(teacher_infos),
            "adapter_params": runtime["adapter_params"],
            "checkpoint_path": relative(paths["checkpoint"]),
            "run_dir": relative(paths["run_dir"]),
            "config_path": relative(paths["config"]),
            "history_path": relative(paths["history"]),
            "per_class_metrics_path": relative(paths["per_class_metrics"]),
            "confusion_matrix_png_path": relative(paths["confusion_matrix_png"]),
            "learning_curves_path": relative(paths["learning_curves"]),
            "tracking_uri": train.tracking_uri,
        }
        append_summary(summary)
        log_metrics(
            {
                "best_val_macro_f1": best.get("macro_f1"),
                "test_accuracy": test.get("accuracy"),
                "test_macro_f1": test.get("macro_f1"),
                "ensemble_test_accuracy": ensemble.get("accuracy"),
                "ensemble_test_macro_f1": ensemble.get("macro_f1"),
            }
        )
        for key, path in paths.items():
            if key != "run_dir" and path.exists():
                log_artifact(path)
        print(
            f"Multi-teacher KD run complete. MLflow run_id={run.info.run_id}",
            flush=True,
        )
    except Exception:
        status = "FAILED"
        raise
    finally:
        if student_capture:
            student_capture.close()
        for capture in teacher_captures:
            capture.close()
        end_run(status)


if __name__ == "__main__":
    main()
