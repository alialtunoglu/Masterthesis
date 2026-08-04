"""Typed configuration for multi-teacher knowledge-distillation experiments."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from training.kd_config import (
    SUPPORTED_MONITORS,
    SUPPORTED_OPTIMIZERS,
    SUPPORTED_SCHEDULERS,
    TrainingConfig,
)


SUPPORTED_DATASET = "plantpathology2021"
SUPPORTED_STUDENT = "mobilenet_v3_small"
SUPPORTED_TEACHERS = frozenset({"resnet50", "densenet201", "regnet_y_8gf"})
SUPPORTED_KD_TYPES = frozenset({"logit_based", "feature_based", "relation_based"})
SUPPORTED_AGGREGATIONS = frozenset({"uniform", "validation_weighted", "manual"})


@dataclass(frozen=True)
class MultiExperimentConfig:
    stage: str = "multi_teacher_knowledge_distillation"
    kd_type: str = "logit_based"
    dataset_name: str = SUPPORTED_DATASET
    seed: int = 42


@dataclass(frozen=True)
class StudentConfig:
    model_name: str = SUPPORTED_STUDENT
    feature_layer: str = "features.12"


@dataclass(frozen=True)
class TeacherConfig:
    model_name: str
    feature_layer: str
    run_id: str | None = None
    checkpoint_path: str | None = None
    best_val_macro_f1: float | None = None
    manual_weight: float | None = None
    resolved_weight: float | None = None


@dataclass(frozen=True)
class MultiDistillationConfig:
    aggregation: str = "uniform"
    temperature: float = 4.0
    alpha: float = 0.5
    hard_label_loss_weight: float = 1.0
    feature_loss: str = "mse"
    feature_loss_weight: float = 1.0
    relation_type: str = "distance_and_angle"
    distance_loss_weight: float = 1.0
    angle_loss_weight: float = 2.0
    feature_normalization: bool = True


@dataclass(frozen=True)
class MultiKDConfig:
    experiment: MultiExperimentConfig
    student: StudentConfig
    teachers: tuple[TeacherConfig, ...]
    training: TrainingConfig
    distillation: MultiDistillationConfig

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["teachers"] = [asdict(teacher) for teacher in self.teachers]
        return payload


def load_multi_kd_config(source: str | Path | Mapping[str, Any]) -> MultiKDConfig:
    if isinstance(source, Mapping):
        payload = dict(source)
    else:
        payload = json.loads(Path(source).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Multi-teacher KD config must contain a JSON object.")
    raw_teachers = payload.get("teachers")
    if not isinstance(raw_teachers, list):
        raise ValueError("teachers must be a JSON array.")
    try:
        teachers = tuple(
            sorted(
                (TeacherConfig(**dict(item)) for item in raw_teachers),
                key=lambda teacher: teacher.model_name,
            )
        )
        config = MultiKDConfig(
            experiment=MultiExperimentConfig(**_mapping(payload, "experiment")),
            student=StudentConfig(**_mapping(payload, "student")),
            teachers=teachers,
            training=TrainingConfig(**_mapping(payload, "training")),
            distillation=MultiDistillationConfig(**_mapping(payload, "distillation")),
        )
    except (TypeError, AttributeError) as exc:
        raise ValueError(f"Invalid multi-teacher KD config field: {exc}") from exc
    validate_multi_kd_config(config)
    return config


def validate_multi_kd_config(config: MultiKDConfig, *, runtime: bool = False) -> None:
    exp, student, train, kd = (
        config.experiment,
        config.student,
        config.training,
        config.distillation,
    )
    errors: list[str] = []
    if exp.stage != "multi_teacher_knowledge_distillation":
        errors.append("experiment.stage must be 'multi_teacher_knowledge_distillation'")
    if exp.dataset_name != SUPPORTED_DATASET:
        errors.append(f"dataset_name must be '{SUPPORTED_DATASET}'")
    if student.model_name != SUPPORTED_STUDENT:
        errors.append(f"student.model_name must be '{SUPPORTED_STUDENT}'")
    if exp.kd_type not in SUPPORTED_KD_TYPES:
        errors.append(f"unsupported kd_type: {exp.kd_type}")
    if len(config.teachers) < 2:
        errors.append("at least two teachers are required")
    names = [teacher.model_name for teacher in config.teachers]
    if len(names) != len(set(names)):
        errors.append("teacher model names must be unique")
    unsupported = sorted(set(names) - SUPPORTED_TEACHERS)
    if unsupported:
        errors.append(f"unsupported teacher models: {', '.join(unsupported)}")
    checkpoints = [
        teacher.checkpoint_path for teacher in config.teachers if teacher.checkpoint_path
    ]
    if len(checkpoints) != len(set(checkpoints)):
        errors.append("teacher checkpoint paths must be unique")
    if kd.aggregation not in SUPPORTED_AGGREGATIONS:
        errors.append(f"unsupported aggregation: {kd.aggregation}")
    if kd.temperature <= 0 or not 0 <= kd.alpha <= 1:
        errors.append("temperature must be positive and alpha must be between 0 and 1")
    if kd.feature_loss not in {"mse", "l1", "cosine"}:
        errors.append("feature_loss must be mse, l1 or cosine")
    if kd.relation_type not in {"distance", "angle", "distance_and_angle", "correlation"}:
        errors.append("unsupported relation_type")
    if train.optimizer.lower() not in SUPPORTED_OPTIMIZERS:
        errors.append(f"unsupported optimizer: {train.optimizer}")
    if train.scheduler.lower() not in SUPPORTED_SCHEDULERS:
        errors.append(f"unsupported scheduler: {train.scheduler}")
    if train.monitor_metric not in SUPPORTED_MONITORS:
        errors.append(f"unsupported monitor_metric: {train.monitor_metric}")
    if train.device not in {"auto", "cpu", "cuda"}:
        errors.append("device must be auto, cpu or cuda")
    if (
        train.image_size <= 0
        or train.batch_size <= 0
        or train.epochs <= 0
        or train.num_workers < 0
    ):
        errors.append("training dimensions and epochs must be positive")
    if train.learning_rate <= 0 or train.weight_decay < 0:
        errors.append("learning_rate must be positive and weight_decay non-negative")
    if runtime:
        for teacher in config.teachers:
            if not teacher.run_id or not teacher.checkpoint_path:
                errors.append(f"{teacher.model_name} requires run_id and checkpoint_path")
            if teacher.resolved_weight is None:
                errors.append(f"{teacher.model_name} requires resolved_weight")
    if errors:
        raise ValueError("; ".join(errors))


def merge_multi_kd_config(
    config: MultiKDConfig, overrides: Mapping[str, Any]
) -> MultiKDConfig:
    payload = config.to_dict()
    for section in ("experiment", "student", "training", "distillation"):
        values = overrides.get(section)
        if isinstance(values, Mapping):
            payload[section].update(
                {key: value for key, value in values.items() if value is not None}
            )
    if "teachers" in overrides:
        payload["teachers"] = overrides["teachers"]
    return load_multi_kd_config(payload)


def _mapping(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Config section '{key}' is required and must be an object.")
    return dict(value)
