"""Typed configuration and validation for knowledge-distillation experiments."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


SUPPORTED_DATASET = "plantpathology2021"
SUPPORTED_STUDENT = "mobilenet_v3_small"
SUPPORTED_TEACHERS = frozenset({"densenet201", "resnet50", "regnet_y_8gf"})
SUPPORTED_KD_TYPES = frozenset({"logit_based", "feature_based", "relation_based"})
SUPPORTED_OPTIMIZERS = frozenset({"adamw", "adam", "sgd"})
SUPPORTED_SCHEDULERS = frozenset({"cosine", "step", "reduce_on_plateau", "none"})
SUPPORTED_MONITORS = frozenset({"val_macro_f1", "val_accuracy", "val_loss"})


@dataclass(frozen=True)
class ExperimentConfig:
    stage: str = "knowledge_distillation"
    kd_type: str = "logit_based"
    dataset_name: str = SUPPORTED_DATASET
    student_model_name: str = SUPPORTED_STUDENT
    teacher_model_name: str = "resnet50"
    teacher_run_id: str | None = None
    teacher_checkpoint_path: str | None = None
    seed: int = 42


@dataclass(frozen=True)
class TrainingConfig:
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 30
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    optimizer_params: dict[str, Any] = field(default_factory=dict)
    scheduler: str = "cosine"
    scheduler_params: dict[str, Any] = field(default_factory=dict)
    early_stopping_patience: int = 7
    monitor_metric: str = "val_macro_f1"
    student_pretrained: bool = True
    data_augmentation: bool = True
    num_workers: int = 4
    device: str = "auto"
    dry_run: bool = False
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    max_test_batches: int | None = None
    experiment_name: str = "MasterThesis-Knowledge-Distillation"
    tracking_uri: str = "sqlite:///mlflow.db"


@dataclass(frozen=True)
class DistillationConfig:
    temperature: float = 4.0
    alpha: float = 0.5
    hard_label_loss_weight: float = 1.0
    feature_loss: str = "mse"
    feature_loss_weight: float = 1.0
    teacher_layer: str | None = None
    student_layer: str = "features.12"
    adapter: str = "automatic"
    spatial_alignment: str = "automatic"
    relation_type: str = "distance_and_angle"
    distance_loss_weight: float = 1.0
    angle_loss_weight: float = 2.0
    feature_normalization: bool = True
    pair_sampling: str = "all"
    include_logit_distillation: bool = False


@dataclass(frozen=True)
class KDConfig:
    experiment: ExperimentConfig
    training: TrainingConfig
    distillation: DistillationConfig

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_kd_config(source: str | Path | Mapping[str, Any]) -> KDConfig:
    """Load a nested KD JSON config or mapping and validate its complete schema."""
    if isinstance(source, Mapping):
        payload = dict(source)
    else:
        path = Path(source)
        payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("KD config must contain a JSON object.")
    try:
        config = KDConfig(
            experiment=ExperimentConfig(**_mapping(payload, "experiment")),
            training=TrainingConfig(**_mapping(payload, "training")),
            distillation=DistillationConfig(**_mapping(payload, "distillation")),
        )
    except TypeError as exc:
        raise ValueError(f"Invalid KD config field: {exc}") from exc
    validate_kd_config(config)
    return config


def validate_kd_config(config: KDConfig) -> None:
    exp, train, kd = config.experiment, config.training, config.distillation
    errors: list[str] = []
    if exp.stage != "knowledge_distillation":
        errors.append("experiment.stage must be 'knowledge_distillation'")
    if exp.dataset_name != SUPPORTED_DATASET:
        errors.append(f"dataset_name must be '{SUPPORTED_DATASET}'")
    if exp.student_model_name != SUPPORTED_STUDENT:
        errors.append(f"student_model_name must be '{SUPPORTED_STUDENT}'")
    if exp.teacher_model_name not in SUPPORTED_TEACHERS:
        errors.append(f"unsupported teacher_model_name: {exp.teacher_model_name}")
    if exp.kd_type not in SUPPORTED_KD_TYPES:
        errors.append(f"unsupported kd_type: {exp.kd_type}")
    if train.optimizer.lower() not in SUPPORTED_OPTIMIZERS:
        errors.append(f"unsupported optimizer: {train.optimizer}")
    if train.scheduler.lower() not in SUPPORTED_SCHEDULERS:
        errors.append(f"unsupported scheduler: {train.scheduler}")
    if train.monitor_metric not in SUPPORTED_MONITORS:
        errors.append(f"unsupported monitor_metric: {train.monitor_metric}")
    if train.device not in {"auto", "cpu", "cuda"}:
        errors.append("device must be auto, cpu or cuda")
    for name, value in {
        "image_size": train.image_size,
        "batch_size": train.batch_size,
        "epochs": train.epochs,
        "num_workers": train.num_workers,
    }.items():
        if value < 0 or (name != "num_workers" and value == 0):
            errors.append(f"training.{name} has an invalid value")
    if train.learning_rate <= 0 or train.weight_decay < 0:
        errors.append("learning_rate must be positive and weight_decay non-negative")
    if kd.temperature <= 0:
        errors.append("distillation.temperature must be positive")
    if not 0 <= kd.alpha <= 1:
        errors.append("distillation.alpha must be between 0 and 1")
    if kd.feature_loss not in {"mse", "l1", "cosine"}:
        errors.append("feature_loss must be mse, l1 or cosine")
    if kd.relation_type not in {"distance", "angle", "distance_and_angle", "correlation"}:
        errors.append("unsupported relation_type")
    if kd.adapter != "automatic":
        errors.append("only the automatic feature adapter is supported")
    if kd.spatial_alignment != "automatic":
        errors.append("only automatic spatial alignment is supported")
    if kd.pair_sampling != "all":
        errors.append("only all-pairs relation sampling is supported")
    if kd.include_logit_distillation:
        errors.append("combined logit distillation is not supported in this version")
    if errors:
        raise ValueError("; ".join(errors))


def merge_kd_config(config: KDConfig, overrides: Mapping[str, Mapping[str, Any]]) -> KDConfig:
    """Return a validated config with section-scoped UI/CLI overrides."""
    payload = config.to_dict()
    for section in ("experiment", "training", "distillation"):
        values = overrides.get(section, {})
        if values:
            payload[section].update({key: value for key, value in values.items() if value is not None})
    return load_kd_config(payload)


def _mapping(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"KD config section '{key}' is required and must be an object.")
    return dict(value)
