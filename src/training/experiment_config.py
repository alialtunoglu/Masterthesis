"""Shared dataset/model validation for flat baseline and nested KD configs."""

from copy import deepcopy
from pathlib import Path
import re

EXTENSIBLE_DATASET_API = 2

try:
    from src.datasets.dataset_registry import DATASETS
    from src.datasets.dataset_runtime import resolve_dataset_runtime
    from src.models.model_catalog import get_model_spec
except ModuleNotFoundError:
    from src.datasets.dataset_registry import DATASETS
    from src.datasets.dataset_runtime import resolve_dataset_runtime
    from src.models.model_catalog import get_model_spec


def validate_dataset_selection(name: str, ref: dict | None) -> None:
    if ref is None:
        if name not in DATASETS:
            raise ValueError("dataset_name requires a registered dataset_ref for custom datasets.")
        return
    if not isinstance(ref, dict) or set(ref) != {"dataset_id", "revision", "split_id"} or ref.get("dataset_id") != name:
        raise ValueError("dataset_ref must match dataset_name and include revision/split_id.")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,127}", name or "") or any(not re.fullmatch(r"[0-9a-f]{64}", str(ref[key])) for key in ("revision", "split_id")):
        raise ValueError("Invalid dataset_ref identity.")


def resolve_experiment_config(root: Path, payload: dict) -> dict:
    result = deepcopy(payload)
    experiment = result.get("experiment", result)
    dataset = experiment.get("dataset_name")
    runtime = resolve_dataset_runtime(root, experiment.get("dataset_ref"), dataset)
    if runtime["dataset_ref"] is not None:
        experiment["dataset_ref"] = runtime["dataset_ref"]
    validate_dataset_selection(dataset, runtime["dataset_ref"])
    stage = experiment.get("stage") or ("multi_teacher_knowledge_distillation" if "teachers" in result else "knowledge_distillation" if "experiment" in result else "baseline")
    train = result.get("training", result)
    if stage == "multi_teacher_knowledge_distillation":
        choices = [(result["student"]["model_name"], "student")] + [(item["model_name"], "teacher") for item in result["teachers"]]
    elif stage == "knowledge_distillation":
        choices = [(experiment["student_model_name"], "student"), (experiment["teacher_model_name"], "teacher")]
    else:
        choices = [(experiment["model_name"], "student" if stage == "baseline" else "teacher")]
    for name, role in choices:
        model = get_model_spec(name)
        if role not in model.roles:
            raise ValueError(f"{name} is not a registered {role} model.")
        size = int(train.get("image_size", 224))
        if size <= 0 or (model.fixed_input_size is not None and size != model.fixed_input_size):
            raise ValueError(f"{name} requires image_size={model.fixed_input_size or 'positive integer'}.")
    return result
