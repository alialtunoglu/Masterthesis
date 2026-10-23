"""Canonical individual-run paths shared by import and removal."""
from pathlib import PurePosixPath

RUN_PREFIXES = {"baseline": "results/baseline/runs", "teacher_cnn": "results/teachers/cnn/runs",
                "teacher_vision_transformer": "results/teachers/vision_transformers/runs",
                "knowledge_distillation": "results/knowledge_distillation/runs",
                "multi_teacher_knowledge_distillation": "results/multi_teacher_knowledge_distillation/runs"}


def validate_run_directory(relative, row, *, strict=False):
    path = PurePosixPath(str(relative).replace("\\", "/"))
    prefix = next((PurePosixPath(value) for value in RUN_PREFIXES.values() if path.is_relative_to(value)), None)
    if prefix is None:
        raise ValueError("Not an experiment run directory")
    parts = path.relative_to(prefix).parts
    run_name = row.get("run_name")
    if not run_name or not parts or parts[-1] != run_name:
        raise ValueError("Run directory must end in its exact run_name")
    if not strict and len(parts) == 1:
        return
    dataset = row.get("dataset_name") or row.get("dataset")
    model = row.get("model_name") or row.get("student_model_name") or row.get("model")
    middle = row.get("kd_type") if "distillation" in str(prefix) else model
    if parts != (dataset, middle, run_name):
        raise ValueError("Run directory must identify exactly dataset/model-or-KD-type/run_name")
