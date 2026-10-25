"""Classify experiments without changing legacy pipeline/storage identifiers."""

from pathlib import PurePosixPath

try:
    from src.models.model_catalog import get_model_family
except ModuleNotFoundError:
    from src.models.model_catalog import get_model_family

STAGE_TAXONOMY = {
    "baseline": ("baseline", "student"),
    "teacher_cnn": ("baseline", "teacher"),
    "teacher_vision_transformer": ("baseline", "teacher"),
    "knowledge_distillation": ("single_teacher_kd", "student"),
    "multi_teacher_knowledge_distillation": ("multi_teacher_kd", "student"),
}


def enrich_experiment_metadata(record: dict) -> dict:
    """Return an enriched copy; unknown pipelines stay unknown, conflicts fail closed."""
    result = dict(record)
    command = record.get("command") or []
    command = command if isinstance(command, (list, tuple)) else []
    script = next((PurePosixPath(str(part).replace("\\", "/")).name
                   for part in command if str(part).lower().endswith(".py")), None)
    model = record.get("student_model") or record.get("model") or record.get("model_name") or record.get("student_model_name")
    if not model and "--model" in command and command.index("--model") + 1 < len(command):
        model = command[command.index("--model") + 1]
    family = get_model_family(model or "")
    inferred = None
    if script == "train_baseline.py":
        inferred = "baseline"
    elif script == "train_multi_kd.py":
        inferred = "multi_teacher_knowledge_distillation"
    elif script == "train_kd.py":
        inferred = "knowledge_distillation"
    elif script == "train_teacher.py" and family:
        inferred = "teacher_vision_transformer" if family == "vision_transformer" else "teacher_cnn"
    stage = record.get("stage") or inferred
    if record.get("stage") and inferred and record["stage"] != inferred:
        raise ValueError("Job stage conflicts with its training command.")
    if stage not in STAGE_TAXONOMY:
        return result
    method, role = STAGE_TAXONOMY[stage]
    fields = {"stage": stage, "training_method": method, "model_role": role}
    if family:
        fields["architecture_family"] = family
    for key, value in fields.items():
        if result.get(key) not in (None, "", value):
            raise ValueError(f"Experiment {key} conflicts with its pipeline/model.")
    result.update(fields)
    return result
