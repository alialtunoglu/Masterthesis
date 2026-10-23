"""Read-only teacher run selection shared by launchers and training entry points."""

import json
from pathlib import Path

import pandas as pd

try:
    from datasets.dataset_registry import DatasetRef
    from datasets.dataset_runtime import resolve_dataset_runtime
    from datasets.dataset_registration import sha256_file
    from models.model_catalog import get_model_spec
except ModuleNotFoundError:
    from src.datasets.dataset_registry import DatasetRef
    from src.datasets.dataset_runtime import resolve_dataset_runtime
    from src.datasets.dataset_registration import sha256_file
    from src.models.model_catalog import get_model_spec


def _path(root, value):
    if value is None or pd.isna(value):
        return None
    path = Path(str(value))
    path = path if path.is_absolute() else root / path
    return path if path.resolve().is_relative_to(root.resolve()) and path.is_file() else None


def _present(value):
    return value is not None and not (isinstance(value, float) and pd.isna(value)) and str(value).lower() not in {"", "none", "nan"}


def _ref(value):
    if not _present(value):
        return None
    try:
        return json.loads(value) if isinstance(value, str) else value
    except ValueError:
        return "invalid"


def _metadata(root, row):
    candidates = []
    run_dir = row.get("run_dir")
    if run_dir is not None and not pd.isna(run_dir):
        candidates.append(str(Path(str(run_dir)) / "run_metadata.json"))
    candidates += [row.get("run_metadata_path"), row.get("config_path")]
    for candidate in candidates:
        path = _path(root, candidate)
        if path:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    return payload
            except (OSError, ValueError):
                continue
    return {}


def find_compatible_teacher_runs(root: Path, dataset_ref: DatasetRef | dict | None, student_model: str,
                                 kd_type: str, *, dataset_name: str | None = None, image_size: int = 224) -> list[dict]:
    root = Path(root)
    if isinstance(dataset_ref, dict):
        dataset_ref = DatasetRef(**dataset_ref)
    if "student" not in get_model_spec(student_model).roles:
        raise ValueError("Unknown student model.")
    runtime = resolve_dataset_runtime(root, dataset_ref, dataset_name)
    try:
        from tracking.run_metadata import split_fingerprint
    except ModuleNotFoundError:
        from src.tracking.run_metadata import split_fingerprint
    current_split_hash = split_fingerprint(json.loads(Path(runtime["split_file"]).read_text(encoding="utf-8")))
    name = dataset_ref.dataset_id if dataset_ref else dataset_name
    result = []
    for family in ("cnn", "vision_transformers"):
        summary = root / f"results/teachers/{family}/teacher_results.csv"
        if not summary.is_file():
            continue
        for row in pd.read_csv(summary).to_dict("records"):
            if str(row.get("dataset_name")) != name:
                continue
            try:
                spec = get_model_spec(row.get("model_name"))
            except ValueError:
                continue
            if "teacher" not in spec.roles or (spec.fixed_input_size and image_size != spec.fixed_input_size):
                continue
            metadata = _metadata(root, row)
            config = metadata.get("config", metadata)
            if not isinstance(config, dict):
                continue
            training = config.get("training", config)
            if not isinstance(training, dict):
                continue
            recorded_split = metadata.get("split_fingerprint") or config.get("split_fingerprint") or row.get("split_fingerprint")
            if _present(recorded_split) and recorded_split != current_split_hash:
                continue
            status = str(metadata.get("status", metadata.get("phase", row.get("status", "completed")))).lower()
            if status not in {"completed", "finished", "success", "nan"}:
                continue
            protocols = (row, metadata, training)
            if any(str(item.get("dry_run")).lower() in {"true", "1"} for item in protocols) or any(_present(item.get(key)) for item in protocols for key in ("max_train_batches", "max_val_batches", "max_test_batches")):
                continue
            try:
                incompatible_size = any(_present(item.get("image_size")) and int(item["image_size"]) != image_size for item in protocols)
            except (ValueError, TypeError):
                continue
            if incompatible_size:
                continue
            checkpoint = _path(root, row.get("checkpoint_path"))
            if not checkpoint or not str(row.get("mlflow_run_id") or "").strip():
                continue
            exp = config.get("experiment", config)
            actual_ref = metadata.get("dataset_ref") or exp.get("dataset_ref")
            if _ref(row.get("dataset_ref")) not in (None, actual_ref):
                continue
            classes = metadata.get("classes") or metadata.get("class_names") or config.get("class_names")
            if dataset_ref:
                if actual_ref != dataset_ref.to_dict() or classes != runtime["class_names"]:
                    continue
                expected = row.get("checkpoint_sha256")
                if not isinstance(expected, str) or len(expected) != 64:
                    continue
            elif actual_ref:
                continue
            if classes and classes != runtime["class_names"]:
                continue
            if kd_type != "logit_based":
                try:
                    from .kd_features import default_feature_layer
                    default_feature_layer(student_model)
                    default_feature_layer(spec.name)
                except ValueError:
                    continue
            actual_hash = sha256_file(checkpoint)
            if dataset_ref and actual_hash != expected:
                continue
            result.append({**row, "checkpoint_sha256": actual_hash, "image_size": image_size,
                           "provenance_level": runtime["provenance_level"], "dataset_ref": runtime["dataset_ref"]})
    return sorted(result, key=lambda row: float(row.get("best_val_macro_f1") or 0), reverse=True)


def validate_teacher_selection(root: Path, config: dict) -> None:
    exp = config["experiment"]
    multi = "teachers" in config
    student = config["student"]["model_name"] if multi else exp["student_model_name"]
    if config.get("teacher_inputs"):
        runtime = resolve_dataset_runtime(root, exp.get("dataset_ref"), exp["dataset_name"])
        selected = config["teachers"] if multi else [{"model_name": exp["teacher_model_name"], "run_id": exp.get("teacher_run_id"),
                   "checkpoint_path": exp.get("teacher_checkpoint_path"), "checkpoint_sha256": exp.get("teacher_checkpoint_sha256")}]
        inputs = config["teacher_inputs"]
        if not isinstance(inputs, list) or len(inputs) != len(selected):
            raise ValueError("Portable teacher inputs do not match the selected teachers.")
        for teacher in selected:
            entries = [item for item in inputs if item.get("model_name") == teacher["model_name"] and item.get("run_id") == teacher["run_id"]]
            if len(entries) != 1:
                raise ValueError("Portable teacher source identity mismatch.")
            entry = entries[0]
            checkpoint = _path(Path(root), teacher.get("checkpoint_path"))
            if (not checkpoint or entry.get("dataset_ref") != runtime["dataset_ref"] or entry.get("dataset_name") != exp["dataset_name"]
                    or entry.get("class_names") != runtime["class_names"] or entry.get("status") != "completed"
                    or entry.get("image_size") != config["training"]["image_size"]
                    or entry.get("checkpoint_sha256") != teacher.get("checkpoint_sha256")
                    or not isinstance(entry.get("checkpoint_sha256"), str) or len(entry["checkpoint_sha256"]) != 64
                    or sha256_file(checkpoint) != entry["checkpoint_sha256"]):
                raise ValueError("Portable teacher checkpoint/protocol mismatch.")
        return
    candidates = find_compatible_teacher_runs(root, exp.get("dataset_ref"), student, exp.get("kd_type", "logit_based"),
                                              dataset_name=exp["dataset_name"], image_size=int(config["training"].get("image_size", 224)))
    teachers = config["teachers"] if multi else [{"model_name": exp["teacher_model_name"], "run_id": exp.get("teacher_run_id"),
                                                 "checkpoint_path": exp.get("teacher_checkpoint_path"),
                                                 "checkpoint_sha256": exp.get("teacher_checkpoint_sha256")}]
    for teacher in teachers:
        matching = [row for row in candidates if str(row["mlflow_run_id"]) == str(teacher.get("run_id")) and row["model_name"] == teacher["model_name"]
                    and (Path(root) / str(row["checkpoint_path"])).resolve() == (Path(root) / str(teacher.get("checkpoint_path"))).resolve()]
        if not matching or (teacher.get("checkpoint_sha256") and teacher["checkpoint_sha256"] != matching[0]["checkpoint_sha256"]):
            raise ValueError(f"No compatible completed teacher checkpoint: {teacher['model_name']}")
