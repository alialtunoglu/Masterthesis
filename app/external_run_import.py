"""Validate and import external experiment bundles without loading model code."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Callable

import pandas as pd
from src.tracking.experiment_taxonomy import enrich_experiment_metadata
from src.models.model_catalog import STUDENT_MODELS


ACCEPTED_STAGES = {"baseline", "teacher_cnn", "teacher_vision_transformer", "knowledge_distillation", "multi_teacher_knowledge_distillation"}
# Imported runs must land in the same experiments local training writes to.
# Kept literal so this module stays free of src/ imports; the pairing with
# src.tracking.mlflow_tracker.DEFAULT_EXPERIMENT_NAME is asserted in the tests.
EXPERIMENT_NAMES = {
    "baseline": "MasterThesis-Baseline",
    "teacher_cnn": "MasterThesis-CNN-Teachers",
    "teacher_vision_transformer": "MasterThesis-Vision-Transformer-Teachers",
    "knowledge_distillation": "MasterThesis-Knowledge-Distillation",
    "multi_teacher_knowledge_distillation": "MasterThesis-Multi-Teacher-Knowledge-Distillation",
}
DATASETS = {"appleleaf9", "plantvillage", "plantpathology2021"}
MODELS = {
    "baseline": set(STUDENT_MODELS),
    "teacher_cnn": {
        "resnet50", "resnet101", "densenet121", "densenet201", "vgg19_bn",
        "efficientnet_b3", "efficientnet_b4", "convnext_tiny", "convnext_base", "regnet_y_8gf",
    },
    "teacher_vision_transformer": {"swin_v2_t", "maxvit_t", "vit_b_16", "dinov2_vitb14"},
}
SUMMARY_PATHS = {
    "baseline": "results/baseline/baseline_results.csv",
    "teacher_cnn": "results/teachers/cnn/teacher_results.csv",
    "teacher_vision_transformer": "results/teachers/vision_transformers/teacher_results.csv",
}
RESULT_PREFIXES = {
    "baseline": "results/baseline/",
    "teacher_cnn": "results/teachers/cnn/",
    "teacher_vision_transformer": "results/teachers/vision_transformers/",
}
CHECKPOINT_PREFIXES = {
    "baseline": "checkpoints/",
    "teacher_cnn": "checkpoints/teachers/cnn/",
    "teacher_vision_transformer": "checkpoints/teachers/vision_transformers/",
}
for _stage, _filename in (("knowledge_distillation", "kd_results.csv"), ("multi_teacher_knowledge_distillation", "multi_kd_results.csv")):
    MODELS[_stage] = MODELS["baseline"]
    SUMMARY_PATHS[_stage] = f"results/{_stage}/{_filename}"
    RESULT_PREFIXES[_stage] = f"results/{_stage}/"
    CHECKPOINT_PREFIXES[_stage] = f"checkpoints/{_stage}/"
MAX_FILES = 100
MAX_PAYLOAD_BYTES = 150 * 1024 * 1024
MAX_CHECKPOINT_BYTES = 3 * 1024 * 1024 * 1024


class ExternalRunImportError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative(value: str) -> Path:
    posix = PurePosixPath(value)
    if not value or ":" in value or "\\" in value or posix.is_absolute() or ".." in posix.parts:
        raise ExternalRunImportError(f"Unsafe bundle path: {value}")
    return Path(*posix.parts)


def _validate_zip_info(info: zipfile.ZipInfo) -> None:
    if info.is_dir() or (info.external_attr >> 16) & 0o170000 == 0o120000:
        raise ExternalRunImportError(f"Directories and symlinks are not allowed: {info.filename}")
    _safe_relative(info.filename)


def _destination(root: Path, value: str) -> Path:
    path = root / _safe_relative(value)
    if not path.resolve().is_relative_to(root.resolve()) or any(parent.is_symlink() for parent in (path, *path.parents) if parent != root and parent.is_relative_to(root)):
        raise ExternalRunImportError(f"Bundle target escapes project or follows a link: {value}")
    return path


def _validate_v2_artifacts(bundle_dir, manifest):
    row, ref = manifest["result_row"], manifest.get("dataset_ref")
    expected_classes = manifest["split_metadata"]["classes"]
    records = {item["target"]: item for item in manifest["files"]}
    paths = [row.get("config_path"), row.get("run_metadata_path")]
    if not row.get("config_path") or row["config_path"] not in records:
        raise ExternalRunImportError("A v2 bundle requires its runtime config artifact.")
    with zipfile.ZipFile(next(bundle_dir.glob("*.zip"))) as archive:
        for path in paths:
            if not path or path not in records:
                continue
            item = records[path]
            raw = archive.read(item["member"])
            if len(raw) != item["size"] or hashlib.sha256(raw).hexdigest() != item["sha256"]:
                raise ExternalRunImportError("V2 metadata artifact hash mismatch.")
            payload = json.loads(raw)
            config = payload.get("config", payload)
            exp = config.get("experiment", config)
            actual_ref = payload.get("dataset_ref") or exp.get("dataset_ref")
            classes = payload.get("classes") or payload.get("class_names") or config.get("class_names")
            model = exp.get("student_model_name") or exp.get("model_name") or config.get("student", {}).get("model_name")
            if actual_ref != ref or classes != expected_classes or model != manifest["model"] or exp.get("dataset_name") != manifest["dataset"]:
                raise ExternalRunImportError("V2 artifact dataset/model/class identity differs from manifest.")
            if exp.get("stage") and exp["stage"] != manifest["stage"]:
                raise ExternalRunImportError("V2 artifact stage mismatch.")
            if manifest["stage"] == "knowledge_distillation":
                if exp.get("teacher_model_name") != row.get("teacher_model_name") or exp.get("teacher_run_id") != row.get("teacher_run_id"):
                    raise ExternalRunImportError("V2 teacher source identity mismatch.")
            elif manifest["stage"] == "multi_teacher_knowledge_distillation":
                names = json.loads(row["teacher_models"]) if isinstance(row.get("teacher_models"), str) else row.get("teacher_models", [])
                ids = json.loads(row["teacher_run_ids"]) if isinstance(row.get("teacher_run_ids"), str) else row.get("teacher_run_ids", [])
                if sorted(zip(names, ids)) != sorted((teacher["model_name"], teacher["run_id"]) for teacher in config.get("teachers", [])):
                    raise ExternalRunImportError("V2 teacher list identity mismatch.")


def _read_manifest(archive: Path) -> tuple[dict, list[zipfile.ZipInfo]]:
    with zipfile.ZipFile(archive) as bundle:
        infos = bundle.infolist()
        if len(infos) > MAX_FILES + 1:
            raise ExternalRunImportError("Bundle contains too many files.")
        for info in infos:
            _validate_zip_info(info)
        if sum(info.file_size for info in infos) > MAX_PAYLOAD_BYTES:
            raise ExternalRunImportError("Bundle payload is too large.")
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise ExternalRunImportError("Bundle contains duplicate ZIP members.")
        if names.count("manifest.json") != 1:
            raise ExternalRunImportError("Bundle must contain exactly one manifest.json.")
        manifest = json.loads(bundle.read("manifest.json"))
    expected = {"manifest.json", *(item.get("member", "") for item in manifest.get("files", []))}
    if set(names) != expected:
        raise ExternalRunImportError("Bundle contains missing or unexpected files.")
    targets = [item.get("target", "") for item in manifest.get("files", [])]
    if len(targets) != len(set(targets)):
        raise ExternalRunImportError("Bundle contains duplicate artifact targets.")
    return manifest, infos


def _validate_manifest(root: Path, bundle_dir: Path, manifest: dict) -> None:
    if manifest.get("schema_version") not in {1, 2}:
        raise ExternalRunImportError("Unsupported bundle schema version.")
    bundle_id = str(manifest.get("bundle_id", ""))
    if not re.fullmatch(r"[0-9a-f]{32}", bundle_id) or bundle_dir.name != bundle_id:
        raise ExternalRunImportError("Bundle directory must match bundle_id.")
    stage = str(manifest.get("stage", ""))
    if stage not in ACCEPTED_STAGES:
        raise ExternalRunImportError(f"Unsupported stage: {stage}")
    row = manifest.get("result_row") or {}
    if row.get("stage") != stage:
        raise ExternalRunImportError("Manifest and result stage do not match.")
    if row.get("dataset_name") != manifest.get("dataset") or (row.get("model_name") or row.get("student_model_name")) != manifest.get("model"):
        raise ExternalRunImportError("Manifest and result identity do not match.")
    if (manifest.get("schema_version") == 1 and manifest.get("dataset") not in DATASETS) or manifest.get("model") not in MODELS[stage]:
        raise ExternalRunImportError("Unsupported dataset/model identity.")
    if not manifest.get("run_name") or not re.fullmatch(
        r"[0-9a-f]{40}", str(manifest.get("source_commit", ""))
    ):
        raise ExternalRunImportError("run_name and a full source commit are required.")
    supplied = str(manifest.get("fingerprint", ""))
    unsigned = dict(manifest)
    unsigned.pop("fingerprint", None)
    actual = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if supplied != actual:
        raise ExternalRunImportError("Manifest fingerprint mismatch.")
    records = {item["target"] for item in manifest.get("files", [])}
    artifact_columns = {"config_path", "history_path", "per_class_metrics_path", "confusion_matrix_csv_path", "confusion_matrix_png_path", "learning_curves_path", "run_metadata_path"}
    if any(row.get(key) and row[key] not in records for key in artifact_columns):
        raise ExternalRunImportError("Result artifact pointer is not in the bundle.")
    run_dir = row.get("run_dir")
    if not run_dir and records:
        parents = {_safe_relative(target).parent.as_posix() for target in records}
        if len(parents) == 1:
            run_dir = parents.pop()
    if run_dir:
        from src.tracking.artifact_paths import validate_run_directory
        try:
            validate_run_directory(run_dir, row, strict=manifest["schema_version"] == 2)
        except ValueError as error:
            raise ExternalRunImportError(str(error)) from error
        directory = _safe_relative(str(run_dir))
        if not records or not any(_safe_relative(target).parent == directory for target in records) or any(not _safe_relative(target).is_relative_to(directory) for target in records):
            raise ExternalRunImportError("Run directory is broader than the bundled artifacts.")
        if not str(run_dir).startswith(RESULT_PREFIXES[stage]) or "runs" not in directory.parts or directory.name == "runs":
            raise ExternalRunImportError("Run directory is not a single stage run.")
        existing = _destination(root, str(run_dir))
        if existing.is_dir() and any(path.is_file() and path.relative_to(root).as_posix() not in records for path in existing.rglob("*")):
            raise ExternalRunImportError("Existing run directory contains unrelated artifacts.")
    if manifest.get("checkpoint") and row.get("checkpoint_path") != manifest["checkpoint"]["target"]:
        raise ExternalRunImportError("Result checkpoint pointer differs from the bundle.")
    if manifest["schema_version"] == 2:
        try:
            ref = manifest.get("dataset_ref")
            if ref:
                from src.datasets.dataset_runtime import validate_dataset_package
                from src.training.experiment_config import validate_dataset_selection
                validate_dataset_selection(manifest["dataset"], ref)
                validate_dataset_package(manifest["dataset_package"], ref)
                row_ref = row.get("dataset_ref")
                if isinstance(row_ref, str):
                    row_ref = json.loads(row_ref)
                if row_ref != ref:
                    raise ValueError("Result dataset reference mismatch")
            elif manifest["dataset"] not in DATASETS:
                raise ValueError("Custom results require a pinned dataset reference")
            split = manifest["split"]
            metadata = manifest["split_metadata"]
            if not split or not metadata or metadata["classes"] != split["class_names"]:
                raise ValueError("Missing portable split/class metadata")
            if ref and metadata != manifest["dataset_package"]["split"]:
                raise ValueError("Portable split metadata differs")
            fingerprint = hashlib.sha256(json.dumps(metadata["classes"], ensure_ascii=False).encode()).hexdigest()
            if fingerprint != split["class_mapping_fingerprint"]:
                raise ValueError("Portable class fingerprint mismatch")
            local_split = _destination(root, split["target"])
            if local_split.is_file() and _sha256(local_split) != split["sha256"]:
                raise ValueError("Local split differs from portable split")
            _validate_v2_artifacts(bundle_dir, manifest)
        except (KeyError, TypeError, ValueError) as error:
            raise ExternalRunImportError(f"Invalid v2 provenance: {error}") from error
        return
    split = manifest.get("split")
    if not split:
        raise ExternalRunImportError("A deterministic split fingerprint is required.")
    split_path = root / _safe_relative(str(split.get("target", "")))
    if not split_path.is_file() or _sha256(split_path) != split.get("sha256"):
        raise ExternalRunImportError("Local deterministic split does not match the external run.")
    class_names = json.loads(split_path.read_text(encoding="utf-8")).get("classes", [])
    fingerprint = hashlib.sha256(
        json.dumps(class_names, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    if fingerprint != split.get("class_mapping_fingerprint"):
        raise ExternalRunImportError("Class mapping mismatch.")


def _default_register_run(manifest: dict, paths: list[Path]) -> str:
    import mlflow

    row = manifest["result_row"]
    from src.tracking.mlflow_tracker import resolve_tracking_uri
    mlflow.set_tracking_uri(resolve_tracking_uri(str(row.get("tracking_uri") or "sqlite:///mlflow.db")))
    mlflow.set_experiment(EXPERIMENT_NAMES[manifest["stage"]])
    tags = {
        "execution_origin": "external",
        "external_bundle_id": manifest["bundle_id"],
        "source_commit": manifest.get("source_commit", ""),
        "source_platform": manifest.get("source_platform", ""),
        "checkpoint_available": str(bool(manifest.get("checkpoint"))).lower(),
    }
    with mlflow.start_run(run_name=manifest.get("run_name"), tags=tags) as run:
        params = {
            key: value
            for key, value in row.items()
            if value is not None and isinstance(value, (str, int, float, bool))
        }
        mlflow.log_params({key: str(value)[:500] for key, value in params.items()})
        metrics = {
            key: float(value)
            for key, value in row.items()
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and any(token in key for token in ("accuracy", "f1", "precision", "recall", "loss"))
        }
        if metrics:
            mlflow.log_metrics(metrics)
        for path in paths:
            mlflow.log_artifact(str(path))
        return run.info.run_id


def _delete_mlflow_run(run_id: str) -> None:
    from mlflow.tracking import MlflowClient

    MlflowClient().delete_run(run_id)


def import_external_bundle(
    root: Path,
    bundle_dir: Path,
    *,
    register_run: Callable[[dict, list[Path]], str] = _default_register_run,
    delete_run: Callable[[str], None] = _delete_mlflow_run,
) -> dict:
    root = root.resolve()
    inbox = (root / "runs" / "import_inbox").resolve()
    bundle_dir = bundle_dir.resolve()
    if bundle_dir.is_symlink() or not bundle_dir.is_relative_to(inbox):
        raise ExternalRunImportError("Bundle must be inside runs/import_inbox.")
    if any(path.is_symlink() or not path.is_file() for path in bundle_dir.iterdir()):
        raise ExternalRunImportError("Bundle directory may contain regular files only.")
    archives = list(bundle_dir.glob("*.zip"))
    if len(archives) != 1:
        raise ExternalRunImportError("Bundle directory must contain exactly one ZIP archive.")
    manifest, _ = _read_manifest(archives[0])
    _validate_manifest(root, bundle_dir, manifest)
    expected_files = {archives[0].name}
    if manifest.get("checkpoint"):
        expected_files.add(str(manifest["checkpoint"].get("filename", "")))
    if {path.name for path in bundle_dir.iterdir()} != expected_files:
        raise ExternalRunImportError("Bundle directory contains unexpected files.")

    receipt_path = root / "runs" / "imported" / f"{manifest['bundle_id']}.json"
    if receipt_path.exists():
        return json.loads(receipt_path.read_text(encoding="utf-8"))

    stage = manifest["stage"]
    created: list[Path] = []
    imported_paths: list[Path] = []
    summary_path = root / SUMMARY_PATHS[stage]
    summary_backup = summary_path.read_bytes() if summary_path.exists() else None
    local_run_id = ""
    try:
        with zipfile.ZipFile(archives[0]) as bundle:
            for item in manifest.get("files", []):
                target_value = str(item.get("target", ""))
                if not target_value.startswith(RESULT_PREFIXES[stage]):
                    raise ExternalRunImportError(f"Artifact target is outside stage results: {target_value}")
                target = _destination(root, target_value)
                if target.exists():
                    if not target.is_file() or _sha256(target) != item.get("sha256"):
                        raise ExternalRunImportError(f"Artifact target collision: {target_value}")
                    imported_paths.append(target)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(item["member"]) as source, tempfile.NamedTemporaryFile(
                    dir=target.parent, delete=False
                ) as temporary:
                    shutil.copyfileobj(source, temporary)
                    temporary_path = Path(temporary.name)
                if temporary_path.stat().st_size != item.get("size") or _sha256(temporary_path) != item.get("sha256"):
                    temporary_path.unlink(missing_ok=True)
                    raise ExternalRunImportError(f"Artifact hash mismatch: {target_value}")
                os.replace(temporary_path, target)
                created.append(target)
                imported_paths.append(target)

        checkpoint = manifest.get("checkpoint")
        if checkpoint:
            source = bundle_dir / str(checkpoint.get("filename", ""))
            target_value = str(checkpoint.get("target", ""))
            if not target_value.startswith(CHECKPOINT_PREFIXES[stage]):
                raise ExternalRunImportError("Checkpoint target is outside the expected stage directory.")
            if not source.is_file() or source.stat().st_size > MAX_CHECKPOINT_BYTES:
                raise ExternalRunImportError("Checkpoint is missing or too large.")
            if _sha256(source) != checkpoint.get("sha256"):
                raise ExternalRunImportError("Checkpoint hash mismatch.")
            target = _destination(root, target_value)
            if target.exists() and _sha256(target) != checkpoint.get("sha256"):
                raise ExternalRunImportError(f"Checkpoint target collision: {target_value}")
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                created.append(target)
            imported_paths.append(target)

        local_run_id = register_run(manifest, imported_paths)
        row = dict(manifest["result_row"])
        if not checkpoint:
            row["checkpoint_path"] = None
        row["source_mlflow_run_id"] = manifest.get("source_mlflow_run_id")
        row["mlflow_run_id"] = local_run_id
        row["execution_origin"] = "external"
        row = enrich_experiment_metadata(row)
        row["external_bundle_id"] = manifest["bundle_id"]
        row["import_receipt_path"] = receipt_path.relative_to(root).as_posix()
        row["external_bundle_dir"] = bundle_dir.relative_to(root).as_posix()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        existing = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
        pd.concat([existing, pd.DataFrame([row])], ignore_index=True).to_csv(summary_path, index=False)

        receipt = {
            "bundle_id": manifest["bundle_id"],
            "mlflow_run_id": local_run_id,
            "stage": stage,
            "dataset": manifest["dataset"],
            "model": manifest["model"],
            "checkpoint_available": bool(checkpoint),
            "source_mlflow_run_id": manifest.get("source_mlflow_run_id"),
            "source_commit": manifest.get("source_commit"),
            "run_metadata_path": row.get("run_metadata_path"),
        }
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        return receipt
    except Exception as exc:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        if summary_backup is None:
            summary_path.unlink(missing_ok=True)
        else:
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_bytes(summary_backup)
        if local_run_id:
            delete_run(local_run_id)
        if isinstance(exc, ExternalRunImportError):
            raise
        raise ExternalRunImportError(str(exc)) from exc


def scan_import_inbox(root: Path) -> list[Path]:
    inbox = root / "runs" / "import_inbox"
    if not inbox.exists():
        return []
    return sorted(path for path in inbox.iterdir() if path.is_dir() and not path.is_symlink())
