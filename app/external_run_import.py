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


ACCEPTED_STAGES = {"baseline", "teacher_cnn", "teacher_vision_transformer"}
DATASETS = {"appleleaf9", "plantvillage", "plantpathology2021"}
MODELS = {
    "baseline": {"mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_b0", "resnet18"},
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
    if not value or "\\" in value or posix.is_absolute() or ".." in posix.parts:
        raise ExternalRunImportError(f"Unsafe bundle path: {value}")
    return Path(*posix.parts)


def _validate_zip_info(info: zipfile.ZipInfo) -> None:
    if info.is_dir() or (info.external_attr >> 16) & 0o170000 == 0o120000:
        raise ExternalRunImportError(f"Directories and symlinks are not allowed: {info.filename}")
    _safe_relative(info.filename)


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
    if manifest.get("schema_version") != 1:
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
    if row.get("dataset_name") != manifest.get("dataset") or row.get("model_name") != manifest.get("model"):
        raise ExternalRunImportError("Manifest and result identity do not match.")
    if manifest.get("dataset") not in DATASETS or manifest.get("model") not in MODELS[stage]:
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
    mlflow.set_tracking_uri(str(row.get("tracking_uri") or "sqlite:///mlflow.db"))
    experiments = {
        "baseline": "MasterThesis-Baselines",
        "teacher_cnn": "MasterThesis-CNN-Teachers",
        "teacher_vision_transformer": "MasterThesis-Vision-Transformer-Teachers",
    }
    mlflow.set_experiment(experiments[manifest["stage"]])
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
                target = root / _safe_relative(target_value)
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
            target = root / _safe_relative(target_value)
            if target.exists() and _sha256(target) != checkpoint.get("sha256"):
                raise ExternalRunImportError(f"Checkpoint target collision: {target_value}")
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                created.append(target)
            imported_paths.append(target)

        local_run_id = register_run(manifest, imported_paths)
        row = dict(manifest["result_row"])
        row["source_mlflow_run_id"] = manifest.get("source_mlflow_run_id")
        row["mlflow_run_id"] = local_run_id
        row["execution_origin"] = "external"
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
