"""Create small, verifiable transfer bundles for externally executed runs."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import shutil
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ACCEPTED_STAGES = {"baseline", "teacher_cnn", "teacher_vision_transformer"}
ARTIFACT_COLUMNS = {
    "config_path",
    "history_path",
    "per_class_metrics_path",
    "confusion_matrix_csv_path",
    "confusion_matrix_png_path",
    "learning_curves_path",
}


@dataclass(frozen=True)
class BundleOutputs:
    bundle_id: str
    archive: Path
    checkpoint: Path | None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _safe_relative(value: object) -> Path:
    path = Path(str(value))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Artifact path must be project-relative: {value}")
    return path


def create_external_bundle(
    project_root: Path,
    result_row: dict[str, Any],
    *,
    source_commit: str,
    output_dir: Path,
) -> BundleOutputs:
    stage = str(result_row.get("stage", ""))
    if stage not in ACCEPTED_STAGES:
        raise ValueError(f"Unsupported external stage: {stage}")
    dataset = str(result_row.get("dataset_name", ""))
    model = str(result_row.get("model_name", ""))
    if not dataset or not model:
        raise ValueError("dataset_name and model_name are required.")
    if not result_row.get("run_name") or not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("run_name and a full source commit are required.")

    bundle_id = uuid.uuid4().hex
    output_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for column in sorted(ARTIFACT_COLUMNS):
        value = result_row.get(column)
        if not value:
            continue
        target = _safe_relative(value)
        source = project_root / target
        if source.is_file():
            files.append(
                {
                    "member": f"payload/{len(files):04d}",
                    "target": target.as_posix(),
                    "size": source.stat().st_size,
                    "sha256": sha256_file(source),
                    "source": source,
                }
            )

    checkpoint_output = None
    checkpoint_meta = None
    checkpoint_value = result_row.get("checkpoint_path")
    if checkpoint_value:
        checkpoint_target = _safe_relative(checkpoint_value)
        checkpoint_source = project_root / checkpoint_target
        if checkpoint_source.is_file():
            checkpoint_output = output_dir / f"{bundle_id}.pt"
            shutil.copy2(checkpoint_source, checkpoint_output)
            checkpoint_meta = {
                "filename": checkpoint_output.name,
                "target": checkpoint_target.as_posix(),
                "size": checkpoint_output.stat().st_size,
                "sha256": sha256_file(checkpoint_output),
            }

    split_meta = None
    split_value = result_row.get("split_file")
    if split_value:
        split_target = _safe_relative(split_value)
        split_source = project_root / split_target
        if split_source.is_file():
            split_data = json.loads(split_source.read_text(encoding="utf-8"))
            class_names = split_data.get("classes", [])
            split_meta = {
                "target": split_target.as_posix(),
                "sha256": sha256_file(split_source),
                "class_names": class_names,
                "class_mapping_fingerprint": hashlib.sha256(
                    json.dumps(class_names, ensure_ascii=False).encode("utf-8")
                ).hexdigest(),
            }

    manifest = {
        "schema_version": 1,
        "bundle_id": bundle_id,
        "stage": stage,
        "dataset": dataset,
        "model": model,
        "run_name": str(result_row.get("run_name", "")),
        "source_mlflow_run_id": str(result_row.get("mlflow_run_id", "")),
        "source_commit": source_commit,
        "source_platform": platform.platform(),
        "result_row": {key: _json_value(value) for key, value in result_row.items()},
        "files": [{key: value for key, value in item.items() if key != "source"} for item in files],
        "checkpoint": checkpoint_meta,
        "split": split_meta,
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["fingerprint"] = hashlib.sha256(canonical).hexdigest()

    archive = output_dir / f"{bundle_id}.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        for item in files:
            bundle.write(item["source"], item["member"])
    return BundleOutputs(bundle_id, archive, checkpoint_output)
