"""Prepare and verify completed-run folders for manual transfer between computers."""
import json
import hashlib
from pathlib import Path
import re
import subprocess
import tempfile
import zipfile

import pandas as pd

from src.tracking.external_bundle import create_external_bundle, sha256_file

SUMMARIES = {"baseline": "results/baseline/baseline_results.csv",
             "teacher_cnn": "results/teachers/cnn/teacher_results.csv",
             "teacher_vision_transformer": "results/teachers/vision_transformers/teacher_results.csv",
             "knowledge_distillation": "results/knowledge_distillation/kd_results.csv",
             "multi_teacher_knowledge_distillation": "results/multi_teacher_knowledge_distillation/multi_kd_results.csv"}


def can_export(job):
    return job.get("status") == "finished" and not job.get("dry_run") and bool(job.get("mlflow_run_id"))


def verify_transfer_folder(root: Path, folder: Path, expected_run: str) -> dict:
    if folder.is_symlink():
        raise ValueError("Transfer klasörü sembolik bağlantı olamaz.")
    root = root.resolve()
    folder = folder.resolve()
    if not folder.is_relative_to(root) or not folder.is_dir() or folder.is_symlink():
        raise ValueError("Transfer klasörü proje içinde olmalı.")
    archives = list(folder.glob("*.zip"))
    if len(archives) != 1 or archives[0].is_symlink():
        raise ValueError("Transfer klasöründe tam bir sonuç ZIP'i bulunmalı.")
    with zipfile.ZipFile(archives[0]) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        canonical = {key: value for key, value in manifest.items() if key != "fingerprint"}
        fingerprint = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if fingerprint != manifest.get("fingerprint"):
            raise ValueError("Transfer manifest bütünlüğü doğrulanamadı.")
        if manifest.get("source_mlflow_run_id") != expected_run:
            raise ValueError("Transfer paketi seçili deneye ait değil.")
        if archive.testzip() is not None:
            raise ValueError("Transfer ZIP bütünlük kontrolü başarısız.")
        for item in manifest.get("files", []):
            data = archive.read(item["member"])
            if len(data) != item["size"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise ValueError("Transfer artifact hash uyuşmazlığı.")
    checkpoint = manifest.get("checkpoint")
    if not checkpoint or Path(checkpoint["filename"]).name != checkpoint["filename"]:
        raise ValueError("Tamamlanmış model checkpoint'i pakette bulunamadı.")
    path = folder / checkpoint["filename"]
    if path.is_symlink() or not path.is_file() or path.stat().st_size != checkpoint["size"] or sha256_file(path) != checkpoint["sha256"]:
        raise ValueError("Transfer checkpoint eksik veya hash uyuşmuyor.")
    if {item.name for item in folder.iterdir()} != {archives[0].name, path.name}:
        raise ValueError("Transfer klasöründe beklenmeyen dosyalar var.")
    return {"folder": str(folder), "archive": str(archives[0]), "checkpoint": str(path),
            "inbox_folder": "runs/import_inbox/" + folder.name}


def prepare_run_transfer(root: Path, job: dict) -> dict:
    root = Path(root).resolve()
    if not can_export(job):
        raise ValueError("Yalnızca başarıyla tamamlanan gerçek eğitimler taşınabilir.")
    run_id = str(job["mlflow_run_id"])
    existing = job.get("lab_export")
    if existing:
        folder = Path(str(existing["archive"])).parent
        return verify_transfer_folder(root, folder, run_id)
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", str(job.get("job_id", ""))):
        raise ValueError("Geçersiz deney kimliği.")
    destination = root / "runs/transfer" / str(job["job_id"])
    if destination.exists():
        return verify_transfer_folder(root, destination, run_id)
    summary = root / SUMMARIES[job["stage"]]
    rows = pd.read_csv(summary)
    rows = rows[rows["mlflow_run_id"].astype(str) == run_id].to_dict("records")
    if len(rows) != 1:
        raise ValueError("Seçili deney için tam bir sonuç satırı bulunamadı.")
    checkpoint = root / str(rows[0].get("checkpoint_path", ""))
    if not checkpoint.is_file() or not checkpoint.resolve().is_relative_to(root):
        raise ValueError("Seçili deneyin checkpoint'i bulunamadı.")
    commit = job.get("source_commit") or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Publish only complete, validated pairs; a failed export leaves no final folder.
    with tempfile.TemporaryDirectory(prefix=".transfer-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        create_external_bundle(root, rows[0], source_commit=commit, output_dir=staging)
        verify_transfer_folder(root, staging, run_id)
        staging.rename(destination)
        staging.mkdir()  # TemporaryDirectory cleans its original name, not the published folder.
    return verify_transfer_folder(root, destination, run_id)
