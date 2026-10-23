"""Epoch-boundary recovery. Snapshots use safe tensor-only torch deserialization."""
from __future__ import annotations

import hashlib
import csv
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np
import torch

try:
    from ..tracking.run_metadata import dataset_fingerprint, protocol_fingerprint, split_fingerprint, source_commit
except ImportError:
    from tracking.run_metadata import dataset_fingerprint, protocol_fingerprint, split_fingerprint, source_commit


def add_recovery_arguments(parser):
    parser.add_argument("--resume-from", default=None, help="Verified epoch generation directory, not a best checkpoint.")
    parser.add_argument("--backup-dir", default=None, help="Mounted durable directory for epoch backups.")
    parser.add_argument("--no-recovery", action="store_true", help="Disable epoch recovery snapshots.")


def recovery_settings(args, config):
    raw_settings = config.get("recovery") or {}
    if not isinstance(raw_settings, dict):
        raise ValueError("recovery must be a JSON object.")
    settings = dict(raw_settings)
    train = config.get("training", config)
    return {"enabled": settings.get("enabled", True) and not getattr(args, "no_recovery", False) and not train.get("dry_run", False),
            "resume_from": getattr(args, "resume_from", None) or os.environ.get("MASTERTHESIS_RESUME_FROM") or settings.get("resume_from"),
            "backup_dir": getattr(args, "backup_dir", None) or os.environ.get("MASTERTHESIS_BACKUP_DIR") or settings.get("backup_dir")}


def recovery_metadata(config, split_file, root):
    split = json.loads(Path(split_file).read_text(encoding="utf-8"))
    exp, train = config.get("experiment", config), config.get("training", config)
    teachers = config.get("teachers", [])
    if exp.get("teacher_checkpoint_path"):
        teachers = [{"model_name": exp.get("teacher_model_name"), "run_id": exp.get("teacher_run_id"),
                     "checkpoint_path": exp["teacher_checkpoint_path"]}]
    hashes = []
    for teacher in teachers:
        path = Path(root) / str(teacher.get("checkpoint_path") or "")
        if not path.is_file():
            raise ValueError(f"Missing resume teacher checkpoint: {path}")
        hashes.append({"model_name": str(teacher["model_name"]),
                       "run_id": str(teacher.get("run_id") or ""),
                       "checkpoint_path": str(teacher.get("checkpoint_path") or ""),
                       "sha256": sha256(path)})
    transport = {"device", "tracking_uri", "experiment_name", "recovery", "teacher_checkpoint_path", "checkpoint_path"}
    def clean(value):
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items() if key not in transport and not (key in {"dataset_ref", "teacher_checkpoint_sha256", "checkpoint_sha256"} and item is None) and not (key == "teacher_inputs" and not item)}
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value
    split_hash = split_fingerprint(split)
    train_transform = {"version": 1, "image_size": train.get("image_size", 224),
                       "augmentation": train.get("use_augmentation", train.get("data_augmentation", True))}
    eval_transform = {"version": 1, "resize_factor": 1.14, "center_crop": train.get("image_size", 224),
                      "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    shared_training = {
        "seed": exp.get("seed", config.get("seed")),
        "image_size": train.get("image_size"),
        "batch_size": train.get("batch_size"),
        "epochs": train.get("epochs"),
        "learning_rate": train.get("learning_rate"),
        "weight_decay": train.get("weight_decay"),
        "optimizer": train.get("optimizer"),
        "optimizer_params": train.get("optimizer_params", {}),
        "scheduler": train.get("scheduler"),
        "scheduler_params": train.get("scheduler_params", {}),
        "monitor_metric": train.get("monitor_metric", "val_macro_f1"),
        "early_stopping_patience": train.get("early_stopping_patience"),
        "augmentation": train_transform["augmentation"],
        "student_pretrained": train.get("student_pretrained", train.get("pretrained")),
    }
    versions = {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "numpy")}
    hardware = {
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    return {"schema_version": 1, "stage": exp.get("stage", "baseline"), "config": config,
            "dataset_ref": split.get("dataset_ref"), "identity_version": split.get("identity_version", 1),
            "class_mapping_hash": split.get("class_mapping_hash"),
            "provenance_level": "content_v2" if split.get("identity_version") == 2 else "legacy",
            "source_commit": source_commit(Path(root)), "versions": versions, "classes": split["classes"],
            "split_fingerprint": split_hash, "teacher_hashes": hashes,
            "dataset_name": exp.get("dataset_name") or config.get("dataset_name") or split.get("dataset_name"),
            "dataset_fingerprint": split.get("dataset_fingerprint") or dataset_fingerprint(split),
            "train_transform": train_transform, "eval_transform": eval_transform,
            "training_protocol": shared_training,
            "hardware": hardware,
            "precision": {"mixed_precision": bool(train.get("mixed_precision", False))},
            "protocol": protocol_fingerprint({"config": clean(config), "split": split_hash,
                                              "train": train_transform, "eval": eval_transform, "recovery_version": 1})}


def create_recovery(args, config, paths, split_file, root, run_id):
    settings = recovery_settings(args, config)
    if settings["resume_from"] and not settings["enabled"]:
        raise ValueError("Resume requires recovery and cannot be combined with dry run.")
    metadata = recovery_metadata(config, split_file, root)
    parent = load_generation(Path(settings["resume_from"])) if settings["resume_from"] else None
    if parent:
        validate_resume(parent["metadata"], metadata)
    attempt = str(run_id)
    metadata.update({"attempt_id": attempt, "mlflow_run_id": attempt,
                     "experiment_id": parent["metadata"]["experiment_id"] if parent else attempt,
                     "parent_attempt_id": parent["metadata"].get("attempt_id") if parent else None,
                     "status": "running", "run_name": paths["run_dir"].name})
    metadata["recovery"] = {"enabled": bool(settings["enabled"]), "backup_enabled": bool(settings["backup_dir"])}
    atomic_json(paths["run_dir"] / "run_metadata.json", metadata)
    backup = Path(settings["backup_dir"]) / metadata["experiment_id"] / attempt if settings["enabled"] and settings["backup_dir"] else None
    if backup:
        backup.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=backup) as probe:
            probe.write(b"ready")
    session = RecoverySession(paths["run_dir"] / "recovery", metadata,
                              resume_from=settings["resume_from"], backup_dir=backup,
                              stop_file=os.environ.get("MASTERTHESIS_STOP_FILE"),
                              enabled=bool(settings["enabled"]))
    if not settings["enabled"]:
        return session
    atomic_json(session.status_path, {"metadata": metadata, "last_local_epoch": None, "last_backup_epoch": None})
    print("RECOVERY_CONTEXT " + json.dumps({"status_path": str(session.status_path.resolve()),
                                           "experiment_id": metadata["experiment_id"], "attempt_id": attempt}), flush=True)
    return session


def mark_recovery(session, status):
    if session is not None:
        session.metadata["status"] = status
        session.metadata["training_time_seconds"] = session.elapsed_seconds()
        session.metadata["backup_time_seconds"] = session.backup_seconds
        if not session.enabled:
            atomic_json(session.root.parent / "run_metadata.json", session.metadata)
            return
        session.status["phase"] = "completed" if status == "FINISHED" else status.lower()
        atomic_json(session.root.parent / "run_metadata.json", session.metadata)
        atomic_json(session.status_path, {**session.status, "metadata": session.metadata})


class TrainingStopped(Exception):
    """A requested graceful stop after a complete, saved epoch."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as output:
            json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _checked_manifest(path: Path) -> dict:
    if path.is_symlink() or not path.is_dir():
        raise ValueError("Recovery generation must be a real directory.")
    try:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Recovery manifest is missing or invalid.") from error
    if not isinstance(manifest, dict):
        raise ValueError("Recovery manifest must be a JSON object.")
    if manifest.get("schema_version") != 1 or manifest.get("kind") != "training_recovery":
        raise ValueError("Unsupported recovery format; best-only checkpoints cannot resume.")
    files = manifest.get("files", {})
    if not isinstance(files, dict):
        raise ValueError("Recovery file manifest must be a JSON object.")
    if "state.pt" not in files or set(files) - {"state.pt", "best.pt", "backup_status.json"}:
        raise ValueError("Invalid recovery members.")
    for name, expected in files.items():
        member = path / name
        if member.is_symlink() or not member.is_file() or sha256(member) != expected:
            raise ValueError(f"Recovery checksum mismatch: {name}")
    if "backup_status.json" in files:
        try:
            timing = json.loads((path / "backup_status.json").read_text(encoding="utf-8"))
            seconds = float(timing["backup_seconds_total"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise ValueError("Invalid recovery backup timing.") from error
        if timing.get("epoch") != manifest.get("epoch") or not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Invalid recovery backup timing.")
    return manifest


def _attach_backup_status(path: Path, epoch: int, seconds: float) -> None:
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Invalid recovery backup timing.")
    manifest = _checked_manifest(path)
    if manifest.get("epoch") != epoch:
        raise ValueError("Recovery backup timing epoch mismatch.")
    timing_path = path / "backup_status.json"
    atomic_json(timing_path, {"epoch": epoch, "backup_seconds_total": seconds})
    manifest["files"][timing_path.name] = sha256(timing_path)
    atomic_json(path / "manifest.json", manifest)
    _checked_manifest(path)


def _prune(root: Path) -> None:
    valid = []
    for candidate in root.iterdir():
        if re.fullmatch(r"epoch-\d{6}", candidate.name) and candidate.is_dir() and not candidate.is_symlink():
            try:
                _checked_manifest(candidate)
                valid.append(candidate)
            except (OSError, ValueError):
                continue
    for candidate in sorted(valid)[:-2]:
        if candidate.resolve().parent != root.resolve():
            raise ValueError("Recovery cleanup escaped its root.")
        shutil.rmtree(candidate)


def write_generation(root: Path, payload: dict, best_path: Path | None = None) -> Path:
    epoch = payload.get("epoch")
    if payload.get("schema_version") != 1 or not isinstance(epoch, int) or not 0 <= epoch <= 999999:
        raise ValueError("Invalid recovery schema or epoch.")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"epoch-{epoch:06d}"
    if target.exists():
        raise ValueError(f"Recovery generation already exists: {target}")
    temporary = Path(tempfile.mkdtemp(prefix=".pending-", dir=root))
    try:
        with (temporary / "state.pt").open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        files = {"state.pt": sha256(temporary / "state.pt")}
        if best_path is not None and best_path.is_file():
            shutil.copy2(best_path, temporary / "best.pt")
            files["best.pt"] = sha256(temporary / "best.pt")
        atomic_json(temporary / "manifest.json", {"schema_version": 1, "kind": "training_recovery",
                    "epoch": epoch, "files": files, "metadata": payload.get("metadata", {})})
        _checked_manifest(temporary)
        os.replace(temporary, target)
        _prune(root)
        return target
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def load_generation(path: Path) -> dict:
    path = Path(path)
    manifest = _checked_manifest(path)
    payload = torch.load(path / "state.pt", map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Recovery state must be a mapping.")
    if payload.get("schema_version") != 1 or payload.get("epoch") != manifest["epoch"]:
        raise ValueError("Recovery manifest/state mismatch.")
    if "backup_status.json" in manifest["files"]:
        timing = json.loads((path / "backup_status.json").read_text(encoding="utf-8"))
        payload["backup_seconds"] = max(float(payload.get("backup_seconds", 0.0)),
                                        float(timing["backup_seconds_total"]))
    return payload


def mirror_generation(source: Path, destination: Path) -> Path:
    source, destination = Path(source), Path(destination)
    manifest = _checked_manifest(source)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / source.name
    if target.exists():
        if _checked_manifest(target) != manifest:
            raise ValueError("Backup generation conflicts with existing data.")
        return target
    temporary = Path(tempfile.mkdtemp(prefix=".pending-", dir=destination))
    try:
        for name in manifest["files"]:
            shutil.copy2(source / name, temporary / name)
        atomic_json(temporary / "manifest.json", manifest)
        _checked_manifest(temporary)
        os.replace(temporary, target)
        _prune(destination)
        return target
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def validate_resume(saved: dict, requested: dict) -> None:
    for key in ("protocol", "source_commit", "versions", "teacher_hashes"):
        if saved.get(key) != requested.get(key):
            raise ValueError(f"Resume mismatch: {key}")


def _rng_state() -> dict:
    numpy = np.random.get_state()
    return {"python": random.getstate(), "numpy": [numpy[0], numpy[1].tolist(), *numpy[2:]],
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state((numpy[0], np.array(numpy[1], dtype=np.uint32), *numpy[2:]))
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        if len(state["cuda"]) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(state["cuda"])


class RecoverySession:
    """Own one attempt's recovery files; trainers own their optimization loop."""
    def __init__(self, root: Path, metadata: dict, *, resume_from=None, backup_dir=None, stop_file=None, enabled=True):
        self.root = Path(root)
        self.metadata = dict(metadata)
        self.resume_from = Path(resume_from) if resume_from else None
        self.backup_dir = Path(backup_dir) if backup_dir else None
        self.stop_file = Path(stop_file) if stop_file else None
        self.enabled = enabled
        self.status_path = self.root.parent / "recovery_status.json"
        self.started = time.monotonic()
        self.previous_seconds = 0.0
        self.backup_seconds = 0.0
        self.previous_backup_seconds = 0.0
        self.status = {}
        self.last_generation = None

    def restore(self, model, optimizer, scheduler, scaler, strategy, loader, best_path) -> dict:
        if not self.enabled or not self.resume_from:
            return {}
        saved = load_generation(self.resume_from)
        validate_resume(saved.get("metadata", {}), self.metadata)
        if "model" not in saved or "optimizer" not in saved:
            raise ValueError("Recovery lacks training state.")
        model.load_state_dict(saved["model"])
        if strategy is not None:
            strategy.load_state_dict(saved["strategy"])
        optimizer.load_state_dict(saved["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(saved["scheduler"])
        if scaler is not None and saved.get("scaler") is not None:
            scaler.load_state_dict(saved["scaler"])
        if saved.get("loader_generator") is not None and loader.generator is not None:
            loader.generator.set_state(saved["loader_generator"])
        source_best = self.resume_from / "best.pt"
        if saved.get("progress", {}).get("best_epoch") is not None and not source_best.is_file():
            raise ValueError("Recovery lacks the best checkpoint required for final evaluation.")
        if source_best.is_file():
            Path(best_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_best, best_path)
        inherited_backup_seconds = float(saved.get("backup_seconds", 0.0))
        # Own the inherited snapshot, so archiving the parent cannot break this attempt.
        inherited = {**saved, "metadata": self.metadata}
        inherited["backup_seconds"] = inherited_backup_seconds
        self.last_generation = write_generation(self.root, inherited, Path(best_path))
        if inherited_backup_seconds:
            _attach_backup_status(self.last_generation, saved["epoch"], inherited_backup_seconds)
        self.previous_seconds = saved.get("training_seconds", 0.0)
        self.backup_seconds = inherited_backup_seconds
        self.previous_backup_seconds = self.backup_seconds
        _restore_rng(saved["rng"])
        self.status.update({"last_local_epoch": saved["epoch"], "generation": str(self.last_generation.resolve()),
                            "updated_at": time.time(), "metadata": self.metadata})
        atomic_json(self.status_path, self.status)
        return saved

    def save(self, epoch, progress, model, optimizer, scheduler, scaler, strategy, loader, best_path):
        if not self.enabled:
            return
        payload = {"schema_version": 1, "epoch": epoch, "phase": "training", "metadata": self.metadata,
                   "progress": progress, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                   "scheduler": scheduler.state_dict() if scheduler is not None else None,
                   "scaler": scaler.state_dict() if scaler is not None else None,
                   "strategy": strategy.state_dict() if strategy is not None else None,
                   "rng": _rng_state(), "loader_generator": loader.generator.get_state() if loader.generator else None,
                   "training_seconds": self.elapsed_seconds(), "backup_seconds": self.backup_seconds}
        self.last_generation = write_generation(self.root, payload, Path(best_path))
        history_path = self.root.parent / "history.csv"
        history = progress.get("history", [])
        if history:
            temporary = history_path.with_suffix(".csv.tmp")
            with temporary.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in history for key in row)))
                writer.writeheader()
                writer.writerows(history)
            os.replace(temporary, history_path)
        self.status.update({"last_local_epoch": epoch, "generation": str(self.last_generation.resolve()),
                            "updated_at": time.time(), "metadata": self.metadata})
        if self.backup_dir:
            start = time.monotonic()
            try:
                backup = mirror_generation(self.last_generation, self.backup_dir)
                self.status.update({"last_backup_epoch": epoch, "backup_generation": str(backup), "backup_error": None})
            except (OSError, ValueError) as error:
                self.status["backup_error"] = str(error)
                print(f"RECOVERY_BACKUP_WARNING {error}", flush=True)
            backup_seconds = time.monotonic() - start
            self.backup_seconds += backup_seconds
            self.status.update({"backup_seconds": backup_seconds, "backup_seconds_total": self.backup_seconds})
            try:
                _attach_backup_status(self.last_generation, epoch, self.backup_seconds)
                if self.status.get("backup_generation"):
                    _attach_backup_status(Path(self.status["backup_generation"]), epoch, self.backup_seconds)
            except (OSError, ValueError) as error:
                self.status["backup_error"] = str(error)
                print(f"RECOVERY_BACKUP_WARNING {error}", flush=True)
        atomic_json(self.status_path, self.status)
        print("RECOVERY_STATUS " + json.dumps(self.status), flush=True)
        if self.stop_file and self.stop_file.exists():
            raise TrainingStopped("Stopped after a saved epoch.")

    def elapsed_seconds(self) -> float:
        current_backup = self.backup_seconds - self.previous_backup_seconds
        return self.previous_seconds + time.monotonic() - self.started - current_backup
