"""Subprocess-based experiment runner for the local dashboard."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil

from ui_utils import get_project_root
from src.datasets.dataset_registry import DATASETS, SPLITS_DIR
from src.training.recovery import load_generation, recovery_metadata, validate_resume
from src.tracking.experiment_taxonomy import enrich_experiment_metadata


RUNS_DIR = get_project_root() / "runs"
JOBS_DIR = RUNS_DIR / "jobs"
LOGS_DIR = RUNS_DIR / "logs"
QUEUE_WORKER_STATE_PATH = RUNS_DIR / "queue_worker.json"
QUEUE_WORKER_HEARTBEAT_PATH = RUNS_DIR / "queue_worker_heartbeat.json"
QUEUE_WORKER_LOG_PATH = LOGS_DIR / "queue_worker.log"
SETTINGS_PATH = get_project_root() / "app" / "dashboard_settings.json"

EPOCH_PATTERN = re.compile(r"epoch\s*[:=/ ]\s*(\d+)(?:/(\d+))?", re.IGNORECASE)
METRIC_PATTERN = re.compile(
    r"(train_loss|total_loss|ce_loss|kd_loss|feature_loss|relation_loss|"
    r"teacher_[a-z0-9_]+_(?:kd|feature|relation)_loss|agreement_[a-z0-9_]+|"
    r"teacher_student_agreement|ensemble_student_agreement|val_loss|val_accuracy|val_macro_precision|val_macro_recall|val_macro_f1|"
    r"learning_rate|test_accuracy|test_macro_precision|test_macro_recall|test_macro_f1|test_weighted_f1)"
    r"\s*[:=]\s*([0-9]*\.?[0-9]+)",
    re.IGNORECASE,
)


def _ensure_run_dirs() -> None:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def _read_json_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def load_dashboard_settings() -> dict[str, Any]:
    """Load dashboard runtime settings."""
    defaults = {"max_parallel_jobs": 1}
    if not SETTINGS_PATH.exists():
        return defaults
    try:
        loaded = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return defaults
    settings = {**defaults, **loaded}
    settings["max_parallel_jobs"] = max(int(settings.get("max_parallel_jobs", 1)), 1)
    return settings


def build_baseline_command(
    dataset: str,
    model: str,
    config: str | None,
    epochs: int,
    batch_size: int,
    image_size: int,
    learning_rate: float,
    weight_decay: float,
    pretrained: bool,
    dry_run: bool,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    max_test_batches: int | None = None,
    tracking_uri: str | None = "sqlite:///mlflow.db",
    scheduler: str | None = None,
    early_stopping_patience: int | None = None,
) -> list[str]:
    """Build a Windows-safe baseline training command."""
    command = [sys.executable, "src/training/train_baseline.py"]
    if config:
        command.extend(["--config", config])
    else:
        command.extend(["--dataset", dataset, "--model", model])

    command.extend(
        [
            "--dataset",
            dataset,
            "--model",
            model,
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
            "--image-size",
            str(image_size),
            "--lr",
            str(learning_rate),
            "--weight-decay",
            str(weight_decay),
        ]
    )
    command.append("--pretrained" if pretrained else "--no-pretrained")
    if scheduler is not None:
        command.extend(["--scheduler", scheduler])
    if early_stopping_patience is not None:
        command.extend(["--early-stopping-patience", str(early_stopping_patience)])
    if dry_run:
        command.append("--dry-run")

    optional_limits = {
        "--max-train-batches": max_train_batches,
        "--max-val-batches": max_val_batches,
        "--max-test-batches": max_test_batches,
    }
    for flag, value in optional_limits.items():
        if value is not None and value > 0:
            command.extend([flag, str(value)])
    if tracking_uri:
        command.extend(["--tracking-uri", tracking_uri])
    return command


def build_teacher_command(
    dataset: str,
    model: str,
    config: str | None,
    epochs: int,
    batch_size: int,
    image_size: int,
    learning_rate: float,
    weight_decay: float,
    pretrained: bool,
    dry_run: bool,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    max_test_batches: int | None = None,
    tracking_uri: str | None = "sqlite:///mlflow.db",
    gradient_accumulation_steps: int = 1,
    label_smoothing: float = 0.0,
    mixed_precision: bool = False,
    gradient_clip_norm: float | None = None,
    warmup_epochs: int = 0,
    scheduler_eta_min: float = 0.0,
) -> list[str]:
    """Build a Windows-safe teacher training command."""
    command = [sys.executable, "src/training/train_teacher.py"]
    if config:
        command.extend(["--config", config])
    else:
        command.extend(["--dataset", dataset, "--model", model])

    command.extend(
        [
            "--dataset",
            dataset,
            "--model",
            model,
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
            "--image-size",
            str(image_size),
            "--lr",
            str(learning_rate),
            "--weight-decay",
            str(weight_decay),
            "--gradient-accumulation-steps",
            str(gradient_accumulation_steps),
            "--label-smoothing",
            str(label_smoothing),
            "--warmup-epochs",
            str(warmup_epochs),
            "--scheduler-eta-min",
            str(scheduler_eta_min),
        ]
    )
    command.append("--pretrained" if pretrained else "--no-pretrained")
    command.append("--mixed-precision" if mixed_precision else "--no-mixed-precision")
    if gradient_clip_norm is not None:
        command.extend(["--gradient-clip-norm", str(gradient_clip_norm)])
    if dry_run:
        command.append("--dry-run")
    for flag, value in {
        "--max-train-batches": max_train_batches,
        "--max-val-batches": max_val_batches,
        "--max-test-batches": max_test_batches,
    }.items():
        if value is not None and value > 0:
            command.extend([flag, str(value)])
    if tracking_uri:
        command.extend(["--tracking-uri", tracking_uri])
    return command


def build_kd_command(
    config: str,
    teacher_run_id: str,
    teacher_checkpoint: str,
    *,
    dry_run: bool = False,
) -> list[str]:
    """Build a Windows-safe KD command from a resolved config snapshot."""
    command = [
        sys.executable,
        "src/training/train_kd.py",
        "--config",
        config,
        "--teacher-run-id",
        teacher_run_id,
        "--teacher-checkpoint",
        teacher_checkpoint,
    ]
    if dry_run:
        command.append("--dry-run")
    return command


def build_multi_kd_command(config: str, *, dry_run: bool = False) -> list[str]:
    """Build a multi-teacher KD command from an immutable config snapshot."""
    command = [
        sys.executable,
        "src/training/train_multi_kd.py",
        "--config",
        config,
    ]
    if dry_run:
        command.append("--dry-run")
    return command


def save_job_config_snapshot(payload: dict[str, Any], job_id: str) -> str:
    """Persist the exact validated UI config consumed by a queued job."""
    config_dir = RUNS_DIR / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / f"{job_id}.json"
    if path.exists():
        raise FileExistsError(f"Job config snapshot already exists: {path}")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path.relative_to(get_project_root()).as_posix()


def _job_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"


def _write_job(job: dict[str, Any]) -> None:
    _ensure_run_dirs()
    _job_path(job["job_id"]).write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")


def _start_queued_job(job: dict[str, Any]) -> dict[str, Any]:
    """Start a queued job subprocess and persist running metadata."""
    _ensure_run_dirs()
    job_id = job["job_id"]
    log_path = LOGS_DIR / f"{job_id}.log"

    if job.get("resume_experiment_id"):
        _assert_no_active_resume(str(job["resume_experiment_id"]), exclude_job_id=job_id)
        saved = _validate_resume_generation(Path(job["resume_from"]))
        if str((saved.get("metadata") or {}).get("experiment_id")) != str(job["resume_experiment_id"]):
            raise ValueError("Resume generation experiment identity changed after enqueue.")
        config_path = Path(str(job.get("config") or ""))
        config_path = config_path if config_path.is_absolute() else get_project_root() / config_path
        if config_path.is_symlink():
            raise ValueError("Resume job config must not be a symlink.")
        config_path = config_path.resolve()
        if not config_path.is_relative_to((RUNS_DIR / "configs").resolve()):
            raise ValueError("Resume job config must be an immutable runs/configs snapshot.")
        queued_config = _read_json_file(config_path)
        if queued_config != (saved.get("metadata") or {}).get("config"):
            raise ValueError("Resume job config changed after enqueue.")
    environment = os.environ.copy()
    environment["MASTERTHESIS_STOP_FILE"] = str((JOBS_DIR / f"{job_id}.stop").resolve())
    source_paths = [str(get_project_root() / "src"), str(get_project_root())]
    environment["PYTHONPATH"] = os.pathsep.join([*source_paths, environment.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    with log_path.open("a", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            job["command"],
            cwd=get_project_root(),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
        )

    job.update(
        {
            "status": "running",
            "start_time": datetime.now().isoformat(timespec="seconds"),
            "log_path": log_path.relative_to(get_project_root()).as_posix(),
            "process_id": process.pid,
        }
    )
    _write_job(job)
    return job


def enqueue_job(command: list[str], metadata: dict[str, Any]) -> dict[str, Any]:
    """Create a queued job and start it if a worker slot is available."""
    metadata = enrich_experiment_metadata({**metadata, "command": command})
    _ensure_run_dirs()
    job_id = metadata.get("job_id") or uuid.uuid4().hex[:12]
    command = list(command)
    if "--config" in command:
        config_index = command.index("--config") + 1
        if config_index >= len(command):
            raise ValueError("--config requires a file path.")
        source = Path(command[config_index])
        source = source if source.is_absolute() else get_project_root() / source
        target = RUNS_DIR / "configs" / f"{job_id}.json"
        if source.resolve() != target.resolve():
            try:
                config_payload = json.loads(source.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(f"Config snapshot could not be created: {source}") from error
            if not isinstance(config_payload, dict):
                raise ValueError("Config snapshot must contain a JSON object.")
            experiment = config_payload.get("experiment", config_payload)
            dataset = experiment.get("dataset_name")
            if dataset and (experiment.get("dataset_ref") or dataset not in DATASETS):
                from src.training.experiment_config import resolve_experiment_config
                config_payload = resolve_experiment_config(get_project_root(), config_payload)
                metadata["dataset_ref"] = config_payload.get("experiment", config_payload).get("dataset_ref")
            command[config_index] = save_job_config_snapshot(config_payload, job_id)
        metadata = {**metadata, "config": command[config_index]}
    if metadata.get("resume_experiment_id"):
        _assert_no_active_resume(str(metadata["resume_experiment_id"]))
    job = {
        **metadata,
        "job_id": job_id,
        "command": command,
        "status": "queued",
        "queued_time": datetime.now().isoformat(timespec="seconds"),
        "log_path": (LOGS_DIR / f"{job_id}.log").relative_to(get_project_root()).as_posix(),
        "process_id": None,
    }
    _write_job(job)
    return json.loads(_job_path(job_id).read_text(encoding="utf-8"))


def start_job(command: list[str], metadata: dict[str, Any]) -> dict[str, Any]:
    """Backward-compatible alias for queue-based job submission."""
    return enqueue_job(command, metadata)


def update_job_status_if_finished(job: dict[str, Any]) -> dict[str, Any]:
    """Update a job status if its process is no longer running."""
    if job.get("status") not in {"running", "stop_requested"}:
        return job
    if _sync_job_context(job):
        _write_job(job)
    process_id = job.get("process_id")
    is_running = False
    if process_id:
        try:
            process = psutil.Process(int(process_id))
            is_running = process.is_running() and process.status() != psutil.STATUS_ZOMBIE
        except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
            is_running = False
    if not is_running:
        log_tail = read_log_tail(job.get("log_path", ""), n_lines=300)
        job["status"] = completed_status(log_tail)
        if job["status"] in {"failed", "stopped"}:
            _reconcile_mlflow_status(job, "FAILED" if job["status"] == "failed" else "KILLED")
        job["end_time"] = datetime.now().isoformat(timespec="seconds")
        job_path = JOBS_DIR / f"{job['job_id']}.json"
        job_path.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    return job


def list_jobs() -> list[dict[str, Any]]:
    """List known jobs from metadata files."""
    _ensure_run_dirs()
    jobs: list[dict[str, Any]] = []
    for job_file in sorted(JOBS_DIR.glob("*.json"), reverse=True):
        try:
            job = json.loads(job_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        jobs.append(update_job_status_if_finished(job))
    return jobs


def jobs_for_stage(
    stage: str, jobs: list[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    """List the jobs belonging to one pipeline stage.

    Baseline jobs queued before the stage key existed carry no stage at all,
    so they are claimed by "baseline" and by nothing else.
    """
    if jobs is None:
        jobs = list_jobs()
    return [job for job in jobs if (job.get("stage") or "baseline") == stage]


def promote_queued_jobs(jobs: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Start queued jobs while respecting the max_parallel_jobs setting."""
    _ensure_run_dirs()
    settings = load_dashboard_settings()
    max_parallel = int(settings["max_parallel_jobs"])
    if jobs is None:
        jobs = []
        for job_file in sorted(JOBS_DIR.glob("*.json")):
            try:
                jobs.append(json.loads(job_file.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError):
                continue
    running_jobs = [update_job_status_if_finished(job) for job in jobs if job.get("status") in {"running", "stop_requested"}]
    running_count = sum(1 for job in running_jobs if job.get("status") in {"running", "stop_requested"})
    available_slots = max(max_parallel - running_count, 0)
    if available_slots <= 0:
        return jobs

    queued_jobs = sorted(
        [job for job in jobs if job.get("status") == "queued"],
        key=lambda item: item.get("queued_time", ""),
    )
    for job in queued_jobs[:available_slots]:
        try:
            _start_queued_job(job)
        except Exception as error:
            job.update({
                "status": "failed",
                "end_time": datetime.now().isoformat(timespec="seconds"),
                "start_error": str(error),
            })
            _write_job(job)
    return jobs


def read_log_tail(log_path: str | Path, n_lines: int = 100) -> str:
    """Read the last n lines of a log file."""
    path = Path(log_path)
    if not path.is_absolute():
        path = get_project_root() / path
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n_lines:])


def resolve_log_path(log_path: str | Path) -> Path:
    """Resolve a job log path relative to the project root."""
    path = Path(log_path)
    if not path.is_absolute():
        path = get_project_root() / path
    return path


def get_log_file_info(log_path: str | Path) -> dict[str, Any]:
    """Return compact file metadata for a job log."""
    path = resolve_log_path(log_path)
    if not path.exists():
        return {"exists": False, "size_bytes": 0, "line_count": 0}
    text = path.read_text(encoding="utf-8", errors="replace")
    return {
        "exists": True,
        "size_bytes": path.stat().st_size,
        "line_count": len(text.splitlines()),
        "modified_time": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
    }


def parse_log_progress(log_text: str) -> dict[str, Any]:
    """Extract lightweight epoch and metric hints from a training log."""
    epochs: list[int] = []
    total_epochs: int | None = None
    for match in EPOCH_PATTERN.finditer(log_text):
        epochs.append(int(match.group(1)))
        if match.group(2):
            total_epochs = int(match.group(2))
    metrics: dict[str, float] = {}
    for name, value in METRIC_PATTERN.findall(log_text):
        metrics[name.lower()] = float(value)
    last_epoch = max(epochs) if epochs else None
    progress_fraction = None
    if last_epoch is not None and total_epochs:
        progress_fraction = min(max(last_epoch / total_epochs, 0.0), 1.0)
    return {
        "last_epoch_seen": last_epoch,
        "total_epochs": total_epochs,
        "progress_fraction": progress_fraction,
        "metrics": metrics,
    }


def get_job_runtime_seconds(job: dict[str, Any]) -> float | None:
    """Calculate elapsed runtime from job metadata."""
    start_time = job.get("start_time")
    if not start_time:
        return None
    try:
        start = datetime.fromisoformat(start_time)
        end = datetime.fromisoformat(job["end_time"]) if job.get("end_time") else datetime.now()
    except ValueError:
        return None
    return max((end - start).total_seconds(), 0.0)


def log_has_error(log_text: str) -> bool:
    """Detect common Python/ML failure markers in a job log."""
    error_markers = [
        "Traceback (most recent call last)",
        "RuntimeError:",
        "ValueError:",
        "FileNotFoundError:",
        "ModuleNotFoundError:",
        "CUDA out of memory",
        "Exception:",
        "ERROR",
    ]
    return any(marker in log_text for marker in error_markers)


def completed_status(log_text: str) -> str:
    """Classify a terminated process, keeping graceful stop distinct from failure."""
    if "TRAINING_STOPPED" in log_text:
        return "stopped"
    return "failed" if log_has_error(log_text) else "finished"


def parse_recovery_events(log_text: str) -> dict[str, Any]:
    """Return the latest well-formed recovery context/status emitted by trainers."""
    recovered: dict[str, Any] = {}
    for line in log_text.splitlines():
        if not line.startswith(("RECOVERY_CONTEXT ", "RECOVERY_STATUS ")):
            continue
        try:
            value = json.loads(line.split(" ", 1)[1])
        except (json.JSONDecodeError, IndexError):
            continue
        if isinstance(value, dict):
            recovered.update(value)
    return recovered


def _sync_job_context(job: dict[str, Any]) -> bool:
    """Persist startup context once so long logs cannot hide MLflow/recovery identity."""
    needs_run = not job.get("mlflow_run_id")
    needs_recovery = not job.get("recovery_status_path")
    if not needs_run and not needs_recovery:
        return False
    path = resolve_log_path(job.get("log_path", ""))
    if not path.is_file():
        return False
    changed = False
    try:
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                prefix, fields = None, ()
                if needs_run and line.startswith("JOB_RUN_CONTEXT "):
                    prefix, fields = "JOB_RUN_CONTEXT ", ("mlflow_run_id", "tracking_uri", "run_name")
                elif needs_recovery and line.startswith("RECOVERY_CONTEXT "):
                    prefix, fields = "RECOVERY_CONTEXT ", ("status_path", "experiment_id", "attempt_id")
                if prefix:
                    try:
                        value = json.loads(line.removeprefix(prefix))
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(value, dict):
                        continue
                    if prefix.startswith("JOB"):
                        for field in fields:
                            if value.get(field) is not None:
                                job[field] = value[field]
                                changed = True
                        needs_run = False
                    else:
                        job["recovery_status_path"] = value.get("status_path")
                        job["recovery_experiment_id"] = value.get("experiment_id")
                        job["recovery_attempt_id"] = value.get("attempt_id")
                        changed = True
                        needs_recovery = False
                if not needs_run and not needs_recovery:
                    break
    except OSError:
        return changed
    return changed


def _reconcile_mlflow_status(job: dict[str, Any], status: str) -> None:
    _sync_job_context(job)
    run_id = job.get("mlflow_run_id")
    tracking_uri = job.get("tracking_uri")
    if not run_id:
        return
    try:
        from mlflow.tracking import MlflowClient
        MlflowClient(tracking_uri=tracking_uri).set_terminated(str(run_id), status=status)
    except Exception as error:
        job["mlflow_status_warning"] = str(error)


def get_recovery_status(job: dict[str, Any]) -> dict[str, Any]:
    """Read persisted recovery status, falling back to emitted log context."""
    event = parse_recovery_events(read_log_tail(job.get("log_path", ""), n_lines=5000))
    path_value = job.get("recovery_status_path") or event.get("status_path")
    if path_value:
        status = _read_json_file(Path(path_value))
        if status:
            return {**event, **status, "status_path": str(path_value)}
    return event


def request_epoch_stop(job: dict[str, Any]) -> dict[str, Any]:
    """Request a stop after the next fully saved epoch; never terminate the process."""
    if job.get("status") not in {"running", "stop_requested"}:
        return job
    recovery = get_recovery_status(job)
    if not bool(((recovery.get("metadata") or {}).get("recovery") or {}).get("enabled")):
        raise ValueError("Epoch-safe stop requires confirmation that recovery is enabled for this job.")
    stop_path = JOBS_DIR / f"{job['job_id']}.stop"
    stop_path.parent.mkdir(parents=True, exist_ok=True)
    stop_path.write_text("stop after saved epoch\n", encoding="utf-8")
    job["status"] = "stop_requested"
    job["stop_requested_time"] = datetime.now().isoformat(timespec="seconds")
    job["stop_file"] = stop_path.relative_to(get_project_root()).as_posix() if stop_path.is_relative_to(get_project_root()) else str(stop_path)
    _write_job(job)
    return job


def _assert_no_active_resume(experiment_id: str, exclude_job_id: str | None = None) -> None:
    for path in JOBS_DIR.glob("*.json") if JOBS_DIR.exists() else ():
        existing = _read_json_file(path) or {}
        if existing.get("job_id") == exclude_job_id:
            continue
        if existing.get("resume_experiment_id") == experiment_id and existing.get("status") in {"queued", "running", "stop_requested"}:
            raise ValueError(f"An active resume already exists for experiment {experiment_id}.")


def _validate_resume_generation(generation: Path) -> dict[str, Any]:
    saved = load_generation(generation)
    metadata = saved.get("metadata") or {}
    config = metadata.get("config")
    if not isinstance(config, dict):
        raise ValueError("Recovery metadata does not contain an immutable config.")
    experiment = config.get("experiment", config)
    dataset = experiment.get("dataset_name") or config.get("dataset_name") or config.get("dataset")
    from src.datasets.dataset_runtime import resolve_dataset_runtime, verify_dataset_content
    from src.datasets.dataset_registry import DatasetRef
    runtime = resolve_dataset_runtime(get_project_root(), experiment.get("dataset_ref"), dataset)
    if experiment.get("dataset_ref"):
        verify_dataset_content(get_project_root(), DatasetRef(**experiment["dataset_ref"]))
    split_file = Path(runtime["split_file"])
    requested = recovery_metadata(config, split_file, get_project_root())
    validate_resume(metadata, requested)
    return saved


def enqueue_resume_job(generation: str | Path, backup_dir: str | Path | None = None) -> dict[str, Any]:
    """Queue a new attempt from a verified generation using its immutable config."""
    generation = Path(generation).resolve()
    saved = _validate_resume_generation(generation)
    metadata = saved.get("metadata") or {}
    config = metadata.get("config")
    if not isinstance(config, dict):
        raise ValueError("Recovery metadata does not contain an immutable config.")
    experiment_id = str(metadata.get("experiment_id") or "")
    if not experiment_id:
        raise ValueError("Recovery metadata does not contain experiment_id.")
    _assert_no_active_resume(experiment_id)
    job_id = uuid.uuid4().hex[:12]
    config_path = save_job_config_snapshot(config, job_id)
    stage = str(metadata.get("stage") or "baseline")
    scripts = {
        "baseline": "src/training/train_baseline.py",
        "teacher_cnn": "src/training/train_teacher.py",
        "teacher_vision_transformer": "src/training/train_teacher.py",
        "knowledge_distillation": "src/training/train_kd.py",
        "multi_teacher_knowledge_distillation": "src/training/train_multi_kd.py",
    }
    if stage not in scripts:
        raise ValueError(f"Unsupported recovery stage: {stage}")
    command = [sys.executable, scripts[stage], "--config", config_path]
    experiment = config.get("experiment", config)
    if stage == "knowledge_distillation":
        teacher_run = experiment.get("teacher_run_id")
        teacher_checkpoint = experiment.get("teacher_checkpoint_path")
        if not teacher_run or not teacher_checkpoint:
            raise ValueError("Single-KD recovery lacks teacher identity/checkpoint.")
        command.extend(["--teacher-run-id", str(teacher_run), "--teacher-checkpoint", str(teacher_checkpoint)])
    command.extend(["--resume-from", str(generation)])
    if backup_dir:
        command.extend(["--backup-dir", str(Path(backup_dir))])
    return enqueue_job(command, {
        "job_id": job_id, "stage": stage, "config": config_path,
        "dataset": experiment.get("dataset_name") or config.get("dataset"),
        "model": experiment.get("student_model_name") or experiment.get("model_name") or config.get("model"),
        "resume_from": str(generation), "resume_experiment_id": experiment_id,
        "parent_attempt_id": metadata.get("attempt_id"),
    })


def stop_job(job: dict[str, Any], timeout_seconds: int = 10) -> dict[str, Any]:
    """Stop a running job process and persist stopped metadata."""
    if job.get("status") not in {"running", "stop_requested"}:
        return job

    process_id = job.get("process_id")
    if not process_id:
        job["status"] = "stop_failed"
        job["stop_error"] = "Missing process_id"
    else:
        try:
            process = psutil.Process(int(process_id))
            children = process.children(recursive=True)
            for child in children:
                child.terminate()
            process.terminate()
            gone, alive = psutil.wait_procs([process, *children], timeout=timeout_seconds)
            for alive_process in alive:
                alive_process.kill()
            job["status"] = "stopped"
            job["end_time"] = datetime.now().isoformat(timespec="seconds")
            job["stop_note"] = "Stopped from Streamlit Job Monitor"
            _reconcile_mlflow_status(job, "KILLED")
        except (psutil.NoSuchProcess, ValueError):
            job["status"] = "finished"
            job["end_time"] = datetime.now().isoformat(timespec="seconds")
        except psutil.AccessDenied as exc:
            job["status"] = "stop_failed"
            job["stop_error"] = str(exc)

    job_path = JOBS_DIR / f"{job['job_id']}.json"
    job_path.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    return job


def cancel_queued_job(job: dict[str, Any]) -> dict[str, Any]:
    """Cancel a queued job without touching any running process."""
    if job.get("status") != "queued":
        job["cancel_error"] = "Only queued jobs can be cancelled."
        return job

    job["status"] = "cancelled"
    job["cancelled_time"] = datetime.now().isoformat(timespec="seconds")
    job["cancel_note"] = "Cancelled from Streamlit Job Monitor before process start"
    job["process_id"] = None
    _write_job(job)
    return job


def is_process_running(process_id: int | str | None) -> bool:
    """Return whether a process id is currently alive."""
    if not process_id:
        return False
    try:
        process = psutil.Process(int(process_id))
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
        return False


def get_queue_worker_status() -> dict[str, Any]:
    """Return queue worker state and heartbeat metadata."""
    state = _read_json_file(QUEUE_WORKER_STATE_PATH) or {}
    heartbeat = _read_json_file(QUEUE_WORKER_HEARTBEAT_PATH) or {}
    process_id = state.get("process_id")
    process_alive = bool(process_id) and is_process_running(process_id)
    fresh = False
    try:
        signal_time = datetime.fromisoformat(heartbeat["timestamp"])
        start_time = datetime.fromisoformat(state["start_time"])
        fresh = start_time <= signal_time and 0 <= (datetime.now() - signal_time).total_seconds() <= 30
    except (KeyError, TypeError, ValueError):
        pass
    running = bool(process_alive and fresh and heartbeat.get("process_id") == process_id
                   and heartbeat.get("status") == "running")
    if process_alive:
        try:
            age = (datetime.now() - datetime.fromisoformat(state["start_time"])).total_seconds()
        except (KeyError, TypeError, ValueError):
            age = 30
        status = "running" if running else "starting" if age < 30 else "unresponsive"
    else:
        status = state.get("status", "stopped")
        if status in {"starting", "running"}:
            status = "failed"
    return {
        **state,
        "status": status,
        "running": running,
        "process_alive": process_alive,
        "heartbeat": heartbeat,
        "state_path": QUEUE_WORKER_STATE_PATH.relative_to(get_project_root()).as_posix(),
        "heartbeat_path": QUEUE_WORKER_HEARTBEAT_PATH.relative_to(get_project_root()).as_posix(),
        "log_path": QUEUE_WORKER_LOG_PATH.relative_to(get_project_root()).as_posix(),
    }


def start_queue_worker() -> dict[str, Any]:
    """Start the queue worker as an independent background process."""
    _ensure_run_dirs()
    status = get_queue_worker_status()
    if status.get("process_alive"):
        return {**status, "started": False, "message": "Kuyruk işleyicisi zaten açık veya başlatılıyor."}

    command = [sys.executable, "-u", "app/queue_worker.py"]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join([
        str(get_project_root()), str(get_project_root() / "src"), environment.get("PYTHONPATH", "")
    ]).rstrip(os.pathsep)
    start_time = datetime.now().isoformat(timespec="seconds")
    creationflags = 0
    if sys.platform.startswith("win"):
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    try:
        with QUEUE_WORKER_LOG_PATH.open("a", encoding="utf-8") as log_file:
            log_file.write(f"\n[{start_time}] Queue worker launch requested.\n")
            log_file.flush()
            process = subprocess.Popen(
                command,
                cwd=get_project_root(),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
                creationflags=creationflags,
                env=environment,
            )
    except OSError as error:
        with QUEUE_WORKER_LOG_PATH.open("a", encoding="utf-8") as log_file:
            log_file.write(f"Queue worker launch failed: {error}\n")
        state = {"process_id": None, "status": "failed", "error": str(error), "start_time": start_time}
        QUEUE_WORKER_STATE_PATH.write_text(json.dumps(state), encoding="utf-8")
        return {**get_queue_worker_status(), "started": False, "message": f"İşleyici başlatılamadı: {error}"}

    state = {
        "process_id": process.pid,
        "command": command,
        "status": "starting",
        "start_time": start_time,
        "log_path": QUEUE_WORKER_LOG_PATH.relative_to(get_project_root()).as_posix(),
    }
    QUEUE_WORKER_STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**get_queue_worker_status(), "started": True,
            "message": "Başlatma istendi. Hazır durumunu görmek için Durumu ve logu yenile düğmesine basın."}


def stop_queue_worker(timeout_seconds: int = 10) -> dict[str, Any]:
    """Stop only the queue worker process, not training jobs."""
    status = get_queue_worker_status()
    process_id = status.get("process_id")
    if not status.get("process_alive"):
        return {**status, "stopped": False, "message": "Queue worker is not running."}
    try:
        process = psutil.Process(int(process_id))
        process.terminate()
        try:
            process.wait(timeout=timeout_seconds)
        except psutil.TimeoutExpired:
            process.kill()
    except (psutil.NoSuchProcess, ValueError):
        pass
    except psutil.AccessDenied as exc:
        return {**status, "stopped": False, "message": f"Access denied: {exc}"}

    state = {
        **(_read_json_file(QUEUE_WORKER_STATE_PATH) or {}),
        "process_id": None,
        "status": "stopped",
        "stop_time": datetime.now().isoformat(timespec="seconds"),
    }
    QUEUE_WORKER_STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**get_queue_worker_status(), "stopped": True, "message": "Queue worker stopped."}
