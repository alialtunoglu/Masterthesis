"""Subprocess-based experiment runner for the local dashboard."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil

from ui_utils import get_project_root


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

    with log_path.open("a", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            job["command"],
            cwd=get_project_root(),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
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
    _ensure_run_dirs()
    job_id = metadata.get("job_id") or uuid.uuid4().hex[:12]
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
    if job.get("status") != "running":
        return job
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
        job["status"] = "failed" if log_has_error(log_tail) else "finished"
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
    running_jobs = [update_job_status_if_finished(job) for job in jobs if job.get("status") == "running"]
    running_count = sum(1 for job in running_jobs if job.get("status") == "running")
    available_slots = max(max_parallel - running_count, 0)
    if available_slots <= 0:
        return jobs

    queued_jobs = sorted(
        [job for job in jobs if job.get("status") == "queued"],
        key=lambda item: item.get("queued_time", ""),
    )
    for job in queued_jobs[:available_slots]:
        _start_queued_job(job)
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


def stop_job(job: dict[str, Any], timeout_seconds: int = 10) -> dict[str, Any]:
    """Stop a running job process and persist stopped metadata."""
    if job.get("status") != "running":
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
    running = is_process_running(process_id)
    return {
        **state,
        "running": running,
        "heartbeat": heartbeat,
        "state_path": QUEUE_WORKER_STATE_PATH.relative_to(get_project_root()).as_posix(),
        "heartbeat_path": QUEUE_WORKER_HEARTBEAT_PATH.relative_to(get_project_root()).as_posix(),
        "log_path": QUEUE_WORKER_LOG_PATH.relative_to(get_project_root()).as_posix(),
    }


def start_queue_worker() -> dict[str, Any]:
    """Start the queue worker as an independent background process."""
    _ensure_run_dirs()
    status = get_queue_worker_status()
    if status.get("running"):
        return {**status, "started": False, "message": "Queue worker is already running."}

    command = [sys.executable, "app/queue_worker.py"]
    creationflags = 0
    if sys.platform.startswith("win"):
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    with QUEUE_WORKER_LOG_PATH.open("a", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=get_project_root(),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            creationflags=creationflags,
        )

    state = {
        "process_id": process.pid,
        "command": command,
        "status": "running",
        "start_time": datetime.now().isoformat(timespec="seconds"),
        "log_path": QUEUE_WORKER_LOG_PATH.relative_to(get_project_root()).as_posix(),
    }
    QUEUE_WORKER_STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**get_queue_worker_status(), "started": True, "message": "Queue worker started."}


def stop_queue_worker(timeout_seconds: int = 10) -> dict[str, Any]:
    """Stop only the queue worker process, not training jobs."""
    status = get_queue_worker_status()
    process_id = status.get("process_id")
    if not status.get("running"):
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
        "status": "stopped",
        "stop_time": datetime.now().isoformat(timespec="seconds"),
    }
    QUEUE_WORKER_STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**get_queue_worker_status(), "stopped": True, "message": "Queue worker stopped."}
