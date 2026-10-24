"""Read-only view of detached notebook attempts; never claim their native job files."""
from datetime import datetime
import json
from pathlib import Path
import re

import psutil


def _read(path):
    try:
        payload = path.read_text(encoding="utf-8")
        value = json.loads(payload)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def runner_identity(state: dict) -> bool | None:
    """A PID alone is not ownership: match both immutable attempt arguments."""
    try:
        process = psutil.Process(int(state["runner_pid"]))
        if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
            return False
        command = process.cmdline()
        return (any(Path(part).name == "run_lab_experiment.py" for part in command)
                and command[command.index("--attempt") + 1] == state["attempt"]
                and command[command.index("--experiment") + 1] == state["experiment"])
    except psutil.AccessDenied:
        return None
    except (psutil.NoSuchProcess, KeyError, ValueError, IndexError, TypeError):
        return False


def _timestamp(value):
    try:
        return datetime.fromtimestamp(float(value)).isoformat(timespec="seconds")
    except (TypeError, ValueError, OSError, OverflowError):
        return None


def list_lab_jobs(root: Path) -> list[dict]:
    root = Path(root).resolve()
    base = root / "lab_state"
    if not base.is_dir() or base.is_symlink():
        return []
    jobs = []
    for attempt in sorted(base.iterdir()):
        if not attempt.is_dir() or attempt.is_symlink() or not re.fullmatch(r"[a-z0-9_]+__[a-f0-9]{12}", attempt.name):
            continue
        if any((attempt / name).is_symlink() for name in ("state.json", "config.json", "training.log")):
            continue
        state = _read(attempt / "state.json")
        config = _read(attempt / "config.json")
        config_invalid = not config or any(key in config and not isinstance(config[key], dict)
                                          for key in ("experiment", "training", "student", "distillation"))
        experiment = config.get("experiment", config)
        experiment = experiment if isinstance(experiment, dict) else {}
        training = config.get("training", config)
        training = training if isinstance(training, dict) else {}
        raw_status = state.get("status")
        warning = ""
        identity = None
        if not state:
            status = "unknown"
            warning = "state.json henüz hazır değil veya okunamadı; koşuyu yeniden başlatmayın."
        elif f"{state.get('experiment')}__{state.get('attempt')}" != attempt.name:
            status = "unknown"
            warning = "Koşu klasörü ile state.json kimliği uyuşmuyor."
        elif raw_status in {"starting", "running"}:
            identity = runner_identity(state)
            status = "running" if identity is True else "interrupted" if identity is False else "unknown"
            if identity is not True:
                warning = "Çalıştırıcı kimliği doğrulanamadı; tamamlandı olarak kabul edilmedi."
        else:
            status = {"finished": "finished", "dry_run_finished": "finished", "failed": "failed", "stopped": "stopped"}.get(raw_status, "unknown")
        if status == "running" and (attempt / "stop.request").exists():
            status = "stop_requested"
        if config_invalid:
            status, warning = "unknown", "config.json henüz hazır değil veya bölümleri geçersiz; mevcut koşuya dokunulmadı."
        log = attempt / "training.log"
        context = state.get("run_context") or {}
        if not isinstance(context, dict):
            context = {}
        if not context and log.is_file():
            with log.open(encoding="utf-8", errors="replace") as stream:
                for line in stream:
                    if line.startswith("JOB_RUN_CONTEXT "):
                        try:
                            context = json.loads(line.split(" ", 1)[1])
                        except ValueError:
                            pass
                        break
        if not isinstance(context, dict):
            context = {}
        teachers = config.get("teachers")
        teachers = teachers if isinstance(teachers, list) else []
        student = config.get("student")
        student = student if isinstance(student, dict) else {}
        distillation = config.get("distillation")
        distillation = distillation if isinstance(distillation, dict) else {}
        teacher_models = [entry.get("model_name") for entry in teachers if isinstance(entry, dict)]
        job = {"job_id": "lab-" + attempt.name.rsplit("__", 1)[1], "execution_origin": "lab_notebook",
               "status": status, "stage": experiment.get("stage", "baseline"),
               "dataset": experiment.get("dataset_name"), "dataset_ref": experiment.get("dataset_ref"),
               "model": experiment.get("student_model_name") or student.get("model_name") or experiment.get("model_name"),
               "teacher_model": experiment.get("teacher_model_name"), "teacher_models": teacher_models,
               "kd_type": experiment.get("kd_type"), "gpu_id": str(state["gpu"]) if state.get("gpu") is not None else None,
               "dry_run": bool(state.get("dry_run", raw_status == "dry_run_finished")),
               "process_id": state.get("runner_pid"), "process_identity_verified": identity is True and not config_invalid,
               "start_time": _timestamp(state.get("started_at")), "end_time": _timestamp(state.get("ended_at")),
               "log_path": log.relative_to(root).as_posix(), "config": (attempt / "config.json").relative_to(root).as_posix(),
               "stop_file": (attempt / "stop.request").relative_to(root).as_posix(),
               "lab_attempt_dir": attempt.relative_to(root).as_posix(), "monitor_warning": warning,
               "source_commit": state.get("source_commit"), "lab_export": state.get("export"),
               "mlflow_run_id": context.get("mlflow_run_id"), "tracking_uri": context.get("tracking_uri"),
               "run_name": context.get("run_name"), "teacher_count": len(teacher_models) or None,
               "teacher_run_id": experiment.get("teacher_run_id"),
               "aggregation": distillation.get("aggregation"),
               "training_epochs": training.get("epochs")}
        jobs.append(job)
    return jobs


def request_notebook_stop(root: Path, job: dict) -> dict:
    matches = [item for item in list_lab_jobs(root) if item["job_id"] == job.get("job_id")]
    if len(matches) != 1 or not matches[0].get("process_identity_verified"):
        raise ValueError("Notebook çalıştırıcısının kimliği doğrulanamadı; durdurma isteği gönderilmedi.")
    current = matches[0]
    stop = Path(root) / current["stop_file"]
    if stop.is_symlink() or not stop.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("Durdurma dosyası proje dışında veya sembolik bağlantı.")
    stop.write_text("stop after saved epoch\n", encoding="utf-8")
    return {**current, "status": "stop_requested"}


def portable_teacher_candidates(root, dataset, ref, image_size):
    """Reuse the transferred teacher inputs; strict checkpoint validation still runs before launch."""
    manifest = _read(Path(root) / "lab_notebooks/lab_manifest.json")
    rows = []
    entries = manifest.get("teachers")
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict) or not entry.get("run_id"):
            continue
        if (entry.get("dataset_name") != dataset or entry.get("dataset_ref") != ref
                or entry.get("image_size") != image_size or entry.get("status") != "completed"):
            continue
        checkpoint = (Path(root) / str(entry.get("checkpoint_path", ""))).resolve()
        if not checkpoint.is_relative_to(Path(root).resolve()) or not checkpoint.is_file():
            continue
        rows.append({**entry, "mlflow_run_id": entry["run_id"], "_portable_teacher_input": entry})
    return rows
