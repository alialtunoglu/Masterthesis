"""Physical GPU inventory and one-owner locks shared with laboratory notebooks."""
from contextlib import contextmanager
import csv
import os
from pathlib import Path
import re
import subprocess


class GpuBusyError(RuntimeError):
    """The selected GPU has an existing experiment owner."""


def discover_gpus() -> list[dict]:
    """Read NVIDIA inventory without importing torch or allocating a CUDA context."""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"], text=True, stderr=subprocess.DEVNULL, timeout=5,
        )
    except FileNotFoundError:
        return []
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise RuntimeError("GPU bilgisi okunamadı; nvidia-smi durumunu kontrol edin.") from error
    cards = []
    for row in csv.reader(output.splitlines(), skipinitialspace=True):
        if len(row) != 5 or not row[0].strip().isdigit():
            raise RuntimeError("nvidia-smi beklenen GPU envanterini döndürmedi.")
        cards.append({"id": row[0].strip(), "name": row[1].strip(),
                      "memory_used_mb": int(row[2]), "memory_total_mb": int(row[3]),
                      "utilization": int(row[4])})
    return cards


def _lock_path(root, gpu_id):
    if not re.fullmatch(r"\d+", str(gpu_id)):
        raise ValueError("Invalid physical GPU index.")
    return Path(root) / "lab_state" / f"gpu_{gpu_id}.lock"


@contextmanager
def gpu_lock(root: Path, gpu_id: str):
    path = _lock_path(root, gpu_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as lock:
        if os.name == "nt":
            import msvcrt
            if path.stat().st_size == 0:
                lock.write(b"0")
                lock.flush()
            lock.seek(0)
            try:
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise GpuBusyError(f"GPU {gpu_id} başka bir deney tarafından kullanılıyor.") from error
        else:
            import fcntl
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise GpuBusyError(f"GPU {gpu_id} başka bir deney tarafından kullanılıyor.") from error
        try:
            yield lock
        finally:
            if os.name == "nt":
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock, fcntl.LOCK_UN)


def gpu_is_available(root: Path, gpu_id: str) -> bool:
    if not _lock_path(root, gpu_id).exists():
        return True
    try:
        with gpu_lock(root, gpu_id):
            return True
    except GpuBusyError:
        return False


def choose_gpu(request: str, inventory: list[dict], running_jobs: list[dict], root: Path) -> str | None:
    request = str(request)
    if request not in {"auto", "cpu"} and not request.isdigit():
        raise ValueError("GPU seçimi otomatik veya geçerli bir fiziksel GPU indeksi olmalı.")
    if request == "cpu":
        return None
    ids = [str(card["id"]) for card in inventory]
    if request != "auto" and request not in ids:
        raise ValueError(f"GPU {request} bu bilgisayarda bulunamadı.")
    if not ids:
        return None
    active = [job for job in running_jobs if job.get("status") in {"running", "stop_requested"}]
    # Old native jobs have no physical assignment; do not guess which GPU they own.
    if any("gpu_id" not in job and job.get("gpu_requested") != "cpu" for job in active):
        raise GpuBusyError("GPU ataması bilinmeyen eski bir koşu hâlâ çalışıyor.")
    occupied = {str(job["gpu_id"]) for job in active if job.get("gpu_id") is not None}
    for candidate in ids if request == "auto" else [request]:
        if candidate not in occupied and gpu_is_available(root, candidate):
            return candidate
    raise GpuBusyError("Seçilen GPU dolu; deney kuyrukta bekliyor.")
