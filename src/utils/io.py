"""Small IO helpers shared by dataset preparation scripts."""

from __future__ import annotations

import csv
import json
import os
import time
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_json(data: Mapping[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def save_csv(rows: Iterable[Mapping[str, Any]], path: str | Path, fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    output_path = Path(path)
    ensure_dir(output_path.parent)

    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def upsert_result_row(row: Mapping[str, Any], path: Path, columns: list[str] | None = None) -> None:
    """Idempotent, process-locked publication of one completed run."""
    import pandas as pd
    if not row.get("mlflow_run_id"):
        raise ValueError("A result requires mlflow_run_id for idempotent publication.")
    path = Path(path)
    ensure_dir(path.parent)
    with path.with_suffix(path.suffix + ".lock").open("a+b") as lock:
        lock.seek(0, os.SEEK_END)
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        deadline = time.monotonic() + 10
        while True:
            try:
                lock.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Result store is busy: {path}")
                time.sleep(.05)
        # OS releases the file lock even if the training process is terminated.
        frame = pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()
        if "mlflow_run_id" in frame:
            frame = frame[frame["mlflow_run_id"].astype(str) != str(row["mlflow_run_id"])]
        frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
        if columns:
            frame = frame.reindex(columns=columns + [name for name in frame.columns if name not in columns])
        descriptor, temporary = tempfile.mkstemp(prefix=".result-", dir=path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
                frame.to_csv(stream, index=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
