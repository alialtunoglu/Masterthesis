"""Background queue worker for Streamlit-launched experiment jobs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from experiment_runner import (  # noqa: E402
    QUEUE_WORKER_HEARTBEAT_PATH,
    QUEUE_WORKER_STATE_PATH,
    get_project_root,
    is_process_running,
    list_jobs,
    load_dashboard_settings,
    promote_queued_jobs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local experiment queue worker.")
    parser.add_argument("--poll-seconds", type=int, default=5)
    return parser.parse_args()


def write_heartbeat(status: str, message: str = "") -> None:
    QUEUE_WORKER_HEARTBEAT_PATH.parent.mkdir(parents=True, exist_ok=True)
    heartbeat = {
        "status": status,
        "message": message,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(get_project_root()),
    }
    QUEUE_WORKER_HEARTBEAT_PATH.write_text(
        json.dumps(heartbeat, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def mark_state(status: str) -> None:
    state = {
        "process_id": None,
        "status": status,
        "updated_time": datetime.now().isoformat(timespec="seconds"),
    }
    if QUEUE_WORKER_STATE_PATH.exists():
        try:
            existing = json.loads(QUEUE_WORKER_STATE_PATH.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                state = {**existing, **state}
        except json.JSONDecodeError:
            pass
    QUEUE_WORKER_STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def claim_worker_slot() -> bool:
    """Claim the single worker slot, returning False when another worker is alive."""
    if QUEUE_WORKER_STATE_PATH.exists():
        try:
            existing = json.loads(QUEUE_WORKER_STATE_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
        existing_pid = existing.get("process_id") if isinstance(existing, dict) else None
        if existing_pid and int(existing_pid) != os.getpid() and is_process_running(existing_pid):
            print(f"Queue worker already running with pid={existing_pid}.", flush=True)
            return False

    state = {
        "process_id": os.getpid(),
        "status": "running",
        "start_time": datetime.now().isoformat(timespec="seconds"),
        "command": [sys.executable, "app/queue_worker.py"],
    }
    QUEUE_WORKER_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    QUEUE_WORKER_STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return True


def main() -> None:
    args = parse_args()
    poll_seconds = max(int(args.poll_seconds), 1)
    if not claim_worker_slot():
        return
    print(f"Queue worker started. poll_seconds={poll_seconds}", flush=True)
    try:
        while True:
            settings = load_dashboard_settings()
            jobs = list_jobs()
            promote_queued_jobs(jobs)
            write_heartbeat(
                "running",
                f"Checked {len(jobs)} jobs. max_parallel_jobs={settings['max_parallel_jobs']}",
            )
            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        write_heartbeat("stopped", "Stopped by KeyboardInterrupt")
        mark_state("stopped")
        print("Queue worker stopped.", flush=True)


if __name__ == "__main__":
    main()
