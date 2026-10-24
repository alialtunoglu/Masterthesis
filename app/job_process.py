"""Hold a GPU admission lock for the entire native training process lifetime."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import subprocess
import sys

from gpu_runtime import gpu_lock


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("A native training command is required.")
    root = Path(__file__).resolve().parents[1]
    lock = nullcontext() if args.gpu == "cpu" else gpu_lock(root, args.gpu)
    try:
        with lock as owned_lock:
            try:
                commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
            except (OSError, subprocess.CalledProcessError):
                commit = None
            print("JOB_PROCESS_CONTEXT " + json.dumps({"source_commit": commit, "gpu_id": args.gpu}), flush=True)
            # Retain ownership if the supervising process dies before its Linux child.
            inherited = {"pass_fds": (owned_lock.fileno(),)} if owned_lock is not None and os.name != "nt" else {}
            result = subprocess.run(command, cwd=root, check=False, **inherited)
            returncode = result.returncode
    except Exception:
        print('JOB_EXIT {"returncode": 1}', flush=True)
        raise
    print("JOB_EXIT " + json.dumps({"returncode": returncode}), flush=True)
    return returncode


if __name__ == "__main__":
    sys.exit(main())
