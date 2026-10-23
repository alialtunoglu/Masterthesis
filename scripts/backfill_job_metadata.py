"""Back up and classify legacy job metadata; never touch experiment artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tracking.experiment_taxonomy import enrich_experiment_metadata


def backfill_job_metadata(root: Path, *, apply: bool = False) -> dict:
    root = Path(root).resolve()
    updates = []
    report = {"candidate_count": 0, "updated_count": 0, "stage_added_count": 0,
              "backup_dir": None, "skipped": []}
    for path in sorted((root / "runs/jobs").glob("*.json")):
        if path.is_symlink():
            report["skipped"].append({"file": path.name, "reason": "Symlink is not migrated."})
            continue
        try:
            original = path.read_bytes()
            record = json.loads(original)
            if not isinstance(record, dict):
                raise ValueError("Expected a JSON object.")
            if record.get("status") in {"running", "queued", "stop_requested"}:
                raise ValueError("Active job is not migrated.")
            enriched = enrich_experiment_metadata(record)
            if not enriched.get("training_method"):
                raise ValueError("Unrecognized pipeline; classification not guessed.")
        except (OSError, ValueError) as error:
            report["skipped"].append({"file": path.name, "reason": str(error)})
            continue
        if enriched != record:
            updates.append((path, original, enriched))
            report["stage_added_count"] += int(not record.get("stage") and bool(enriched.get("stage")))
    report["candidate_count"] = len(updates)
    if not apply or not updates:
        return report
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = root / "runs/metadata_backups" / stamp
    (backup / "jobs").mkdir(parents=True)
    entries = []
    for path, original, _ in updates:
        (backup / "jobs" / path.name).write_bytes(original)
        entries.append({"file": path.name, "sha256": hashlib.sha256(original).hexdigest()})
    (backup / "manifest.json").write_text(json.dumps({"files": entries}, indent=2), encoding="utf-8")
    report["backup_dir"] = str(backup)
    for path, original, enriched in updates:
        if path.read_bytes() != original:
            report["skipped"].append({"file": path.name, "reason": "Changed after scan; not overwritten."})
            continue
        descriptor, temporary = tempfile.mkstemp(prefix=".taxonomy-", dir=path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(enriched, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
        report["updated_count"] += 1
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--apply", action="store_true", help="Write metadata after preserving byte-exact backups.")
    args = parser.parse_args()
    print(json.dumps(backfill_job_metadata(args.root, apply=args.apply), ensure_ascii=False, indent=2))
