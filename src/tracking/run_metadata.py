"""Portable experiment identity and protocol fingerprints (no dataset writes)."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


def protocol_fingerprint(metadata: dict) -> str:
    return hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def split_fingerprint(metadata: dict) -> str:
    def portable(path):
        value = str(path).replace("\\", "/")
        if "/datasets/" in value:
            value = "datasets/" + value.split("/datasets/", 1)[1]
        return value
    return protocol_fingerprint({
        "dataset": metadata.get("dataset_name"), "classes": metadata["classes"],
        "splits": {name: sorted([(portable(row["path"]), int(row["label_id"])) for row in rows])
                   for name, rows in metadata["splits"].items()},
    })


def dataset_fingerprint(metadata: dict) -> str:
    """Portable manifest fingerprint over sample identities and labels, not image bytes."""
    samples = []
    for rows in metadata["splits"].values():
        for row in rows:
            path = str(row["path"]).replace("\\", "/")
            if "/datasets/" in path:
                path = "datasets/" + path.split("/datasets/", 1)[1]
            samples.append((path, int(row["label_id"])))
    return protocol_fingerprint({"classes": metadata["classes"], "samples": sorted(samples)})


def source_commit(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
