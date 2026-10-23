"""Resolve pinned dataset versions; legacy split protocols stay unchanged."""

from collections import Counter
import json
import math
from pathlib import Path

from .dataset_registry import DATASETS, DatasetRef, resolve_dataset
from .dataset_registration import atomic_json, checked_id, contained_path, digest_json, registry_lock, scan_dataset


def _revision_paths(root, dataset_id, revision):
    checked_id(dataset_id)
    checked_id(revision)
    base = contained_path(root, f"datasets/.registry/{dataset_id}/{revision}")
    manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    definition = json.loads(contained_path(root, f"configs/datasets/{dataset_id}.json").read_text(encoding="utf-8"))
    expected = digest_json([(row["relative_path"], row["label_id"], row["content_sha256"]) for row in manifest["samples"]])
    if manifest["dataset_id"] != dataset_id or manifest["revision"] != revision or expected != revision:
        raise ValueError("Dataset manifest identity is invalid.")
    return base, manifest, definition


def create_registered_split(root: Path, dataset_id: str, revision: str, ratios: dict, seed: int) -> DatasetRef:
    root = Path(root).resolve()
    if set(ratios) != {"train", "val", "test"} or any(not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0 for value in ratios.values()) or not math.isclose(sum(ratios.values()), 1., abs_tol=1e-9):
        raise ValueError("Split ratios must be positive and sum to1.")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("Split seed must be a non-negative integer.")
    with registry_lock(root, dataset_id):
        base, manifest, definition = _revision_paths(root, dataset_id, revision)
        candidate = contained_path(root, definition["root_relative_path"])
        current = scan_dataset(root, candidate, definition["layout"])
        if current.revision != revision:
            raise ValueError("Dataset content changed; register a new revision.")
        if definition["layout"] == "presplit":
            splits = {part: [row for row in current.samples if row.get("split") == part] for part in ("train", "val", "test")}
        else:
            from .create_splits import stratified_split
            splits = stratified_split(current.samples, seed, ratios)
        for part, samples in splits.items():
            if set(Counter(row["label"] for row in samples)) != set(current.classes):
                raise ValueError(f"Insufficient samples: every class must appear in {part}.")
        assignments = {part: sorted((row["relative_path"], row["label_id"]) for row in rows) for part, rows in splits.items()}
        split_id = digest_json({"revision": revision, "assignments": assignments, "ratios": ratios, "seed": seed})
        ref = DatasetRef(dataset_id, revision, split_id)
        payload = {"identity_version": 2, "dataset_name": dataset_id, "dataset_path": definition["root_relative_path"],
                   "dataset_ref": ref.to_dict(), "dataset_fingerprint": revision, "class_mapping_hash": current.class_mapping_hash,
                   "seed": seed, "split_ratio": ratios, "num_classes": len(current.classes), "classes": current.classes,
                   "total_samples": len(current.samples), "splits": splits}
        target = base / "splits" / f"{split_id}.json"
        if target.exists():
            if json.loads(target.read_text(encoding="utf-8")) != payload:
                raise ValueError("Immutable split metadata differs.")
        else:
            atomic_json(target, payload)
        atomic_json(root / "configs/datasets" / f"{dataset_id}.json",
                    {**definition, "default_revision": revision, "default_split": split_id, "ready": True})
    return ref


def verify_dataset_content(root: Path, ref: DatasetRef) -> None:
    _, manifest, definition = _revision_paths(root, ref.dataset_id, ref.revision)
    scan = scan_dataset(root, contained_path(root, definition["root_relative_path"]), definition["layout"])
    if scan.revision != ref.revision or scan.class_mapping_hash != manifest["class_mapping_hash"]:
        raise ValueError("Dataset content/class mapping changed; register a new revision.")


def dataset_package(root: Path, ref: DatasetRef | dict) -> dict:
    if isinstance(ref, dict):
        ref = DatasetRef(**ref)
    runtime = resolve_dataset_runtime(root, ref)
    _, manifest, descriptor = _revision_paths(root, ref.dataset_id, ref.revision)
    return {"descriptor": descriptor, "manifest": manifest,
            "split": json.loads(Path(runtime["split_file"]).read_text(encoding="utf-8"))}


def validate_dataset_package(package: dict, ref: dict) -> None:
    """Validate portable metadata without creating a local training-ready dataset."""
    definition, manifest, split = (package[key] for key in ("descriptor", "manifest", "split"))
    for key in ("dataset_id", "revision", "split_id"):
        checked_id(ref[key])
    relative_root = Path(definition["root_relative_path"])
    if relative_root.parts[:1] != ("datasets",) or relative_root.is_absolute() or ".." in relative_root.parts or "\\" in definition["root_relative_path"] or ":" in definition["root_relative_path"]:
        raise ValueError("Invalid portable dataset root.")
    identity = [(row["relative_path"], row["label_id"], row["content_sha256"]) for row in manifest["samples"]]
    classes = manifest["classes"]
    if (definition["dataset_id"] != ref["dataset_id"] or manifest["dataset_id"] != ref["dataset_id"]
            or manifest["revision"] != ref["revision"] or digest_json(identity) != ref["revision"]
            or split["dataset_ref"] != ref or split["classes"] != classes
            or digest_json(classes) != manifest["class_mapping_hash"] or split["class_mapping_hash"] != manifest["class_mapping_hash"]):
        raise ValueError("Portable dataset identity/class mapping mismatch.")
    if len(classes) < 2 or len(classes) != len(set(classes)) or set(split["splits"]) != {"train", "val", "test"}:
        raise ValueError("Invalid portable class/split metadata.")
    samples = {}
    hashes = set()
    for row in manifest["samples"]:
        path = Path(row["relative_path"])
        if path.is_absolute() or ".." in path.parts or "\\" in row["relative_path"] or ":" in row["relative_path"]:
            raise ValueError("Invalid portable image path.")
        index = row["label_id"]
        if (not isinstance(index, int) or not 0 <= index < len(classes) or classes[index] != row["label"]
                or row["relative_path"] in samples or row["content_sha256"] in hashes):
            raise ValueError("Invalid/duplicate portable sample.")
        samples[row["relative_path"]] = row
        hashes.add(row["content_sha256"])
    seen = set()
    for rows in split["splits"].values():
        if {row["label"] for row in rows} != set(classes):
            raise ValueError("Each split must contain all classes.")
        for row in rows:
            path = row["relative_path"]
            if path in seen or path not in samples or any(row.get(key) != samples[path][key] for key in ("label", "label_id", "size", "content_sha256")):
                raise ValueError("Portable split samples differ from manifest.")
            seen.add(path)
    assignments = {part: sorted((row["relative_path"], row["label_id"]) for row in rows) for part, rows in split["splits"].items()}
    expected = digest_json({"revision": ref["revision"], "assignments": assignments, "ratios": split["split_ratio"], "seed": split["seed"]})
    if seen != set(samples) or expected != ref["split_id"]:
        raise ValueError("Portable split identity mismatch.")


def resolve_dataset_runtime(root: Path, ref: DatasetRef | dict | None = None, legacy_name: str | None = None) -> dict:
    root = Path(root).resolve()
    if isinstance(ref, dict):
        ref = DatasetRef(**ref)
    if ref is None and legacy_name not in DATASETS:
        definition_path = contained_path(root, f"configs/datasets/{checked_id(legacy_name or '')}.json")
        definition = json.loads(definition_path.read_text(encoding="utf-8"))
        if not definition.get("default_revision") or not definition.get("default_split"):
            raise ValueError("Dataset is validated but has no ready split.")
        ref = DatasetRef(legacy_name, definition["default_revision"], definition["default_split"])
    if ref is not None:
        config = resolve_dataset(root, ref)
        split_file = Path(config.split_filename)
    else:
        if legacy_name not in DATASETS:
            raise ValueError("Unknown dataset.")
        config = DATASETS[legacy_name]
        split_file = root / "splits" / config.split_filename
    metadata = json.loads(split_file.read_text(encoding="utf-8"))
    if ref and (metadata.get("dataset_ref") != ref.to_dict() or metadata.get("dataset_fingerprint") != ref.revision):
        raise ValueError("Dataset split identity does not match the requested reference.")
    if ref:
        _, manifest, _ = _revision_paths(root, ref.dataset_id, ref.revision)
        assignments = {part: sorted((row["relative_path"], row["label_id"]) for row in rows)
                       for part, rows in metadata["splits"].items()}
        expected = digest_json({"revision": ref.revision, "assignments": assignments,
                                "ratios": metadata["split_ratio"], "seed": metadata["seed"]})
        samples = {row["relative_path"]: row for row in manifest["samples"]}
        seen = set()
        for rows in metadata["splits"].values():
            for row in rows:
                source = samples.get(row.get("relative_path"))
                if not source or row["relative_path"] in seen or any(row.get(key) != source[key] for key in ("label", "label_id", "content_sha256", "size")):
                    raise ValueError("Split samples differ from the dataset manifest.")
                seen.add(row["relative_path"])
        if expected != ref.split_id or metadata["classes"] != manifest["classes"] or seen != set(samples):
            raise ValueError("Immutable split/class mapping was modified.")
    classes = metadata["classes"]
    if not classes or len(classes) != len(set(classes)):
        raise ValueError("Invalid class mapping.")
    return {"dataset_ref": ref.to_dict() if ref else None, "split_file": str(split_file),
            "dataset_id": ref.dataset_id if ref else legacy_name,
            "dataset_revision": ref.revision if ref else None, "split_id": ref.split_id if ref else None,
            "class_names": classes, "class_mapping_hash": digest_json(classes),
            "identity_version": 2 if ref else 1, "provenance_level": "content_v2" if ref else "legacy",
            "dataset_fingerprint": ref.revision if ref else None}
