"""Create an independent, content-verified dataset; never modify legacy data."""

from collections import defaultdict
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datasets.dataset_registration import (
    _linked, atomic_json, contained_path, digest_json, register_dataset_revision,
    registry_lock, scan_dataset, sha256_file,
)
from datasets.dataset_registry import DatasetDefinition
from datasets.dataset_runtime import create_registered_split

DATASET_ID = "plantpathology2021_clean"
SOURCE_SPLIT = "splits/plantpathology2021_seed42_70_15_15.json"
DESTINATION = "datasets/PlantPathology2021Clean"


def prepare_clean_dataset(root: Path, progress=print) -> dict:
    root = Path(root).resolve()
    source = root / SOURCE_SPLIT
    metadata = json.loads(source.read_text(encoding="utf-8"))
    classes = metadata["classes"]
    if classes != sorted(set(classes), key=lambda value: (value.casefold(), value)):
        raise ValueError("Source class mapping must match the imagefolder class order.")
    by_hash, seen = defaultdict(list), set()
    for rows in metadata["splits"].values():
        for row in rows:
            path = contained_path(root, row["path"])
            label_id = row["label_id"]
            if (not isinstance(label_id, int) or isinstance(label_id, bool) or not 0 <= label_id < len(classes)
                    or row["path"] in seen or classes[label_id] != row["label"]):
                raise ValueError("Invalid/duplicate source path or class mapping.")
            seen.add(row["path"])
            by_hash[sha256_file(path)].append(dict(row))
    kept, excluded = [], []
    for digest, rows in sorted(by_hash.items()):
        rows.sort(key=lambda row: row["path"])
        if len({row["label"] for row in rows}) > 1:
            excluded.extend({**row, "content_sha256": digest, "reason": "conflicting_labels"} for row in rows)
        else:
            kept.append({**rows[0], "content_sha256": digest})
            excluded.extend({**row, "content_sha256": digest, "reason": "duplicate_same_label"} for row in rows[1:])
    identity = {"source_split": SOURCE_SPLIT, "source_split_sha256": sha256_file(source),
                "source_content_fingerprint": digest_json(sorted((row["path"], row["label_id"], digest) for digest, rows in by_hash.items() for row in rows))}
    destination = contained_path(root, DESTINATION)
    with registry_lock(root, DATASET_ID + "_preparation"):
        marker = destination / ".preparation.json"
        if destination.exists():
            if destination.is_symlink() or destination.resolve() != destination.absolute() or not marker.is_file():
                raise ValueError("Destination is not an owned preparation directory.")
            if json.loads(marker.read_text(encoding="utf-8")) != identity:
                raise ValueError("Source changed; destination belongs to another source revision.")
        else:
            destination.mkdir(parents=True)
            atomic_json(marker, identity)
        for index, row in enumerate(kept, 1):
            target = contained_path(destination, f"images/{row['label']}/{Path(row['path']).name}")
            if any((path.exists() or path.is_symlink()) and _linked(path)
                   for path in (target, target.parent, target.parent.parent)):
                raise ValueError("Destination image links/junctions are not allowed.")
            if target.exists():
                if target.is_symlink() or sha256_file(target) != row["content_sha256"]:
                    raise ValueError(f"Copied image differs; will not overwrite: {target}")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                # shortcut: interrupted partial copies fail safely on retry; atomic copy staging is deferred.
                shutil.copyfile(contained_path(root, row["path"]), target)
                if sha256_file(target) != row["content_sha256"]:
                    raise ValueError(f"Source changed while copying: {row['path']}")
            if progress and index % 3000 == 0:
                progress(f"COPY_VERIFIED {index}/{len(kept)}", flush=True)
        expected = {f"images/{row['label']}/{Path(row['path']).name}" for row in kept}
        actual = {p.relative_to(destination).as_posix() for p in (destination / "images").rglob("*") if p.is_file()}
        if actual != expected:
            raise ValueError("Destination inventory differs from the approved selection.")
        if progress:
            progress("FULL_PIXEL_VALIDATION scanning clean imagefolder", flush=True)
        scan = scan_dataset(root, destination, "imagefolder")
        definition = DatasetDefinition(DATASET_ID, "PlantPathology2021 Clean", DESTINATION, "imagefolder")
        if progress:
            progress("REGISTER_REVISION revalidating source content", flush=True)
        revision = register_dataset_revision(root, definition, scan)
        if progress:
            progress("CREATE_PINNED_SPLIT revalidating content; seed=42", flush=True)
        ref = create_registered_split(root, DATASET_ID, revision.revision, {"train": .7, "val": .15, "test": .15}, 42)
        result = {**identity, "dataset_ref": ref.to_dict(), "source_count": len(seen),
                  "kept_count": len(kept), "excluded_count": len(excluded), "excluded": excluded,
                  "policy": "Exclude all conflicting-label records; retain one canonical copy of same-label duplicates. Never relabel or delete source images."}
        atomic_json(destination / "curation.json", result)
        return result


if __name__ == "__main__":
    print(json.dumps(prepare_clean_dataset(ROOT), ensure_ascii=False, indent=2))
