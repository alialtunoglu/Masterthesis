"""Read-only image validation and explicit, immutable dataset registration."""

from contextlib import contextmanager
from dataclasses import dataclass, asdict
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import time

from PIL import Image

from .dataset_registry import DATASETS, DatasetDefinition, DatasetRevisionRef
from .imagefolder_utils import IMAGE_EXTENSIONS


def checked_id(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,127}", value):
        raise ValueError("Invalid dataset identity.")
    return value


def contained_path(root: Path, relative: str | Path) -> Path:
    raw = relative.as_posix() if isinstance(relative, Path) else str(relative)
    relative = Path(raw)
    if relative.is_absolute() or relative.drive or ".." in relative.parts or "\\" in raw:
        raise ValueError("Path must be project-relative.")
    path = Path(root) / relative
    if not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("Path escapes the project root.")
    return path


def digest_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".dataset-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def registry_lock(root: Path, dataset_id: str):
    checked_id(dataset_id)
    path = Path(root) / "configs/datasets" / f".{dataset_id}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as lock:
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
                if time.monotonic() > deadline:
                    raise TimeoutError("Dataset registry is busy; retry registration.")
                time.sleep(.05)
        yield


def _linked(path: Path) -> bool:
    return path.is_symlink() or bool(getattr(path.lstat(), "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 1024))


def discover_datasets(root: Path) -> list[Path]:
    directory = Path(root) / "datasets"
    if not directory.is_dir():
        return []
    from .dataset_registry import list_datasets
    known = {item.root_relative_path for item in list_datasets(root)}
    return [path for path in sorted(directory.iterdir()) if path.is_dir() and not path.name.startswith(".")
            and not _linked(path) and path.relative_to(root).as_posix() not in known
            and ((path / "images").is_dir() or (path / "train").is_dir())]


@dataclass(frozen=True)
class DatasetScan:
    root_relative_path: str
    layout: str
    classes: list[str]
    samples: list[dict]
    revision: str
    class_mapping_hash: str


def scan_dataset(root: Path, candidate: Path, layout: str, progress=None) -> DatasetScan:
    root = Path(root).resolve()
    candidate = Path(candidate)
    if not candidate.is_absolute():
        candidate = root / candidate
    if not candidate.resolve().is_relative_to((root / "datasets").resolve()) or candidate.resolve() == (root / "datasets").resolve():
        raise ValueError("Dataset must be a child of datasets/.")
    for parent in (candidate, *candidate.parents):
        if parent == root:
            break
        if _linked(parent):
            raise ValueError("Dataset links/junctions are not allowed.")
    if layout not in {"imagefolder", "presplit"}:
        raise ValueError("Choose imagefolder or presplit layout.")
    if (candidate / "images").is_dir() and (candidate / "train").is_dir():
        raise ValueError("Ambiguous dataset layout; use only one template.")
    parts = [(None, candidate / "images")] if layout == "imagefolder" else [(part, candidate / part) for part in ("train", "val", "test")]
    classes = None
    samples, hashes = [], {}
    for split, directory in parts:
        if not directory.is_dir():
            raise ValueError(f"Missing dataset directory: {directory.name}")
        if _linked(directory) or not directory.resolve().is_relative_to(candidate.resolve()):
            raise ValueError("Dataset layout links/junctions or root escapes are not allowed.")
        folders = sorted([path for path in directory.iterdir() if path.is_dir() and not path.name.startswith(".")], key=lambda path: (path.name.casefold(), path.name))
        names = [path.name for path in folders]
        if len(names) < 2 or len(set(name.casefold() for name in names)) != len(names):
            raise ValueError("At least two distinct non-empty classes are required; case collisions are invalid.")
        if classes is not None and names != classes:
            raise ValueError("Class folders must match in train/val/test.")
        classes = names
        for index, folder in enumerate(folders):
            found = 0
            for current, dirs, files in os.walk(folder, followlinks=False):
                current = Path(current)
                if _linked(current) or any(_linked(current / name) for name in dirs):
                    raise ValueError("Dataset links/junctions are not allowed.")
                dirs[:] = sorted(name for name in dirs if not name.startswith("."))
                for name in sorted(files):
                    path = current / name
                    if name.startswith(".") or path.suffix.lower() not in IMAGE_EXTENSIONS:
                        continue
                    if _linked(path):
                        raise ValueError("Dataset links are not allowed.")
                    try:
                        with Image.open(path) as image:
                            image.load()
                    except (OSError, ValueError) as error:
                        raise ValueError(f"Unreadable image: {path.relative_to(candidate)}") from error
                    content_hash = sha256_file(path)
                    if content_hash in hashes:
                        raise ValueError(f"Duplicate image: {hashes[content_hash]} and {path.relative_to(candidate)}")
                    hashes[content_hash] = path.relative_to(candidate).as_posix()
                    row = {"path": path.relative_to(root).as_posix(), "relative_path": path.relative_to(candidate).as_posix(),
                           "label": folder.name, "label_id": index, "size": path.stat().st_size, "content_sha256": content_hash}
                    if split:
                        row["split"] = split
                    samples.append(row)
                    found += 1
                    if progress:
                        progress(len(samples), row["relative_path"])
            if not found:
                raise ValueError(f"Empty class: {folder.name}")
    samples.sort(key=lambda row: row["relative_path"])
    identity = [(row["relative_path"], row["label_id"], row["content_sha256"]) for row in samples]
    return DatasetScan(candidate.relative_to(root).as_posix(), layout, classes, samples,
                       digest_json(identity), digest_json(classes))


def register_dataset_revision(root: Path, definition: DatasetDefinition, scan: DatasetScan) -> DatasetRevisionRef:
    root = Path(root).resolve()
    checked_id(definition.dataset_id)
    if definition.dataset_id in DATASETS or definition.task_type != "single_label_classification" or definition.schema_version != 2:
        raise ValueError("Unsupported or reserved dataset definition.")
    if definition.root_relative_path != scan.root_relative_path or definition.layout != scan.layout:
        raise ValueError("Definition does not match the scanned dataset.")
    candidate = contained_path(root, definition.root_relative_path)
    with registry_lock(root, definition.dataset_id):
        if scan_dataset(root, candidate, definition.layout).revision != scan.revision:
            raise ValueError("Dataset changed after validation; scan again.")
        descriptor = root / "configs/datasets" / f"{definition.dataset_id}.json"
        if descriptor.exists():
            existing = json.loads(descriptor.read_text(encoding="utf-8"))
            if existing["root_relative_path"] != definition.root_relative_path or existing["layout"] != definition.layout:
                raise ValueError("Dataset ID is already bound to a different location/layout.")
        else:
            atomic_json(descriptor, {**asdict(definition), "ready": False, "default_revision": None, "default_split": None})
        target = root / "datasets/.registry" / definition.dataset_id / scan.revision / "manifest.json"
        payload = {"schema_version": 2, "dataset_id": definition.dataset_id, **asdict(scan)}
        if target.exists():
            if json.loads(target.read_text(encoding="utf-8")) != payload:
                raise ValueError("Immutable dataset revision already exists with different metadata.")
        else:
            atomic_json(target, payload)
    return DatasetRevisionRef(definition.dataset_id, scan.revision)
