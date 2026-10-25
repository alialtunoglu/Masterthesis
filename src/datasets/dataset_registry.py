"""Central dataset configuration for the thesis project."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path


from src.utils.project import get_project_root

PROJECT_ROOT = get_project_root()
SEED = 42
SPLIT_RATIO = {"train": 0.70, "val": 0.15, "test": 0.15}


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    kind: str
    path: Path
    csv_path: Path | None = None
    image_dir: Path | None = None
    split_filename: str | None = None
    single_label_only: bool = False
    dataset_ref: dict | None = None
    manifest_file: str | None = None

    @property
    def absolute_path(self) -> Path:
        return PROJECT_ROOT / self.path

    @property
    def absolute_csv_path(self) -> Path | None:
        return PROJECT_ROOT / self.csv_path if self.csv_path else None

    @property
    def absolute_image_dir(self) -> Path | None:
        return PROJECT_ROOT / self.image_dir if self.image_dir else None


DATASETS: dict[str, DatasetConfig] = {
    "plantvillage": DatasetConfig(
        name="PlantVillage",
        kind="imagefolder",
        path=Path("datasets/PlantVillage/raw/color"),
        split_filename="plantvillage_seed42_70_15_15.json",
    ),
    "appleleaf9": DatasetConfig(
        name="AppleLeaf9",
        kind="imagefolder",
        path=Path("datasets/AppleLeaf9/raw"),
        split_filename="appleleaf9_seed42_70_15_15.json",
    ),
    "plantpathology2021": DatasetConfig(
        name="PlantPathology2021",
        kind="plantpathology2021",
        path=Path("datasets/PlantPathology2021"),
        csv_path=Path("datasets/PlantPathology2021/train.csv"),
        image_dir=Path("datasets/PlantPathology2021/train_images"),
        split_filename="plantpathology2021_seed42_70_15_15.json",
        single_label_only=True,
    ),
}

SPLITS_DIR = PROJECT_ROOT / "splits"
DATASET_ANALYSIS_DIR = PROJECT_ROOT / "results" / "dataset_analysis"


@dataclass(frozen=True)
class DatasetRevisionRef:
    dataset_id: str
    revision: str


@dataclass(frozen=True)
class DatasetRef(DatasetRevisionRef):
    split_id: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DatasetDefinition:
    dataset_id: str
    display_name: str
    root_relative_path: str
    layout: str
    task_type: str = "single_label_classification"
    schema_version: int = 2
    default_revision: str | None = None
    default_split: str | None = None
    ready: bool = False


def list_datasets(root: Path = PROJECT_ROOT, *, include_legacy: bool = True) -> list[DatasetDefinition]:
    root = Path(root)
    result = []
    if include_legacy:
        result = [DatasetDefinition(key, config.name, config.path.as_posix(), "legacy",
                                    ready=(root / "splits" / config.split_filename).is_file())
                  for key, config in DATASETS.items()]
    for path in sorted((root / "configs/datasets").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        definition = DatasetDefinition(**payload)
        if definition.dataset_id in DATASETS:
            raise ValueError("Custom dataset ID conflicts with a built-in dataset.")
        revision, split = definition.default_revision, definition.default_split
        ready = bool(revision and split and (root / "datasets/.registry" / definition.dataset_id /
                                             revision / "splits" / f"{split}.json").is_file())
        result.append(DatasetDefinition(**{**payload, "ready": ready}))
    return result


def resolve_dataset(root: Path, ref: DatasetRef) -> DatasetConfig:
    from .dataset_registration import checked_id, contained_path
    checked_id(ref.dataset_id)
    checked_id(ref.revision)
    checked_id(ref.split_id)
    root = Path(root).resolve()
    descriptor = contained_path(root, f"configs/datasets/{ref.dataset_id}.json")
    definition = DatasetDefinition(**json.loads(descriptor.read_text(encoding="utf-8")))
    base = contained_path(root, f"datasets/.registry/{ref.dataset_id}/{ref.revision}")
    split = base / "splits" / f"{ref.split_id}.json"
    manifest = base / "manifest.json"
    if not split.is_file() or not manifest.is_file():
        raise ValueError("Dataset revision/split is not ready.")
    contained_path(root, definition.root_relative_path)
    return DatasetConfig(definition.display_name, "imagefolder", Path(definition.root_relative_path),
                         split_filename=str(split), dataset_ref=ref.to_dict(), manifest_file=str(manifest))


def get_dataset_catalog(root: Path = PROJECT_ROOT) -> dict[str, DatasetConfig]:
    result = dict(DATASETS)
    for definition in list_datasets(root, include_legacy=False):
        if definition.ready:
            ref = DatasetRef(definition.dataset_id, definition.default_revision, definition.default_split)
            result[definition.dataset_id] = resolve_dataset(root, ref)
    return result
