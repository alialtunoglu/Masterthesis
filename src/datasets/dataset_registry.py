"""Central dataset configuration for the thesis project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
