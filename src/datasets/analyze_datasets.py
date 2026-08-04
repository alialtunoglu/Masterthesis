"""Analyze raw datasets without modifying them."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
PROJECT_ROOT_BOOTSTRAP = SRC_DIR.parent
for path in (SRC_DIR, CURRENT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dataset_registry import DATASET_ANALYSIS_DIR, DATASETS, PROJECT_ROOT
from imagefolder_utils import collect_imagefolder_samples, count_by_class
from plantpathology_utils import (
    collect_plantpathology_samples,
    list_image_files,
    load_train_csv,
    summarize_labels,
)
from utils.io import ensure_dir, save_csv


def analyze_imagefolder(dataset_key: str) -> tuple[dict, list[dict]]:
    config = DATASETS[dataset_key]
    samples, classes = collect_imagefolder_samples(config.absolute_path, PROJECT_ROOT)
    counts = count_by_class(samples)
    summary = {
        "dataset_name": config.name,
        "dataset_path": config.path.as_posix(),
        "total_images": len(samples),
        "num_classes": len(classes),
        "total_csv_rows": "",
        "available_image_files": "",
        "single_label_samples": "",
        "multi_label_samples": "",
        "missing_images_in_folder": "",
        "notes": "imagefolder",
    }
    distribution = [
        {"dataset_name": config.name, "class_name": class_name, "count": counts[class_name]}
        for class_name in classes
    ]
    return summary, distribution


def analyze_plantpathology2021() -> tuple[dict, list[dict], list[dict], list[str]]:
    config = DATASETS["plantpathology2021"]
    dataframe = load_train_csv(config.absolute_csv_path)
    label_summary = summarize_labels(dataframe)
    image_files = list_image_files(config.absolute_image_dir)
    samples, classes, missing_images = collect_plantpathology_samples(
        config.absolute_csv_path,
        config.absolute_image_dir,
        PROJECT_ROOT,
    )
    counts = Counter(sample["label"] for sample in samples)

    summary = {
        "dataset_name": config.name,
        "dataset_path": config.path.as_posix(),
        "total_images": len(samples),
        "num_classes": len(classes),
        "total_csv_rows": label_summary["total_csv_rows"],
        "available_image_files": len(image_files),
        "single_label_samples": label_summary["single_label_count"],
        "multi_label_samples": label_summary["multi_label_count"],
        "missing_images_in_folder": len(missing_images),
        "notes": "only single-label samples are used for first-stage splits",
    }

    distribution = [
        {"dataset_name": config.name, "class_name": class_name, "count": counts[class_name]}
        for class_name in classes
    ]

    report_rows = [
        {"section": "summary", "item": "total_csv_rows", "value": label_summary["total_csv_rows"]},
        {"section": "summary", "item": "available_image_files", "value": len(image_files)},
        {"section": "summary", "item": "single_label_samples", "value": label_summary["single_label_count"]},
        {"section": "summary", "item": "multi_label_samples", "value": label_summary["multi_label_count"]},
        {"section": "summary", "item": "empty_label_samples", "value": label_summary["empty_label_count"]},
        {"section": "summary", "item": "missing_images_in_folder", "value": len(missing_images)},
    ]
    report_rows.extend(
        {"section": "unique_label", "item": label, "value": ""}
        for label in label_summary["unique_labels"]
    )
    report_rows.extend(
        {"section": "single_label_class_count", "item": class_name, "value": label_summary["single_label_class_counts"][class_name]}
        for class_name in sorted(label_summary["single_label_class_counts"])
    )
    report_rows.extend(
        {"section": "missing_image", "item": image_name, "value": ""}
        for image_name in missing_images
    )

    return summary, distribution, report_rows, missing_images


def print_summary(summary_rows: list[dict], distribution_rows: list[dict], missing_images: list[str]) -> None:
    print("\nDataset analysis summary")
    print("=" * 80)
    for row in summary_rows:
        print(f"\n{row['dataset_name']}")
        print(f"  path: {row['dataset_path']}")
        print(f"  total images used: {row['total_images']}")
        print(f"  classes: {row['num_classes']}")
        if row["dataset_name"] == "PlantPathology2021":
            print(f"  total CSV rows: {row['total_csv_rows']}")
            print(f"  files in train_images: {row['available_image_files']}")
            print(f"  single-label rows: {row['single_label_samples']}")
            print(f"  multi-label rows: {row['multi_label_samples']}")
            print(f"  missing CSV images: {row['missing_images_in_folder']}")

        print("  class distribution:")
        for class_row in [item for item in distribution_rows if item["dataset_name"] == row["dataset_name"]]:
            print(f"    - {class_row['class_name']}: {class_row['count']}")

    if missing_images:
        print("\nWARNING: train.csv contains images missing from train_images:")
        for image_name in missing_images[:20]:
            print(f"  - {image_name}")
        if len(missing_images) > 20:
            print(f"  ... and {len(missing_images) - 20} more")


def main() -> None:
    ensure_dir(DATASET_ANALYSIS_DIR)

    summary_rows: list[dict] = []
    distribution_rows: list[dict] = []

    for dataset_key in ("plantvillage", "appleleaf9"):
        summary, distribution = analyze_imagefolder(dataset_key)
        summary_rows.append(summary)
        distribution_rows.extend(distribution)

    plant_summary, plant_distribution, plant_report, missing_images = analyze_plantpathology2021()
    summary_rows.append(plant_summary)
    distribution_rows.extend(plant_distribution)

    save_csv(summary_rows, DATASET_ANALYSIS_DIR / "dataset_summary.csv")
    save_csv(distribution_rows, DATASET_ANALYSIS_DIR / "class_distribution.csv")
    save_csv(plant_report, DATASET_ANALYSIS_DIR / "plantpathology2021_label_report.csv")

    print_summary(summary_rows, distribution_rows, missing_images)
    print(f"\nSaved analysis files to: {DATASET_ANALYSIS_DIR.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
