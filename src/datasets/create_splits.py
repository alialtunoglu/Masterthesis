"""Create deterministic stratified split JSON files without moving images."""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
for path in (SRC_DIR, CURRENT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dataset_registry import DATASETS, PROJECT_ROOT, SEED, SPLIT_RATIO, SPLITS_DIR
from imagefolder_utils import collect_imagefolder_samples
from plantpathology_utils import collect_plantpathology_samples
from utils.io import ensure_dir, save_json
from utils.seed import set_seed


def largest_remainder_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    raw_counts = {name: total * ratio for name, ratio in ratios.items()}
    counts = {name: int(value) for name, value in raw_counts.items()}
    remaining = total - sum(counts.values())
    remainders = sorted(
        ((name, raw_counts[name] - counts[name]) for name in ratios),
        key=lambda item: (-item[1], item[0]),
    )
    for index in range(remaining):
        counts[remainders[index % len(remainders)][0]] += 1
    return counts


def stratified_split(samples: list[dict], seed: int) -> dict[str, list[dict]]:
    by_label: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        by_label[sample["label"]].append(sample)

    splits = {"train": [], "val": [], "test": []}
    rng = random.Random(seed)

    for label in sorted(by_label):
        class_samples = sorted(by_label[label], key=lambda sample: sample["path"])
        rng.shuffle(class_samples)
        counts = largest_remainder_counts(len(class_samples), SPLIT_RATIO)

        train_end = counts["train"]
        val_end = train_end + counts["val"]
        splits["train"].extend(class_samples[:train_end])
        splits["val"].extend(class_samples[train_end:val_end])
        splits["test"].extend(class_samples[val_end:])

    for split_name in splits:
        splits[split_name] = sorted(splits[split_name], key=lambda sample: (sample["label_id"], sample["path"]))

    return splits


def load_samples(dataset_key: str) -> tuple[list[dict], list[str], list[str]]:
    config = DATASETS[dataset_key]
    if config.kind == "imagefolder":
        samples, classes = collect_imagefolder_samples(config.absolute_path, PROJECT_ROOT)
        return samples, classes, []

    if config.kind == "plantpathology2021":
        samples, classes, missing_images = collect_plantpathology_samples(
            config.absolute_csv_path,
            config.absolute_image_dir,
            PROJECT_ROOT,
        )
        return samples, classes, missing_images

    raise ValueError(f"Unsupported dataset kind: {config.kind}")


def build_split_payload(dataset_key: str) -> tuple[dict, list[str]]:
    config = DATASETS[dataset_key]
    samples, classes, missing_images = load_samples(dataset_key)
    splits = stratified_split(samples, SEED)
    payload = {
        "dataset_name": config.name,
        "dataset_path": config.path.as_posix(),
        "seed": SEED,
        "split_ratio": SPLIT_RATIO,
        "num_classes": len(classes),
        "classes": classes,
        "total_samples": len(samples),
        "splits": splits,
    }
    return payload, missing_images


def print_split_summary(payload: dict, missing_images: list[str]) -> None:
    print(f"\n{payload['dataset_name']}")
    print("-" * 80)
    for split_name in ("train", "val", "test"):
        split_samples = payload["splits"][split_name]
        counts = Counter(sample["label"] for sample in split_samples)
        print(f"{split_name}: {len(split_samples)} samples")
        for class_name in payload["classes"]:
            print(f"  - {class_name}: {counts[class_name]}")

    if missing_images:
        print(f"WARNING: skipped {len(missing_images)} CSV rows because images were missing from train_images.")


def create_split(dataset_key: str, overwrite: bool) -> Path:
    config = DATASETS[dataset_key]
    output_path = SPLITS_DIR / config.split_filename

    if output_path.exists() and not overwrite:
        print(f"\nSkipping existing split file: {output_path.relative_to(PROJECT_ROOT).as_posix()}")
        print("Use --overwrite to regenerate it.")
        return output_path

    payload, missing_images = build_split_payload(dataset_key)
    save_json(payload, output_path)
    print_split_summary(payload, missing_images)
    print(f"Saved: {output_path.relative_to(PROJECT_ROOT).as_posix()}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create deterministic dataset split JSON files.")
    parser.add_argument(
        "--dataset",
        choices=["all", "plantvillage", "appleleaf9", "plantpathology2021"],
        default="all",
        help="Dataset split to create.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing split files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(SEED)
    ensure_dir(SPLITS_DIR)

    dataset_keys = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    for dataset_key in dataset_keys:
        create_split(dataset_key, args.overwrite)


if __name__ == "__main__":
    main()
