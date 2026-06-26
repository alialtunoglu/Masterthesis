# Dataset Preparation Notes

This thesis project compares knowledge distillation methods for deep learning models in plant leaf disease diagnosis.

## Existing Dataset Structure

The raw datasets are expected under:

```text
datasets/
  AppleLeaf9/
    raw/
      Alternaria leaf spot/
      Brown spot/
      Frogeye leaf spot/
      Grey spot/
      Health/
      Mosaic/
      Powdery mildew/
      Rust/
      Scab/
  PlantVillage/
    raw/
      color/
      grayscale/
      segmented/
  PlantPathology2021/
    train.csv
    train_images/
```

## Used Folders

- PlantVillage uses only `datasets/PlantVillage/raw/color`.
- AppleLeaf9 uses `datasets/AppleLeaf9/raw`.
- PlantPathology2021 uses `datasets/PlantPathology2021/train.csv` and `datasets/PlantPathology2021/train_images`.

The `grayscale` and `segmented` PlantVillage folders are intentionally ignored in this first stage.

## Why Raw Datasets Are Not Physically Split

The raw datasets are not moved, copied, renamed, or split into physical `train`, `val`, and `test` directories. Instead, deterministic JSON split files are created under `splits/`.

This keeps the original datasets intact, avoids duplicated images, and makes later experiments reproducible. Training scripts can read the same split files and use exactly the same samples across baseline, teacher, student, and distillation experiments.

## PlantPathology2021 Single-Label Policy

PlantPathology2021 contains rows whose `labels` value can include multiple space-separated labels. In the first project stage, only rows with exactly one label are used.

This keeps the initial setup as a standard single-label classification problem. Multi-label learning can be added later as a separate, explicit experimental setting.

## Analyze Datasets

Run:

```bash
python src/datasets/analyze_datasets.py
```

The script writes:

- `results/dataset_analysis/dataset_summary.csv`
- `results/dataset_analysis/class_distribution.csv`
- `results/dataset_analysis/plantpathology2021_label_report.csv`

It also prints a readable summary to the terminal and warns if an image listed in `train.csv` is missing from `train_images`.

## Create Splits

Run all dataset splits:

```bash
python src/datasets/create_splits.py --dataset all
```

Run a single dataset:

```bash
python src/datasets/create_splits.py --dataset plantvillage
python src/datasets/create_splits.py --dataset appleleaf9
python src/datasets/create_splits.py --dataset plantpathology2021
```

Regenerate existing split files:

```bash
python src/datasets/create_splits.py --dataset all --overwrite
```

The split ratio is fixed at 70% train, 15% validation, and 15% test with seed `42`. Split JSON files store image paths relative to the project root.
