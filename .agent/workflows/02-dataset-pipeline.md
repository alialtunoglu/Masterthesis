# Workflow: Dataset Pipeline

Use this workflow for dataset analysis, split generation, and dataset loading.

Steps:

1. Inspect dataset paths.
2. Verify that the expected folders/files exist.
3. Count images and classes.
4. For CSV-based datasets, verify image filenames against the image folder.
5. Generate or load deterministic split JSON files.
6. Use stratified splitting.
7. Do not overwrite existing split files unless --overwrite is passed.
8. Print class distributions for train, val, and test.
9. Save analysis outputs to results/dataset_analysis/.
10. Keep raw datasets unchanged.

Expected datasets:

PlantVillage:
- path: datasets/PlantVillage/raw/color
- type: ImageFolder

AppleLeaf9:
- path: datasets/AppleLeaf9/raw
- type: ImageFolder

PlantPathology2021:
- images: datasets/PlantPathology2021/train_images
- csv: datasets/PlantPathology2021/train.csv
- type: CSV-based
- first stage: single-label samples only
