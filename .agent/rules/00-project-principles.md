# Project Principles

This project is a master's thesis project on:

"Comparative Analysis of Knowledge Distillation Methods for Deep Learning Models in Plant Leaf Disease Diagnosis"

The codebase must be designed for reproducible machine learning experiments.

Core principles:

1. Never modify, move, rename, delete, or duplicate raw datasets unless explicitly instructed.
2. Keep raw datasets immutable.
3. Use deterministic seeds for all dataset splits and experiments.
4. Store train/validation/test splits as JSON files.
5. Do not physically split image folders unless explicitly requested.
6. Keep experiment outputs organized under results/ and checkpoints/.
7. Every training or evaluation script must produce machine-readable outputs, preferably CSV or JSON.
8. All model comparisons must be fair and use the same dataset split for the same dataset.
9. Prefer readable, maintainable, modular code over quick scripts.
10. Do not introduce unnecessary complexity before the baseline pipeline works.

Current datasets:

- PlantVillage:
  - use only datasets/PlantVillage/raw/color
  - ImageFolder-style multi-class dataset

- AppleLeaf9:
  - use datasets/AppleLeaf9/raw
  - ImageFolder-style multi-class dataset

- PlantPathology2021:
  - use datasets/PlantPathology2021/train.csv
  - use datasets/PlantPathology2021/train_images
  - initially use only single-label samples
  - exclude multi-label samples for the first multi-class classification pipeline

Do not start model training unless the user explicitly asks.
