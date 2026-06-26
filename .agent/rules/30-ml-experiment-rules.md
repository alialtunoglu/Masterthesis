# Machine Learning Experiment Rules

All ML experiments must be reproducible and comparable.

Dataset rules:

1. Use the same split JSON for all models on the same dataset.
2. Use stratified train/validation/test split whenever possible.
3. Use seed=42 unless the user specifies otherwise.
4. Store split metadata inside the JSON file.
5. For PlantPathology2021, use only single-label samples in the first stage.

Training rules:

1. Always log dataset name, model name, number of classes, seed, batch size, optimizer, learning rate, scheduler, epoch count, and image size.
2. Save the best checkpoint according to validation macro F1 or validation accuracy.
3. Save final test metrics only after model selection using validation metrics.
4. Do not use the test set for model selection.
5. Use the same preprocessing and image size for fair comparisons unless intentionally comparing preprocessing strategies.
6. Validation and test transforms must not include random augmentation.
7. Train transforms may include augmentation.

Metrics:

For multi-class classification, report:

- accuracy
- macro precision
- macro recall
- macro F1
- weighted F1
- per-class precision/recall/F1 when useful
- confusion matrix when requested

Efficiency metrics:

- parameter count
- model size in MB
- inference time per image or batch
- FLOPs if implemented

Result files:

1. Save results as CSV under results/.
2. Save experiment config as JSON.
3. Save logs in a readable format.
4. Use consistent column names across experiments.

MLflow tracking:

1. All training and evaluation scripts should be designed to log parameters, metrics, and artifacts to MLflow.
2. Keep MLflow calls abstracted under src/tracking instead of scattering them through training loops.
3. Use MLflow for experiment UI tracking while preserving CSV/JSON outputs for thesis tables and reproducibility.
4. Do not commit mlruns/ or checkpoint files.
5. Each experiment should log dataset_name, model_name, seed, split_file, hyperparameters, metrics, and checkpoint_path.
