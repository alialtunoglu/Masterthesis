# Skill: Experiment Tracking

For every experiment or dataset operation:

1. Save configuration.
2. Save metrics.
3. Save warnings.
4. Save output file paths.
5. Use consistent names.

Naming conventions:

Dataset names:
- plantvillage
- appleleaf9
- plantpathology2021

Split files:
- splits/plantvillage_seed42_70_15_15.json
- splits/appleleaf9_seed42_70_15_15.json
- splits/plantpathology2021_seed42_70_15_15.json

Results:
- results/dataset_analysis/
- results/baseline/
- results/teachers/
- results/kd_single/
- results/kd_multi/
- results/quantization/

Checkpoints:
- checkpoints/{dataset_name}/{model_name}/

MLflow:
- Use src/tracking for MLflow setup and logging helpers.
- MLflow is for the interactive experiment UI; CSV/JSON outputs must still be saved.
- Log dataset_name, model_name, seed, split_file, hyperparameters, metrics, artifacts, and checkpoint_path.
- Never commit mlruns/, mlruns.db, or checkpoint files.
