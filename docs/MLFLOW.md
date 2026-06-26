# MLflow Tracking

This project uses MLflow to inspect and compare experiment runs through a UI. CSV and JSON outputs are still saved separately because they are better suited for thesis tables, reproducibility records, and long-term archiving.

## Start the MLflow UI

Run from the project root:

```bash
mlflow ui --host 127.0.0.1 --port 5000
```

Then open:

```text
http://127.0.0.1:5000
```

Local MLflow runs are stored under `mlruns/`. This directory must not be committed to Git.

## Baseline Smoke Runs

These commands run only a few batches and are intended to validate the pipeline:

```bash
python src/training/train_baseline.py --dataset appleleaf9 --model mobilenet_v3_small --epochs 1 --batch-size 8 --max-train-batches 2 --max-val-batches 1 --max-test-batches 1
```

```bash
python src/training/train_baseline.py --dataset plantvillage --model mobilenet_v3_small --epochs 1 --batch-size 8 --max-train-batches 2 --max-val-batches 1 --max-test-batches 1
```

```bash
python src/training/train_baseline.py --dataset plantpathology2021 --model mobilenet_v3_small --epochs 1 --batch-size 8 --max-train-batches 2 --max-val-batches 1 --max-test-batches 1
```

The training script logs parameters, metrics, per-class metrics, result CSVs, and checkpoint paths through `src/tracking`. Checkpoints are saved under `checkpoints/`, which must also stay out of Git.

## Baseline Real Run

The AppleLeaf9 MobileNetV3-Small baseline can be started from config:

```bash
python src/training/train_baseline.py --config configs/baseline/mobilenetv3_small_appleleaf9.json
```

Command-line arguments override config values. For a smaller CPU-controlled run:

```bash
python src/training/train_baseline.py --config configs/baseline/mobilenetv3_small_appleleaf9.json --max-train-batches 100 --max-val-batches 30 --max-test-batches 30
```

The config file stores the default experiment settings. The final merged config used by the run is saved under `results/baseline/` and logged as an MLflow artifact.

## CUDA AppleLeaf9 Full Baseline

After installing CUDA-enabled PyTorch, verify the GPU:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

Dry-run:

```bash
train-baseline --config configs/baseline/mobilenetv3_small_appleleaf9.json --dry-run
```

Full 5-epoch CUDA baseline:

```bash
train-baseline --config configs/baseline/mobilenetv3_small_appleleaf9.json
```

If the console script is not available, use:

```bash
python src/training/train_baseline.py --config configs/baseline/mobilenetv3_small_appleleaf9.json
```

The earlier CPU mini experiment used `max_train_batches`, `max_val_batches`, and `max_test_batches` to keep runtime short. The CUDA full baseline does not use batch limits and processes the complete AppleLeaf9 train, validation, and test splits for 5 epochs. Run names include the device and mode, for example `baseline__appleleaf9__mobilenet_v3_small__seed42__cuda__full__epochs5`, so CPU mini runs and CUDA full runs remain separate.

## Logged MLflow Content

Parameters include:

- dataset name, model name, seed, split file, number of classes
- image size, batch size, epochs, optimizer, learning rate, weight decay
- pretrained flag, device, parameter count, and estimated model size

Metrics include:

- best validation accuracy and macro F1
- final test accuracy, macro precision, macro recall, macro F1, and weighted F1
- per-epoch train loss, validation loss, validation accuracy, and validation macro F1

Artifacts include:

- final config JSON
- training history CSV
- per-class metrics CSV
- confusion matrix CSV and PNG
- learning curve PNG
- best checkpoint
- cumulative `baseline_results.csv`

## `results/baseline/` Files

- `baseline_results.csv`: cumulative machine-readable summary of baseline runs.
- `{run_name}_config.json`: final merged config used by the run.
- `{run_name}_history.csv`: per-epoch training and validation metrics.
- `{run_name}_per_class_metrics.csv`: per-class test precision, recall, F1, and support.
- `{run_name}_confusion_matrix.csv`: raw test confusion matrix.
- `{run_name}_confusion_matrix.png`: heatmap visualization of the test confusion matrix.
- `{run_name}_learning_curves.png`: loss and validation metric curves.
