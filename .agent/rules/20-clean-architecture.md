# Clean Architecture Rules

The project should follow a modular architecture.

Preferred structure:

src/
  datasets/
    dataset loading, dataset analysis, split creation

  models/
    model factory, CNN models, ViT models, student models

  training/
    baseline training, teacher training, KD training

  kd/
    knowledge distillation losses and methods

  evaluation/
    metrics, inference speed, model size, evaluation reports

  utils/
    seed, io, logging, device, reproducibility helpers

configs/
  experiment configuration files

results/
  CSV, JSON, tables, figures

checkpoints/
  trained model weights

Architecture rules:

1. Dataset logic must not be mixed with model training logic.
2. Model creation must be handled through model factory functions.
3. Training loops must be reusable across models and datasets.
4. Metrics must be calculated in separate evaluation utilities.
5. Result writing must be centralized.
6. Avoid hardcoded dataset paths inside training scripts.
7. Use dataset_registry.py or config files for dataset definitions.
8. Keep command-line scripts thin; move logic into reusable functions.
9. Avoid circular imports.
10. Any new component should be easy to test independently.

When adding new code, think about future stages:

- baseline student training
- CNN teacher training
- ViT teacher training
- single-teacher KD
- multi-teacher KD
- adaptive temperature KD
- self-distillation
- quantization
