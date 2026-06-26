# Workflow: Training Experiment

Use this workflow only when the user explicitly asks for training code or training execution.

Steps:

1. Load dataset definition from dataset_registry.py or config.
2. Load the existing split JSON.
3. Create train, validation, and test datasets from the same split.
4. Use train transforms with augmentation.
5. Use validation/test transforms without random augmentation.
6. Create model through a model factory.
7. Train using a reusable training loop.
8. Select best checkpoint using validation metric.
9. Evaluate once on the test set after training.
10. Save:
    - checkpoint
    - metrics CSV
    - config JSON
    - optional confusion matrix
11. Log parameters, metrics, artifacts, and checkpoint paths through src/tracking MLflow helpers.
12. Keep CSV/JSON results even when MLflow logging is enabled.

Do not train all models at once unless explicitly requested.

Start with a small baseline model to validate the pipeline before large teacher models.
