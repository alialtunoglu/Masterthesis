# Skill: PyTorch ML Development

When writing PyTorch code:

1. Use torch, torchvision, and timm when appropriate.
2. Keep dataset, model, training, and evaluation code separate.
3. Use DataLoader with configurable batch_size and num_workers.
4. Set random seeds for reproducibility.
5. Move tensors and models to the selected device.
6. Use torch.no_grad() during validation and testing.
7. Use model.train() and model.eval() correctly.
8. Do not apply random transforms to validation or test sets.
9. Save checkpoints with model state_dict, config, class names, and metrics.
10. Keep model factories flexible for different num_classes.

For multi-class classification, use CrossEntropyLoss.
For multi-label classification, use BCEWithLogitsLoss, but do not introduce multi-label training unless requested.
