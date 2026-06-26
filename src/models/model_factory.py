"""Factory for baseline student models."""

from __future__ import annotations

import torch.nn as nn
from torchvision import models


SUPPORTED_MODELS = {
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "efficientnet_b0",
    "resnet18",
}


def _weights_or_none(weights_enum, pretrained: bool):
    return weights_enum.DEFAULT if pretrained else None


def _replace_classifier(model: nn.Module, num_classes: int, model_name: str) -> nn.Module:
    if model_name.startswith("mobilenet_v3"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    if model_name == "efficientnet_b0":
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    if model_name == "resnet18":
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Unsupported model for classifier replacement: {model_name}")


def create_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Create a supported torchvision baseline model with a custom classifier head."""
    normalized_name = model_name.lower().strip()
    if normalized_name not in SUPPORTED_MODELS:
        supported = ", ".join(sorted(SUPPORTED_MODELS))
        raise ValueError(f"Unsupported model_name '{model_name}'. Supported models: {supported}")

    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}")

    if normalized_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(
            weights=_weights_or_none(models.MobileNet_V3_Small_Weights, pretrained)
        )
    elif normalized_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(
            weights=_weights_or_none(models.MobileNet_V3_Large_Weights, pretrained)
        )
    elif normalized_name == "efficientnet_b0":
        model = models.efficientnet_b0(
            weights=_weights_or_none(models.EfficientNet_B0_Weights, pretrained)
        )
    elif normalized_name == "resnet18":
        model = models.resnet18(weights=_weights_or_none(models.ResNet18_Weights, pretrained))
    else:
        raise ValueError(f"Unsupported model_name '{model_name}'")

    return _replace_classifier(model, num_classes, normalized_name)
