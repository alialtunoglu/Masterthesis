"""Factory for supported student and teacher models."""

from __future__ import annotations

import torch.nn as nn
from torchvision import models


STUDENT_MODELS = {
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "efficientnet_b0",
    "resnet18",
}

VISION_TRANSFORMER_TEACHER_MODELS = {
    "swin_v2_t",
    "maxvit_t",
    "vit_b_16",
    "dinov2_vitb14",
}

CNN_TEACHER_MODELS = {
    "resnet50",
    "resnet101",
    "densenet121",
    "densenet201",
    "vgg19_bn",
    "efficientnet_b3",
    "efficientnet_b4",
    "convnext_tiny",
    "convnext_base",
    "regnet_y_8gf",
}

SUPPORTED_MODELS = STUDENT_MODELS | CNN_TEACHER_MODELS | VISION_TRANSFORMER_TEACHER_MODELS


def _weights_or_none(weights_enum, pretrained: bool):
    return weights_enum.DEFAULT if pretrained else None


def _replace_classifier(model: nn.Module, num_classes: int, model_name: str) -> nn.Module:
    if model_name.startswith("mobilenet_v3"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    if model_name in {"efficientnet_b0", "efficientnet_b3", "efficientnet_b4"}:
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    if model_name in {"resnet18", "resnet50", "resnet101", "regnet_y_8gf"}:
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    if model_name in {"densenet121", "densenet201"}:
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        return model

    if model_name == "vgg19_bn":
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    if model_name in {"convnext_tiny", "convnext_base"}:
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    if model_name == "swin_v2_t":
        model.head = nn.Linear(model.head.in_features, num_classes)
        return model

    if model_name == "maxvit_t":
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model

    if model_name == "vit_b_16":
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        return model

    raise ValueError(f"Unsupported model for classifier replacement: {model_name}")


def get_model_type(model_name: str) -> str:
    """Return the project model type for a supported model."""
    normalized_name = model_name.lower().strip()
    if normalized_name in STUDENT_MODELS:
        return "baseline_student"
    if normalized_name in CNN_TEACHER_MODELS:
        return "cnn_teacher"
    if normalized_name in VISION_TRANSFORMER_TEACHER_MODELS:
        return "vision_transformer_teacher"
    supported = ", ".join(sorted(SUPPORTED_MODELS))
    raise ValueError(f"Unsupported model_name '{model_name}'. Supported models: {supported}")


def create_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Create a supported model with a custom classifier head."""
    normalized_name = model_name.lower().strip()
    if normalized_name not in SUPPORTED_MODELS:
        supported = ", ".join(sorted(SUPPORTED_MODELS))
        raise ValueError(f"Unsupported model_name '{model_name}'. Supported models: {supported}")

    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}")

    if normalized_name == "dinov2_vitb14":
        import timm

        return timm.create_model(
            "vit_base_patch14_dinov2.lvd142m",
            pretrained=pretrained,
            num_classes=num_classes,
            img_size=224,
        )

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
    elif normalized_name == "resnet50":
        model = models.resnet50(weights=_weights_or_none(models.ResNet50_Weights, pretrained))
    elif normalized_name == "resnet101":
        model = models.resnet101(weights=_weights_or_none(models.ResNet101_Weights, pretrained))
    elif normalized_name == "densenet121":
        model = models.densenet121(weights=_weights_or_none(models.DenseNet121_Weights, pretrained))
    elif normalized_name == "densenet201":
        model = models.densenet201(weights=_weights_or_none(models.DenseNet201_Weights, pretrained))
    elif normalized_name == "vgg19_bn":
        model = models.vgg19_bn(weights=_weights_or_none(models.VGG19_BN_Weights, pretrained))
    elif normalized_name == "efficientnet_b3":
        model = models.efficientnet_b3(
            weights=_weights_or_none(models.EfficientNet_B3_Weights, pretrained)
        )
    elif normalized_name == "efficientnet_b4":
        model = models.efficientnet_b4(
            weights=_weights_or_none(models.EfficientNet_B4_Weights, pretrained)
        )
    elif normalized_name == "convnext_tiny":
        model = models.convnext_tiny(
            weights=_weights_or_none(models.ConvNeXt_Tiny_Weights, pretrained)
        )
    elif normalized_name == "convnext_base":
        model = models.convnext_base(
            weights=_weights_or_none(models.ConvNeXt_Base_Weights, pretrained)
        )
    elif normalized_name == "regnet_y_8gf":
        model = models.regnet_y_8gf(weights=_weights_or_none(models.RegNet_Y_8GF_Weights, pretrained))
    elif normalized_name == "swin_v2_t":
        model = models.swin_v2_t(weights=_weights_or_none(models.Swin_V2_T_Weights, pretrained))
    elif normalized_name == "maxvit_t":
        model = models.maxvit_t(weights=_weights_or_none(models.MaxVit_T_Weights, pretrained))
    elif normalized_name == "vit_b_16":
        model = models.vit_b_16(weights=_weights_or_none(models.ViT_B_16_Weights, pretrained))
    else:
        raise ValueError(f"Unsupported model_name '{model_name}'")

    return _replace_classifier(model, num_classes, normalized_name)
