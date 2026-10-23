"""Feature extraction boundaries used by feature and relational KD strategies."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


FEATURE_LAYER_REGISTRY = {
    "mobilenet_v3_small": "features.12",
    "shufflenet_v2_x0_5": "conv5",
    "micronet_m0": "features",
    "resnet50": "layer4",
    "densenet201": "features.norm5",
    "regnet_y_8gf": "trunk_output",
    "mobilenet_v3_large": "input:classifier.3",
    "efficientnet_b0": "input:classifier.1",
    "efficientnet_b3": "input:classifier.1",
    "efficientnet_b4": "input:classifier.1",
    "resnet18": "input:fc", "resnet101": "input:fc",
    "densenet121": "input:classifier", "vgg19_bn": "input:classifier.6",
    "convnext_tiny": "input:classifier.2", "convnext_base": "input:classifier.2",
    "swin_v2_t": "input:head", "maxvit_t": "input:classifier.5",
    "vit_b_16": "input:heads.head", "dinov2_vitb14": "input:head",
    "deit_tiny_patch16_224": "input:head", "deit_small_patch16_224": "input:head",
}


def default_feature_layer(model_name: str) -> str:
    try:
        return FEATURE_LAYER_REGISTRY[model_name]
    except KeyError as exc:
        raise ValueError(f"No validated KD feature layer for model '{model_name}'.") from exc


def resolve_module(model: nn.Module, layer_name: str) -> nn.Module:
    layer_name = layer_name.removeprefix("input:")
    modules = dict(model.named_modules())
    if layer_name not in modules:
        raise ValueError(f"Layer '{layer_name}' does not exist in model.")
    return modules[layer_name]


class FeatureCapture:
    """Capture one module output while keeping hook lifecycle explicit."""

    def __init__(self, model: nn.Module, layer_name: str) -> None:
        self.output: torch.Tensor | None = None
        module = resolve_module(model, layer_name)
        self._handle = module.register_forward_pre_hook(self._capture_input) if layer_name.startswith("input:") else module.register_forward_hook(self._capture)

    def _capture_input(self, _module, inputs):
        self._capture(_module, inputs, inputs[0])

    def _capture(self, _module, _inputs, output) -> None:
        if not isinstance(output, torch.Tensor):
            raise TypeError("KD feature layer must return a Tensor.")
        self.output = output

    def require_output(self) -> torch.Tensor:
        if self.output is None:
            raise RuntimeError("Feature output is unavailable; run a model forward pass first.")
        return self.output

    def close(self) -> None:
        self._handle.remove()


@dataclass(frozen=True)
class FeatureShapes:
    student: tuple[int, ...]
    teacher: tuple[int, ...]

    @property
    def feature_mode(self) -> str:
        return "spatial" if len(self.student) == len(self.teacher) == 4 else "pooled_vector"


def extract_feature_vector(features: torch.Tensor, layout: str, pooling: str = "mean") -> torch.Tensor:
    if layout == "NC" and features.ndim == 2:
        return features
    if layout == "NCHW" and features.ndim == 4:
        return features.mean(dim=(2, 3))
    if layout == "NHWC" and features.ndim == 4:
        return features.mean(dim=(1, 2))
    if layout == "NLC" and features.ndim == 3:
        if pooling == "cls":
            return features[:, 0]
        if pooling == "patch_mean":
            return features[:, 1:].mean(dim=1)
        return features.mean(dim=1)
    raise ValueError(f"Unsupported feature layout/shape: {layout}, {tuple(features.shape)}")


def pooled_feature(features: torch.Tensor) -> torch.Tensor:
    if features.ndim not in (2, 4):
        raise ValueError("Use a validated pooled/classifier feature node; raw token layouts require an explicit pooling spec.")
    return extract_feature_vector(features, "NC" if features.ndim == 2 else "NCHW")


class VectorFeatureAdapter(nn.Module):
    def __init__(self, student_channels: int, teacher_channels: int):
        super().__init__()
        self.projection = nn.Linear(student_channels, teacher_channels, bias=False)

    def forward(self, student, teacher):
        return self.projection(pooled_feature(student))


class ConvFeatureAdapter(nn.Module):
    """Align student channels and spatial dimensions to a teacher feature map."""

    def __init__(self, student_channels: int, teacher_channels: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, bias=False)

    def forward(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        if student.ndim != 4 or teacher.ndim != 4:
            raise ValueError("ConvFeatureAdapter requires 4D CNN feature maps.")
        aligned = self.projection(student)
        if aligned.shape[-2:] != teacher.shape[-2:]:
            aligned = torch.nn.functional.adaptive_avg_pool2d(aligned, teacher.shape[-2:])
        return aligned


def infer_feature_shapes(
    student: nn.Module,
    teacher: nn.Module,
    student_layer: str,
    teacher_layer: str,
    image_size: int,
    device: torch.device,
) -> FeatureShapes:
    student_capture = teacher_capture = None
    student_training, teacher_training = student.training, teacher.training
    try:
        student_capture = FeatureCapture(student, student_layer)
        teacher_capture = FeatureCapture(teacher, teacher_layer)
        sample = torch.zeros(1, 3, image_size, image_size, device=device)
        student.eval()
        teacher.eval()
        with torch.no_grad():
            student(sample)
            teacher(sample)
        return FeatureShapes(
            tuple(student_capture.require_output().shape),
            tuple(teacher_capture.require_output().shape),
        )
    finally:
        student.train(student_training)
        teacher.train(teacher_training)
        if student_capture:
            student_capture.close()
        if teacher_capture:
            teacher_capture.close()
