"""Feature extraction boundaries used by feature and relational KD strategies."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


FEATURE_LAYER_REGISTRY = {
    "mobilenet_v3_small": "features.12",
    "resnet50": "layer4",
    "densenet201": "features.norm5",
    "regnet_y_8gf": "trunk_output",
}


def default_feature_layer(model_name: str) -> str:
    try:
        return FEATURE_LAYER_REGISTRY[model_name]
    except KeyError as exc:
        raise ValueError(f"No validated KD feature layer for model '{model_name}'.") from exc


def resolve_module(model: nn.Module, layer_name: str) -> nn.Module:
    modules = dict(model.named_modules())
    if layer_name not in modules:
        raise ValueError(f"Layer '{layer_name}' does not exist in model.")
    return modules[layer_name]


class FeatureCapture:
    """Capture one module output while keeping hook lifecycle explicit."""

    def __init__(self, model: nn.Module, layer_name: str) -> None:
        self.output: torch.Tensor | None = None
        self._handle = resolve_module(model, layer_name).register_forward_hook(self._capture)

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
    student_capture = FeatureCapture(student, student_layer)
    teacher_capture = FeatureCapture(teacher, teacher_layer)
    try:
        sample = torch.zeros(1, 3, image_size, image_size, device=device)
        with torch.no_grad():
            student(sample)
            teacher(sample)
        return FeatureShapes(
            tuple(student_capture.require_output().shape),
            tuple(teacher_capture.require_output().shape),
        )
    finally:
        student_capture.close()
        teacher_capture.close()
