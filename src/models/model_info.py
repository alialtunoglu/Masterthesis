"""Model metadata utilities."""

from __future__ import annotations

import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    """Return the total number of model parameters."""
    return sum(parameter.numel() for parameter in model.parameters())


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of trainable model parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def estimate_model_size_mb(model: nn.Module) -> float:
    """Estimate model state size in megabytes from parameters and buffers."""
    total_bytes = 0
    for tensor in list(model.parameters()) + list(model.buffers()):
        total_bytes += tensor.numel() * tensor.element_size()
    return total_bytes / (1024**2)


def get_model_summary_dict(model: nn.Module) -> dict[str, float | int]:
    """Return a compact model summary dictionary."""
    return {
        "params": count_parameters(model),
        "trainable_params": count_trainable_parameters(model),
        "model_size_mb": estimate_model_size_mb(model),
    }
