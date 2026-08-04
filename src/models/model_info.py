"""Model metadata utilities."""

from __future__ import annotations

import torch.nn as nn
from torchinfo import summary


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


def get_model_compute_dict(model: nn.Module, image_size: int) -> dict[str, float | int]:
    """Return MACs and an explicitly documented two-FLOPs-per-MAC estimate."""
    training = model.training
    try:
        statistics = summary(
            model,
            input_size=(1, 3, image_size, image_size),
            verbose=0,
            device=next(model.parameters()).device,
        )
    finally:
        model.train(training)
    macs = int(statistics.total_mult_adds)
    return {"macs": macs, "estimated_flops": 2 * macs}
