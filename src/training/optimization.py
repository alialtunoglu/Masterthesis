"""Optimizer, scheduler, and monitored-metric factories shared by KD training."""

from __future__ import annotations

from collections.abc import Iterable

import torch

from training.kd_config import TrainingConfig


def create_optimizer(parameters: Iterable[torch.nn.Parameter], config: TrainingConfig):
    params = list(parameters)
    name = config.optimizer.lower()
    options = dict(config.optimizer_params)
    if name == "adamw":
        return torch.optim.AdamW(
            params, lr=config.learning_rate, weight_decay=config.weight_decay, **options
        )
    if name == "adam":
        return torch.optim.Adam(
            params, lr=config.learning_rate, weight_decay=config.weight_decay, **options
        )
    if name == "sgd":
        options.setdefault("momentum", 0.9)
        return torch.optim.SGD(
            params, lr=config.learning_rate, weight_decay=config.weight_decay, **options
        )
    raise ValueError(f"Unsupported optimizer: {config.optimizer}")


def create_scheduler(optimizer, config: TrainingConfig):
    name = config.scheduler.lower()
    options = dict(config.scheduler_params)
    if name == "none":
        return None
    if name == "cosine":
        options.setdefault("T_max", config.epochs)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **options)
    if name == "step":
        options.setdefault("step_size", 10)
        options.setdefault("gamma", 0.1)
        return torch.optim.lr_scheduler.StepLR(optimizer, **options)
    if name == "reduce_on_plateau":
        options.setdefault("mode", monitor_mode(config.monitor_metric))
        options.setdefault("factor", 0.1)
        options.setdefault("patience", 3)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **options)
    raise ValueError(f"Unsupported scheduler: {config.scheduler}")


def monitor_mode(metric: str) -> str:
    return "min" if metric == "val_loss" else "max"


def is_improvement(value: float, best: float | None, metric: str) -> bool:
    if best is None:
        return True
    return value < best if monitor_mode(metric) == "min" else value > best


def step_scheduler(scheduler, metric_value: float) -> None:
    if scheduler is None:
        return
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(metric_value)
    else:
        scheduler.step()
