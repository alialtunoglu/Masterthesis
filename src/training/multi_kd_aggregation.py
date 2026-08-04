"""Teacher-weight resolution policies for multi-teacher distillation."""

from __future__ import annotations

import math
from typing import Sequence

from training.multi_kd_config import TeacherConfig


def resolve_teacher_weights(
    aggregation: str, teachers: Sequence[TeacherConfig]
) -> tuple[float, ...]:
    if len(teachers) < 2:
        raise ValueError("At least two teachers are required.")
    if aggregation == "uniform":
        values = [1.0] * len(teachers)
    elif aggregation == "validation_weighted":
        values = [
            _positive_finite(teacher.best_val_macro_f1, teacher.model_name)
            for teacher in teachers
        ]
    elif aggregation == "manual":
        values = [
            _non_negative(teacher.manual_weight, teacher.model_name)
            for teacher in teachers
        ]
    else:
        raise ValueError(f"Unsupported aggregation: {aggregation}")
    total = sum(values)
    if total <= 0:
        raise ValueError("Teacher weights must contain at least one positive value.")
    return tuple(value / total for value in values)


def _positive_finite(value: float | None, name: str) -> float:
    if value is None or not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"{name} requires a positive finite validation macro-F1.")
    return float(value)


def _non_negative(value: float | None, name: str) -> float:
    if value is None or not math.isfinite(float(value)) or float(value) < 0:
        raise ValueError(f"{name} requires a non-negative finite manual weight.")
    return float(value)
