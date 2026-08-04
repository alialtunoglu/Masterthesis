"""Loss strategies for multi-teacher knowledge distillation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.kd_features import ConvFeatureAdapter
from training.multi_kd_config import MultiDistillationConfig


@dataclass
class MultiTeacherLoss:
    total: torch.Tensor
    components: dict[str, torch.Tensor]


class MultiTeacherStrategy(nn.Module):
    def __init__(
        self,
        config: MultiDistillationConfig,
        teacher_names: Sequence[str],
        weights: Sequence[float],
    ) -> None:
        super().__init__()
        self.config = config
        self.teacher_names = tuple(teacher_names)
        self.register_buffer("teacher_weights", torch.tensor(weights, dtype=torch.float32))

    def _weight(self, index: int, reference: torch.Tensor) -> torch.Tensor:
        return self.teacher_weights[index].to(reference)


class MultiLogitStrategy(MultiTeacherStrategy):
    def compute(self, student_logits, teacher_logits, targets, **_kwargs):
        ce = F.cross_entropy(student_logits, targets)
        kd_total = torch.zeros_like(ce)
        components: dict[str, torch.Tensor] = {"ce_loss": ce}
        temperature = self.config.temperature
        student_log_prob = F.log_softmax(student_logits / temperature, dim=1)
        for index, (name, logits) in enumerate(zip(self.teacher_names, teacher_logits)):
            loss = F.kl_div(
                student_log_prob,
                F.softmax(logits / temperature, dim=1),
                reduction="batchmean",
            ) * temperature**2
            components[f"teacher_{name}_kd_loss"] = loss
            kd_total = kd_total + self._weight(index, loss) * loss
        total = (1 - self.config.alpha) * ce + self.config.alpha * kd_total
        components["kd_loss"] = kd_total
        return MultiTeacherLoss(total, components)


class MultiFeatureStrategy(MultiTeacherStrategy):
    def __init__(
        self,
        config,
        teacher_names,
        weights,
        student_channels: int,
        teacher_channels: Sequence[int],
    ):
        super().__init__(config, teacher_names, weights)
        self.adapters = nn.ModuleDict(
            {
                name: ConvFeatureAdapter(student_channels, channels)
                for name, channels in zip(self.teacher_names, teacher_channels)
            }
        )

    def compute(
        self,
        student_logits,
        teacher_logits,
        targets,
        student_features=None,
        teacher_features=None,
    ):
        if student_features is None or teacher_features is None:
            raise ValueError("Multi-teacher feature KD requires all feature tensors.")
        ce = F.cross_entropy(student_logits, targets)
        feature_total = torch.zeros_like(ce)
        components: dict[str, torch.Tensor] = {"ce_loss": ce}
        for index, (name, feature) in enumerate(
            zip(self.teacher_names, teacher_features)
        ):
            aligned = self.adapters[name](student_features, feature)
            target = feature.detach()
            if self.config.feature_loss == "mse":
                loss = F.mse_loss(aligned, target)
            elif self.config.feature_loss == "l1":
                loss = F.l1_loss(aligned, target)
            else:
                loss = 1 - F.cosine_similarity(
                    aligned.flatten(1), target.flatten(1), dim=1
                ).mean()
            components[f"teacher_{name}_feature_loss"] = loss
            feature_total = feature_total + self._weight(index, loss) * loss
        total = (
            self.config.hard_label_loss_weight * ce
            + self.config.feature_loss_weight * feature_total
        )
        components["feature_loss"] = feature_total
        return MultiTeacherLoss(total, components)


class MultiRelationStrategy(MultiTeacherStrategy):
    def compute(
        self,
        student_logits,
        teacher_logits,
        targets,
        student_features=None,
        teacher_features=None,
    ):
        if student_features is None or teacher_features is None:
            raise ValueError("Multi-teacher relation KD requires all feature tensors.")
        ce = F.cross_entropy(student_logits, targets)
        student_vectors = _vectors(
            student_features, self.config.feature_normalization
        )
        relation_total = torch.zeros_like(ce)
        components: dict[str, torch.Tensor] = {"ce_loss": ce}
        for index, (name, feature) in enumerate(
            zip(self.teacher_names, teacher_features)
        ):
            teacher_vectors = _vectors(
                feature.detach(), self.config.feature_normalization
            )
            distance = F.smooth_l1_loss(
                _distances(student_vectors), _distances(teacher_vectors)
            )
            angle = F.smooth_l1_loss(
                _angles(student_vectors), _angles(teacher_vectors)
            )
            if self.config.relation_type == "distance":
                loss = self.config.distance_loss_weight * distance
            elif self.config.relation_type == "angle":
                loss = self.config.angle_loss_weight * angle
            elif self.config.relation_type == "correlation":
                loss = F.mse_loss(
                    student_vectors @ student_vectors.T,
                    teacher_vectors @ teacher_vectors.T,
                )
            else:
                loss = (
                    self.config.distance_loss_weight * distance
                    + self.config.angle_loss_weight * angle
                )
            components[f"teacher_{name}_relation_loss"] = loss
            relation_total = relation_total + self._weight(index, loss) * loss
        total = self.config.hard_label_loss_weight * ce + relation_total
        components["relation_loss"] = relation_total
        return MultiTeacherLoss(total, components)


def create_multi_teacher_strategy(
    kd_type: str,
    config: MultiDistillationConfig,
    teacher_names: Sequence[str],
    weights: Sequence[float],
    *,
    student_channels: int | None = None,
    teacher_channels: Sequence[int] | None = None,
) -> MultiTeacherStrategy:
    if kd_type == "logit_based":
        return MultiLogitStrategy(config, teacher_names, weights)
    if kd_type == "feature_based":
        if student_channels is None or teacher_channels is None:
            raise ValueError("Feature channel dimensions are required.")
        return MultiFeatureStrategy(
            config, teacher_names, weights, student_channels, teacher_channels
        )
    if kd_type == "relation_based":
        return MultiRelationStrategy(config, teacher_names, weights)
    raise ValueError(f"Unsupported multi-teacher KD type: {kd_type}")


def weighted_ensemble_probabilities(
    teacher_logits: Sequence[torch.Tensor], weights: torch.Tensor
) -> torch.Tensor:
    probabilities = [
        weight.to(logits) * F.softmax(logits, dim=1)
        for weight, logits in zip(weights, teacher_logits)
    ]
    return torch.stack(probabilities).sum(dim=0)


def _vectors(features: torch.Tensor, normalize: bool) -> torch.Tensor:
    vectors = features.mean(dim=(-2, -1)) if features.ndim == 4 else features.flatten(1)
    return F.normalize(vectors, dim=1) if normalize else vectors


def _distances(vectors: torch.Tensor) -> torch.Tensor:
    distances = torch.cdist(vectors, vectors)
    positive = distances[distances > 0]
    scale = positive.mean() if positive.numel() else distances.new_tensor(1.0)
    return distances / scale.clamp_min(1e-12)


def _angles(vectors: torch.Tensor) -> torch.Tensor:
    differences = vectors.unsqueeze(0) - vectors.unsqueeze(1)
    normalized = F.normalize(differences, dim=2)
    return torch.bmm(normalized, normalized.transpose(1, 2))
