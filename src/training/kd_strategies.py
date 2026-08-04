"""Composable loss strategies for the supported knowledge-distillation types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.kd_config import DistillationConfig
from training.kd_features import ConvFeatureAdapter


@dataclass
class DistillationLoss:
    total: torch.Tensor
    components: dict[str, torch.Tensor]


class DistillationStrategy(Protocol):
    def parameters(self): ...

    def compute(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
        student_features: torch.Tensor | None = None,
        teacher_features: torch.Tensor | None = None,
    ) -> DistillationLoss: ...


class LogitDistillationStrategy(nn.Module):
    def __init__(self, config: DistillationConfig) -> None:
        super().__init__()
        self.temperature = config.temperature
        self.alpha = config.alpha

    def compute(self, student_logits, teacher_logits, targets, **_kwargs) -> DistillationLoss:
        ce = F.cross_entropy(student_logits, targets)
        temperature = self.temperature
        kd = F.kl_div(
            F.log_softmax(student_logits / temperature, dim=1),
            F.softmax(teacher_logits / temperature, dim=1),
            reduction="batchmean",
        ) * temperature**2
        total = (1.0 - self.alpha) * ce + self.alpha * kd
        return DistillationLoss(total, {"ce_loss": ce, "kd_loss": kd})


class FeatureDistillationStrategy(nn.Module):
    def __init__(
        self,
        config: DistillationConfig,
        student_channels: int,
        teacher_channels: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.adapter = ConvFeatureAdapter(student_channels, teacher_channels)

    def compute(
        self,
        student_logits,
        teacher_logits,
        targets,
        student_features=None,
        teacher_features=None,
    ) -> DistillationLoss:
        if student_features is None or teacher_features is None:
            raise ValueError("Feature-based KD requires student and teacher features.")
        ce = F.cross_entropy(student_logits, targets)
        aligned = self.adapter(student_features, teacher_features)
        teacher_features = teacher_features.detach()
        if self.config.feature_loss == "mse":
            feature = F.mse_loss(aligned, teacher_features)
        elif self.config.feature_loss == "l1":
            feature = F.l1_loss(aligned, teacher_features)
        else:
            feature = 1.0 - F.cosine_similarity(
                aligned.flatten(1), teacher_features.flatten(1), dim=1
            ).mean()
        total = self.config.hard_label_loss_weight * ce + self.config.feature_loss_weight * feature
        return DistillationLoss(total, {"ce_loss": ce, "feature_loss": feature})


class RelationDistillationStrategy(nn.Module):
    def __init__(self, config: DistillationConfig) -> None:
        super().__init__()
        self.config = config

    def compute(
        self,
        student_logits,
        teacher_logits,
        targets,
        student_features=None,
        teacher_features=None,
    ) -> DistillationLoss:
        if student_features is None or teacher_features is None:
            raise ValueError("Relation-based KD requires student and teacher features.")
        ce = F.cross_entropy(student_logits, targets)
        student_vectors = _feature_vectors(student_features, self.config.feature_normalization)
        teacher_vectors = _feature_vectors(teacher_features.detach(), self.config.feature_normalization)
        distance = F.smooth_l1_loss(
            _normalized_distances(student_vectors), _normalized_distances(teacher_vectors)
        )
        angle = F.smooth_l1_loss(_angles(student_vectors), _angles(teacher_vectors))
        relation = torch.zeros_like(ce)
        if self.config.relation_type in {"distance", "distance_and_angle"}:
            relation = relation + self.config.distance_loss_weight * distance
        if self.config.relation_type in {"angle", "distance_and_angle"}:
            relation = relation + self.config.angle_loss_weight * angle
        if self.config.relation_type == "correlation":
            relation = F.mse_loss(
                student_vectors @ student_vectors.T,
                teacher_vectors @ teacher_vectors.T,
            )
        total = self.config.hard_label_loss_weight * ce + relation
        return DistillationLoss(
            total,
            {"ce_loss": ce, "relation_loss": relation, "distance_loss": distance, "angle_loss": angle},
        )


def create_distillation_strategy(
    kd_type: str,
    config: DistillationConfig,
    *,
    student_channels: int | None = None,
    teacher_channels: int | None = None,
) -> nn.Module:
    if kd_type == "logit_based":
        return LogitDistillationStrategy(config)
    if kd_type == "feature_based":
        if student_channels is None or teacher_channels is None:
            raise ValueError("Feature channel dimensions are required for feature-based KD.")
        return FeatureDistillationStrategy(config, student_channels, teacher_channels)
    if kd_type == "relation_based":
        return RelationDistillationStrategy(config)
    raise ValueError(f"Unsupported KD type: {kd_type}")


def _feature_vectors(features: torch.Tensor, normalize: bool) -> torch.Tensor:
    vectors = features.mean(dim=(-2, -1)) if features.ndim == 4 else features.flatten(1)
    return F.normalize(vectors, dim=1) if normalize else vectors


def _normalized_distances(vectors: torch.Tensor) -> torch.Tensor:
    distances = torch.cdist(vectors, vectors, p=2)
    positive = distances[distances > 0]
    scale = positive.mean() if positive.numel() else distances.new_tensor(1.0)
    return distances / scale.clamp_min(1e-12)


def _angles(vectors: torch.Tensor) -> torch.Tensor:
    differences = vectors.unsqueeze(0) - vectors.unsqueeze(1)
    normalized = F.normalize(differences, dim=2)
    return torch.bmm(normalized, normalized.transpose(1, 2))
