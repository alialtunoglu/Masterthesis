"""Training loop for frozen multi-teacher knowledge distillation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import torch
import torch.nn as nn

from evaluation.metrics import compute_classification_metrics
from training.kd_features import FeatureCapture
from training.kd_trainer import evaluate_student, freeze_teacher
from training.multi_kd_config import MultiKDConfig
from training.multi_kd_strategies import weighted_ensemble_probabilities
from training.optimization import is_improvement, step_scheduler
from utils.io import ensure_dir


MetricLogger = Callable[[dict[str, float], int | None], None]


def train_multi_teacher_epoch(
    student: nn.Module,
    teachers: Sequence[nn.Module],
    teacher_names: Sequence[str],
    strategy: nn.Module,
    dataloader,
    optimizer,
    device: torch.device,
    student_capture: FeatureCapture | None = None,
    teacher_captures: Sequence[FeatureCapture] | None = None,
    max_batches: int | None = None,
) -> dict[str, float]:
    student.train()
    for teacher in teachers:
        teacher.eval()
    totals: dict[str, float] = {}
    samples = 0
    agreements = {name: 0 for name in teacher_names}
    ensemble_agreements = 0
    captures = list(teacher_captures or [])
    for batch_index, (inputs, targets) in enumerate(dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        student_logits = student(inputs)
        student_features = (
            student_capture.require_output() if student_capture else None
        )
        teacher_logits: list[torch.Tensor] = []
        teacher_features: list[torch.Tensor] = []
        with torch.no_grad():
            for index, teacher in enumerate(teachers):
                logits = teacher(inputs)
                teacher_logits.append(logits)
                if captures:
                    teacher_features.append(captures[index].require_output().detach())
        losses = strategy.compute(
            student_logits,
            teacher_logits,
            targets,
            student_features=student_features,
            teacher_features=teacher_features or None,
        )
        losses.total.backward()
        optimizer.step()
        batch_size = targets.size(0)
        samples += batch_size
        student_predictions = student_logits.argmax(1)
        for name, logits in zip(teacher_names, teacher_logits):
            agreements[name] += int(
                (student_predictions == logits.argmax(1)).sum().item()
            )
        ensemble = weighted_ensemble_probabilities(
            teacher_logits, strategy.teacher_weights
        )
        ensemble_agreements += int(
            (student_predictions == ensemble.argmax(1)).sum().item()
        )
        values = {"total_loss": losses.total, **losses.components}
        for name, value in values.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach()) * batch_size
    if samples == 0:
        raise ValueError("No multi-teacher training samples were processed.")
    return {
        **{name: value / samples for name, value in totals.items()},
        **{
            f"agreement_{name}": value / samples
            for name, value in agreements.items()
        },
        "ensemble_student_agreement": ensemble_agreements / samples,
    }


def evaluate_teacher_ensemble(
    teachers: Sequence[nn.Module],
    weights: torch.Tensor,
    dataloader,
    device: torch.device,
    class_names: list[str],
    max_batches: int | None,
) -> dict:
    y_true: list[int] = []
    y_pred: list[int] = []
    for teacher in teachers:
        teacher.eval()
    with torch.no_grad():
        for batch_index, (inputs, targets) in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break
            inputs = inputs.to(device)
            logits = [teacher(inputs) for teacher in teachers]
            predictions = weighted_ensemble_probabilities(logits, weights).argmax(1)
            y_true.extend(targets.tolist())
            y_pred.extend(predictions.cpu().tolist())
    if not y_true:
        raise ValueError("No ensemble evaluation samples were processed.")
    return {
        "metrics": compute_classification_metrics(y_true, y_pred, class_names),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def fit_multi_teacher_kd(
    student: nn.Module,
    teachers: Sequence[nn.Module],
    teacher_names: Sequence[str],
    strategy: nn.Module,
    train_loader,
    val_loader,
    test_loader,
    class_names: list[str],
    optimizer,
    scheduler,
    device: torch.device,
    config: MultiKDConfig,
    checkpoint_path: Path,
    student_capture: FeatureCapture | None = None,
    teacher_captures: Sequence[FeatureCapture] | None = None,
    logger: MetricLogger | None = None,
) -> dict:
    for teacher in teachers:
        freeze_teacher(teacher)
    train = config.training
    best_value: float | None = None
    best_epoch = -1
    best_val_metrics: dict[str, float] = {}
    history: list[dict[str, float | int]] = []
    without_improvement = 0
    for epoch in range(1, train.epochs + 1):
        train_metrics = train_multi_teacher_epoch(
            student,
            teachers,
            teacher_names,
            strategy,
            train_loader,
            optimizer,
            device,
            student_capture,
            teacher_captures,
            train.max_train_batches,
        )
        validation = evaluate_student(
            student, val_loader, device, class_names, train.max_val_batches
        )
        metrics = validation["metrics"]
        epoch_metrics = {
            **train_metrics,
            "train_loss": train_metrics["total_loss"],
            "val_loss": validation["loss"],
            "val_accuracy": metrics["accuracy"],
            "val_macro_precision": metrics["macro_precision"],
            "val_macro_recall": metrics["macro_recall"],
            "val_macro_f1": metrics["macro_f1"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append({"epoch": epoch, **epoch_metrics})
        if logger:
            logger(epoch_metrics, epoch)
        print(_epoch_summary(epoch, train.epochs, epoch_metrics), flush=True)
        monitored = float(epoch_metrics[train.monitor_metric])
        if is_improvement(monitored, best_value, train.monitor_metric):
            best_value = monitored
            best_epoch = epoch
            best_val_metrics = metrics
            without_improvement = 0
            _save_checkpoint(
                checkpoint_path, student, strategy, epoch, class_names, config
            )
        else:
            without_improvement += 1
        step_scheduler(scheduler, monitored)
        if without_improvement >= train.early_stopping_patience:
            print(
                f"EARLY_STOPPING epoch={epoch}/{train.epochs} best_epoch={best_epoch}",
                flush=True,
            )
            break
    checkpoint = torch.load(checkpoint_path, map_location=device)
    student.load_state_dict(checkpoint["student_state_dict"])
    strategy.load_state_dict(checkpoint["strategy_state_dict"])
    test = evaluate_student(
        student, test_loader, device, class_names, train.max_test_batches
    )
    ensemble = evaluate_teacher_ensemble(
        teachers,
        strategy.teacher_weights,
        test_loader,
        device,
        class_names,
        train.max_test_batches,
    )
    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_metrics": best_val_metrics,
        "test_loss": test["loss"],
        "test_metrics": test["metrics"],
        "test_y_true": test["y_true"],
        "test_y_pred": test["y_pred"],
        "ensemble_metrics": ensemble["metrics"],
    }


def _save_checkpoint(path, student, strategy, epoch, class_names, config) -> None:
    ensure_dir(path.parent)
    torch.save(
        {
            "epoch": epoch,
            "student_state_dict": student.state_dict(),
            "strategy_state_dict": strategy.state_dict(),
            "class_names": class_names,
            "teacher_models": [
                teacher.model_name for teacher in config.teachers
            ],
            "config": config.to_dict(),
        },
        path,
    )


def _epoch_summary(epoch: int, epochs: int, metrics: dict[str, float]) -> str:
    values = " ".join(
        f"{name}={value:.6f}"
        for name, value in metrics.items()
        if name != "train_loss"
    )
    return f"EPOCH_SUMMARY epoch={epoch}/{epochs} {values}"
