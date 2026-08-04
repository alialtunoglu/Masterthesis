"""Training loop for frozen-teacher, single-student knowledge distillation."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn as nn

from evaluation.metrics import compute_classification_metrics
from training.kd_config import KDConfig
from training.kd_features import FeatureCapture
from training.optimization import is_improvement, step_scheduler
from utils.io import ensure_dir


MetricLogger = Callable[[dict[str, float], int | None], None]


def freeze_teacher(teacher: nn.Module) -> None:
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)


def train_kd_epoch(
    student: nn.Module,
    teacher: nn.Module,
    strategy: nn.Module,
    dataloader,
    optimizer,
    device: torch.device,
    student_capture: FeatureCapture | None,
    teacher_capture: FeatureCapture | None,
    max_batches: int | None,
) -> dict[str, float]:
    student.train()
    teacher.eval()
    totals: dict[str, float] = {}
    samples = 0
    agreements = 0
    for batch_index, (inputs, targets) in enumerate(dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            teacher_logits = teacher(inputs)
        student_logits = student(inputs)
        losses = strategy.compute(
            student_logits,
            teacher_logits,
            targets,
            student_features=(
                student_capture.require_output() if student_capture else None
            ),
            teacher_features=(
                teacher_capture.require_output() if teacher_capture else None
            ),
        )
        losses.total.backward()
        optimizer.step()
        batch_size = targets.size(0)
        samples += batch_size
        agreements += int(
            (student_logits.argmax(1) == teacher_logits.argmax(1)).sum().item()
        )
        values = {"total_loss": losses.total, **losses.components}
        for name, value in values.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach()) * batch_size
    if samples == 0:
        raise ValueError("No KD training samples were processed.")
    return {
        **{name: value / samples for name, value in totals.items()},
        "teacher_student_agreement": agreements / samples,
    }


def evaluate_student(
    student: nn.Module,
    dataloader,
    device: torch.device,
    class_names: list[str],
    max_batches: int | None,
) -> dict:
    student.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    samples = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for batch_index, (inputs, targets) in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            logits = student(inputs)
            batch_size = targets.size(0)
            total_loss += float(criterion(logits, targets)) * batch_size
            samples += batch_size
            y_true.extend(targets.cpu().tolist())
            y_pred.extend(logits.argmax(1).cpu().tolist())
    if samples == 0:
        raise ValueError("No KD evaluation samples were processed.")
    return {
        "loss": total_loss / samples,
        "metrics": compute_classification_metrics(y_true, y_pred, class_names),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def fit_kd(
    student: nn.Module,
    teacher: nn.Module,
    strategy: nn.Module,
    train_loader,
    val_loader,
    test_loader,
    class_names: list[str],
    optimizer,
    scheduler,
    device: torch.device,
    config: KDConfig,
    checkpoint_path: Path,
    student_capture: FeatureCapture | None = None,
    teacher_capture: FeatureCapture | None = None,
    logger: MetricLogger | None = None,
) -> dict:
    freeze_teacher(teacher)
    train_config = config.training
    best_value: float | None = None
    best_epoch = -1
    best_val_metrics: dict[str, float] = {}
    history: list[dict[str, float | int]] = []
    without_improvement = 0

    for epoch in range(1, train_config.epochs + 1):
        train_metrics = train_kd_epoch(
            student,
            teacher,
            strategy,
            train_loader,
            optimizer,
            device,
            student_capture,
            teacher_capture,
            train_config.max_train_batches,
        )
        validation = evaluate_student(
            student,
            val_loader,
            device,
            class_names,
            train_config.max_val_batches,
        )
        val_metrics = validation["metrics"]
        epoch_metrics = {
            **train_metrics,
            "train_loss": train_metrics["total_loss"],
            "val_loss": validation["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_precision": val_metrics["macro_precision"],
            "val_macro_recall": val_metrics["macro_recall"],
            "val_macro_f1": val_metrics["macro_f1"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append({"epoch": epoch, **epoch_metrics})
        if logger:
            logger(epoch_metrics, epoch)
        print(_epoch_summary(epoch, train_config.epochs, epoch_metrics), flush=True)

        monitored = float(epoch_metrics[train_config.monitor_metric])
        if is_improvement(monitored, best_value, train_config.monitor_metric):
            best_value = monitored
            best_epoch = epoch
            best_val_metrics = val_metrics
            without_improvement = 0
            _save_kd_checkpoint(
                checkpoint_path, student, strategy, epoch, class_names, config
            )
        else:
            without_improvement += 1
        step_scheduler(scheduler, monitored)
        if without_improvement >= train_config.early_stopping_patience:
            print(
                f"EARLY_STOPPING epoch={epoch}/{train_config.epochs} best_epoch={best_epoch}",
                flush=True,
            )
            break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    student.load_state_dict(checkpoint["student_state_dict"])
    strategy.load_state_dict(checkpoint["strategy_state_dict"])
    test = evaluate_student(
        student, test_loader, device, class_names, train_config.max_test_batches
    )
    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_metrics": best_val_metrics,
        "test_loss": test["loss"],
        "test_metrics": test["metrics"],
        "test_y_true": test["y_true"],
        "test_y_pred": test["y_pred"],
    }


def _save_kd_checkpoint(path, student, strategy, epoch, class_names, config) -> None:
    ensure_dir(path.parent)
    torch.save(
        {
            "epoch": epoch,
            "student_state_dict": student.state_dict(),
            "strategy_state_dict": strategy.state_dict(),
            "class_names": class_names,
            "config": config.to_dict(),
        },
        path,
    )


def _epoch_summary(epoch: int, epochs: int, metrics: dict[str, float]) -> str:
    ordered = [
        "total_loss",
        "ce_loss",
        "kd_loss",
        "feature_loss",
        "relation_loss",
        "teacher_student_agreement",
        "val_loss",
        "val_accuracy",
        "val_macro_f1",
        "learning_rate",
    ]
    values = " ".join(
        f"{name}={metrics[name]:.6f}" for name in ordered if name in metrics
    )
    return f"EPOCH_SUMMARY epoch={epoch}/{epochs} {values}"
