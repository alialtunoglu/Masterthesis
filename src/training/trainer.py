"""Reusable baseline training and evaluation loop."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from evaluation.metrics import compute_classification_metrics
from utils.io import ensure_dir


MetricLogger = Callable[[dict[str, float], int | None], None]


def _show_progress_bars() -> bool:
    """Show tqdm bars only in an interactive terminal, not redirected log files."""
    return sys.stderr.isatty()


def _batch_limit_reached(batch_index: int, max_batches: int | None) -> bool:
    return max_batches is not None and batch_index >= max_batches


def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    """Train for one epoch and return average loss."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    progress = tqdm(dataloader, desc="train", leave=False, disable=not _show_progress_bars())
    for batch_index, (inputs, targets) in enumerate(progress):
        if _batch_limit_reached(batch_index, max_batches):
            break

        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("No training samples were processed. Check dataloader and max_batches.")
    return total_loss / total_samples


def evaluate(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
    class_names: list[str],
    max_batches: int | None = None,
) -> dict:
    """Evaluate a model and return loss, aggregate metrics, and predictions."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        progress = tqdm(dataloader, desc="eval", leave=False, disable=not _show_progress_bars())
        for batch_index, (inputs, targets) in enumerate(progress):
            if _batch_limit_reached(batch_index, max_batches):
                break

            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            predictions = torch.argmax(outputs, dim=1)

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            y_true.extend(targets.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())

    if total_samples == 0:
        raise ValueError("No evaluation samples were processed. Check dataloader and max_batches.")

    metrics = compute_classification_metrics(y_true, y_pred, class_names)
    return {
        "loss": total_loss / total_samples,
        "metrics": metrics,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def fit(
    model: nn.Module,
    train_loader,
    val_loader,
    test_loader,
    class_names: list[str],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    scheduler=None,
    checkpoint_path: str | Path | None = None,
    logger: MetricLogger | None = None,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    max_test_batches: int | None = None,
    checkpoint_metadata: dict | None = None,
) -> dict:
    """Train, select the best validation macro-F1 checkpoint, and test once."""
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")

    criterion = nn.CrossEntropyLoss()
    best_val_macro_f1 = -1.0
    best_epoch = -1
    best_val_metrics: dict[str, float] = {}
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            max_batches=max_train_batches,
        )
        val_output = evaluate(
            model,
            val_loader,
            criterion,
            device,
            class_names,
            max_batches=max_val_batches,
        )
        val_metrics = val_output["metrics"]
        if scheduler is not None:
            scheduler.step()

        learning_rate = optimizer.param_groups[0]["lr"]
        epoch_metrics = {
            "train_loss": train_loss,
            "val_loss": val_output["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_precision": val_metrics["macro_precision"],
            "val_macro_recall": val_metrics["macro_recall"],
            "val_macro_f1": val_metrics["macro_f1"],
            "learning_rate": learning_rate,
        }
        history.append({"epoch": epoch, **epoch_metrics})
        if logger is not None:
            logger(epoch_metrics, epoch)

        print(
            "EPOCH_SUMMARY "
            f"epoch={epoch}/{epochs} "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_output['loss']:.6f} "
            f"val_accuracy={val_metrics['accuracy']:.6f} "
            f"val_macro_precision={val_metrics['macro_precision']:.6f} "
            f"val_macro_recall={val_metrics['macro_recall']:.6f} "
            f"val_macro_f1={val_metrics['macro_f1']:.6f} "
            f"learning_rate={learning_rate:.8f}",
            flush=True,
        )

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_val_metrics = val_metrics
            if checkpoint_path is not None:
                save_checkpoint(
                    model=model,
                    checkpoint_path=checkpoint_path,
                    epoch=epoch,
                    class_names=class_names,
                    metrics=val_metrics,
                    metadata=checkpoint_metadata,
                )

    if checkpoint_path is not None and Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    test_output = evaluate(
        model,
        test_loader,
        criterion,
        device,
        class_names,
        max_batches=max_test_batches,
    )

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_metrics": best_val_metrics,
        "test_metrics": test_output["metrics"],
        "test_loss": test_output["loss"],
        "test_y_true": test_output["y_true"],
        "test_y_pred": test_output["y_pred"],
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else "",
    }


def save_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    epoch: int,
    class_names: list[str],
    metrics: dict[str, float],
    metadata: dict | None = None,
) -> None:
    """Save a PyTorch checkpoint for the current best model."""
    checkpoint_path = Path(checkpoint_path)
    ensure_dir(checkpoint_path.parent)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "metrics": metrics,
            "metadata": metadata or {},
        },
        checkpoint_path,
    )
