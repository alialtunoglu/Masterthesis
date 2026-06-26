"""Evaluation report artifacts for baseline classification runs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

Path("logs/matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(Path("logs/matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

from utils.io import ensure_dir


def save_confusion_matrix_artifacts(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
    csv_path: str | Path,
    png_path: str | Path,
) -> tuple[Path, Path]:
    """Save confusion matrix as a CSV table and a heatmap PNG."""
    labels = list(range(len(class_names)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    csv_path = Path(csv_path)
    png_path = Path(png_path)
    ensure_dir(csv_path.parent)
    ensure_dir(png_path.parent)

    matrix_dataframe = pd.DataFrame(matrix, index=list(class_names), columns=list(class_names))
    matrix_dataframe.to_csv(csv_path)

    figure_size = max(7, min(24, len(class_names) * 0.55))
    fig, ax = plt.subplots(figsize=(figure_size, figure_size))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(labels)
    ax.set_yticks(labels)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    max_value = matrix.max() if matrix.size else 0
    if len(class_names) <= 12:
        threshold = max_value / 2 if max_value else 0
        for row_index in labels:
            for column_index in labels:
                value = matrix[row_index, column_index]
                color = "white" if value > threshold else "black"
                ax.text(column_index, row_index, str(value), ha="center", va="center", color=color)

    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    return csv_path, png_path


def save_learning_curves(history: list[dict], output_path: str | Path) -> Path:
    """Save loss and validation metric curves for a training run."""
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    history_dataframe = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(history_dataframe["epoch"], history_dataframe["train_loss"], marker="o", label="train_loss")
    axes[0].plot(history_dataframe["epoch"], history_dataframe["val_loss"], marker="o", label="val_loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history_dataframe["epoch"], history_dataframe["val_accuracy"], marker="o", label="val_accuracy")
    axes[1].plot(history_dataframe["epoch"], history_dataframe["val_macro_f1"], marker="o", label="val_macro_f1")
    axes[1].set_title("Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path
