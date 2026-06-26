"""Classification metrics for multi-class experiments."""

from __future__ import annotations

from typing import Sequence

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
) -> dict[str, float]:
    """Compute aggregate multi-class classification metrics."""
    labels = list(range(len(class_names)))
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="weighted",
        zero_division=0,
    )

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
    }


def compute_per_class_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
) -> pd.DataFrame:
    """Return per-class precision, recall, F1, and support as a DataFrame."""
    labels = list(range(len(class_names)))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )
    rows = []
    for class_id, class_name in enumerate(class_names):
        class_report = report[class_name]
        rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "precision": class_report["precision"],
                "recall": class_report["recall"],
                "f1": class_report["f1-score"],
                "support": int(class_report["support"]),
            }
        )
    return pd.DataFrame(rows)
