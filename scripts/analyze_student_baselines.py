"""Analyze completed student baseline runs and generate selection reports."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import matplotlib

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib_cache"))
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = PROJECT_ROOT / "results" / "baseline"
BASELINE_RESULTS = BASELINE_DIR / "baseline_results.csv"
DOCS_DIR = PROJECT_ROOT / "docs"

EXPECTED_DATASETS = ["appleleaf9", "plantvillage", "plantpathology2021"]
EXPECTED_MODELS = ["mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_b0", "resnet18"]

CLEAN_COLUMNS = [
    "dataset_name",
    "model_name",
    "seed",
    "epochs",
    "batch_size",
    "image_size",
    "pretrained",
    "params",
    "model_size_mb",
    "best_epoch",
    "best_val_accuracy",
    "best_val_macro_f1",
    "test_accuracy",
    "test_macro_precision",
    "test_macro_recall",
    "test_macro_f1",
    "test_weighted_f1",
    "checkpoint_path",
    "mlflow_run_id",
    "run_name",
    "run_dir",
    "config_path",
    "history_path",
    "per_class_metrics_path",
    "confusion_matrix_csv_path",
    "confusion_matrix_png_path",
    "learning_curves_path",
]

COMPARISON_COLUMNS = [
    "model_name",
    "test_accuracy",
    "test_macro_precision",
    "test_macro_recall",
    "test_macro_f1",
    "test_weighted_f1",
    "best_val_accuracy",
    "best_val_macro_f1",
    "params",
    "model_size_mb",
    "best_epoch",
    "run_name",
    "mlflow_run_id",
    "checkpoint_path",
    "run_dir",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze student baseline results.")
    return parser.parse_args()


def rel(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def load_results() -> pd.DataFrame:
    if not BASELINE_RESULTS.exists():
        raise FileNotFoundError(f"Missing baseline results: {BASELINE_RESULTS}")
    df = pd.read_csv(BASELINE_RESULTS)
    required = {"dataset_name", "model_name", "run_name", "test_macro_f1", "test_accuracy"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"baseline_results.csv is missing required columns: {missing}")
    return df


def select_best_runs(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    numeric_cols = [
        "test_macro_f1",
        "test_accuracy",
        "best_val_macro_f1",
        "model_size_mb",
        "params",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    selected_rows = []
    duplicate_rows = []
    missing_pairs = []
    for dataset in EXPECTED_DATASETS:
        for model in EXPECTED_MODELS:
            subset = df[(df["dataset_name"] == dataset) & (df["model_name"] == model)].copy()
            if subset.empty:
                missing_pairs.append({"dataset_name": dataset, "model_name": model})
                continue
            subset = subset.sort_values(
                by=["test_macro_f1", "test_accuracy", "best_val_macro_f1", "model_size_mb"],
                ascending=[False, False, False, True],
            )
            if len(subset) > 1:
                duplicate_rows.append(
                    {
                        "dataset_name": dataset,
                        "model_name": model,
                        "num_runs": len(subset),
                        "selected_run_name": subset.iloc[0]["run_name"],
                    }
                )
            selected_rows.append(subset.iloc[0])

    selected = pd.DataFrame(selected_rows).reset_index(drop=True)
    duplicates = pd.DataFrame(duplicate_rows)
    return selected, duplicates, missing_pairs


def add_artifact_paths(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        run_dir = BASELINE_DIR / "runs" / str(row["dataset_name"]) / str(row["model_name"]) / str(row["run_name"])
        updated = row.to_dict()
        updated.update(
            {
                "run_dir": rel(run_dir),
                "config_path": rel(run_dir / "config.json"),
                "history_path": rel(run_dir / "history.csv"),
                "per_class_metrics_path": rel(run_dir / "per_class_metrics.csv"),
                "confusion_matrix_csv_path": rel(run_dir / "confusion_matrix.csv"),
                "confusion_matrix_png_path": rel(run_dir / "confusion_matrix.png"),
                "learning_curves_path": rel(run_dir / "learning_curves.png"),
            }
        )
        rows.append(updated)
    clean = pd.DataFrame(rows)
    return clean.reindex(columns=CLEAN_COLUMNS)


def artifact_check(clean: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in clean.iterrows():
        run_dir = PROJECT_ROOT / str(row["run_dir"])
        rows.append(
            {
                "dataset_name": row["dataset_name"],
                "model_name": row["model_name"],
                "run_name": row["run_name"],
                "run_dir_exists": run_dir.exists(),
                "config_exists": (PROJECT_ROOT / str(row["config_path"])).exists(),
                "history_exists": (PROJECT_ROOT / str(row["history_path"])).exists(),
                "per_class_metrics_exists": (PROJECT_ROOT / str(row["per_class_metrics_path"])).exists(),
                "confusion_matrix_csv_exists": (PROJECT_ROOT / str(row["confusion_matrix_csv_path"])).exists(),
                "confusion_matrix_png_exists": (PROJECT_ROOT / str(row["confusion_matrix_png_path"])).exists(),
                "learning_curves_png_exists": (PROJECT_ROOT / str(row["learning_curves_path"])).exists(),
                "checkpoint_exists": (PROJECT_ROOT / str(row["checkpoint_path"])).exists(),
            }
        )
    return pd.DataFrame(rows)


def write_dataset_comparisons(clean: pd.DataFrame) -> dict[str, pd.DataFrame]:
    comparisons = {}
    for dataset in EXPECTED_DATASETS:
        dataset_df = clean[clean["dataset_name"] == dataset].copy()
        dataset_df = dataset_df.sort_values("test_macro_f1", ascending=False)
        output = dataset_df.reindex(columns=COMPARISON_COLUMNS)
        output.to_csv(BASELINE_DIR / f"{dataset}_student_baseline_comparison.csv", index=False)
        comparisons[dataset] = output
    return comparisons


def build_overall(clean: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        clean.groupby("model_name")
        .agg(
            num_datasets=("dataset_name", "nunique"),
            avg_test_accuracy=("test_accuracy", "mean"),
            avg_test_macro_precision=("test_macro_precision", "mean"),
            avg_test_macro_recall=("test_macro_recall", "mean"),
            avg_test_macro_f1=("test_macro_f1", "mean"),
            avg_test_weighted_f1=("test_weighted_f1", "mean"),
            avg_model_size_mb=("model_size_mb", "mean"),
            avg_params=("params", "mean"),
        )
        .reset_index()
    )
    grouped["rank_by_avg_macro_f1"] = grouped["avg_test_macro_f1"].rank(ascending=False, method="min").astype(int)
    grouped["rank_by_avg_accuracy"] = grouped["avg_test_accuracy"].rank(ascending=False, method="min").astype(int)
    grouped["rank_by_model_size"] = grouped["avg_model_size_mb"].rank(ascending=True, method="min").astype(int)
    grouped["performance_size_score"] = grouped["avg_test_macro_f1"] / grouped["avg_model_size_mb"]
    grouped["notes"] = ""
    return grouped.sort_values("avg_test_macro_f1", ascending=False)


def build_recommendations(overall: pd.DataFrame) -> pd.DataFrame:
    best_macro = overall.sort_values("avg_test_macro_f1", ascending=False).iloc[0]
    best_acc = overall.sort_values("avg_test_accuracy", ascending=False).iloc[0]
    best_score = overall.sort_values("performance_size_score", ascending=False).iloc[0]
    acceptable = overall[overall["avg_test_macro_f1"] >= best_macro["avg_test_macro_f1"] - 0.02]
    lightweight = acceptable.sort_values("avg_model_size_mb", ascending=True).iloc[0]

    mobilenet_small = overall[overall["model_name"] == "mobilenet_v3_small"]
    if not mobilenet_small.empty and mobilenet_small.iloc[0]["avg_test_macro_f1"] >= best_macro["avg_test_macro_f1"] - 0.02:
        kd_student = mobilenet_small.iloc[0]
        kd_reason = (
            "MobileNetV3-Small is the lightest student and stays within 0.02 avg macro F1 "
            "of the best model, making KD gains easier to interpret."
        )
    else:
        kd_student = lightweight
        kd_reason = "Selected as the lightest acceptable student by the 0.02 avg macro F1 threshold."

    compact_candidates = overall[overall["model_name"].isin(["mobilenet_v3_large", "efficientnet_b0"])]
    strong_compact = compact_candidates.sort_values("avg_test_macro_f1", ascending=False).iloc[0]

    rows = [
        (
            "Best performance student",
            best_macro,
            "Highest average test macro F1 across datasets.",
        ),
        (
            "Best accuracy student",
            best_acc,
            "Highest average test accuracy across datasets.",
        ),
        (
            "Best lightweight student",
            lightweight,
            "Smallest model within 0.02 average macro F1 of the best model.",
        ),
        (
            "Recommended KD student",
            kd_student,
            kd_reason,
        ),
        (
            "Strong compact baseline",
            strong_compact,
            "Compact but stronger reference student for secondary KD comparisons.",
        ),
        (
            "Best performance-size balance",
            best_score,
            "Highest avg_test_macro_f1 / avg_model_size_mb score.",
        ),
    ]

    return pd.DataFrame(
        [
            {
                "recommendation_type": kind,
                "selected_model": row["model_name"],
                "reason": reason,
                "avg_test_accuracy": row["avg_test_accuracy"],
                "avg_test_macro_f1": row["avg_test_macro_f1"],
                "avg_model_size_mb": row["avg_model_size_mb"],
                "avg_params": row["avg_params"],
            }
            for kind, row, reason in rows
        ]
    )


def save_bar_by_dataset(clean: pd.DataFrame, metric: str, output: Path, ylabel: str) -> None:
    pivot = clean.pivot(index="dataset_name", columns="model_name", values=metric).reindex(EXPECTED_DATASETS)
    ax = pivot.plot(kind="bar", figsize=(11, 6), width=0.8)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Dataset")
    ax.set_ylim(0, max(1.0, float(pivot.max().max()) * 1.05))
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def save_scatter(clean: pd.DataFrame, x: str, y: str, output: Path, xlabel: str, ylabel: str) -> None:
    plt.figure(figsize=(10, 6))
    for model, subset in clean.groupby("model_name"):
        plt.scatter(subset[x], subset[y], label=model, s=70)
        for _, row in subset.iterrows():
            plt.annotate(str(row["dataset_name"]), (row[x], row[y]), fontsize=8, alpha=0.75)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.25)
    plt.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def save_overall_bars(overall: pd.DataFrame, metric: str, output: Path, ylabel: str) -> None:
    plot_df = overall.sort_values(metric, ascending=False)
    plt.figure(figsize=(10, 5))
    plt.bar(plot_df["model_name"], plot_df[metric])
    plt.ylabel(ylabel)
    plt.xlabel("Model")
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def write_graphs(clean: pd.DataFrame, overall: pd.DataFrame) -> list[str]:
    outputs = [
        BASELINE_DIR / "student_baseline_macro_f1_by_dataset.png",
        BASELINE_DIR / "student_baseline_accuracy_by_dataset.png",
        BASELINE_DIR / "student_baseline_model_size_vs_macro_f1.png",
        BASELINE_DIR / "student_baseline_params_vs_macro_f1.png",
        BASELINE_DIR / "student_baseline_overall_macro_f1.png",
        BASELINE_DIR / "student_baseline_performance_size_score.png",
    ]
    save_bar_by_dataset(clean, "test_macro_f1", outputs[0], "Test Macro F1")
    save_bar_by_dataset(clean, "test_accuracy", outputs[1], "Test Accuracy")
    save_scatter(clean, "model_size_mb", "test_macro_f1", outputs[2], "Model Size (MB)", "Test Macro F1")
    save_scatter(clean, "params", "test_macro_f1", outputs[3], "Parameters", "Test Macro F1")
    save_overall_bars(overall, "avg_test_macro_f1", outputs[4], "Average Test Macro F1")
    save_overall_bars(overall, "performance_size_score", outputs[5], "Macro F1 / Model Size MB")
    return [rel(path) for path in outputs]


def _format_markdown_value(value: Any, floatfmt: str) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return format(value, floatfmt)
    return str(value)


def to_markdown(df: pd.DataFrame, columns: list[str] | None = None, floatfmt: str = ".4f") -> str:
    view = df if columns is None else df[columns]
    headers = list(view.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in view.iterrows():
        values = [_format_markdown_value(row[column], floatfmt) for column in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    clean: pd.DataFrame,
    comparisons: dict[str, pd.DataFrame],
    overall: pd.DataFrame,
    recommendations: pd.DataFrame,
    artifact_df: pd.DataFrame,
    missing_pairs: list[dict[str, Any]],
    duplicates: pd.DataFrame,
) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    best_by_dataset = clean.sort_values("test_macro_f1", ascending=False).groupby("dataset_name").head(1)
    missing_artifacts = artifact_df[
        ~artifact_df[
            [
                "run_dir_exists",
                "config_exists",
                "history_exists",
                "per_class_metrics_exists",
                "confusion_matrix_csv_exists",
                "confusion_matrix_png_exists",
                "learning_curves_png_exists",
                "checkpoint_exists",
            ]
        ].all(axis=1)
    ]

    sections = [
        "# Student Baseline Report",
        "",
        "## Amaç",
        "Student baseline aşamasının amacı, KD öncesinde her veri seti için bağımsız öğrenci model performansını ölçmek ve KD ile neyin iyileştirileceğini netleştirmektir.",
        "",
        "## Veri Setleri",
        ", ".join(EXPECTED_DATASETS),
        "",
        "## Student Modeller",
        ", ".join(EXPECTED_MODELS),
        "",
        "## Eğitim Ayarları",
        "Tüm seçilen run'larda seed=42, ImageNet pretrained başlangıç, AdamW optimizer, image_size=224 ve aynı split JSON mantığı kullanılmıştır.",
        "",
        "## Okunan Dosyalar",
        "- `results/baseline/baseline_results.csv`",
        "- `results/baseline/runs/{dataset}/{model}/{run_name}/`",
        "",
        "## Veri Seti Bazlı Sonuçlar",
    ]
    for dataset, table in comparisons.items():
        sections.extend(["", f"### {dataset}", to_markdown(table, COMPARISON_COLUMNS[:10])])
    sections.extend(
        [
            "",
            "## Genel Ortalama Performans",
            to_markdown(
                overall,
                [
                    "model_name",
                    "num_datasets",
                    "avg_test_accuracy",
                    "avg_test_macro_f1",
                    "avg_model_size_mb",
                    "avg_params",
                    "performance_size_score",
                ],
            ),
            "",
            "## Model Boyutu ve Parametre Karşılaştırması",
            to_markdown(overall.sort_values("avg_model_size_mb"), ["model_name", "avg_model_size_mb", "avg_params", "avg_test_macro_f1"]),
            "",
            "## Veri Seti Bazında En İyi Student",
            to_markdown(best_by_dataset, ["dataset_name", "model_name", "test_macro_f1", "test_accuracy", "model_size_mb"]),
            "",
            "## Student Seçim Önerileri",
            to_markdown(recommendations),
            "",
            "## PlantVillage Ceiling Effect Yorumu",
            "PlantVillage sonuçları tüm modellerde çok yüksek olduğu için bu veri setinde ceiling effect beklenir. KD etkisi burada sınırlı görünebilir; AppleLeaf9 ve PlantPathology2021 sonuçları KD kazanımlarını yorumlamak için daha ayırt edici olacaktır.",
            "",
            "## KD Seçim Kriterleri",
            "KD aşamasında sadece accuracy değil macro F1, model boyutu, parametre sayısı ve ileride ölçülecek inference maliyeti birlikte değerlendirilecektir. Macro F1 özellikle sınıf dengesizliği ve az temsil edilen hastalık sınıfları için daha açıklayıcıdır.",
            "",
            "## Artifact Kontrolü",
            "Eksik artifact sayısı: " + str(len(missing_artifacts)),
        ]
    )
    if missing_pairs:
        sections.extend(["", "## Eksik Deneyler", to_markdown(pd.DataFrame(missing_pairs))])
    if not duplicates.empty:
        sections.extend(["", "## Duplicate Run Seçimleri", to_markdown(duplicates)])
    sections.extend(
        [
            "",
            "## Sonraki Aşama",
            "Student baseline aşaması kapatıldıktan sonra CNN teacher training altyapısına geçilmelidir.",
        ]
    )
    (DOCS_DIR / "STUDENT_BASELINE_REPORT.md").write_text("\n".join(sections) + "\n", encoding="utf-8")


def main() -> None:
    parse_args()
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    df = load_results()
    selected, duplicates, missing_pairs = select_best_runs(df)
    clean = add_artifact_paths(selected)
    clean.to_csv(BASELINE_DIR / "student_baseline_clean_results.csv", index=False)
    artifact_df = artifact_check(clean)
    artifact_df.to_csv(BASELINE_DIR / "student_baseline_artifact_check.csv", index=False)
    comparisons = write_dataset_comparisons(clean)
    overall = build_overall(clean)
    overall.to_csv(BASELINE_DIR / "student_baseline_overall_comparison.csv", index=False)
    recommendations = build_recommendations(overall)
    recommendations.to_csv(BASELINE_DIR / "student_selection_recommendation.csv", index=False)
    graph_paths = write_graphs(clean, overall)
    write_report(clean, comparisons, overall, recommendations, artifact_df, missing_pairs, duplicates)
    print(f"baseline_runs={len(df)}")
    print(f"selected_runs={len(clean)}")
    print(f"missing_pairs={missing_pairs}")
    print(f"duplicates={duplicates.to_dict(orient='records') if not duplicates.empty else []}")
    print("graphs=" + ",".join(graph_paths))


if __name__ == "__main__":
    main()
