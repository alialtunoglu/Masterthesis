"""Baseline and student selection result explorer page."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from result_loader import load_csv_if_exists
from ui_utils import get_project_root, show_dataframe_or_warning


BASELINE_TABLES = {
    "Raw baseline results": "results/baseline/baseline_results.csv",
    "Clean student baseline results": "results/baseline/student_baseline_clean_results.csv",
    "Overall student comparison": "results/baseline/student_baseline_overall_comparison.csv",
    "Student recommendation": "results/baseline/student_selection_recommendation.csv",
    "Artifact check": "results/baseline/student_baseline_artifact_check.csv",
}

STUDENT_FIGURES = [
    "results/baseline/student_baseline_macro_f1_by_dataset.png",
    "results/baseline/student_baseline_accuracy_by_dataset.png",
    "results/baseline/student_baseline_model_size_vs_macro_f1.png",
    "results/baseline/student_baseline_params_vs_macro_f1.png",
    "results/baseline/student_baseline_overall_macro_f1.png",
    "results/baseline/student_baseline_performance_size_score.png",
]


def _filter_multiselect(df: pd.DataFrame, column: str, label: str) -> pd.DataFrame:
    if column not in df.columns:
        return df
    options = sorted(df[column].dropna().astype(str).unique().tolist())
    selected = st.multiselect(label, options, default=options)
    if not selected:
        return df.iloc[0:0]
    return df[df[column].astype(str).isin(selected)]


def _resolve_path(value: object) -> Path | None:
    if value is None or pd.isna(value):
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = get_project_root() / path
    return path


def _show_run_artifacts(df: pd.DataFrame) -> None:
    if df.empty or "run_name" not in df.columns:
        return
    st.subheader("Run Artifactleri")
    run_labels = df["run_name"].astype(str).tolist()
    selected_label = st.selectbox("Run seç", run_labels)
    selected_row = df[df["run_name"].astype(str) == selected_label].iloc[0]

    artifact_cols = [
        "checkpoint_path",
        "run_dir",
        "config_path",
        "history_path",
        "per_class_metrics_path",
        "confusion_matrix_png_path",
        "learning_curves_path",
    ]
    artifact_rows = []
    for column in artifact_cols:
        if column in df.columns:
            path = _resolve_path(selected_row.get(column))
            artifact_rows.append(
                {
                    "artifact": column,
                    "path": str(path) if path else "",
                    "exists": bool(path and path.exists()),
                }
            )
    st.dataframe(pd.DataFrame(artifact_rows), use_container_width=True)

    image_cols = st.columns(2)
    confusion_path = _resolve_path(selected_row.get("confusion_matrix_png_path"))
    curves_path = _resolve_path(selected_row.get("learning_curves_path"))
    with image_cols[0]:
        st.subheader("Confusion Matrix")
        if confusion_path and confusion_path.exists():
            st.image(str(confusion_path))
        else:
            st.info("Confusion matrix PNG bulunamadı.")
    with image_cols[1]:
        st.subheader("Learning Curves")
        if curves_path and curves_path.exists():
            st.image(str(curves_path))
        else:
            st.info("Learning curves PNG bulunamadı.")


st.set_page_config(page_title="Results Explorer", layout="wide")
st.title("Results Explorer")

tabs = st.tabs(["Tables", "Student Figures", "Run Artifacts"])

with tabs[0]:
    selected_table_name = st.selectbox("Tablo seç", list(BASELINE_TABLES))
    selected_table_path = BASELINE_TABLES[selected_table_name]
    df = load_csv_if_exists(selected_table_path)
    st.caption(selected_table_path)
    if df is None or df.empty:
        st.info(f"{selected_table_path} henüz yok veya boş.")
    else:
        with st.sidebar:
            st.header("Filtreler")
            filtered = _filter_multiselect(df, "dataset_name", "Dataset")
            filtered = _filter_multiselect(filtered, "model_name", "Model")
            filtered = _filter_multiselect(filtered, "device", "Device")
            if "pretrained" in filtered.columns:
                pretrained_values = sorted(filtered["pretrained"].dropna().astype(str).unique().tolist())
                selected_pretrained = st.multiselect("Pretrained", pretrained_values, default=pretrained_values)
                filtered = filtered[filtered["pretrained"].astype(str).isin(selected_pretrained)]

            sort_options = [
                column
                for column in ["test_macro_f1", "test_accuracy", "best_val_macro_f1", "avg_test_macro_f1"]
                if column in filtered.columns
            ]
            if sort_options:
                sort_metric = st.selectbox("Sıralama", sort_options)
                ascending = st.checkbox("Artan sırala", value=False)
                filtered = filtered.sort_values(sort_metric, ascending=ascending)

        show_dataframe_or_warning(filtered, "Seçilen filtrelerle sonuç bulunamadı.")

        if {"model_name", "test_accuracy", "test_macro_f1"}.issubset(filtered.columns):
            chart_cols = st.columns(2)
            grouped = filtered.copy()
            grouped["test_accuracy"] = pd.to_numeric(grouped["test_accuracy"], errors="coerce")
            grouped["test_macro_f1"] = pd.to_numeric(grouped["test_macro_f1"], errors="coerce")
            model_scores = grouped.groupby("model_name", as_index=True)[["test_accuracy", "test_macro_f1"]].mean(numeric_only=True)
            with chart_cols[0]:
                st.subheader("Model Bazında Test Accuracy")
                st.bar_chart(model_scores["test_accuracy"])
            with chart_cols[1]:
                st.subheader("Model Bazında Test Macro F1")
                st.bar_chart(model_scores["test_macro_f1"])

with tabs[1]:
    st.subheader("Student Baseline Figures")
    for figure in STUDENT_FIGURES:
        path = get_project_root() / figure
        if path.exists():
            st.caption(figure)
            st.image(str(path))
        else:
            st.warning(f"Eksik grafik: {figure}")

with tabs[2]:
    df = load_csv_if_exists("results/baseline/student_baseline_clean_results.csv")
    if df is None or df.empty:
        df = load_csv_if_exists("results/baseline/baseline_results.csv")
    if df is None or df.empty:
        st.info("Gösterilecek run artifact tablosu yok.")
    else:
        filtered = _filter_multiselect(df, "dataset_name", "Dataset")
        filtered = _filter_multiselect(filtered, "model_name", "Model")
        _show_run_artifacts(filtered)
