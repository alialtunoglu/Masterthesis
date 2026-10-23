"""Streamlit presentation for the read-only experiment comparison core."""

from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import streamlit as st

from src.evaluation.experiment_comparison import build_comparison, export_comparison, load_comparison_runs


def table_frames(report: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.DataFrame(report["rows"]), pd.DataFrame(report["class_rows"])


def report_downloads(report: dict) -> dict[str, bytes]:
    return {label: export_comparison(report, format_) for label, format_ in (("Markdown", "markdown"), ("CSV", "csv"), ("LaTeX", "latex"), ("JSON", "json"))}


def student_options(runs: list[dict]) -> list[str]:
    return sorted({str(run["student"]) for run in runs
                   if run.get("student") and run.get("role") in {"baseline", "single_kd", "multi_kd"}})


def render_comparison(root: Path) -> None:
    st.subheader("Karşılaştırma ve Rapor")
    runs = load_comparison_runs(root)
    if not runs:
        st.info("Karşılaştırılabilir tamamlanmış sonuç bulunamadı.")
        return
    datasets = sorted({str(run["dataset"]) for run in runs})
    dataset = st.selectbox("Dataset", datasets, key="comparison_dataset")
    available = [run for run in runs if str(run["dataset"]) == dataset]
    identities = sorted({json.dumps(run.get("dataset_ref"), sort_keys=True) for run in available})
    if any(value != "null" for value in identities):
        identity = st.selectbox("Veri sürümü / bölme", identities, key="comparison_identity",
                                format_func=lambda value: "Legacy (byte hash yok)" if value == "null" else json.loads(value)["revision"][:12] + " / " + json.loads(value)["split_id"][:12])
        available = [run for run in available if json.dumps(run.get("dataset_ref"), sort_keys=True) == identity]
    filters = st.columns(3)
    students = student_options(available)
    if not students:
        st.info("Bu veri setinde karşılaştırılabilir student sonucu yok.")
        return
    student = filters[0].selectbox("Student", students, key="comparison_student")
    seeds = sorted({run["seed"] for run in available if run.get("seed") is not None})
    seed = filters[1].selectbox("Seed", seeds, key="comparison_seed")
    kd_types = sorted({str(run["kd_type"]) for run in available if run.get("kd_type")})
    kd_type = filters[2].selectbox("KD türü", ["Tümü", *kd_types], key="comparison_kd_type")
    available = [run for run in available if run.get("student") == student and run.get("seed") == seed]
    if kd_type != "Tümü":
        available = [run for run in available if run.get("role") == "baseline" or run.get("kd_type") == kd_type]
    labels = {
        run["run_id"]: " | ".join(filter(None, (
            str(run["role"]), str(run["model"]),
            " + ".join(run.get("teacher_models", [])), str(run.get("kd_type") or ""),
            f"seed={run['seed']}", str(run["run_id"]),
        )))
        for run in available
    }
    selected = st.multiselect("Run / job ID seç", list(labels), default=list(labels), format_func=labels.get, key="comparison_runs")
    baselines = [run["run_id"] for run in available if run["role"] == "baseline"]
    baseline = st.selectbox("Baseline", [None, *baselines], format_func=lambda value: "Otomatik (validation Macro-F1)" if value is None else labels[value], key="comparison_baseline")
    report = build_comparison(runs, selected, baseline)
    if any(run.get("compatibility") == "legacy_unknown" for run in available):
        st.info("Bazı eski kayıtların dataset/protokol fingerprint'i yok; bunlar otomatik kontrollü KD karşılaştırmasına alınmaz.")
    for warning in report["warnings"]:
        st.warning(warning)
    main, classes = table_frames(report)
    display_main = main.copy()
    for column in ("val_macro_f1", "test_accuracy", "test_macro_f1"):
        if column in display_main:
            display_main[column] = 100 * display_main[column]
    st.dataframe(display_main, hide_index=True, width="stretch")
    st.subheader("Sınıf Bazlı Sonuçlar")
    display_classes = classes.copy()
    for column in ("precision", "recall", "f1"):
        if column in display_classes:
            display_classes[column] = 100 * display_classes[column]
    st.dataframe(display_classes, hide_index=True, width="stretch")
    if not main.empty and "test_macro_f1" in main:
        st.bar_chart(main.set_index("run_id")["test_macro_f1"])
        if "params" in main:
            st.scatter_chart(main, x="params", y="test_macro_f1", color="role")
    for label, data in report_downloads(report).items():
        extension = {"Markdown": "md", "CSV": "csv", "LaTeX": "tex", "JSON": "json"}[label]
        st.download_button(f"{label} indir", data, f"experiment_comparison.{extension}", key=f"comparison_download_{extension}")
