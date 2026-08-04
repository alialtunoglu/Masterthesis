"""Baseline experiment launcher page."""

from __future__ import annotations

import shlex
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from experiment_runner import (
    build_baseline_command,
    load_dashboard_settings,
    list_jobs,
    read_log_tail,
    start_job,
)
from portable_notebook import render_notebook_download
from result_loader import safe_read_json
from ui_utils import get_project_root


DATASETS = ["appleleaf9", "plantvillage", "plantpathology2021"]
MODELS = ["mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_b0", "resnet18"]
CONFIG_ROOTS = {
    "baseline": "configs/baseline",
}


def _quote_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _load_config_index() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    project_root = get_project_root()
    for stage, relative_dir in CONFIG_ROOTS.items():
        config_dir = project_root / relative_dir
        for path in sorted(config_dir.glob("*.json")):
            relative_path = path.relative_to(project_root).as_posix()
            data = safe_read_json(relative_path) or {}
            rows.append(
                {
                    "path": relative_path,
                    "file": path.name,
                    "stage": data.get("stage", stage),
                    "dataset_name": data.get("dataset_name", ""),
                    "model_name": data.get("model_name", ""),
                    "epochs": data.get("epochs", ""),
                    "pretrained": data.get("pretrained", ""),
                }
            )
    return rows


def _filter_configs(
    configs: list[dict[str, Any]],
    stage: str,
    dataset: str,
    model: str,
    search: str,
) -> list[dict[str, Any]]:
    search = search.strip().lower()
    filtered = []
    for row in configs:
        if stage != "all" and row["stage"] != stage:
            continue
        if dataset != "all" and row["dataset_name"] != dataset:
            continue
        if model != "all" and row["model_name"] != model:
            continue
        haystack = " ".join(str(row.get(key, "")) for key in ["path", "stage", "dataset_name", "model_name"]).lower()
        if search and search not in haystack:
            continue
        filtered.append(row)
    return filtered


def _config_label(row: dict[str, Any]) -> str:
    return (
        f"{row['stage']} | {row['dataset_name']} | {row['model_name']} | "
        f"epochs={row['epochs']} | {row['file']}"
    )


def _load_config_defaults(config_path: str | None) -> dict:
    if not config_path:
        return {}
    return safe_read_json(config_path) or {}


st.set_page_config(page_title="Baseline Experiments", layout="wide")
st.title("Baseline Experiments")
st.caption("Bu sayfa yalnızca baseline eğitim komutlarını hazırlar ve yerel subprocess olarak başlatır.")
settings = load_dashboard_settings()
st.info(f"Queue aktif. Aynı anda çalışacak maksimum eğitim sayısı: {settings['max_parallel_jobs']}")

config_index = _load_config_index()
st.subheader("Config Seçimi")
filter_cols = st.columns(4)
with filter_cols[0]:
    stage_filter = st.selectbox("Stage filtresi", ["baseline", "all"])
with filter_cols[1]:
    dataset_filter = st.selectbox("Dataset filtresi", ["all"] + DATASETS)
with filter_cols[2]:
    model_filter = st.selectbox("Model filtresi", ["all"] + MODELS)
with filter_cols[3]:
    search_filter = st.text_input("Config ara", value="")

filtered_configs = _filter_configs(config_index, stage_filter, dataset_filter, model_filter, search_filter)
st.caption(f"{len(filtered_configs)} config gösteriliyor / toplam {len(config_index)} config")
config_labels = [_config_label(row) for row in filtered_configs]
selected_label = st.selectbox("Config dosyası", [""] + config_labels)
selected_config = ""
if selected_label:
    selected_config = filtered_configs[config_labels.index(selected_label)]["path"]
defaults = _load_config_defaults(selected_config)

left, right = st.columns(2)
with left:
    dataset = st.selectbox(
        "Dataset",
        DATASETS,
        index=DATASETS.index(defaults.get("dataset_name", "appleleaf9"))
        if defaults.get("dataset_name", "appleleaf9") in DATASETS
        else 0,
    )
    model = st.selectbox(
        "Model",
        MODELS,
        index=MODELS.index(defaults.get("model_name", "mobilenet_v3_small"))
        if defaults.get("model_name", "mobilenet_v3_small") in MODELS
        else 0,
    )
    pretrained = st.checkbox("Pretrained", value=bool(defaults.get("pretrained", True)))
    dry_run = st.checkbox("Dry run", value=False)

with right:
    epochs = st.number_input("Epochs", min_value=1, max_value=200, value=int(defaults.get("epochs", 5)))
    batch_size = st.number_input("Batch size", min_value=1, max_value=512, value=int(defaults.get("batch_size", 32)))
    image_size = st.number_input("Image size", min_value=32, max_value=1024, value=int(defaults.get("image_size", 224)))
    learning_rate = st.number_input(
        "Learning rate",
        min_value=0.0,
        value=float(defaults.get("learning_rate", 0.001)),
        format="%.6f",
    )
    weight_decay = st.number_input(
        "Weight decay",
        min_value=0.0,
        value=float(defaults.get("weight_decay", 0.0001)),
        format="%.6f",
    )
    tracking_uri = st.text_input("MLflow tracking URI", value=str(defaults.get("tracking_uri") or "sqlite:///mlflow.db"))

st.subheader("Smoke Test Limitleri")
limit_cols = st.columns(3)
with limit_cols[0]:
    max_train_batches = st.number_input("max_train_batches", min_value=0, value=0)
with limit_cols[1]:
    max_val_batches = st.number_input("max_val_batches", min_value=0, value=0)
with limit_cols[2]:
    max_test_batches = st.number_input("max_test_batches", min_value=0, value=0)

command = build_baseline_command(
    dataset=dataset,
    model=model,
    config=selected_config or None,
    epochs=int(epochs),
    batch_size=int(batch_size),
    image_size=int(image_size),
    learning_rate=float(learning_rate),
    weight_decay=float(weight_decay),
    pretrained=pretrained,
    dry_run=dry_run,
    max_train_batches=int(max_train_batches) or None,
    max_val_batches=int(max_val_batches) or None,
    max_test_batches=int(max_test_batches) or None,
    tracking_uri=tracking_uri,
)

st.subheader("Komut Önizlemesi")
st.code(_quote_command(command), language="powershell")
render_notebook_download(
    stage="baseline",
    dataset=dataset,
    model=model,
    command=command,
    key="baseline_notebook_download",
)

button_cols = st.columns(2)
with button_cols[0]:
    if st.button("Dry Run Başlat", type="secondary"):
        dry_command = build_baseline_command(
            dataset=dataset,
            model=model,
            config=selected_config or None,
            epochs=int(epochs),
            batch_size=int(batch_size),
            image_size=int(image_size),
            learning_rate=float(learning_rate),
            weight_decay=float(weight_decay),
            pretrained=pretrained,
            dry_run=True,
            max_train_batches=int(max_train_batches) or None,
            max_val_batches=int(max_val_batches) or None,
            max_test_batches=int(max_test_batches) or None,
            tracking_uri=tracking_uri,
        )
        job = start_job(
            dry_command,
            {
                "dataset": dataset,
                "model": model,
                "config": selected_config,
                "dry_run": True,
            },
        )
        st.success(f"Dry-run kuyruğa eklendi: {job['job_id']} | status={job['status']}")

with button_cols[1]:
    st.warning("Bu işlem GPU kullanabilir ve uzun sürebilir.")
    if st.button("Eğitimi Başlat", type="primary"):
        job = start_job(
            command,
            {
                "dataset": dataset,
                "model": model,
                "config": selected_config,
                "dry_run": dry_run,
            },
        )
        st.success(f"Eğitim işi kuyruğa eklendi: {job['job_id']} | status={job['status']}")

st.subheader("Aktif / Son İşler")
jobs = list_jobs()
if jobs:
    jobs_df = pd.DataFrame(jobs)
    display_cols = ["job_id", "dataset", "model", "status", "queued_time", "start_time", "process_id", "log_path"]
    st.dataframe(jobs_df[[col for col in display_cols if col in jobs_df.columns]], use_container_width=True)

    selected_job = st.selectbox("Log görüntülenecek job", [job["job_id"] for job in jobs])
    selected = next(job for job in jobs if job["job_id"] == selected_job)
    st.page_link("pages/3_Job_Monitor.py", label="Canlı takip için Job Monitor'a git")
    n_lines = st.slider("Log satır sayısı", min_value=20, max_value=500, value=20, step=20)
    st.code(read_log_tail(selected.get("log_path", ""), n_lines=n_lines), language="text")
else:
    st.info("Henüz job metadata dosyası yok.")
