"""CNN teacher experiment launcher page."""

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

from experiment_runner import build_teacher_command, list_jobs, read_log_tail, start_job
from portable_notebook import render_notebook_download
from result_loader import safe_read_json
from ui_utils import get_project_root


DATASETS = ["appleleaf9", "plantvillage", "plantpathology2021"]
CNN_TEACHERS = [
    "resnet50",
    "resnet101",
    "densenet121",
    "densenet201",
    "vgg19_bn",
    "efficientnet_b3",
    "efficientnet_b4",
    "convnext_tiny",
    "convnext_base",
    "regnet_y_8gf",
]


def quote_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def load_config_index() -> list[dict[str, Any]]:
    rows = []
    root = get_project_root()
    config_root = root / "configs" / "teachers" / "cnn"
    for path in sorted(config_root.glob("*/*.json")):
        relative = path.relative_to(root).as_posix()
        data = safe_read_json(relative) or {}
        rows.append(
            {
                "path": relative,
                "file": path.name,
                "dataset_name": data.get("dataset_name", ""),
                "model_name": data.get("model_name", ""),
                "epochs": data.get("epochs", ""),
                "stage": data.get("stage", "teacher_cnn"),
            }
        )
    return rows


def config_label(row: dict[str, Any]) -> str:
    return f"{row['stage']} | {row['dataset_name']} | {row['model_name']} | epochs={row['epochs']} | {row['file']}"


st.set_page_config(page_title="CNN Teacher Experiments", layout="wide")
st.title("CNN Teacher Experiments")
st.caption("CNN teacher dry-run ve eğitim job'ları hazırlar. Bu sayfa queue sistemini kullanır.")

configs = load_config_index()
filter_cols = st.columns(3)
with filter_cols[0]:
    dataset_filter = st.selectbox("Dataset filtresi", ["all"] + DATASETS)
with filter_cols[1]:
    model_filter = st.selectbox("Model filtresi", ["all"] + CNN_TEACHERS)
with filter_cols[2]:
    search = st.text_input("Config ara", value="")

filtered_configs = []
for row in configs:
    if dataset_filter != "all" and row["dataset_name"] != dataset_filter:
        continue
    if model_filter != "all" and row["model_name"] != model_filter:
        continue
    if search and search.lower() not in config_label(row).lower():
        continue
    filtered_configs.append(row)

labels = [config_label(row) for row in filtered_configs]
selected_label = st.selectbox("Config dosyası", [""] + labels)
selected_config = ""
if selected_label:
    selected_config = filtered_configs[labels.index(selected_label)]["path"]
defaults = safe_read_json(selected_config) if selected_config else {}
defaults = defaults or {}

left, right = st.columns(2)
with left:
    dataset = st.selectbox("Dataset", DATASETS, index=DATASETS.index(defaults.get("dataset_name", "appleleaf9")) if defaults.get("dataset_name", "appleleaf9") in DATASETS else 0)
    model = st.selectbox("CNN teacher model", CNN_TEACHERS, index=CNN_TEACHERS.index(defaults.get("model_name", "resnet50")) if defaults.get("model_name", "resnet50") in CNN_TEACHERS else 0)
    pretrained = st.checkbox("Pretrained", value=bool(defaults.get("pretrained", True)))
    dry_run = st.checkbox("Dry run", value=False)
with right:
    epochs = st.number_input("Epochs", min_value=1, max_value=200, value=int(defaults.get("epochs", 15)))
    batch_size = st.number_input("Batch size", min_value=1, max_value=256, value=int(defaults.get("batch_size", 32)))
    image_size = st.number_input("Image size", min_value=32, max_value=1024, value=int(defaults.get("image_size", 224)))
    learning_rate = st.number_input("Learning rate", min_value=0.0, value=float(defaults.get("learning_rate", 0.0003)), format="%.6f")
    weight_decay = st.number_input("Weight decay", min_value=0.0, value=float(defaults.get("weight_decay", 0.0001)), format="%.6f")
    tracking_uri = st.text_input("MLflow tracking URI", value=str(defaults.get("tracking_uri") or "sqlite:///mlflow.db"))

st.subheader("Smoke Test Limitleri")
limit_cols = st.columns(3)
with limit_cols[0]:
    max_train_batches = st.number_input("max_train_batches", min_value=0, value=0)
with limit_cols[1]:
    max_val_batches = st.number_input("max_val_batches", min_value=0, value=0)
with limit_cols[2]:
    max_test_batches = st.number_input("max_test_batches", min_value=0, value=0)

command = build_teacher_command(
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
st.code(quote_command(command), language="powershell")

button_cols = st.columns(2)
with button_cols[0]:
    if st.button("Teacher Dry Run Başlat", type="secondary"):
        dry_command = build_teacher_command(
            dataset,
            model,
            selected_config or None,
            int(epochs),
            int(batch_size),
            int(image_size),
            float(learning_rate),
            float(weight_decay),
            pretrained,
            True,
            int(max_train_batches) or None,
            int(max_val_batches) or None,
            int(max_test_batches) or None,
            tracking_uri,
        )
        job = start_job(dry_command, {"stage": "teacher_cnn", "dataset": dataset, "model": model, "config": selected_config, "dry_run": True})
        st.success(f"Teacher dry-run kuyruğa eklendi: {job['job_id']} | status={job['status']}")
with button_cols[1]:
    st.warning("Gerçek teacher eğitimi uzun sürebilir ve GPU kullanır.")
    confirmed = st.checkbox("Uzun süren eğitimi onaylıyorum")
    if dry_run:
        st.caption("Dry run açıkken gerçek eğitim başlatılamaz; dry-run butonunu kullanın.")
    if st.button("Teacher Eğitimi Başlat", type="primary", disabled=dry_run or not confirmed):
        job = start_job(command, {"stage": "teacher_cnn", "dataset": dataset, "model": model, "config": selected_config, "dry_run": dry_run})
        st.success(f"Teacher eğitim işi kuyruğa eklendi: {job['job_id']} | status={job['status']}")

render_notebook_download(
    stage="teacher_cnn",
    dataset=dataset,
    model=model,
    command=command,
    key="cnn_teacher_notebook_download",
)

st.subheader("Aktif / Son İşler")
jobs = list_jobs()
if jobs:
    jobs_df = pd.DataFrame(jobs)
    display_cols = ["job_id", "stage", "dataset", "model", "status", "queued_time", "start_time", "process_id", "log_path"]
    st.dataframe(jobs_df[[col for col in display_cols if col in jobs_df.columns]], width="stretch")
    selected_job = st.selectbox("Log görüntülenecek job", [job["job_id"] for job in jobs])
    selected = next(job for job in jobs if job["job_id"] == selected_job)
    st.page_link("pages/3_Job_Monitor.py", label="Canlı takip için Job Monitor'a git")
    n_lines = st.slider("Log satır sayısı", min_value=20, max_value=500, value=20, step=20)
    st.code(read_log_tail(selected.get("log_path", ""), n_lines=n_lines), language="text")
else:
    st.info("Henüz job metadata dosyası yok.")
