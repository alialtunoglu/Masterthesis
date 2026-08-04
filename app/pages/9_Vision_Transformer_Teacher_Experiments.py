"""Vision Transformer teacher experiment launcher."""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

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
VIT_TEACHERS = ["swin_v2_t", "maxvit_t", "vit_b_16", "dinov2_vitb14"]


def quote_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


st.set_page_config(page_title="Vision Transformer Teacher Experiments", layout="wide")
st.title("Vision Transformer Teacher Experiments")
st.caption("ViT teacher dry-run ve eğitimlerini mevcut queue sistemiyle çalıştırır.")

selection = st.columns(2)
with selection[0]:
    dataset = st.selectbox("Dataset", DATASETS)
with selection[1]:
    model = st.selectbox("Vision Transformer teacher model", VIT_TEACHERS)

config_root = get_project_root() / "configs" / "teachers" / "vision_transformers"
config_paths = sorted((config_root / dataset).glob(f"{model}.json"))
config_options = [path.relative_to(get_project_root()).as_posix() for path in config_paths]
selected_config = st.selectbox("Config dosyası", config_options)
defaults = safe_read_json(selected_config) or {}

left, right = st.columns(2)
with left:
    pretrained = st.checkbox("Pretrained", value=bool(defaults.get("pretrained", True)))
    mixed_precision = st.checkbox(
        "Mixed precision (AMP)", value=bool(defaults.get("mixed_precision", True))
    )
    dry_run = st.checkbox("Dry run", value=False)
    epochs = st.number_input("Epochs", 1, 200, int(defaults.get("epochs", 30)))
    batch_size = st.number_input("Physical batch size", 1, 256, int(defaults.get("batch_size", 16)))
    accumulation = st.number_input(
        "Gradient accumulation", 1, 32, int(defaults.get("gradient_accumulation_steps", 2))
    )
    st.caption(f"Effective batch size: {int(batch_size) * int(accumulation)}")
with right:
    image_size = st.number_input(
        "Image size", 224, 224, int(defaults.get("image_size", 224)), disabled=True
    )
    learning_rate = st.number_input(
        "Learning rate", min_value=0.0, value=float(defaults.get("learning_rate", 1e-4)), format="%.6f"
    )
    weight_decay = st.number_input(
        "Weight decay", min_value=0.0, value=float(defaults.get("weight_decay", 0.05)), format="%.6f"
    )
    label_smoothing = st.number_input(
        "Label smoothing", 0.0, 0.99, float(defaults.get("label_smoothing", 0.1)), step=0.05
    )
    max_warmup = max(0, int(epochs) - 1)
    warmup_epochs = st.number_input(
        "Warmup epochs",
        0,
        max_warmup,
        min(int(defaults.get("warmup_epochs", 3)), max_warmup),
    )
    gradient_clip = st.number_input(
        "Gradient clip norm", min_value=0.0, value=float(defaults.get("gradient_clip_norm", 1.0))
    )
    eta_min = st.number_input(
        "Cosine eta_min", min_value=0.0, value=float(defaults.get("scheduler_eta_min", 1e-6)), format="%.7f"
    )
    tracking_uri = st.text_input(
        "MLflow tracking URI", value=str(defaults.get("tracking_uri", "sqlite:///mlflow.db"))
    )

with st.expander("Smoke test limitleri"):
    limits = st.columns(3)
    max_train_batches = limits[0].number_input("max_train_batches", min_value=0, value=0)
    max_val_batches = limits[1].number_input("max_val_batches", min_value=0, value=0)
    max_test_batches = limits[2].number_input("max_test_batches", min_value=0, value=0)


def make_command(force_dry_run: bool) -> list[str]:
    return build_teacher_command(
        dataset=dataset,
        model=model,
        config=selected_config,
        epochs=int(epochs),
        batch_size=int(batch_size),
        image_size=int(image_size),
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
        pretrained=pretrained,
        dry_run=force_dry_run,
        max_train_batches=int(max_train_batches) or None,
        max_val_batches=int(max_val_batches) or None,
        max_test_batches=int(max_test_batches) or None,
        tracking_uri=tracking_uri,
        gradient_accumulation_steps=int(accumulation),
        label_smoothing=float(label_smoothing),
        mixed_precision=mixed_precision,
        gradient_clip_norm=float(gradient_clip) or None,
        warmup_epochs=int(warmup_epochs),
        scheduler_eta_min=float(eta_min),
    )


command = make_command(dry_run)
st.subheader("Komut Önizlemesi")
st.code(quote_command(command), language="powershell")
render_notebook_download(
    stage="teacher_vision_transformer",
    dataset=dataset,
    model=model,
    command=command,
    key="vit_notebook_download",
)

metadata = {
    "stage": "teacher_vision_transformer",
    "teacher_family": "vision_transformer",
    "dataset": dataset,
    "model": model,
    "config": selected_config,
}
buttons = st.columns(2)
with buttons[0]:
    if st.button("ViT Dry Run Başlat"):
        job = start_job(make_command(True), {**metadata, "dry_run": True})
        st.success(f"ViT dry-run kuyruğa eklendi: {job['job_id']} | {job['status']}")
with buttons[1]:
    st.warning("Gerçek ViT eğitimi uzun sürebilir ve yüksek GPU belleği kullanabilir.")
    if st.button("ViT Eğitimi Başlat", type="primary"):
        job = start_job(command, {**metadata, "dry_run": dry_run})
        st.success(f"ViT eğitimi kuyruğa eklendi: {job['job_id']} | {job['status']}")

st.subheader("Aktif / Son ViT İşleri")
jobs = [job for job in list_jobs() if job.get("stage") == "teacher_vision_transformer"]
if jobs:
    frame = pd.DataFrame(jobs)
    columns = ["job_id", "dataset", "model", "status", "queued_time", "start_time", "log_path"]
    st.dataframe(frame[[column for column in columns if column in frame]], width="stretch")
    selected_id = st.selectbox("Log görüntülenecek job", [job["job_id"] for job in jobs])
    selected = next(job for job in jobs if job["job_id"] == selected_id)
    st.code(read_log_tail(selected.get("log_path", ""), n_lines=20), language="text")
else:
    st.info("Henüz ViT job metadata dosyası yok.")
