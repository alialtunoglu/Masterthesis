"""Live-ish job monitor for Streamlit-launched experiments."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from experiment_runner import (
    cancel_queued_job,
    get_job_runtime_seconds,
    get_log_file_info,
    get_queue_worker_status,
    load_dashboard_settings,
    list_jobs,
    parse_log_progress,
    read_log_tail,
    start_queue_worker,
    stop_job,
    stop_queue_worker,
)


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "N/A"
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {sec}s"
    if minutes:
        return f"{minutes}m {sec}s"
    return f"{sec}s"


def unique_values(jobs: list[dict], key: str) -> list[str]:
    return sorted({str(job.get(key)) for job in jobs if job.get(key) not in (None, "")})


def filter_jobs(
    jobs: list[dict],
    status_filter: str,
    dataset_filter: str,
    model_filter: str,
    search_text: str,
) -> list[dict]:
    search = search_text.strip().lower()
    filtered = []
    for job in jobs:
        if status_filter != "all" and str(job.get("status")) != status_filter:
            continue
        if dataset_filter != "all" and str(job.get("dataset")) != dataset_filter:
            continue
        if model_filter != "all" and str(job.get("model")) != model_filter:
            continue
        haystack = " ".join(
            str(job.get(key, ""))
            for key in ["job_id", "dataset", "model", "status", "config", "log_path"]
        ).lower()
        if search and search not in haystack:
            continue
        filtered.append(job)
    return filtered


st.set_page_config(page_title="Job Monitor", layout="wide")
st.title("Job Monitor")
st.caption("Streamlit üzerinden başlatılan eğitim işlerini, job metadata ve log dosyalarıyla takip eder.")
settings = load_dashboard_settings()
st.info(f"Queue aktif. Aynı anda çalışacak maksimum eğitim sayısı: {settings['max_parallel_jobs']}")

worker_status = get_queue_worker_status()
worker_cols = st.columns(5)
worker_cols[0].metric("Queue Worker", "running" if worker_status.get("running") else "stopped")
worker_cols[1].metric("Worker PID", str(worker_status.get("process_id") or "N/A"))
heartbeat = worker_status.get("heartbeat") or {}
worker_cols[2].metric("Heartbeat", heartbeat.get("timestamp", "N/A"))
worker_cols[3].metric("Worker Log", worker_status.get("log_path", "N/A"))
worker_cols[4].metric("Max Parallel", str(settings["max_parallel_jobs"]))

worker_control_cols = st.columns(2)
with worker_control_cols[0]:
    if st.button("Queue Worker Başlat", disabled=bool(worker_status.get("running"))):
        result = start_queue_worker()
        st.success(result.get("message", "Queue worker start requested."))
        st.rerun()
with worker_control_cols[1]:
    confirm_worker_stop = st.checkbox("Worker durdurmayı onayla")
    if st.button(
        "Queue Worker Durdur",
        disabled=not worker_status.get("running") or not confirm_worker_stop,
    ):
        result = stop_queue_worker()
        st.warning(result.get("message", "Queue worker stop requested."))
        st.rerun()

jobs = list_jobs()
if not jobs:
    st.info("Henüz izlenecek job yok. Baseline Experiments sayfasından bir iş başlatabilirsin.")
    st.stop()

top_cols = st.columns([1, 1, 2])
with top_cols[0]:
    auto_refresh = st.checkbox("Auto-refresh", value=True)
with top_cols[1]:
    refresh_seconds = st.number_input("Yenileme saniyesi", min_value=2, max_value=60, value=5)
with top_cols[2]:
    st.caption("Auto-refresh açıkken sayfa seçilen aralıkta kendini yeniler.")

st.subheader("Job Filtresi")
filter_cols = st.columns(4)
with filter_cols[0]:
    status_filter = st.selectbox("Status", ["all"] + unique_values(jobs, "status"))
with filter_cols[1]:
    dataset_filter = st.selectbox("Dataset", ["all"] + unique_values(jobs, "dataset"))
with filter_cols[2]:
    model_filter = st.selectbox("Model", ["all"] + unique_values(jobs, "model"))
with filter_cols[3]:
    search_text = st.text_input("Job ara", value="", placeholder="job id, model, config...")

filtered_jobs = filter_jobs(jobs, status_filter, dataset_filter, model_filter, search_text)
st.caption(f"{len(filtered_jobs)} job gösteriliyor / toplam {len(jobs)} job")
if not filtered_jobs:
    st.warning("Seçilen filtrelerle job bulunamadı.")
    st.stop()

job_options = [
    f"{job.get('job_id')} | {job.get('dataset')} | {job.get('model')} | {job.get('status')}"
    for job in filtered_jobs
]
selected_label = st.selectbox("Job seç", job_options)
selected_index = job_options.index(selected_label)
job = filtered_jobs[selected_index]

log_path = job.get("log_path", "")
log_info = get_log_file_info(log_path)
n_lines = st.slider("Gösterilecek log satırı", min_value=20, max_value=1000, value=20, step=20)
log_text = read_log_tail(log_path, n_lines=n_lines)
progress = parse_log_progress(log_text)

metric_cols = st.columns(5)
metric_cols[0].metric("Status", job.get("status", "unknown"))
metric_cols[1].metric("PID", str(job.get("process_id", "N/A")))
metric_cols[2].metric("Runtime", format_duration(get_job_runtime_seconds(job)))
metric_cols[3].metric("Log lines", str(log_info.get("line_count", 0)))
epoch_label = "N/A"
if progress.get("last_epoch_seen"):
    epoch_label = str(progress["last_epoch_seen"])
    if progress.get("total_epochs"):
        epoch_label = f"{progress['last_epoch_seen']}/{progress['total_epochs']}"
metric_cols[4].metric("Epoch", epoch_label)

if progress.get("progress_fraction") is not None:
    st.progress(float(progress["progress_fraction"]), text=f"Epoch progress: {epoch_label}")

st.subheader("Job Control")
control_cols = st.columns([1, 3])
with control_cols[0]:
    confirm_stop = st.checkbox("Durdurmayı onayla")
with control_cols[1]:
    if st.button("Seçili Job'ı Durdur", disabled=job.get("status") != "running" or not confirm_stop):
        updated = stop_job(job)
        st.warning(f"Job status güncellendi: {updated.get('status')}")
        st.rerun()

cancel_cols = st.columns([1, 3])
with cancel_cols[0]:
    confirm_cancel = st.checkbox("Queue iptalini onayla")
with cancel_cols[1]:
    if st.button(
        "Queued Job'ı İptal Et",
        disabled=job.get("status") != "queued" or not confirm_cancel,
    ):
        updated = cancel_queued_job(job)
        st.warning(f"Job status güncellendi: {updated.get('status')}")
        st.rerun()

st.subheader("Job Metadata")
metadata_cols = [
    "job_id",
    "dataset",
    "model",
    "status",
    "queued_time",
    "start_time",
    "end_time",
    "cancelled_time",
    "process_id",
    "dry_run",
    "config",
    "log_path",
]
metadata_row = {column: job.get(column) for column in metadata_cols if column in job}
st.dataframe(pd.DataFrame([metadata_row]), use_container_width=True)

if progress["metrics"]:
    st.subheader("Son Görülen Metrikler")
    metrics = progress["metrics"]
    visible_metrics = [
        "train_loss",
        "val_loss",
        "val_accuracy",
        "val_macro_f1",
        "learning_rate",
    ]
    metric_cards = st.columns(len(visible_metrics))
    for index, name in enumerate(visible_metrics):
        value = metrics.get(name)
        metric_cards[index].metric(name, "N/A" if value is None else f"{value:.6f}")
    st.dataframe(pd.DataFrame([metrics]), use_container_width=True)

st.subheader("Log")
if not log_info.get("exists"):
    st.warning(f"Log dosyası henüz yok: {log_path}")
elif not log_text.strip():
    st.info("Log dosyası var ama şu an boş. Süreç yeni başlamış veya henüz çıktı flush edilmemiş olabilir.")
else:
    st.code(log_text, language="text")

st.subheader("Tüm İşler")
jobs_df = pd.DataFrame(filtered_jobs)
display_cols = ["job_id", "dataset", "model", "status", "queued_time", "start_time", "end_time", "process_id", "log_path"]
st.dataframe(jobs_df[[col for col in display_cols if col in jobs_df.columns]], use_container_width=True)

if auto_refresh:
    time.sleep(int(refresh_seconds))
    st.rerun()
