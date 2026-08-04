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


def is_kd_job(job: dict) -> bool:
    return str(job.get("stage", "")).lower() == "knowledge_distillation" or bool(
        job.get("kd_type")
    )


def is_multi_kd_job(job: dict) -> bool:
    return str(job.get("stage", "")).lower() == "multi_teacher_knowledge_distillation"


def job_type_label(job: dict) -> str:
    if is_multi_kd_job(job):
        return "Multi-Teacher KD"
    if is_kd_job(job):
        return "Knowledge Distillation"
    if str(job.get("stage", "")).lower() == "teacher_cnn":
        return "CNN Teacher"
    if str(job.get("stage", "")).lower() == "teacher_vision_transformer":
        return "Vision Transformer Teacher"
    return "Student Baseline"


def job_option_label(job: dict) -> str:
    prefix = f"{job.get('job_id')} | {job_type_label(job)}"
    if is_multi_kd_job(job):
        teachers = " + ".join(job.get("teacher_models") or [])
        student = job.get("student_model") or job.get("model") or "unknown student"
        kd_type = str(job.get("kd_type") or "unknown").replace("_", " ")
        return (
            f"{prefix} ({kd_type}) | {teachers} → {student} | "
            f"{job.get('dataset')} | {job.get('status')}"
        )
    if is_kd_job(job):
        kd_type = str(job.get("kd_type") or "unknown").replace("_", " ")
        teacher = job.get("teacher_model") or "unknown teacher"
        student = job.get("student_model") or job.get("model") or "unknown student"
        return (
            f"{prefix} ({kd_type}) | Teacher: {teacher} → Student: {student} | "
            f"{job.get('dataset')} | {job.get('status')}"
        )
    return f"{prefix} | {job.get('dataset')} | {job.get('model')} | {job.get('status')}"


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
            for key in [
                "job_id",
                "dataset",
                "model",
                "student_model",
                "teacher_model",
                "teacher_models",
                "teacher_run_ids",
                "aggregation",
                "kd_type",
                "status",
                "config",
                "log_path",
            ]
        ).lower()
        if search and search not in haystack:
            continue
        filtered.append(job)
    return filtered


def find_job_by_id(jobs: list[dict], job_id: str | None) -> dict | None:
    if not job_id:
        return None
    return next((job for job in jobs if str(job.get("job_id")) == str(job_id)), None)


def latest_running_job(jobs: list[dict]) -> dict | None:
    running_jobs = [job for job in jobs if job.get("status") == "running"]
    if not running_jobs:
        return None
    return max(
        running_jobs,
        key=lambda job: str(job.get("start_time") or job.get("queued_time") or ""),
    )


def choose_monitor_job_id(jobs: list[dict], follow_running: bool) -> str:
    current_id = st.session_state.get("job_monitor_selected_job_id")
    current_job = find_job_by_id(jobs, current_id)
    running_job = latest_running_job(jobs)

    if follow_running and running_job:
        selected_id = str(running_job.get("job_id"))
        st.session_state["job_monitor_selected_job_id"] = selected_id
        st.session_state["job_monitor_last_running_job_id"] = selected_id
        return selected_id

    if current_job:
        return str(current_job.get("job_id"))

    last_running_id = st.session_state.get("job_monitor_last_running_job_id")
    if find_job_by_id(jobs, last_running_id):
        st.session_state["job_monitor_selected_job_id"] = last_running_id
        return str(last_running_id)

    selected_id = str(jobs[0].get("job_id"))
    st.session_state["job_monitor_selected_job_id"] = selected_id
    return selected_id


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

top_cols = st.columns([1, 1, 1, 2])
with top_cols[0]:
    auto_refresh = st.checkbox("Auto-refresh", value=True)
with top_cols[1]:
    refresh_seconds = st.number_input("Yenileme saniyesi", min_value=2, max_value=60, value=5)
with top_cols[2]:
    follow_running = st.checkbox("Running job'ı otomatik takip et", value=True)
with top_cols[3]:
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

job_options = [job_option_label(job) for job in filtered_jobs]
selected_job_id = choose_monitor_job_id(filtered_jobs, follow_running=follow_running)
job_ids = [str(job.get("job_id")) for job in filtered_jobs]
selected_index = job_ids.index(selected_job_id) if selected_job_id in job_ids else 0
selected_label = st.selectbox("Job seç", job_options, index=selected_index)
selected_index = job_options.index(selected_label)
job = filtered_jobs[selected_index]
st.session_state["job_monitor_selected_job_id"] = str(job.get("job_id"))
if job.get("status") == "running":
    st.session_state["job_monitor_last_running_job_id"] = str(job.get("job_id"))

st.subheader("Seçili İş")
if is_multi_kd_job(job):
    identity_cols = st.columns(5)
    identity_cols[0].metric("İş Türü", "Multi-Teacher KD")
    identity_cols[1].metric(
        "KD Türü", str(job.get("kd_type") or "N/A").replace("_", " ")
    )
    identity_cols[2].metric(
        "Teacher'lar", " + ".join(job.get("teacher_models") or [])
    )
    identity_cols[3].metric(
        "Student", str(job.get("student_model") or job.get("model") or "N/A")
    )
    identity_cols[4].metric("Aggregation", str(job.get("aggregation") or "N/A"))
    st.caption(
        "Teacher run ID'leri: " + ", ".join(job.get("teacher_run_ids") or [])
    )
elif is_kd_job(job):
    identity_cols = st.columns(5)
    identity_cols[0].metric("İş Türü", "Knowledge Distillation")
    identity_cols[1].metric(
        "KD Türü", str(job.get("kd_type") or "N/A").replace("_", " ")
    )
    identity_cols[2].metric("Teacher", str(job.get("teacher_model") or "N/A"))
    identity_cols[3].metric(
        "Student", str(job.get("student_model") or job.get("model") or "N/A")
    )
    identity_cols[4].metric("Dataset", str(job.get("dataset") or "N/A"))
    teacher_run_id = str(job.get("teacher_run_id") or "N/A")
    st.caption(f"Teacher run ID: {teacher_run_id}")
else:
    st.info(
        f"{job_type_label(job)} · Model: {job.get('model', 'N/A')} · "
        f"Dataset: {job.get('dataset', 'N/A')}"
    )

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
    "student_model",
    "teacher_model",
    "teacher_run_id",
    "teacher_models",
    "teacher_run_ids",
    "teacher_count",
    "aggregation",
    "resolved_teacher_weights",
    "kd_type",
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
st.dataframe(pd.DataFrame([metadata_row]), width="stretch")

if progress["metrics"]:
    st.subheader("Son Görülen Metrikler")
    metrics = progress["metrics"]
    visible_metrics = [
        "train_loss",
        "total_loss",
        "ce_loss",
        "kd_loss",
        "feature_loss",
        "relation_loss",
        "teacher_student_agreement",
        "ensemble_student_agreement",
        "val_loss",
        "val_accuracy",
        "val_macro_f1",
        "learning_rate",
    ]
    visible_metrics.extend(
        sorted(
            name
            for name in metrics
            if name.startswith("teacher_") or name.startswith("agreement_")
        )
    )
    metric_cards = st.columns(min(len(visible_metrics), 6))
    for index, name in enumerate(visible_metrics):
        value = metrics.get(name)
        metric_cards[index % len(metric_cards)].metric(
            name, "N/A" if value is None else f"{value:.6f}"
        )
    st.dataframe(pd.DataFrame([metrics]), width="stretch")

st.subheader("Log")
if not log_info.get("exists"):
    st.warning(f"Log dosyası henüz yok: {log_path}")
elif not log_text.strip():
    st.info("Log dosyası var ama şu an boş. Süreç yeni başlamış veya henüz çıktı flush edilmemiş olabilir.")
else:
    st.code(log_text, language="text")

st.subheader("Tüm İşler")
jobs_df = pd.DataFrame(filtered_jobs)
display_cols = [
    "job_id",
    "stage",
    "dataset",
    "kd_type",
    "teacher_model",
    "teacher_models",
    "aggregation",
    "student_model",
    "model",
    "status",
    "queued_time",
    "start_time",
    "end_time",
    "process_id",
    "log_path",
]
st.dataframe(
    jobs_df[[col for col in display_cols if col in jobs_df.columns]], width="stretch"
)

if auto_refresh:
    time.sleep(int(refresh_seconds))
    st.rerun()
