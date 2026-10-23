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
    get_recovery_status,
    enqueue_resume_job,
    load_dashboard_settings,
    list_jobs,
    parse_log_progress,
    request_epoch_stop,
    read_log_tail,
    start_queue_worker,
    stop_job,
    stop_queue_worker,
)
from src.tracking.experiment_taxonomy import enrich_experiment_metadata
from experiment_ui import FAMILY_LABELS, job_identity, open_queued_job, status_label
from job_filters import render_job_filters


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
        return "Teacher baseline — CNN"
    if str(job.get("stage", "")).lower() == "teacher_vision_transformer":
        return "Teacher baseline — ViT"
    if job.get("stage") != "baseline":
        return "Sınıflandırılmamış deney"
    family = FAMILY_LABELS.get(job.get("architecture_family"), "")
    return "Student baseline" + (f" — {family}" if family else "")


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


st.set_page_config(page_title="Deneyler", layout="wide")
st.title("Deneyler")
st.caption("Job ID ile arayın; durumunu, ilerlemesini ve kayıtlarını aynı yerden izleyin.")
jobs = []
for record in list_jobs():
    try:
        jobs.append(enrich_experiment_metadata(record))
    except ValueError as error:
        st.warning(f"Job {record.get('job_id', '?')} sınıflandırılamadı: {error}")
        jobs.append(record)
if not jobs:
    st.info("Henüz deney yok. Yeni deney ile bir eğitim veya kısa test hazırlayın.")
    st.page_link("pages/0_New_Experiment.py", label="Yeni deney oluştur")
    st.stop()

filtered_jobs = render_job_filters(jobs)
st.caption(f"{len(filtered_jobs)} / {len(jobs)} deney")
if not filtered_jobs:
    st.info("Bu filtrelerle deney bulunamadı. Aramayı temizleyin veya filtreleri genişletin.")
    st.stop()

follow_running = st.session_state.get("monitor_follow_running", False)
selected_job_id = choose_monitor_job_id(filtered_jobs, follow_running)
job_ids = [str(job["job_id"]) for job in filtered_jobs]
selected_id = st.selectbox("Deney seç", job_ids, index=job_ids.index(selected_job_id),
                          format_func=lambda value: job_option_label(find_job_by_id(filtered_jobs, value)))
job = find_job_by_id(filtered_jobs, selected_id)
st.session_state["job_monitor_selected_job_id"] = selected_id
st.subheader(job_identity(job))
st.caption(f"{job.get('dataset', '—')} · Job ID: {selected_id}")
log_path = job.get("log_path") or ""
log_text = read_log_tail(log_path, n_lines=5000) if log_path else ""
progress = parse_log_progress(log_text)
recovery = get_recovery_status(job)
epoch_label = str(progress.get("last_epoch_seen") or "—")
if progress.get("last_epoch_seen") and progress.get("total_epochs"):
    epoch_label += f"/{progress['total_epochs']}"
summary = st.columns(3)
summary[0].metric("Durum", status_label(job.get("status")))
summary[1].metric("Epoch", epoch_label)
summary[2].metric("Geçen süre", format_duration(get_job_runtime_seconds(job)))
if progress.get("progress_fraction") is not None:
    st.progress(float(progress["progress_fraction"]), text=f"Epoch: {epoch_label}")
if job.get("status") == "queued" and not get_queue_worker_status().get("running"):
    st.warning("Kuyruk işleyicisi henüz hazır değil; durum ve hata logu için Kuyruk yönetimi bölümünü kontrol edin.")
    st.page_link("pages/10_Settings.py", label="Ayarlar → Kuyruk yönetimi")

metrics = progress.get("metrics") or {}
primary = [name for name in ("val_macro_f1", "val_accuracy", "val_loss", "train_loss", "total_loss") if metrics.get(name) is not None]
if primary:
    st.subheader("Son epoch metrikleri")
    metric_columns = st.columns(min(len(primary), 3))
    for index, name in enumerate(primary):
        metric_columns[index % len(metric_columns)].metric(name, f"{metrics[name]:.6f}")
if metrics:
    with st.expander("Tüm loss ve teacher metrikleri"):
        st.dataframe(pd.DataFrame([metrics]), hide_index=True, width="stretch")

status = job.get("status")
if status in {"running", "stop_requested"}:
    with st.expander("Eğitimi durdur"):
        safe_available = bool(((recovery.get("metadata") or {}).get("recovery") or {}).get("enabled"))
        if status == "running" and safe_available:
            confirm_safe = st.checkbox("Epoch sonunda durdurmayı onayla")
            if st.button("Epoch Sonunda Kaydet ve Durdur", disabled=not confirm_safe):
                try:
                    request_epoch_stop(job)
                except ValueError as error:
                    st.error(str(error))
                else:
                    st.info("Mevcut epoch ve yedek tamamlanınca duracak.")
                    st.rerun()
        elif status == "running":
            st.caption("Epoch sonunda güvenli durdurma için hazır bir kurtarma kaydı gerekli.")
        st.warning("Zorla durdurma, tamamlanmamış epoch veya yedeği kaybettirebilir.")
        confirm_stop = st.checkbox("Durdurmayı onayla")
        if st.button("Zorla Durdur", disabled=not confirm_stop):
            stop_job(job)
            st.rerun()
elif status == "queued":
    confirm_cancel = st.checkbox("Kuyruktaki deneyi iptal etmeyi onayla")
    if st.button("Kuyruktan çıkar", disabled=not confirm_cancel):
        cancel_queued_job(job)
        st.rerun()

generation = recovery.get("generation") or recovery.get("backup_generation")
if status in {"stopped", "failed"} and generation:
    st.subheader("Kaydedilmiş eğitimden devam et")
    generation_path = Path(str(generation))
    generations = sorted(generation_path.parent.glob("epoch-*"), reverse=True) if generation_path.parent.is_dir() else [generation_path]
    selected_generation = st.selectbox("Kurtarma kaydı", generations, format_func=lambda path: path.name, key=f"resume_generation_{selected_id}")
    backup_dir = st.text_input("Kalıcı yedek klasörü (isteğe bağlı)", key=f"resume_backup_{selected_id}")
    if st.button("Devam Et", key=f"resume_{selected_id}"):
        try:
            resumed = enqueue_resume_job(selected_generation, backup_dir or None)
        except (OSError, ValueError) as error:
            st.error(str(error))
        else:
            open_queued_job(resumed)
if status in {"finished", "failed", "stopped", "cancelled"}:
    st.page_link("pages/4_Results_Explorer.py", label="Sonuçlar, artifactler ve güvenli kaldırma")

with st.expander("Eğitim logu"):
    n_lines = st.slider("Gösterilecek log satırı", 20, 1000, 100, 20)
    visible_log = read_log_tail(log_path, n_lines=n_lines) if log_path else ""
    if visible_log.strip():
        st.code(visible_log, language="text")
    else:
        st.info("Henüz log çıktısı yok. Kuyruktaki deneylerde eğitim başlamamış olabilir.")
with st.expander("Teknik kayıt ve kurtarma ayrıntıları"):
    st.json(job)
    if recovery:
        import json
        st.json(json.dumps(recovery, default=str))

with st.expander("Takip seçenekleri"):
    auto_refresh = st.checkbox("Otomatik yenile", value=False, key="monitor_auto_refresh")
    refresh_seconds = st.number_input("Yenileme saniyesi", 2, 60, 5)
    follow_running = st.checkbox("Çalışan deneyi otomatik seç", value=False, key="monitor_follow_running")

if auto_refresh:
    time.sleep(int(refresh_seconds))
    st.rerun()
