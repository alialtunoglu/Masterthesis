"""Overview of real experiment activity."""

from pathlib import Path
import sys

import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from experiment_runner import cancel_queued_job, get_queue_worker_status, list_jobs
from experiment_ui import job_identity, open_queued_job, status_label
from gpu_view import execution_label, execution_phase, render_gpu_overview
from experiment_runner import parse_log_progress, read_log_tail

st.title("Genel bakış")
st.caption("Yeni bir deney hazırlayın, devam eden eğitimi izleyin veya sonuçları karşılaştırın.")
if st.button("Yeni deney oluştur"):
    st.switch_page("pages/0_New_Experiment.py")
if notice := st.session_state.pop("overview_queue_notice", None):
    st.success(notice)

jobs = list_jobs()
render_gpu_overview(jobs)
uncertain = [job for job in jobs if job.get("execution_origin") == "lab_notebook" and job.get("status") in {"unknown", "interrupted"}]
if uncertain:
    st.warning(f"{len(uncertain)} notebook koşusunun durumu doğrulanamadı veya kesintiye uğramış. Deneyler sayfasındaki kayıt ve logları inceleyin; tamamlanmış sayılmadılar.")
active = [job for job in jobs if job.get("status") in {"running", "queued", "stop_requested"}]
active.sort(key=lambda job: {"running": 0, "stop_requested": 1, "queued": 2}[job["status"]])
running = sum(job.get("status") in {"running", "stop_requested"} for job in active)
queued = sum(job.get("status") == "queued" for job in active)
st.subheader("Devam eden deneyler")
st.write(f"{running} çalışıyor · {queued} kuyrukta")
if queued and not get_queue_worker_status().get("running"):
    st.warning("Kuyruk işleyicisi henüz hazır değil; durum ve hata logu için Kuyruk yönetimi bölümünü kontrol edin.")
    st.page_link("pages/10_Settings.py", label="Ayarlar → Kuyruk yönetimi")
if not active:
    st.info("Çalışan veya kuyrukta bekleyen deney yok. Yeni deney ile başlayabilirsiniz.")
for job in active:
    if job["status"] == "queued":
        continue
    with st.container(border=True):
        st.markdown(f"#### {job_identity(job)}")
        st.badge(
            status_label(job["status"]),
            icon=":material/play_circle:" if job["status"] == "running" else ":material/pause_circle:",
            color="green" if job["status"] == "running" else "orange",
        )
        st.write(f"Veri seti: {job.get('dataset', '—')}")
        st.caption(f"Job ID: {job['job_id']}")
        st.caption(execution_label(job))
        progress = parse_log_progress(read_log_tail(job.get("log_path", ""), n_lines=100)) if job.get("log_path") else {}
        phase = execution_phase(job, progress)
        if progress.get("last_epoch_seen"):
            phase += f" · Epoch {progress['last_epoch_seen']}/{progress.get('total_epochs') or '—'}"
        st.caption(phase)
        if st.button("Eğitimi izle", key=f"overview_{job['job_id']}", type="primary"):
            open_queued_job(job)
if queued:
    with st.expander(f"Kuyrukta bekleyenler ({queued})", expanded=True):
        st.caption("Bekleyen deneyler aşağıdaki sırayla başlatılır. Kuyruktan çıkarmak deney kaydını veya dosyalarını silmez.")
        pending = sorted(
            (job for job in active if job["status"] == "queued"),
            key=lambda job: (job.get("queued_time") or "", job["job_id"]),
        )
        for position, job in enumerate(pending, start=1):
            st.markdown(f"**{position}. {job_identity(job)}**")
            st.badge("Kuyrukta", icon=":material/schedule:", color="orange")
            st.caption(f"{job.get('dataset', '—')} · Kuyrukta · Job ID: {job['job_id']}")
            with st.container(horizontal=True):
                if st.button("Deney ayrıntıları", key=f"overview_{job['job_id']}"):
                    open_queued_job(job)
                with st.popover("Kuyruktan çıkar", icon=":material/playlist_remove:"):
                    st.write(job_identity(job))
                    st.write(f"Job ID: {job['job_id']}")
                    st.write("Bu deney başlatılmayacak. Kaydı ve dosyaları korunacak; çalışan eğitim etkilenmeyecek.")
                    if st.button("Deneyi kuyruktan çıkar", key=f"overview_cancel_{job['job_id']}"):
                        try:
                            cancelled = cancel_queued_job(job)
                        except OSError:
                            st.error("Kuyruk kaydı güncellenemedi. Sayfayı yenileyip tekrar deneyin.")
                        else:
                            if cancelled.get("cancel_error"):
                                st.error("Deney artık kuyrukta değil. Güncel durumu görmek için sayfayı yenileyin.")
                            else:
                                st.session_state["overview_queue_notice"] = f"{job['job_id']} kuyruktan çıkarıldı. Deney kaydı ve dosyaları korundu."
                                st.rerun()

recent = sorted(
    (job for job in jobs if job not in active),
    key=lambda job: str(job.get("queued_time") or job.get("start_time") or ""),
    reverse=True,
)[:5]
with st.expander(f"Son deneyler ({len(recent)})"):
    if not recent:
        st.caption("Tamamlanan veya durdurulan deneyler burada görünecek.")
    for job in recent:
        st.write(f"{job_identity(job)} — {status_label(job.get('status'))}")
        st.caption(f"{job.get('dataset', '—')} · Job ID: {job['job_id']}")
st.page_link("pages/3_Job_Monitor.py", label="Tüm deneyleri incele")
st.page_link("pages/4_Results_Explorer.py", label="Sonuçları karşılaştır ve raporla")
