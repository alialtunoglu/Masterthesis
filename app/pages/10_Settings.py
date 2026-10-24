"""Technical controls, separate from experiment tracking."""

from pathlib import Path
import sys

import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from experiment_runner import get_queue_worker_status, load_dashboard_settings, start_queue_worker, stop_queue_worker, read_log_tail
from result_loader import load_dataset_summary, load_split_summaries
from experiment_runner import save_dashboard_settings
from gpu_view import render_gpu_overview
from experiment_runner import list_jobs

st.title("Ayarlar")
st.caption("Yerel çalışma ortamı ve servisler. Bu sayfayı açmak hiçbir işlemi otomatik başlatmaz.")
st.subheader("Kuyruk yönetimi")
worker = get_queue_worker_status()
settings = load_dashboard_settings()
worker_labels = {"running": "Çalışıyor", "starting": "Başlatılıyor", "failed": "Hata nedeniyle kapandı",
                 "unresponsive": "Yanıt alınamıyor", "stopped": "Kapalı"}
st.write("Kuyruk işleyicisi: " + worker_labels.get(worker.get("status"), "Kapalı"))
if worker.get("status") in {"failed", "unresponsive"}:
    st.error(worker.get("error") or "Kuyruk işleyicisi hazır değil. Hata ayrıntıları aşağıdaki işleyici logunda.")
if notice := st.session_state.pop("queue_worker_notice", None):
    st.info(notice)
st.caption(f"Aynı anda en fazla {settings['max_parallel_jobs']} eğitim çalışabilir. İşleyiciyi durdurmak kuyruktaki deneyleri silmez.")
parallel = st.number_input("Toplam paralel deney sınırı", min_value=1, max_value=64,
                           value=int(settings["max_parallel_jobs"]), key="settings_parallel_jobs")
st.caption("İki GPU'da birer koşu için 2 seçin. GPU başına bir koşu koruması ayrıca uygulanır; bu değer onu kaldırmaz.")
confirm_capacity = st.checkbox("Kuyruk kapasitesi değişikliğini onaylıyorum")
st.caption("İşleyici açıksa sınırı artırmak bekleyen deneylerin başlamasına yol açabilir.")
if st.button("Kuyruk kapasitesini kaydet", disabled=not confirm_capacity):
    save_dashboard_settings(int(parallel))
    st.success("Kuyruk kapasitesi kaydedildi.")
st.warning("Queue Worker Başlat düğmesi, kuyrukta bekleyen deneylerin eğitimini başlatabilir.")
left, right = st.columns(2)
with left:
    if st.button("Queue Worker Başlat", disabled=bool(worker.get("process_alive", worker.get("running")))):
        st.session_state["queue_worker_notice"] = start_queue_worker().get("message", "Başlatma istendi.")
        st.rerun()
with right:
    confirmed = st.checkbox("Worker durdurmayı onayla")
    if st.button("Queue Worker Durdur", disabled=not worker.get("process_alive", worker.get("running")) or not confirmed):
        st.warning(stop_queue_worker().get("message", "Durdurma istendi."))
        st.rerun()
if st.button("Durumu ve logu yenile"):
    st.rerun()
with st.expander("Kuyruk işleyicisi logu", expanded=worker.get("status") in {"failed", "unresponsive"}):
    st.caption("İşleyicinin kendi hataları burada; eğitim çıktıları Deneyler sayfasındaki deney logundadır.")
    st.caption("Log önceki denemeleri de içerir; son başlatmanın tarihinden sonraki satırları inceleyin.")
    st.code(read_log_tail(worker["log_path"], n_lines=200) or "Henüz işleyici logu yok.", language="text")
with st.expander("İşleyici ayrıntıları"):
    st.json(worker)

st.subheader("MLflow")
st.link_button("MLflow'u aç", "http://127.0.0.1:5000")
st.caption("MLflow ayrı bir yerel servistir; dış ağa açmayın.")
with st.expander("Başlatma komutu ve kayıtlı deneyler"):
    st.code("mlflow ui --host 127.0.0.1 --port 5000", language="powershell")
    st.page_link("pages/7_MLflow_Helper.py", label="MLflow kayıtlarını incele")
with st.expander("GPU ve Python ortamı"):
    st.code(sys.executable, language="text")
    render_gpu_overview(list_jobs())
with st.expander("Veri setleri ve sabit bölmeler"):
    summary = load_dataset_summary()
    if summary is not None:
        st.dataframe(summary, hide_index=True, width="stretch")
    st.dataframe(load_split_summaries(), hide_index=True, width="stretch")
