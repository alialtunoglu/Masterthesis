"""Small native GPU controls shared by all launchers and monitoring pages."""
import streamlit as st

from app.gpu_runtime import discover_gpus, gpu_is_available
from app.ui_utils import get_project_root


@st.cache_data(ttl=5, show_spinner=False)
def gpu_inventory():
    return discover_gpus()


def render_gpu_selection(key, *, device="auto") -> str:
    try:
        cards = gpu_inventory()
    except (RuntimeError, ValueError) as error:
        st.error(str(error))
        st.caption("GPU envanteri okunamıyor; kuyruk işleyicisi de GPU atamasını doğrulayacak.")
        cards = []
    options = ["auto", *(card["id"] for card in cards)]
    labels = {"auto": "Otomatik — boştaki GPU", **{card["id"]: f"GPU {card['id']} — {card['name']}" for card in cards}}
    selected = st.selectbox("Çalıştırılacak GPU", options, format_func=labels.get,
                            key=f"{key}_gpu", disabled=device == "cpu")
    if device == "cpu":
        st.caption("Device CPU seçili; GPU ayrılmayacak.")
        return "cpu"
    st.caption("Seçilen GPU doluysa deney kuyrukta bekler. GPU başına bir koşu çalışır.")
    if not cards:
        st.caption("NVIDIA GPU bulunamadı; otomatik seçim CPU ile çalışır. CUDA zorunlu config'leri kontrol edin.")
    return selected


def execution_label(job):
    selected = job.get("gpu_id")
    if selected is not None:
        target = f"GPU {selected}"
    elif "gpu_id" in job or job.get("gpu_requested") == "cpu":
        target = "CPU"
    else:
        request = job.get("gpu_requested", "auto")
        target = "Otomatik GPU" if request == "auto" else f"GPU {request} (bekliyor)"
    origin = "Notebook" if job.get("execution_origin") == "lab_notebook" else "Streamlit"
    return f"{target} · {origin} · {'Dry run' if job.get('dry_run') else 'Gerçek eğitim'}"


def execution_phase(job, progress):
    if job.get("status") == "queued":
        return job.get("queue_reason") or "Kuyrukta — uygun işleyici ve GPU bekleniyor"
    if job.get("status") in {"running", "stop_requested"}:
        return "Eğitim / validation" if progress.get("last_epoch_seen") else "Hazırlanıyor / ilk epoch — henüz epoch özeti yok"
    return ""


def render_gpu_overview(jobs):
    try:
        cards = gpu_inventory()
    except (RuntimeError, ValueError) as error:
        st.warning(str(error))
        return
    if not cards:
        return
    rows = []
    for card in cards:
        owners = [job for job in jobs if str(job.get("gpu_id")) == card["id"] and job.get("status") in {"running", "stop_requested"}]
        locked = not gpu_is_available(get_project_root(), card["id"])
        rows.append({"GPU": f"GPU {card['id']}", "Model": card["name"],
                     "Durum": "Çalışıyor" if owners else "Başka koşu kilidi var" if locked else "Kuyruk için uygun",
                     "Deney": ", ".join(str(job["job_id"]) for job in owners) or "—",
                     "Bellek (MiB)": f"{card['memory_used_mb']} / {card['memory_total_mb']}",
                     "GPU kullanımı (%)": card["utilization"]})
    st.subheader("GPU ve koşular")
    st.dataframe(rows, hide_index=True, width="stretch")
    st.caption("Uygunluk uygulamanın deney/kilit kayıtlarına göredir; diğer kullanıcıların GPU tüketimini bellek ve kullanım sütunlarından kontrol edin.")
