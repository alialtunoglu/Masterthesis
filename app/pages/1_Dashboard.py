"""Project overview page for the local Streamlit dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from result_loader import load_baseline_results, load_csv_if_exists, load_dataset_summary, load_split_summaries
from ui_utils import show_dataframe_or_warning


st.set_page_config(page_title="Dashboard", layout="wide")

st.title("Dashboard")

st.subheader("CUDA Durumu")
try:
    # The first import costs a couple of seconds; later reruns are instant.
    with st.spinner("PyTorch yükleniyor..."):
        import torch

    cuda_available = torch.cuda.is_available()
    cols = st.columns(3)
    cols[0].metric("CUDA available", str(cuda_available))
    cols[1].metric("CUDA version", torch.version.cuda or "N/A")
    cols[2].metric(
        "GPU",
        torch.cuda.get_device_name(0) if cuda_available else "N/A",
    )
except ImportError as exc:
    st.warning(f"PyTorch import edilemedi: {exc}")

st.subheader("Veri Seti Özeti")
show_dataframe_or_warning(
    load_dataset_summary(),
    "results/dataset_analysis/dataset_summary.csv bulunamadı.",
)

st.subheader("Split Dosyaları")
split_df = load_split_summaries()
show_dataframe_or_warning(split_df, "splits klasöründe okunabilir split JSON bulunamadı.")

st.subheader("Son Baseline Sonuçları")
baseline_df = load_baseline_results()
if baseline_df is not None and not baseline_df.empty:
    st.dataframe(baseline_df.tail(10), width="stretch")
else:
    st.info("results/baseline/baseline_results.csv henüz yok.")

st.subheader("Son CNN Teacher Sonuçları")
teacher_df = load_csv_if_exists("results/teachers/cnn/teacher_results.csv")
if teacher_df is not None and not teacher_df.empty:
    st.dataframe(teacher_df.tail(10), width="stretch")
else:
    st.info("results/teachers/cnn/teacher_results.csv henüz yok veya boş.")

st.subheader("Son Vision Transformer Teacher Sonuçları")
vit_df = load_csv_if_exists("results/teachers/vision_transformers/teacher_results.csv")
if vit_df is not None and not vit_df.empty:
    st.dataframe(vit_df.tail(10), width="stretch")
else:
    st.info("results/teachers/vision_transformers/teacher_results.csv henüz yok veya boş.")

st.subheader("MLflow")
st.link_button("MLflow UI Aç", "http://127.0.0.1:5000")
st.page_link(
    "pages/7_MLflow_Helper.py",
    label="Başlatma komutu ve kayıtlı deneyler için MLflow Helper'a git",
)
