"""Project overview page for the local Streamlit dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from result_loader import load_baseline_results, load_dataset_summary, load_split_summaries
from ui_utils import show_dataframe_or_warning


st.set_page_config(page_title="Dashboard", layout="wide")

st.title("Dashboard")

st.subheader("CUDA Durumu")
try:
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
    st.dataframe(baseline_df.tail(10), use_container_width=True)
else:
    st.info("results/baseline/baseline_results.csv henüz yok.")

st.subheader("MLflow UI")
st.code("mlflow ui --host 127.0.0.1 --port 5000", language="powershell")
st.link_button("MLflow UI Aç", "http://127.0.0.1:5000")
