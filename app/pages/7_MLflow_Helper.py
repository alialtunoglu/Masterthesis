"""MLflow usage helper page."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from result_loader import mlflow_experiment_overview

TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_URL = "http://127.0.0.1:5000"


@st.cache_data(ttl=30, show_spinner=False)
def _overview(tracking_uri: str) -> list[dict]:
    return mlflow_experiment_overview(tracking_uri)


st.set_page_config(page_title="MLflow Helper", layout="wide")
st.title("MLflow Helper")
st.caption("MLflow UI ayrı bir terminalde, Streamlit dashboard ayrı bir terminalde çalışır.")

st.subheader("MLflow UI Başlatma")
st.code("mlflow ui --host 127.0.0.1 --port 5000", language="powershell")
st.link_button("MLflow UI Aç", MLFLOW_URL)

st.subheader("Kayıtlı Deneyler")
refresh_column, _ = st.columns([1, 4])
if refresh_column.button("Yenile"):
    _overview.clear()
    st.rerun()

experiments = _overview(TRACKING_URI)
if experiments:
    st.dataframe(pd.DataFrame(experiments), hide_index=True, width="stretch")
    st.caption(f"Kaynak: {TRACKING_URI} — silinen run'lar sayılmaz.")
else:
    st.info(
        f"{TRACKING_URI} okunamadı veya henüz deney yok. "
        "Bir eğitim çalıştırdıktan sonra burada görünür."
    )

st.subheader("Her Run'da Kayıtlı Olanlar")
st.markdown(
    """
- params
- metrics
- artifacts
- checkpoints
- confusion matrix
- learning curves
"""
)
