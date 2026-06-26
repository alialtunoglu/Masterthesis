"""MLflow usage helper page."""

from __future__ import annotations

import streamlit as st


st.set_page_config(page_title="MLflow Helper", layout="wide")
st.title("MLflow Helper")

st.subheader("MLflow UI Başlatma")
st.code("mlflow ui --host 127.0.0.1 --port 5000", language="powershell")
st.link_button("http://127.0.0.1:5000", "http://127.0.0.1:5000")

st.subheader("Deneyler")
st.write("MLflow içinde baseline deneyleri `MasterThesis-Baseline` experiment adı altında görülebilir.")

st.subheader("MLflow İçeriği")
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

st.info("MLflow UI ayrı terminalde, Streamlit dashboard ayrı terminalde başlatılır.")
