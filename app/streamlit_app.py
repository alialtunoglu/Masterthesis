"""Main Streamlit entry point for the local experiment dashboard."""

from __future__ import annotations

import streamlit as st


st.set_page_config(page_title="MasterThesis Experiment Dashboard", layout="wide")

st.title("MasterThesis Experiment Dashboard")
st.write("Bitki yaprak hastalığı teşhisi için baseline, teacher, ViT ve KD deneylerinin takibi.")

st.info(
    "Bu dashboard local kullanım içindir. İlk sürüm baseline deney yönetimi, sonuç inceleme "
    "ve MLflow yönlendirmesi sağlar."
)

st.subheader("Pages")
st.page_link("pages/1_Dashboard.py", label="Dashboard")
st.page_link("pages/2_Baseline_Experiments.py", label="Baseline Experiments")
st.page_link("pages/3_Job_Monitor.py", label="Job Monitor")
st.page_link("pages/4_Results_Explorer.py", label="Results Explorer")
st.page_link("pages/5_MLflow_Helper.py", label="MLflow Helper")

st.subheader("MLflow UI")
st.markdown("[http://127.0.0.1:5000](http://127.0.0.1:5000)")

st.caption("Streamlit ve MLflow ayrı local servislerdir. Public network'e açmayın.")
