"""Task-oriented entry point; legacy page URLs remain registered."""

from pathlib import Path
import sys

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

st.set_page_config(page_title="MasterThesis · Deney çalışma alanı", layout="wide")
pages = [
    st.Page("pages/1_Dashboard.py", title="Genel bakış", default=True),
    st.Page("pages/3_Job_Monitor.py", title="Deneyler"),
    st.Page("pages/4_Results_Explorer.py", title="Sonuçlar"),
    st.Page("pages/12_Datasets.py", title="Veri Setleri"),
    st.Page("pages/10_Settings.py", title="Ayarlar"),
    st.Page("pages/0_New_Experiment.py", title="Yeni deney"),
    st.Page("pages/2_Baseline_Experiments.py", title="Student baseline"),
    st.Page("pages/11_Student_ViT_Baseline.py", title="Student baseline — ViT"),
    st.Page("pages/5_CNN_Teacher_Experiments.py", title="Teacher baseline — CNN"),
    st.Page("pages/6_Vision_Transformer_Teacher_Experiments.py", title="Teacher baseline — ViT"),
    st.Page("pages/7_MLflow_Helper.py", title="MLflow"),
    st.Page("pages/8_Knowledge_Distillation.py", title="Single-teacher KD"),
    st.Page("pages/9_Multi_Teacher_Knowledge_Distillation.py", title="Multi-teacher KD"),
]
page = st.navigation(pages, position="hidden")
with st.sidebar:
    st.header("MasterThesis")
    st.caption("Deney çalışma alanı")
    if st.button("Yeni deney", type="primary", width="stretch"):
        st.switch_page("pages/0_New_Experiment.py")
    for path, label in [
        ("1_Dashboard.py", "Genel bakış"), ("3_Job_Monitor.py", "Deneyler"),
        ("4_Results_Explorer.py", "Sonuçlar"), ("12_Datasets.py", "Veri Setleri"), ("10_Settings.py", "Ayarlar"),
    ]:
        st.page_link(f"pages/{path}", label=label)
page.run()
