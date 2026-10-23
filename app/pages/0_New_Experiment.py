"""One entry point for existing, independently testable launchers."""

from pathlib import Path
import runpy

import streamlit as st

st.title("Yeni deney")
st.caption("Önce eğitim yaklaşımını, ardından modelin rolünü ve mimari ailesini seçin.")
launchers = {
    "Single-teacher KD": "8_Knowledge_Distillation.py",
    "Multi-teacher KD": "9_Multi_Teacher_Knowledge_Distillation.py",
}
kind = st.selectbox("Eğitim türü", ["Baseline", *launchers], key="new_experiment_kind")
if kind == "Baseline":
    selection = st.columns(2)
    role = selection[0].selectbox("Model rolü", ["Student", "Teacher"], key="new_baseline_role")
    family = selection[1].selectbox("Mimari ailesi", ["CNN", "ViT"], key="new_baseline_family")
    path = {
        ("Student", "CNN"): "2_Baseline_Experiments.py",
        ("Student", "ViT"): "11_Student_ViT_Baseline.py",
        ("Teacher", "CNN"): "5_CNN_Teacher_Experiments.py",
        ("Teacher", "ViT"): "6_Vision_Transformer_Teacher_Experiments.py",
    }[(role, family)]
    st.caption(f"Baseline → {role} → {family}: KD olmadan, gerçek etiketlerle standart eğitim.")
else:
    path = launchers[kind]
runpy.run_path(str(Path(__file__).parent / path), init_globals={"UNIFIED_LAUNCHER": True})
