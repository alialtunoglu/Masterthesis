"""Small presentation helpers; training services stay unchanged."""

import pandas as pd
import streamlit as st
from src.tracking.experiment_taxonomy import enrich_experiment_metadata

STAGE_LABELS = {
    "baseline": "Student baseline", "teacher_cnn": "Teacher baseline — CNN",
    "teacher_vision_transformer": "Teacher baseline — ViT",
    "knowledge_distillation": "Single-teacher KD",
    "multi_teacher_knowledge_distillation": "Multi-teacher KD",
}
METHOD_LABELS = {"baseline": "Baseline", "single_teacher_kd": "Single-teacher KD", "multi_teacher_kd": "Multi-teacher KD"}
ROLE_LABELS = {"student": "Student", "teacher": "Teacher"}
FAMILY_LABELS = {"cnn": "CNN", "vision_transformer": "ViT"}


def status_label(status):
    return {"queued": "Kuyrukta", "running": "Çalışıyor", "finished": "Tamamlandı",
            "failed": "Başarısız", "stopped": "Durduruldu", "cancelled": "İptal edildi",
            "stop_requested": "Durdurma bekleniyor", "interrupted": "Kesintiye uğradı",
            "unknown": "Durum doğrulanamadı"}.get(status, str(status or "Bilinmiyor"))


def job_identity(job):
    try:
        job = enrich_experiment_metadata(job)
        stage = job.get("stage") or "unknown"
        label = STAGE_LABELS.get(stage, "Sınıflandırılmamış deney")
    except ValueError:
        stage = "unknown"
        label = "Sınıflandırılamayan deney"
    if stage == "baseline" and job.get("architecture_family"):
        label += " — " + FAMILY_LABELS[job["architecture_family"]]
    student = job.get("student_model") or job.get("model") or "—"
    teachers = job.get("teacher_models") or ([job["teacher_model"]] if job.get("teacher_model") else [])
    models = f"{' + '.join(teachers)} → {student}" if teachers else student
    kind = f" · {str(job['kd_type']).replace('_', ' ')}" if job.get("kd_type") else ""
    return f"{label} · {models}{kind}"


def open_queued_job(job):
    st.session_state["job_monitor_selected_job_id"] = str(job["job_id"])
    st.session_state["monitor_follow_running"] = False
    st.switch_page("pages/3_Job_Monitor.py")


def compact_results(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["model_name", "student_model", "student_model_name", "teacher_model", "teacher_model_name",
               "teacher_models", "kd_type", "test_macro_f1", "test_accuracy", "best_val_macro_f1",
               "val_macro_f1", "seed", "num_parameters", "model_size_mb"]
    columns += ["dataset_revision", "split_id", "provenance_level", "feature_representation", "feature_representations"]
    return frame[[column for column in columns if column in frame.columns]].copy()


def classify_results(frame: pd.DataFrame, stage: str) -> pd.DataFrame:
    result = frame.copy()
    model_column = "student_model_name" if "student_model_name" in frame else "model_name"
    classifications = [enrich_experiment_metadata({"stage": stage, "model_name": row.get(model_column)})
                       for row in frame.to_dict("records")]
    for key in ("training_method", "model_role", "architecture_family"):
        result[key] = [row.get(key) for row in classifications]
    return result
