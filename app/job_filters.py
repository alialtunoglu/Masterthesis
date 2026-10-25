"""Dataset-first job filters; each choice narrows the next field's options."""

import streamlit as st

from app.experiment_ui import FAMILY_LABELS, METHOD_LABELS, ROLE_LABELS, status_label

PREFIX = "monitor_filter_"
KD_LABELS = {"logit_based": "Logit-based", "feature_based": "Feature-based", "relation_based": "Relation-based"}
AGGREGATION_LABELS = {"uniform": "Uniform", "manual": "Manual", "validation_weighted": "Validation macro-F1 weighted"}
DETAIL_FIELDS = ("model_role", "architecture_family", "model", "student", "teacher",
                 "teacher_count", "teacher_combination", "kd_type", "aggregation", "status")


def job_value(job, field):
    if field == "student":
        return job.get("student_model") or job.get("model")
    if field == "teacher":
        return job.get("teacher_model")
    if field in {"teacher_count", "teacher_combination"}:
        teachers = job.get("teacher_models")
        teachers = sorted({str(model) for model in teachers if model}) if isinstance(teachers, (list, tuple)) else []
        if field == "teacher_combination":
            return " + ".join(teachers) or None
        if teachers:
            return str(len(teachers))
        count = str(job.get("teacher_count") or "")
        return str(int(count)) if count.isascii() and count.isdecimal() and int(count) > 0 else None
    return job.get(field)


def clear_filters():
    for key in list(st.session_state):
        if key.startswith(PREFIX):
            del st.session_state[key]
    st.session_state["monitor_job_search"] = ""


def select_method(method):
    st.session_state[PREFIX + "training_method"] = method
    for field in DETAIL_FIELDS:
        st.session_state.pop(PREFIX + field, None)


def select_field(jobs, field, label, labels=None):
    values = {str(value) for job in jobs if (value := job_value(job, field)) not in (None, "")}
    options = ["all"] + sorted(values, key=int if field == "teacher_count" else None)
    key = PREFIX + field
    if st.session_state.get(key, "all") not in options:
        st.session_state[key] = "all"
    selected = st.selectbox(label, options, key=key, disabled=not values,
                            format_func=lambda value: "Tümü" if value == "all" else (labels or {}).get(value, value))
    return jobs if selected == "all" else [job for job in jobs if str(job_value(job, field)) == selected]


def render_job_filters(jobs: list[dict]) -> list[dict]:
    search = st.text_input("Deney ara", placeholder="Job ID, model veya teacher adı", key="monitor_job_search")
    with st.expander("Filtreler"):
        st.button("Filtreleri temizle", on_click=clear_filters)
        remaining = select_field(jobs, "dataset", "Veri seti")
        available = {job.get("training_method") for job in remaining}
        method_key = PREFIX + "training_method"
        if st.session_state.get(method_key, "all") not in available | {"all"}:
            select_method("all")
        method = st.session_state.get(method_key, "all")
        st.markdown("Eğitim türü")
        with st.container(horizontal=True):
            for value, label in {"all": "Tümü", **METHOD_LABELS}.items():
                st.button(label, key=PREFIX + "choose_" + value,
                          type="primary" if method == value else "secondary",
                          disabled=value != "all" and value not in available,
                          help="Bu veri setinde kayıtlı deney yok." if value != "all" and value not in available else None,
                          on_click=select_method, args=(value,))
        unavailable = [label for value, label in METHOD_LABELS.items() if value not in available]
        if unavailable:
            st.caption("Bu veri setinde kayıtlı deney yok: " + ", ".join(unavailable))
        if method != "all":
            remaining = [job for job in remaining if job.get("training_method") == method]
        if method == "baseline":
            columns = st.columns(2)
            with columns[0]:
                remaining = select_field(remaining, "model_role", "Model rolü", ROLE_LABELS)
            with columns[1]:
                remaining = select_field(remaining, "architecture_family", "Mimari ailesi", FAMILY_LABELS)
            remaining = select_field(remaining, "model", "Model")
        elif method in {"single_teacher_kd", "multi_teacher_kd"}:
            columns = st.columns(2)
            with columns[0]:
                remaining = select_field(remaining, "student", "Student model")
            with columns[1]:
                if method == "single_teacher_kd":
                    remaining = select_field(remaining, "teacher", "Teacher model")
                else:
                    remaining = select_field(remaining, "teacher_count", "Teacher sayısı")
            if method == "multi_teacher_kd":
                remaining = select_field(remaining, "teacher_combination", "Teacher kombinasyonu")
            remaining = select_field(remaining, "kd_type", "KD türü", KD_LABELS)
            if method == "multi_teacher_kd":
                remaining = select_field(remaining, "aggregation", "Aggregation", AGGREGATION_LABELS)
        remaining = select_field(remaining, "status", "Durum", {job.get("status"): status_label(job.get("status")) for job in remaining})
    search = search.strip().lower()
    fields = ("job_id", "dataset", "model", "student_model", "teacher_model", "teacher_models",
              "teacher_run_ids", "aggregation", "kd_type", "status", "config", "log_path")
    return [job for job in remaining if not search or search in " ".join(str(job.get(field, "")) for field in fields).lower()]
