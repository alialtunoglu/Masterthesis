"""Baseline and student selection result explorer page."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from experiment_removal import (
    ExperimentActiveError,
    ExperimentRemovalError,
    ExperimentRemovalService,
    InterruptedJobRemovalService,
    RemovalMode,
    RunJobMatch,
    find_jobs_for_run,
)
from experiment_runner import list_jobs, parse_log_progress, read_log_tail
from external_run_import import (
    ExternalRunImportError,
    import_external_bundle,
    scan_import_inbox,
)
from result_loader import load_csv_if_exists
from ui_utils import get_project_root, show_dataframe_or_warning


BASELINE_TABLES = {
    "Raw baseline results": "results/baseline/baseline_results.csv",
    "Clean student baseline results": "results/baseline/student_baseline_clean_results.csv",
    "Overall student comparison": "results/baseline/student_baseline_overall_comparison.csv",
    "Student recommendation": "results/baseline/student_selection_recommendation.csv",
    "Artifact check": "results/baseline/student_baseline_artifact_check.csv",
    "CNN teacher results": "results/teachers/cnn/teacher_results.csv",
    "Vision Transformer teacher results": "results/teachers/vision_transformers/teacher_results.csv",
    "Knowledge Distillation results": "results/knowledge_distillation/kd_results.csv",
    "Multi-Teacher KD results": "results/multi_teacher_knowledge_distillation/multi_kd_results.csv",
}

STUDENT_FIGURES = [
    "results/baseline/student_baseline_macro_f1_by_dataset.png",
    "results/baseline/student_baseline_accuracy_by_dataset.png",
    "results/baseline/student_baseline_model_size_vs_macro_f1.png",
    "results/baseline/student_baseline_params_vs_macro_f1.png",
    "results/baseline/student_baseline_overall_macro_f1.png",
    "results/baseline/student_baseline_performance_size_score.png",
]

RUN_SUMMARIES = {
    "Baseline": "results/baseline/baseline_results.csv",
    "CNN Teacher": "results/teachers/cnn/teacher_results.csv",
    "Vision Transformer Teacher": "results/teachers/vision_transformers/teacher_results.csv",
    "Knowledge Distillation": "results/knowledge_distillation/kd_results.csv",
    "Multi-Teacher KD": "results/multi_teacher_knowledge_distillation/multi_kd_results.csv",
}


def _filter_multiselect(
    df: pd.DataFrame,
    column: str,
    label: str,
    *,
    key: str | None = None,
) -> pd.DataFrame:
    if column not in df.columns:
        return df
    options = sorted(df[column].dropna().astype(str).unique().tolist())
    selected = st.multiselect(label, options, default=options, key=key)
    if not selected:
        return df.iloc[0:0]
    return df[df[column].astype(str).isin(selected)]


def _resolve_path(value: object) -> Path | None:
    if value is None or pd.isna(value):
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = get_project_root() / path
    return path


def _linked_jobs(selected_row: pd.Series, jobs: list[dict[str, object]]) -> list[RunJobMatch]:
    run_id = str(selected_row.get("mlflow_run_id", "")).strip()
    run_name = str(selected_row.get("run_name", "")).strip()
    if not run_id or run_id.lower() == "nan":
        return []

    return find_jobs_for_run(
        get_project_root(),
        run_id,
        run_name,
        jobs=jobs,
    )


def _render_linked_jobs(
    selected_row: pd.Series,
    jobs: list[dict[str, object]],
    *,
    widget_prefix: str,
) -> None:
    matches = _linked_jobs(selected_row, jobs)
    run_id = str(selected_row.get("mlflow_run_id", "")).strip()
    st.subheader("İlişkili Job")
    if not matches:
        st.info("Bu run için kesin eşleşen job metadata/log kaydı bulunamadı.")
        return

    selected_job = st.selectbox(
        "Job seç",
        matches,
        format_func=lambda job: f"{job.job_id} | status={job.status}",
        key=f"{widget_prefix}_job_{run_id}",
    )
    _render_job_details(selected_job)


def _render_job_details(job: RunJobMatch) -> None:
    details = st.columns(2)
    details[0].metric("Job ID", job.job_id)
    details[1].metric("Status", job.status)
    if job.log_path:
        try:
            log_label = job.log_path.relative_to(get_project_root()).as_posix()
        except ValueError:
            log_label = str(job.log_path)
        st.caption(f"Log: {log_label}")


def _render_removal_controls(selected_row: pd.Series) -> None:
    run_name = str(selected_row.get("run_name", "")).strip()
    run_id = str(selected_row.get("mlflow_run_id", "")).strip()
    if not run_name or not run_id or run_id.lower() == "nan":
        st.warning(
            "Bu kayıt güvenli kaldırma için gerekli run_name/mlflow_run_id "
            "alanlarını içermiyor."
        )
        return

    st.divider()
    st.subheader("Deneyi Kaldır")
    st.warning(
        "Bu işlem seçili run'ı sonuç tablolarından ve MLflow görünümünden kaldırır. "
        "Çalışan veya kuyruktaki deneyler kaldırılamaz."
    )
    mode_label = st.radio(
        "Kaldırma modu",
        ("Arşivle (önerilen)", "Yerel dosyaları kalıcı sil"),
        horizontal=True,
        key=f"removal_mode_{run_id}",
    )
    mode = RemovalMode.ARCHIVE if mode_label.startswith("Arşivle") else RemovalMode.DELETE
    if mode is RemovalMode.ARCHIVE:
        st.caption("Artefaktlar `_archive/experiment_removals/` altına taşınır ve geri alınabilir.")
    else:
        st.error(
            "Checkpoint, run klasörü, log ve job dosyaları yerel diskten kalıcı olarak silinir. "
            "MLflow kaydı soft-delete ile arayüzden gizlenir."
        )

    confirmation = f"KALDIR {run_name}"
    typed_confirmation = st.text_input(
        f"Onaylamak için `{confirmation}` yazın",
        key=f"removal_confirmation_{run_id}",
    )
    acknowledged = st.checkbox(
        "Doğru deneyi seçtiğimi ve işlemin etkilerini anlıyorum.",
        key=f"removal_acknowledged_{run_id}",
    )
    confirmed = typed_confirmation == confirmation and acknowledged
    if st.button(
        "Seçili Deneyi Kaldır",
        type="primary",
        disabled=not confirmed,
        key=f"remove_experiment_{run_id}",
    ):
        service = ExperimentRemovalService(get_project_root())
        try:
            with st.spinner("Deney güvenli şekilde kaldırılıyor..."):
                report = service.remove(selected_row.to_dict(), mode)
        except ExperimentActiveError as exc:
            st.error(str(exc))
            return
        except ExperimentRemovalError as exc:
            st.error(str(exc))
            return

        if report.archive_path:
            archive_relative = report.archive_path.relative_to(get_project_root()).as_posix()
            st.success(f"Deney kaldırıldı ve arşivlendi: {archive_relative}")
        else:
            st.success("Deney ve yerel artefaktları kalıcı olarak kaldırıldı.")
        for warning in report.warnings:
            st.warning(warning)
        st.cache_data.clear()
        st.rerun()


def _run_choice_label(
    row: pd.Series,
    jobs: list[dict[str, object]],
) -> str:
    run_name = str(row.get("run_name", ""))
    matches = _linked_jobs(row, jobs)
    if not matches:
        return f"{run_name} | job eşleşmesi yok"
    job_summary = ", ".join(f"{job.job_id} ({job.status})" for job in matches)
    return f"{run_name} | {job_summary}"


def _show_run_artifacts(
    df: pd.DataFrame,
    jobs: list[dict[str, object]],
    *,
    widget_prefix: str,
) -> None:
    if df.empty or "run_name" not in df.columns:
        return
    st.subheader("Run Artifactleri")
    run_indices = df.index.tolist()
    selected_index = st.selectbox(
        "Run seç (run adı veya job ID yazarak arayabilirsiniz)",
        run_indices,
        format_func=lambda index: _run_choice_label(df.loc[index], jobs),
        key=f"{widget_prefix}_run",
    )
    selected_row = df.loc[selected_index]

    _render_linked_jobs(selected_row, jobs, widget_prefix=widget_prefix)
    if str(selected_row.get("stage", "")) == "knowledge_distillation":
        _render_kd_comparison(selected_row)
    if str(selected_row.get("stage", "")) == "multi_teacher_knowledge_distillation":
        _render_multi_kd_comparison(selected_row)

    artifact_cols = [
        "checkpoint_path",
        "run_dir",
        "config_path",
        "history_path",
        "per_class_metrics_path",
        "confusion_matrix_png_path",
        "learning_curves_path",
    ]
    artifact_rows = []
    for column in artifact_cols:
        if column in df.columns:
            path = _resolve_path(selected_row.get(column))
            artifact_rows.append(
                {
                    "artifact": column,
                    "path": str(path) if path else "",
                    "exists": bool(path and path.exists()),
                }
            )
    st.dataframe(pd.DataFrame(artifact_rows), width="stretch")

    image_cols = st.columns(2)
    confusion_path = _resolve_path(selected_row.get("confusion_matrix_png_path"))
    curves_path = _resolve_path(selected_row.get("learning_curves_path"))
    with image_cols[0]:
        st.subheader("Confusion Matrix")
        if confusion_path and confusion_path.exists():
            st.image(str(confusion_path))
        else:
            st.info("Confusion matrix PNG bulunamadı.")
    with image_cols[1]:
        st.subheader("Learning Curves")
        if curves_path and curves_path.exists():
            st.image(str(curves_path))
        else:
            st.info("Learning curves PNG bulunamadı.")

    _render_removal_controls(selected_row)


def _render_kd_comparison(selected_row: pd.Series) -> None:
    teacher_frame = load_csv_if_exists("results/teachers/cnn/teacher_results.csv")
    baseline_frame = load_csv_if_exists("results/baseline/baseline_results.csv")
    rows: list[dict[str, object]] = []
    teacher_run_id = str(selected_row.get("teacher_run_id", ""))
    if teacher_frame is not None and "mlflow_run_id" in teacher_frame.columns:
        matched = teacher_frame[
            teacher_frame["mlflow_run_id"].astype(str) == teacher_run_id
        ]
        if not matched.empty:
            teacher = matched.iloc[0]
            rows.append(_comparison_row("Teacher", teacher, "params", "model_size_mb"))

    student_model = str(selected_row.get("student_model_name", ""))
    dataset = str(selected_row.get("dataset_name", ""))
    seed = str(selected_row.get("seed", ""))
    if baseline_frame is not None:
        matched = baseline_frame[
            (baseline_frame["model_name"].astype(str) == student_model)
            & (baseline_frame["dataset_name"].astype(str) == dataset)
            & (baseline_frame["seed"].astype(str) == seed)
        ]
        if not matched.empty:
            metric = pd.to_numeric(matched["best_val_macro_f1"], errors="coerce")
            baseline = matched.loc[metric.idxmax()]
            rows.append(
                _comparison_row("Student baseline", baseline, "params", "model_size_mb")
            )
    rows.append(
        _comparison_row(
            "Student + KD",
            selected_row,
            "student_params",
            "student_model_size_mb",
        )
    )
    st.subheader("Teacher / Baseline / KD Karşılaştırması")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _json_list(value: object) -> list:
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(str(value))
    except (json.JSONDecodeError, TypeError):
        return []
    return parsed if isinstance(parsed, list) else []


def _render_multi_kd_comparison(selected_row: pd.Series) -> None:
    teacher_frame = load_csv_if_exists("results/teachers/cnn/teacher_results.csv")
    baseline_frame = load_csv_if_exists("results/baseline/baseline_results.csv")
    single_frame = load_csv_if_exists("results/knowledge_distillation/kd_results.csv")
    run_ids = {str(value) for value in _json_list(selected_row.get("teacher_run_ids"))}
    teacher_models = {
        str(value) for value in _json_list(selected_row.get("teacher_models"))
    }
    rows: list[dict[str, object]] = []
    if teacher_frame is not None and "mlflow_run_id" in teacher_frame.columns:
        for _, teacher in teacher_frame[
            teacher_frame["mlflow_run_id"].astype(str).isin(run_ids)
        ].iterrows():
            rows.append(
                _comparison_row(
                    f"Teacher: {teacher.get('model_name')}",
                    teacher,
                    "params",
                    "model_size_mb",
                )
            )
    rows.append(
        {
            "model": "Weighted teacher ensemble",
            "accuracy": selected_row.get("ensemble_test_accuracy"),
            "macro_f1": selected_row.get("ensemble_test_macro_f1"),
            "params": selected_row.get("teacher_total_params"),
            "model_size_mb": None,
            "estimated_flops": selected_row.get("teacher_total_estimated_flops"),
        }
    )
    dataset = str(selected_row.get("dataset_name", ""))
    student = str(selected_row.get("student_model_name", ""))
    seed = str(selected_row.get("seed", ""))
    if baseline_frame is not None:
        matched = baseline_frame[
            (baseline_frame["dataset_name"].astype(str) == dataset)
            & (baseline_frame["model_name"].astype(str) == student)
            & (baseline_frame["seed"].astype(str) == seed)
        ]
        if not matched.empty:
            metric = pd.to_numeric(matched["best_val_macro_f1"], errors="coerce")
            rows.append(
                _comparison_row(
                    "Student baseline",
                    matched.loc[metric.idxmax()],
                    "params",
                    "model_size_mb",
                )
            )
    if single_frame is not None:
        matched = single_frame[
            (single_frame["dataset_name"].astype(str) == dataset)
            & (single_frame["student_model_name"].astype(str) == student)
            & (single_frame["seed"].astype(str) == seed)
            & (single_frame["kd_type"].astype(str) == str(selected_row.get("kd_type", "")))
            & (single_frame["teacher_model_name"].astype(str).isin(teacher_models))
        ]
        if not matched.empty:
            metric = pd.to_numeric(matched["best_val_macro_f1"], errors="coerce")
            rows.append(
                _comparison_row(
                    "Best matching single-teacher KD",
                    matched.loc[metric.idxmax()],
                    "student_params",
                    "student_model_size_mb",
                )
            )
    rows.append(
        _comparison_row(
            "Student + Multi-Teacher KD",
            selected_row,
            "student_params",
            "student_model_size_mb",
        )
    )
    st.subheader("Teacher / Ensemble / Baseline / KD Karşılaştırması")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _comparison_row(
    label: str,
    row: pd.Series,
    params_column: str,
    size_column: str,
) -> dict[str, object]:
    return {
        "model": label,
        "accuracy": row.get("test_accuracy"),
        "macro_f1": row.get("test_macro_f1"),
        "params": row.get(params_column),
        "model_size_mb": row.get(size_column),
        "estimated_flops": row.get(
            "student_estimated_flops"
            if label in {"Student + KD", "Student + Multi-Teacher KD", "Best matching single-teacher KD"}
            else "estimated_flops"
        ),
    }


def _job_belongs_to_source(job: dict[str, object], run_source: str) -> bool:
    stage = str(job.get("stage", "")).strip().lower()
    command_parts = job.get("command")
    command = (
        " ".join(str(part) for part in command_parts)
        if isinstance(command_parts, list)
        else ""
    )
    if run_source == "CNN Teacher":
        return stage == "teacher_cnn"
    if run_source == "Vision Transformer Teacher":
        return stage == "teacher_vision_transformer"
    if run_source == "Knowledge Distillation":
        return stage == "knowledge_distillation" or "train_kd.py" in command
    if run_source == "Multi-Teacher KD":
        return (
            stage == "multi_teacher_knowledge_distillation"
            or "train_multi_kd.py" in command
        )
    return (
        stage not in {
            "teacher_cnn",
            "teacher_vision_transformer",
            "knowledge_distillation",
            "multi_teacher_knowledge_distillation",
        }
        and "train_teacher.py" not in command
        and "train_kd.py" not in command
        and "train_multi_kd.py" not in command
    )


def _render_interrupted_jobs(
    jobs: list[dict[str, object]],
    run_source: str,
    *,
    widget_prefix: str,
) -> None:
    interrupted_statuses = {"stopped", "failed", "cancelled"}
    interrupted = [
        job
        for job in jobs
        if str(job.get("status", "")).lower() in interrupted_statuses
        and _job_belongs_to_source(job, run_source)
    ]

    st.divider()
    st.subheader("Durdurulan / Başarısız Job'lar")
    st.caption(
        "Bu kayıtlar tamamlanmış run sonuçlarından değil, doğrudan job geçmişinden gelir."
    )
    if not interrupted:
        st.info("Bu deney türü için durdurulmuş, başarısız veya iptal edilmiş job yok.")
        return

    selected_job = st.selectbox(
        "Kesintiye uğrayan job seç",
        interrupted,
        format_func=lambda job: (
            f"{job.get('job_id', '?')} | {job.get('status', 'unknown')} | "
            f"{job.get('dataset', '?')} | {job.get('model', '?')}"
        ),
        key=f"{widget_prefix}_interrupted_job",
    )
    summary = st.columns(4)
    summary[0].metric("Job ID", str(selected_job.get("job_id", "-")))
    summary[1].metric("Status", str(selected_job.get("status", "unknown")))
    summary[2].metric("Dataset", str(selected_job.get("dataset", "-")))
    summary[3].metric("Model", str(selected_job.get("model", "-")))

    log_text = read_log_tail(str(selected_job.get("log_path", "")), n_lines=5000)
    progress = parse_log_progress(log_text)
    last_epoch = progress.get("last_epoch_seen")
    total_epochs = progress.get("total_epochs")
    if last_epoch:
        epoch_label = f"{last_epoch}/{total_epochs}" if total_epochs else str(last_epoch)
        st.metric("Kaydedilen son epoch", epoch_label)
    else:
        st.info("Logda tamamlanmış epoch kaydı bulunamadı.")

    job_details = {
        "queued_time": selected_job.get("queued_time"),
        "start_time": selected_job.get("start_time"),
        "end_time": selected_job.get("end_time"),
        "log_path": selected_job.get("log_path"),
        "checkpoint_path": selected_job.get("checkpoint_path"),
        "run_dir": selected_job.get("run_dir"),
    }
    st.dataframe(
        pd.DataFrame(
            [
                {"alan": key, "değer": value}
                for key, value in job_details.items()
                if value not in (None, "")
            ]
        ),
        width="stretch",
        hide_index=True,
    )
    with st.expander("Logun son kısmını göster"):
        st.code(read_log_tail(str(selected_job.get("log_path", "")), n_lines=200) or "Log yok.")
    _render_interrupted_removal_controls(selected_job)


def _render_interrupted_removal_controls(selected_job: dict[str, object]) -> None:
    job_id = str(selected_job.get("job_id", "")).strip()
    if not job_id:
        return

    st.subheader("Job Kalıntılarını Kaldır")
    st.warning(
        "Job kaydı ve log kaldırılır. Yalnızca bu job'a kesin bağlanan ve tamamlanmış "
        "başka bir run tarafından kullanılmayan artifactler temizlenir."
    )
    mode_label = st.radio(
        "Job kaldırma modu",
        ("Arşivle (önerilen)", "Kalıcı sil"),
        horizontal=True,
        key=f"interrupted_removal_mode_{job_id}",
    )
    mode = RemovalMode.ARCHIVE if mode_label.startswith("Arşivle") else RemovalMode.DELETE
    confirmation = f"JOB SİL {job_id}"
    typed_confirmation = st.text_input(
        f"Onaylamak için `{confirmation}` yazın",
        key=f"interrupted_removal_confirmation_{job_id}",
    )
    acknowledged = st.checkbox(
        "Seçili kesintili job'ın kalıntılarını kaldırmak istediğimi onaylıyorum.",
        key=f"interrupted_removal_ack_{job_id}",
    )
    if st.button(
        "Job Kalıntılarını Kaldır",
        type="primary",
        disabled=typed_confirmation != confirmation or not acknowledged,
        key=f"remove_interrupted_job_{job_id}",
    ):
        service = InterruptedJobRemovalService(get_project_root())
        try:
            with st.spinner("Job kalıntıları güvenli şekilde kaldırılıyor..."):
                report = service.remove_job(selected_job, mode)
        except ExperimentRemovalError as exc:
            st.error(str(exc))
            return

        if report.archive_path:
            relative_archive = report.archive_path.relative_to(get_project_root()).as_posix()
            st.success(f"Job kalıntıları arşivlendi: {relative_archive}")
        else:
            st.success("Job kalıntıları kalıcı olarak silindi.")
        for warning in report.warnings:
            st.warning(warning)
        st.cache_data.clear()
        st.rerun()


st.set_page_config(page_title="Results Explorer", layout="wide")
st.title("Results Explorer")

tabs = st.tabs(["Tables", "Student Figures", "Run Artifacts", "External Run Import"])

with tabs[0]:
    selected_table_name = st.selectbox("Tablo seç", list(BASELINE_TABLES))
    selected_table_path = BASELINE_TABLES[selected_table_name]
    table_widget_prefix = selected_table_path.replace("/", "_").replace(".", "_")
    df = load_csv_if_exists(selected_table_path)
    st.caption(selected_table_path)
    if df is None or df.empty:
        st.info(f"{selected_table_path} henüz yok veya boş.")
    else:
        with st.sidebar:
            st.header("Filtreler")
            filtered = _filter_multiselect(
                df, "dataset_name", "Dataset", key=f"{table_widget_prefix}_dataset"
            )
            filtered = _filter_multiselect(
                filtered, "model_name", "Model", key=f"{table_widget_prefix}_model"
            )
            filtered = _filter_multiselect(
                filtered,
                "teacher_family",
                "Teacher family",
                key=f"{table_widget_prefix}_teacher_family",
            )
            filtered = _filter_multiselect(
                filtered, "device", "Device", key=f"{table_widget_prefix}_device"
            )
            if "pretrained" in filtered.columns:
                pretrained_values = sorted(
                    filtered["pretrained"].dropna().astype(str).unique().tolist()
                )
                selected_pretrained = st.multiselect(
                    "Pretrained",
                    pretrained_values,
                    default=pretrained_values,
                    key=f"{table_widget_prefix}_pretrained",
                )
                filtered = filtered[filtered["pretrained"].astype(str).isin(selected_pretrained)]

            sort_options = [
                column
                for column in [
                    "test_macro_f1",
                    "test_accuracy",
                    "best_val_macro_f1",
                    "avg_test_macro_f1",
                ]
                if column in filtered.columns
            ]
            if sort_options:
                sort_metric = st.selectbox(
                    "Sıralama", sort_options, key=f"{table_widget_prefix}_sort_metric"
                )
                ascending = st.checkbox(
                    "Artan sırala", value=False, key=f"{table_widget_prefix}_ascending"
                )
                filtered = filtered.sort_values(sort_metric, ascending=ascending)

        show_dataframe_or_warning(filtered, "Seçilen filtrelerle sonuç bulunamadı.")

        if not filtered.empty and {
            "model_name",
            "test_accuracy",
            "test_macro_f1",
        }.issubset(filtered.columns):
            chart_cols = st.columns(2)
            grouped = filtered.copy()
            grouped["test_accuracy"] = pd.to_numeric(grouped["test_accuracy"], errors="coerce")
            grouped["test_macro_f1"] = pd.to_numeric(grouped["test_macro_f1"], errors="coerce")
            model_scores = grouped.groupby("model_name", as_index=True)[
                ["test_accuracy", "test_macro_f1"]
            ].mean(numeric_only=True)
            with chart_cols[0]:
                st.subheader("Model Bazında Test Accuracy")
                st.bar_chart(model_scores["test_accuracy"])
            with chart_cols[1]:
                st.subheader("Model Bazında Test Macro F1")
                st.bar_chart(model_scores["test_macro_f1"])

with tabs[1]:
    st.subheader("Student Baseline Figures")
    for figure in STUDENT_FIGURES:
        path = get_project_root() / figure
        if path.exists():
            st.caption(figure)
            st.image(str(path))
        else:
            st.warning(f"Eksik grafik: {figure}")

with tabs[2]:
    run_source = st.selectbox("Deney türü", list(RUN_SUMMARIES), key="artifact_source")
    run_summary_path = RUN_SUMMARIES[run_source]
    st.caption(run_summary_path)
    df = load_csv_if_exists(run_summary_path)
    jobs = list_jobs()
    if df is None or df.empty:
        st.info("Gösterilecek run artifact tablosu yok.")
    else:
        filtered = _filter_multiselect(
            df, "dataset_name", "Dataset", key=f"artifact_dataset_{run_source}"
        )
        model_column = (
            "student_model_name"
            if run_source in {"Knowledge Distillation", "Multi-Teacher KD"}
            else "model_name"
        )
        filtered = _filter_multiselect(
            filtered, model_column, "Model", key=f"artifact_model_{run_source}"
        )
        _show_run_artifacts(filtered, jobs, widget_prefix=f"artifact_{run_source}")
    _render_interrupted_jobs(
        jobs,
        run_source,
        widget_prefix=f"artifact_{run_source}",
    )

with tabs[3]:
    st.subheader("External Run Import")
    st.caption(
        "Colab/Kaggle çıktısını `runs/import_inbox/<bundle_id>/` altına koyun. "
        "ZIP zorunlu, checkpoint isteğe bağlıdır."
    )
    bundles = scan_import_inbox(get_project_root())
    if not bundles:
        st.info("İçe aktarılmayı bekleyen bundle yok.")
    else:
        selected_bundle = st.selectbox(
            "Bundle seç",
            bundles,
            format_func=lambda path: path.name,
            key="external_bundle_selection",
        )
        contents = [path.name for path in selected_bundle.iterdir() if path.is_file()]
        st.write({"bundle_id": selected_bundle.name, "files": contents})
        confirmed = st.checkbox(
            "Bundle doğrulandıktan sonra sonuçlara ve MLflow'a aktarılsın.",
            key=f"external_bundle_confirm_{selected_bundle.name}",
        )
        if st.button(
            "Bundle'ı İçe Aktar",
            type="primary",
            disabled=not confirmed,
            key=f"external_bundle_import_{selected_bundle.name}",
        ):
            try:
                with st.spinner("Hash, split ve class mapping doğrulanıyor..."):
                    receipt = import_external_bundle(get_project_root(), selected_bundle)
            except ExternalRunImportError as exc:
                st.error(str(exc))
            else:
                st.success(
                    f"İçe aktarıldı: {receipt['bundle_id']} | MLflow run: "
                    f"{receipt['mlflow_run_id']}"
                )
                st.cache_data.clear()
