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
from comparison_view import render_comparison
from dataset_view import filter_result_identity
from experiment_ui import classify_results, FAMILY_LABELS


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
RESULT_STAGES = dict(zip(RUN_SUMMARIES.values(), (
    "baseline", "teacher_cnn", "teacher_vision_transformer",
    "knowledge_distillation", "multi_teacher_knowledge_distillation",
)))
SOURCE_LABELS = {
    "Baseline": "Student baseline",
    "CNN Teacher": "Teacher baseline — CNN",
    "Vision Transformer Teacher": "Teacher baseline — ViT",
    "Knowledge Distillation": "Single-teacher KD",
    "Multi-Teacher KD": "Multi-teacher KD",
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
    with st.expander("Artifact dosyaları ve yolları"):
        st.dataframe(pd.DataFrame(artifact_rows), hide_index=True, width="stretch")

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

    with st.expander("Deneyi arşivle veya kaldır"):
        _render_removal_controls(selected_row)


def _render_kd_comparison(selected_row: pd.Series) -> None:
    from src.evaluation.experiment_comparison import load_comparison_runs, build_comparison
    runs = load_comparison_runs(get_project_root())
    run_id = str(selected_row.get("mlflow_run_id", ""))
    report = build_comparison(runs, [run_id], None)
    st.subheader("Teacher / Baseline / KD karşılaştırması")
    for warning in report["warnings"]:
        st.warning(warning)
    st.dataframe(pd.DataFrame(report["rows"]), hide_index=True, width="stretch")


def _json_list(value: object) -> list:
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(str(value))
    except (json.JSONDecodeError, TypeError):
        return []
    return parsed if isinstance(parsed, list) else []


def _render_multi_kd_comparison(selected_row: pd.Series) -> None:
    from src.evaluation.experiment_comparison import load_comparison_runs, build_comparison
    runs = load_comparison_runs(get_project_root())
    run_id = str(selected_row.get("mlflow_run_id", ""))
    report = build_comparison(runs, [run_id], None)
    st.subheader("Teacher / Baseline / KD karşılaştırması")
    for warning in report["warnings"]:
        st.warning(warning)
    st.dataframe(pd.DataFrame(report["rows"]), hide_index=True, width="stretch")


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
    with st.expander("Kesintili deneyin kalıntılarını kaldır"):
        _render_interrupted_removal_controls(selected_job)


def _render_interrupted_removal_controls(selected_job: dict[str, object]) -> None:
    if selected_job.get("execution_origin") == "lab_notebook":
        st.info("Bu notebook kaydı salt okunur izlenir; native job kaldırma akışıyla silinmez.")
        return
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


st.set_page_config(page_title="Sonuçlar", layout="wide")
st.title("Sonuçlar")
st.caption("Veri seti bazında sonuçları inceleyin; artifactlere, içe aktarmaya ve karşılaştırma raporuna aşağıdaki sekmelerden ulaşın.")
from experiment_ui import compact_results

tabs = st.tabs(["Sonuçlar", "Grafikler", "Artifactler", "İçe Aktar", "Karşılaştırma ve Rapor"])

with tabs[0]:
    source_labels = {
        "Raw baseline results": "Student baseline",
        "CNN teacher results": "Teacher baseline — CNN",
        "Vision Transformer teacher results": "Teacher baseline — ViT",
        "Knowledge Distillation results": "Single-teacher KD",
        "Multi-Teacher KD results": "Multi-teacher KD",
    }
    ordered_sources = list(source_labels) + [name for name in BASELINE_TABLES if name not in source_labels]
    selected_table_name = st.selectbox("Sonuç kaynağı", ordered_sources, format_func=lambda name: source_labels.get(name, name))
    selected_table_path = BASELINE_TABLES[selected_table_name]
    prefix = selected_table_path.replace("/", "_").replace(".", "_")
    df = load_csv_if_exists(selected_table_path)
    if df is not None and selected_table_path in RESULT_STAGES:
        df = classify_results(df, RESULT_STAGES[selected_table_path])
    if df is None or df.empty:
        st.info("Bu deney türü için sonuç yok. Tamamlanan eğitimler veya içe aktarılan sonuçlar burada görünür.")
    else:
        filtered = df
        if "dataset_name" in df.columns:
            datasets = sorted(df["dataset_name"].dropna().astype(str).unique().tolist())
            selected_dataset = st.selectbox("Veri seti", datasets, key=f"{prefix}_dataset_single")
            filtered = df[df["dataset_name"].astype(str) == selected_dataset]
        filtered = filter_result_identity(filtered, key=f"{prefix}_identity")
        with st.expander("Model ve sıralama filtreleri"):
            if "architecture_family" in filtered:
                family_options = sorted(filtered["architecture_family"].dropna().unique().tolist())
                family = st.selectbox("Mimari ailesi", ["all", *family_options], key=f"{prefix}_family",
                                      format_func=lambda value: "Tümü" if value == "all" else FAMILY_LABELS.get(value, value))
                if family != "all":
                    filtered = filtered[filtered["architecture_family"] == family]
            model_column = "student_model_name" if "student_model_name" in filtered else "model_name"
            filtered = _filter_multiselect(filtered, model_column, "Model", key=f"{prefix}_model")
            sort_options = [name for name in ("test_macro_f1", "test_accuracy", "best_val_macro_f1", "avg_test_macro_f1") if name in filtered]
            if sort_options:
                sort_metric = st.selectbox("Sıralama metriği", sort_options, key=f"{prefix}_sort")
                filtered = filtered.sort_values(sort_metric, ascending=False)
        st.caption(f"{len(filtered)} sonuç · Metrikler 0–1 aralığındadır; farklı veri setleri birleştirilmez.")
        compact = compact_results(filtered)
        if not compact.empty:
            st.dataframe(compact, hide_index=True, width="stretch")
        elif not filtered.empty:
            st.info("Bu türetilmiş raporun sütunları aşağıdaki ayrıntı tablosunda gösterilir.")
        else:
            st.info("Bu filtrelerle sonuç bulunamadı.")
        with st.expander("Tüm sütunlar ve kaynak dosya"):
            st.caption(selected_table_path)
            st.dataframe(filtered, hide_index=True, width="stretch")
        model_column = "student_model_name" if "student_model_name" in filtered else "model_name"
        if "dataset_name" in filtered and model_column in filtered and "test_macro_f1" in filtered and not filtered.empty:
            st.subheader("Koşu bazında Test Macro-F1")
            chart = filtered.reset_index(drop=True).copy()
            labels = chart[model_column].astype(str)
            if "teacher_model_name" in chart:
                labels += " ← " + chart["teacher_model_name"].astype(str)
            if "teacher_models" in chart:
                labels += " ← " + chart["teacher_models"].astype(str)
            if "kd_type" in chart:
                labels += " · " + chart["kd_type"].astype(str)
            chart["Koşu"] = labels + " · #" + (chart.index + 1).astype(str)
            chart["Test Macro-F1"] = pd.to_numeric(chart["test_macro_f1"], errors="coerce")
            st.bar_chart(chart.set_index("Koşu")["Test Macro-F1"])
            st.caption("Her sütun ayrı bir koşudur; model veya veri setleri arasında ortalama alınmaz.")

with tabs[1]:
    st.subheader("Student Baseline Grafikleri")
    st.caption("Önceden üretilmiş rapor görselleridir; Sonuçlar sekmesindeki filtrelerden bağımsızdır.")
    for figure in STUDENT_FIGURES:
        path = get_project_root() / figure
        if path.exists():
            st.caption(figure)
            st.image(str(path))
        else:
            st.warning(f"Eksik grafik: {figure}")

with tabs[2]:
    run_source = st.selectbox("Deney türü", list(RUN_SUMMARIES), key="artifact_source", format_func=SOURCE_LABELS.get)
    run_summary_path = RUN_SUMMARIES[run_source]
    st.caption("Run adı veya job ID ile artifact ve güvenli kaldırma işlemlerine ulaşın.")
    df = load_csv_if_exists(run_summary_path)
    jobs = list_jobs()
    if df is None or df.empty:
        st.info("Gösterilecek run artifact tablosu yok.")
    else:
        filtered = df
        if "dataset_name" in df:
            dataset_options = sorted(df["dataset_name"].dropna().astype(str).unique().tolist())
            artifact_dataset = st.selectbox("Veri seti", dataset_options, key=f"artifact_dataset_{run_source}")
            filtered = df[df["dataset_name"].astype(str) == artifact_dataset]
        filtered = filter_result_identity(filtered, key=f"artifact_identity_{run_source}")
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
    st.subheader("Harici Run İçe Aktarma")
    st.caption(
        "Laboratuvar veya Colab/Kaggle sonuç klasörünü `runs/import_inbox/<klasör_adı>/` altına koyun. "
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

with tabs[4]:
    render_comparison(get_project_root())
