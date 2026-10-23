"""Shared baseline/CNN/ViT form using the existing command builders."""

import shlex
import uuid

import streamlit as st

from experiment_runner import build_baseline_command, build_teacher_command, start_job, save_job_config_snapshot
from dataset_view import select_dataset
from experiment_ui import STAGE_LABELS, open_queued_job
from portable_notebook import render_notebook_download
from recovery_view import add_recovery_flags, render_recovery_settings
from result_loader import safe_read_json
from ui_utils import get_project_root
from src.models.model_catalog import get_model_family, get_model_spec
from src.training.experiment_presets import list_presets
from src.training.experiment_config import resolve_experiment_config


def render_standard_launcher(stage, datasets, models, *, unified=False):
    if not unified:
        st.title(STAGE_LABELS[stage])
        st.caption("Veri seti ve modeli seçin; uygun config üzerinden deneyinizi hazırlayın.")
    dataset, dataset_ref, dataset_runtime = select_dataset(key=stage)
    model_label = "Student model" if stage == "baseline" else "Teacher model"
    model = st.selectbox(model_label, models, key=f"{stage}_model")
    root = get_project_root()
    folders = {"baseline": "configs/baseline", "teacher_cnn": "configs/teachers/cnn",
               "teacher_vision_transformer": "configs/teachers/vision_transformers"}
    configs = []
    for path in sorted((root / folders[stage]).rglob("*.json")):
        config = safe_read_json(path) or {}
        if config.get("dataset_name") == dataset and config.get("model_name") == model:
            configs.append(path.relative_to(root).as_posix())
    if not configs:
        configs = [preset["path"] for preset in list_presets(stage, model_name=model)]
    selected_config = st.selectbox("Config dosyası", configs or [""],
                                   format_func=lambda value: value or "Varsayılan ayarlar",
                                   key=f"{stage}_config_{dataset}_{model}")
    defaults = safe_read_json(selected_config) if selected_config else {}
    defaults = defaults or {}
    # Scope edits to the preset so another model never inherits stale hyperparameters.
    scope = f"{stage}_{dataset}_{model}_{selected_config}_{(dataset_ref or {}).get('split_id', '')}"
    st.subheader("Eğitim ayarları")
    common = st.columns(3)
    epochs = common[0].number_input("Epochs", 1, 200, int(defaults.get("epochs", 5 if stage == "baseline" else 30)), key=f"{scope}_epochs")
    batch_label = "Physical batch size" if stage == "teacher_vision_transformer" else "Batch size"
    batch = common[1].number_input(batch_label, 1, 512, int(defaults.get("batch_size", 32)), key=f"{scope}_batch")
    lr = common[2].number_input("Learning rate", min_value=0.0, value=float(defaults.get("learning_rate", .0003)), format="%.6f", key=f"{scope}_lr")
    actions = st.container()
    vit = get_model_family(model) == "vision_transformer"
    teacher_vit = stage == "teacher_vision_transformer"
    options = {}
    with st.expander("Gelişmiş eğitim ayarları"):
        left, right = st.columns(2)
        with left:
            pretrained = st.checkbox("Pretrained", bool(defaults.get("pretrained", True)), key=f"{scope}_pretrained")
            if get_model_spec(model).fixed_input_size:
                image_size = get_model_spec(model).fixed_input_size
                st.caption(f"Image size: {image_size} (seçili model için sabit)")
            else:
                image_size = st.number_input("Image size", 32, 1024, int(defaults.get("image_size", 224)), key=f"{scope}_image")
            if teacher_vit:
                options["mixed_precision"] = st.checkbox("Mixed precision (AMP)", bool(defaults.get("mixed_precision", True)), key=f"{scope}_amp")
                options["gradient_accumulation_steps"] = st.number_input("Gradient accumulation", 1, 32, int(defaults.get("gradient_accumulation_steps", 2)), key=f"{scope}_accum")
                st.caption(f"Effective batch size: {batch * options['gradient_accumulation_steps']}")
                options["label_smoothing"] = st.number_input("Label smoothing", 0.0, .99, float(defaults.get("label_smoothing", .1)), key=f"{scope}_smoothing")
        with right:
            weight_decay = st.number_input("Weight decay", min_value=0.0, value=float(defaults.get("weight_decay", .0001)), format="%.6f", key=f"{scope}_decay")
            if stage == "baseline":
                schedulers = ["none", "cosine"]
                options["scheduler"] = st.selectbox("Scheduler", schedulers,
                    index=schedulers.index(defaults.get("scheduler", "none")), key=f"{scope}_scheduler")
                options["early_stopping_patience"] = st.number_input("Early stopping patience", min_value=0,
                    value=int(defaults.get("early_stopping_patience") or 0), key=f"{scope}_patience")
                st.caption("İzlenen metrik: validation macro-F1. Patience 0: early stopping kapalı.")
            if teacher_vit:
                max_warmup = int(epochs) - 1
                options["warmup_epochs"] = st.number_input("Warmup epochs", 0, max_warmup, min(int(defaults.get("warmup_epochs", 3)), max_warmup), key=f"{scope}_warmup_{epochs}")
                options["gradient_clip_norm"] = st.number_input("Gradient clip norm", min_value=0.0, value=float(defaults.get("gradient_clip_norm", 1.0)), key=f"{scope}_clip") or None
                options["scheduler_eta_min"] = st.number_input("Cosine eta_min", min_value=0.0, value=float(defaults.get("scheduler_eta_min", 1e-6)), format="%.7f", key=f"{scope}_eta")
            tracking_uri = st.text_input("MLflow tracking URI", str(defaults.get("tracking_uri") or "sqlite:///mlflow.db"), key=f"{scope}_tracking")
    with st.expander("Kısa test seçenekleri"):
        dry_run = st.checkbox("Dry run", value=False, key=f"{scope}_dry")
        st.caption("Dry run, uzun eğitimden önce kurulum ve modelin çalışabilirliğini kontrol eder. Sıfır limit, sınır koymaz.")
        limits = st.columns(3)
        train_limit = limits[0].number_input("max_train_batches", min_value=0, value=0, key=f"{scope}_train_limit")
        val_limit = limits[1].number_input("max_val_batches", min_value=0, value=0, key=f"{scope}_val_limit")
        test_limit = limits[2].number_input("max_test_batches", min_value=0, value=0, key=f"{scope}_test_limit")
    recovery_enabled, backup_dir = render_recovery_settings(defaults.get("recovery"), key=scope)
    builder = build_baseline_command if stage == "baseline" else build_teacher_command

    runtime_payload = {**defaults, "stage": stage, "dataset_name": dataset, "model_name": model,
                       "epochs": int(epochs), "batch_size": int(batch), "image_size": int(image_size),
                       "learning_rate": float(lr), "weight_decay": float(weight_decay), "pretrained": pretrained,
                       "tracking_uri": tracking_uri, "max_train_batches": int(train_limit) or None,
                       "max_val_batches": int(val_limit) or None, "max_test_batches": int(test_limit) or None,
                       "recovery": {"enabled": recovery_enabled, "backup_dir": backup_dir or None}, **options}
    if dataset_ref:
        runtime_payload["dataset_ref"] = dataset_ref

    def command_for(dry):
        return builder(dataset=dataset, model=model, config=selected_config or None, epochs=int(epochs),
                       batch_size=int(batch), image_size=int(image_size), learning_rate=float(lr),
                       weight_decay=float(weight_decay), pretrained=pretrained, dry_run=dry,
                       max_train_batches=int(train_limit) or None, max_val_batches=int(val_limit) or None,
                       max_test_batches=int(test_limit) or None, tracking_uri=tracking_uri, **options)

    base_command = command_for(dry_run)
    command = add_recovery_flags(base_command, enabled=recovery_enabled, backup_dir=backup_dir)
    with st.expander("Çalıştırılacak komut"):
        st.code(" ".join(shlex.quote(part) for part in command), language="powershell")
    with actions:
        st.caption("Yerel eğitim GPU kullanabilir ve uzun sürebilir. Kuyruğa ekledikten sonra Deneyler ekranına geçilir.")
        confirmed = st.checkbox("Uzun süren eğitimi onaylıyorum", key=f"{scope}_confirm")
        dry_label, train_label = {
            "baseline": ("Dry Run Başlat", "Eğitimi Başlat"),
            "teacher_cnn": ("Teacher Dry Run Başlat", "Teacher Eğitimi Başlat"),
            "teacher_vision_transformer": ("ViT Dry Run Başlat", "ViT Eğitimi Başlat"),
        }[stage]
        buttons = st.columns(2)
        training_clicked = buttons[0].button(train_label, type="primary", disabled=dry_run or not confirmed)
        dry_clicked = buttons[1].button(dry_label)
        if dry_run:
            st.caption("Dry run açık; gerçek eğitim için bu seçeneği kapatın.")
        if training_clicked or dry_clicked:
            job_id = uuid.uuid4().hex[:12]
            payload = resolve_experiment_config(root, {**runtime_payload, "dry_run": bool(dry_clicked)})
            snapshot = save_job_config_snapshot(payload, job_id)
            launch_command = add_recovery_flags(command_for(True), enabled=recovery_enabled, backup_dir=backup_dir) if dry_clicked else command
            launch_command[launch_command.index("--config") + 1] = snapshot
            job = start_job(launch_command, {"stage": stage, "dataset": dataset, "model": model,
                                            "job_id": job_id, "dataset_ref": dataset_ref,
                                            "config": snapshot, "dry_run": bool(dry_clicked)})
            open_queued_job(job)
    with st.expander("Colab / Kaggle için notebook indir"):
        from src.datasets.dataset_runtime import dataset_package
        render_notebook_download(stage=stage, dataset=dataset, model=model, command=base_command,
                                 runtime_config=runtime_payload, dataset_package=dataset_package(root, dataset_ref) if dataset_ref else None,
                                 key=f"{stage}_notebook_download")
    st.page_link("pages/3_Job_Monitor.py", label="Deneyleri ve kuyruğu izle")
