"""Single and multi-teacher forms share dataset, validation and queue boundaries."""
import hashlib
import json
import uuid

import pandas as pd
import streamlit as st

from dataset_view import select_dataset
from experiment_runner import build_kd_command, build_multi_kd_command, save_job_config_snapshot, start_job
from experiment_ui import open_queued_job
from recovery_view import render_recovery_settings
from ui_utils import get_project_root
from src.models.model_catalog import STUDENT_MODELS, CNN_TEACHER_MODELS, VISION_TRANSFORMER_TEACHER_MODELS, get_model_spec
from src.training.experiment_config import resolve_experiment_config
from src.training.experiment_presets import list_presets, compose_preset
from src.training.kd_config import load_kd_config
from src.training.multi_kd_config import load_multi_kd_config
from src.training.teacher_catalog import find_compatible_teacher_runs, validate_teacher_selection
from src.training.kd_features import default_feature_layer
from src.training.multi_kd_aggregation import resolve_teacher_weights
from gpu_view import render_gpu_selection

KD_LABELS = {"logit_based": "Logit / Response-based", "feature_based": "Feature-based", "relation_based": "Relation-based"}


@st.cache_data(ttl=30, show_spinner=False)
def teacher_candidates(root, dataset, ref, student, image_size):
    from lab_runs import portable_teacher_candidates
    rows = find_compatible_teacher_runs(root, ref, student, "logit_based", dataset_name=dataset, image_size=image_size)
    known = {str(row["mlflow_run_id"]) for row in rows}
    return rows + [row for row in portable_teacher_candidates(root, dataset, ref, image_size) if str(row["mlflow_run_id"]) not in known]


def _select(label, values, default):
    return st.selectbox(label, values, index=values.index(default) if default in values else 0)


def _teacher_input(row):
    return {"model_name": row["model_name"], "run_id": str(row["mlflow_run_id"]),
            "checkpoint_path": str(row["checkpoint_path"]), "checkpoint_sha256": row["checkpoint_sha256"],
            "portable_input": row.get("_portable_teacher_input"),
            "best_val_macro_f1": float(row["best_val_macro_f1"]) if pd.notna(row.get("best_val_macro_f1")) else None}


def validate_models(root, payload, classes):
    """Explicit preflight; no downloads and no optimizer/training side effects."""
    import torch
    from src.datasets.dataset_registry import DatasetRef
    from src.datasets.dataset_runtime import verify_dataset_content
    from src.models.model_factory import create_model
    from src.training.train_kd import load_teacher_checkpoint
    from src.training.kd_features import infer_feature_shapes
    validate_teacher_selection(root, payload)
    exp = payload["experiment"]
    if exp.get("dataset_ref"):
        verify_dataset_content(root, DatasetRef(**exp["dataset_ref"]))
    multi = "teachers" in payload
    student_name = payload["student"]["model_name"] if multi else exp["student_model_name"]
    teachers = payload["teachers"] if multi else [{"model_name": exp["teacher_model_name"], "checkpoint_path": exp["teacher_checkpoint_path"]}]
    student = create_model(student_name, num_classes=len(classes), pretrained=False).eval()
    size = payload["training"]["image_size"]
    with torch.no_grad():
        if student(torch.zeros(1, 3, size, size)).shape != (1, len(classes)):
            raise ValueError("Student output class count mismatch")
    details = []
    for entry in teachers:
        teacher = create_model(entry["model_name"], num_classes=len(classes), pretrained=False).eval()
        load_teacher_checkpoint(teacher, root / entry["checkpoint_path"], classes)
        with torch.no_grad():
            if teacher(torch.zeros(1, 3, size, size)).shape != (1, len(classes)):
                raise ValueError("Teacher output class count mismatch")
        if exp["kd_type"] != "logit_based":
            shapes = infer_feature_shapes(student, teacher, default_feature_layer(student_name),
                                          default_feature_layer(entry["model_name"]), size, torch.device("cpu"))
            details.append({"Teacher": entry["model_name"], "Student feature": str(shapes.student),
                            "Teacher feature": str(shapes.teacher), "Adapter": shapes.feature_mode})
        del teacher
    return details


def _training_fields(defaults, image_size):
    values = dict(defaults)
    values["image_size"] = image_size
    cols = st.columns(3)
    values["epochs"] = cols[0].number_input("Epochs", 1, 200, int(values.get("epochs", 30)))
    values["batch_size"] = cols[1].number_input("Batch size", 2, 512, int(values.get("batch_size", 32)))
    values["learning_rate"] = cols[2].number_input("Learning rate", min_value=0.000001, value=float(values.get("learning_rate", .0003)), format="%.6f")
    with st.expander("Gelişmiş eğitim ayarları"):
        st.caption(f"Image size: {image_size} · Teacher adayları bu ön işleme boyutuna göre filtrelenir.")
        values["weight_decay"] = st.number_input("Weight decay", min_value=0.0, value=float(values.get("weight_decay", .0001)), format="%.6f")
        values["optimizer"] = _select("Optimizer", ["adamw", "adam", "sgd"], values.get("optimizer"))
        params = values.get("optimizer_params", {})
        values["optimizer_params"] = {"momentum": st.number_input("SGD momentum", 0.0, .999, float(params.get("momentum", .9)))} if values["optimizer"] == "sgd" else {}
        values["scheduler"] = _select("Scheduler", ["cosine", "step", "reduce_on_plateau", "none"], values.get("scheduler"))
        params = values.get("scheduler_params", {})
        values["scheduler_params"] = {}
        if values["scheduler"] == "step":
            values["scheduler_params"] = {"step_size": st.number_input("Step size", 1, 100, int(params.get("step_size", 10))),
                                          "gamma": st.number_input("Step gamma", .001, 1.0, float(params.get("gamma", .1)))}
        elif values["scheduler"] == "reduce_on_plateau":
            values["scheduler_params"] = {"patience": st.number_input("Plateau patience", 1, 100, int(params.get("patience", 3))),
                                          "factor": st.number_input("Plateau factor", .001, .999, float(params.get("factor", .1)))}
        values["monitor_metric"] = _select("Monitor metric", ["val_macro_f1", "val_accuracy", "val_loss"], values.get("monitor_metric"))
        values["early_stopping_patience"] = st.number_input("Early stopping patience", 1, 100, int(values.get("early_stopping_patience", 7)))
        values["num_workers"] = st.number_input("Num workers", 0, 32, int(values.get("num_workers", 4)))
        values["student_pretrained"] = st.checkbox("Student pretrained", values.get("student_pretrained", True))
        values["data_augmentation"] = st.checkbox("Data augmentation", values.get("data_augmentation", True))
        values["device"] = _select("Device", ["auto", "cuda", "cpu"], values.get("device", "auto"))
        values["tracking_uri"] = st.text_input("MLflow tracking URI", values.get("tracking_uri", "sqlite:///mlflow.db"))
    return values


def render_kd_launcher(*, multi=False, unified=False):
    root = get_project_root()
    stage = "multi_teacher_knowledge_distillation" if multi else "knowledge_distillation"
    if not unified:
        st.title("Multi-teacher KD" if multi else "Single-teacher KD")
    st.caption("Veri ve modelleri seçin; aynı bölmede tamamlanmış teacher koşularıyla deney hazırlayın.")
    dataset, ref, runtime = select_dataset(key=stage)
    student = st.selectbox("Student model", sorted(STUDENT_MODELS), index=sorted(STUDENT_MODELS).index("mobilenet_v3_small"))
    fixed_size = get_model_spec(student).fixed_input_size
    image_size = fixed_size or st.number_input("Giriş görüntü boyutu", 32, 1024, 224)
    if fixed_size:
        st.caption(f"Image size: {fixed_size} (seçili model için sabit)")
    with st.spinner("Uyumlu teacher koşuları okunuyor..."):
        candidates = teacher_candidates(root, dataset, ref, student, int(image_size))
    available = sorted({row["model_name"] for row in candidates})
    enough = len(available) >= (2 if multi else 1)
    if not enough:
        st.info("Bu veri sürümü/bölmesi için yeterli tamamlanmış teacher yok. Önce Teacher baseline eğitin veya sonuç paketini içe aktarın.")
        st.page_link("pages/0_New_Experiment.py", label="Yeni Teacher baseline oluştur")
    if multi:
        count = st.number_input("Teacher sayısı", 2, max(2, len(available)), 2, disabled=not enough)
        selected_models = st.multiselect("Teacher modelleri", available, default=available[:int(count)], max_selections=int(count), disabled=not enough)
        enough = enough and len(selected_models) == count
    else:
        selected = st.selectbox("Teacher model", available or ["Uygun teacher yok"], disabled=not enough)
        selected_models = [selected] if enough else []
    chosen = []
    for model in sorted(selected_models):
        rows = [row for row in candidates if row["model_name"] == model]
        index = st.selectbox(f"Teacher run · {model}", range(len(rows)),
                             format_func=lambda index: f"{rows[index]['mlflow_run_id']} · val F1 {float(rows[index].get('best_val_macro_f1') or 0):.4f}")
        chosen.append(_teacher_input(rows[index]))
    portable_inputs = [item.pop("portable_input", None) for item in chosen]
    kd_type = st.selectbox("KD türü", list(KD_LABELS), format_func=KD_LABELS.get)
    aggregation = st.selectbox("Aggregation", ["uniform", "validation_weighted", "manual"]) if multi else None
    presets = list_presets(stage, kd_type=kd_type)
    paths = []
    legacy_root = root / "configs" / stage / dataset / kd_type
    for path in sorted(legacy_root.glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            exp = value["experiment"]
            names = sorted(item["model_name"] for item in value["teachers"]) if multi else [exp["teacher_model_name"]]
            baseline_student = value["student"]["model_name"] if multi else exp["student_model_name"]
            if baseline_student == student and names == sorted(selected_models):
                paths.append(path.relative_to(root).as_posix())
        except (ValueError, KeyError, TypeError):
            continue
    paths += [item["path"] for item in presets]
    source = st.radio("Config kaynağı", ["Hazır config", "JSON yükle"], horizontal=True)
    payload = presets[0]["payload"]
    if source == "Hazır config":
        path = st.selectbox("Config dosyası", paths)
        payload = json.loads((root / path).read_text(encoding="utf-8"))
    else:
        uploaded = st.file_uploader("KD JSON config", type=["json"])
        if uploaded:
            try:
                payload = json.loads(uploaded.getvalue())
                if not isinstance(payload, dict):
                    raise ValueError("JSON object required")
            except (ValueError, UnicodeError) as error:
                st.error(f"Config okunamadı: {error}")
                enough = False
        else:
            st.caption("Dosya yüklenene kadar genel ayarlar gösterilir; eğitim başlatılamaz.")
            enough = False
    payload = compose_preset(payload, ref, {"stage": stage, "dataset_name": dataset, "kd_type": kd_type})
    # Build portable provenance from current selections, never from an uploaded template.
    payload["teacher_inputs"] = []
    if any(portable_inputs):
        payload["teacher_inputs"] = [portable or {**item, "dataset_name": dataset, "dataset_ref": ref,
            "class_names": runtime["class_names"], "status": "completed", "image_size": int(image_size)}
            for item, portable in zip(chosen, portable_inputs)]
    payload["experiment"]["seed"] = st.number_input("Eğitim seed", 0, 2147483647, int(payload["experiment"].get("seed", 42)))
    # Model and run identities come from the current selection, never an uploaded template.
    fallback = sorted(CNN_TEACHER_MODELS | VISION_TRANSFORMER_TEACHER_MODELS)
    if multi:
        payload["student"] = {"model_name": student}
        payload["teachers"] = chosen or [{"model_name": name} for name in fallback[:2]]
        payload["distillation"]["aggregation"] = aggregation
    else:
        teacher = chosen[0] if chosen else {"model_name": fallback[0]}
        payload["experiment"].update(student_model_name=student, teacher_model_name=teacher["model_name"],
                                     teacher_run_id=teacher.get("run_id"), teacher_checkpoint_path=teacher.get("checkpoint_path"),
                                     teacher_checkpoint_sha256=teacher.get("checkpoint_sha256"))
    loader = load_multi_kd_config if multi else load_kd_config
    try:
        defaults = loader(payload).to_dict()
    except (ValueError, TypeError, KeyError) as error:
        st.error(f"Config doğrulanamadı: {error}")
        return
    payload = defaults
    st.subheader("Eğitim ayarları")
    payload["training"] = _training_fields(defaults["training"], int(image_size))
    recovery, backup = render_recovery_settings(defaults.get("recovery"), key=stage)
    payload["recovery"] = {"enabled": recovery, "backup_dir": backup or None}
    with st.expander("KD yöntem ayarları"):
        kd = payload["distillation"]
        if kd_type == "logit_based":
            kd["temperature"] = st.number_input("Temperature", min_value=.1, value=float(kd["temperature"]))
            kd["alpha"] = st.slider("KD alpha", 0.0, 1.0, float(kd["alpha"]), .05)
        else:
            st.caption(f"Student layer: {default_feature_layer(student)}")
            if multi:
                payload["student"]["feature_layer"] = default_feature_layer(student)
                for teacher in payload["teachers"]:
                    teacher["feature_layer"] = default_feature_layer(teacher["model_name"])
            else:
                kd["student_layer"] = default_feature_layer(student)
                kd["teacher_layer"] = default_feature_layer(payload["experiment"]["teacher_model_name"])
            if kd_type == "feature_based":
                kd["feature_loss"] = _select("Feature loss", ["mse", "l1", "cosine"], kd["feature_loss"])
                kd["feature_loss_weight"] = st.number_input("Feature loss weight", min_value=0.0, value=float(kd["feature_loss_weight"]))
            else:
                kd["relation_type"] = _select("Relation type", ["distance_and_angle", "distance", "angle", "correlation"], kd["relation_type"])
                kd["distance_loss_weight"] = st.number_input("Distance weight", min_value=0.0, value=float(kd["distance_loss_weight"]))
                kd["angle_loss_weight"] = st.number_input("Angle weight", min_value=0.0, value=float(kd["angle_loss_weight"]))
                kd["feature_normalization"] = st.checkbox("Feature normalization", kd["feature_normalization"])
                if payload["training"]["batch_size"] < 8:
                    st.warning("Relation KD için küçük batch az sayıda ilişki üretir.")
    error = None
    try:
        if multi:
            if aggregation == "manual":
                for teacher in payload["teachers"]:
                    teacher["manual_weight"] = st.number_input(f"Ağırlık · {teacher['model_name']}", min_value=0.0, value=1.0)
            weights = resolve_teacher_weights(aggregation, loader(payload).teachers)
            for teacher, weight in zip(payload["teachers"], weights):
                teacher["resolved_weight"] = weight
            st.dataframe([{"Teacher": item["model_name"], "Val macro F1": item.get("best_val_macro_f1"), "Ağırlık": weight} for item, weight in zip(payload["teachers"], weights)], hide_index=True)
        payload = resolve_experiment_config(root, loader(payload).to_dict())
    except (ValueError, TypeError) as exc:
        error = str(exc)
        st.error(error)
    st.subheader("Başlatma kontrolü")
    signature = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    validation_key = f"{stage}:validated:{signature}"
    if st.button("Teacher ve veri uyumluluğunu doğrula", disabled=not enough or bool(error)):
        try:
            with st.spinner("Checkpoint, sınıf sırası, output ve feature boyutları denetleniyor..."):
                details = validate_models(root, payload, runtime["class_names"])
            st.session_state[validation_key] = details
        except (ValueError, OSError, RuntimeError) as exc:
            st.session_state.pop(validation_key, None)
            st.error(f"Doğrulama başarısız: {exc}. Teacher run veya ayarları değiştirip yeniden deneyin.")
    valid = enough and not error and validation_key in st.session_state
    if valid:
        st.success("Veri kimliği, checkpoint ve model boyutları uyumlu.")
        if st.session_state[validation_key]:
            st.dataframe(st.session_state[validation_key], hide_index=True)
    else:
        st.caption("Eğitimden önce uyumluluğu doğrulayın. Ayar değişirse kontrol yenilenir.")
    with st.expander("Loss formülü ve hesaplama maliyeti"):
        kd = payload["distillation"]
        term = " + ".join(f"{item.get('resolved_weight') or 0:.3f} × {item['model_name']}" for item in payload["teachers"]) if multi else payload["experiment"]["teacher_model_name"]
        formula = f"{1-kd['alpha']:.3f} × CE + {kd['alpha']:.3f} × {kd['temperature']}² × KL({term})" if kd_type == "logit_based" else f"{kd['hard_label_loss_weight']} × CE + weighted {kd_type} loss ({term})"
        st.code(formula, language="text")
        if st.button("Parametre, MACs ve FLOPs bilgilerini hesapla"):
            from src.models.model_factory import create_model
            from src.models.model_info import get_model_summary_dict, get_model_compute_dict
            profiles = []
            with st.spinner("Hesaplama maliyetleri ölçülüyor..."):
                for name in [student, *selected_models]:
                    model = create_model(name, num_classes=len(runtime["class_names"]), pretrained=False)
                    profiles.append({"Model": name, **get_model_summary_dict(model), **get_model_compute_dict(model, payload["training"]["image_size"])})
                    del model
            st.dataframe(profiles, hide_index=True)
            st.caption("Tahmini FLOPs = 2 × MACs. Teacher ve adapter maliyetleri student inference modeline dahil değildir.")
    confirmed = st.checkbox("Uzun süren eğitimi onaylıyorum")
    gpu_requested = render_gpu_selection(stage, device=payload["training"].get("device", "auto"))
    cols = st.columns(2)
    train_clicked = cols[0].button("Multi-KD Eğitimini Başlat" if multi else "KD Eğitimini Başlat", type="primary", disabled=not valid or not confirmed)
    dry_clicked = cols[1].button("Multi-KD Dry Run Başlat" if multi else "KD Dry Run Başlat", disabled=not valid)
    if train_clicked or dry_clicked:
        payload["training"]["dry_run"] = bool(dry_clicked)
        validate_teacher_selection(root, payload)
        job_id = uuid.uuid4().hex[:12]
        snapshot = save_job_config_snapshot(payload, job_id)
        exp = payload["experiment"]
        command = build_multi_kd_command(snapshot, dry_run=bool(dry_clicked)) if multi else build_kd_command(snapshot, exp["teacher_run_id"], exp["teacher_checkpoint_path"], dry_run=bool(dry_clicked))
        metadata = {"job_id": job_id, "stage": stage, "dataset": dataset, "dataset_ref": ref, "model": student, "student_model": student,
                    "gpu_requested": gpu_requested,
                    "kd_type": kd_type, "config": snapshot, "dry_run": bool(dry_clicked)}
        if multi:
            metadata.update(teacher_models=selected_models, teacher_run_ids=[item["run_id"] for item in payload["teachers"]],
                            teacher_count=len(selected_models), aggregation=aggregation, resolved_teacher_weights=weights)
        else:
            metadata.update(teacher_model=exp["teacher_model_name"], teacher_run_id=exp["teacher_run_id"])
        open_queued_job(start_job(command, metadata))
    st.page_link("pages/3_Job_Monitor.py", label="Deneyleri ve kuyruğu izle")
    with st.expander("Colab / Kaggle için notebook indir"):
        if enough and not error:
            from portable_notebook import render_notebook_download
            from src.datasets.dataset_runtime import dataset_package
            teachers = payload["teachers"] if multi else chosen
            inputs = [{**item, "dataset_name": dataset, "dataset_ref": ref, "class_names": runtime["class_names"],
                       "status": "completed", "image_size": int(image_size)} for item in teachers]
            exp = payload["experiment"]
            command = build_multi_kd_command("runtime.json") if multi else build_kd_command("runtime.json", exp["teacher_run_id"], exp["teacher_checkpoint_path"])
            st.caption("Teacher ağırlıkları notebook’a eklenmez. Uzak ortamda checkpoint dosyalarını ayrıca bağlayıp yollarını girmeniz gerekir.")
            render_notebook_download(stage=stage, dataset=dataset, model=student, command=command, runtime_config=payload,
                                     dataset_package=dataset_package(root, ref) if ref else None, teacher_inputs=inputs, key=f"{stage}_notebook")
        else:
            st.caption("Notebook için önce uyumlu teacher koşularını seçin ve geçerli ayarları tamamlayın.")
