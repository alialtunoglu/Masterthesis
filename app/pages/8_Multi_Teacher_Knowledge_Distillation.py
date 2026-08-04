"""Multi-teacher knowledge-distillation experiment launcher."""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_path in (PROJECT_ROOT / "app", PROJECT_ROOT / "src"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from experiment_runner import (
    build_multi_kd_command,
    save_job_config_snapshot,
    start_job,
)
from models.model_factory import create_model
from models.model_info import get_model_compute_dict, get_model_summary_dict
from training.kd_features import FEATURE_LAYER_REGISTRY, infer_feature_shapes
from training.multi_kd_aggregation import resolve_teacher_weights
from training.multi_kd_config import (
    TeacherConfig,
    load_multi_kd_config,
    merge_multi_kd_config,
    validate_multi_kd_config,
)


CONFIG_ROOT = (
    PROJECT_ROOT
    / "configs/multi_teacher_knowledge_distillation/plantpathology2021"
)
TEACHER_RESULTS = PROJECT_ROOT / "results/teachers/cnn/teacher_results.csv"
SPLIT_FILE = PROJECT_ROOT / "splits/plantpathology2021_seed42_70_15_15.json"
TEACHER_MODELS = ["resnet50", "densenet201", "regnet_y_8gf"]
KD_TYPES = ["logit_based", "feature_based", "relation_based"]
AGGREGATIONS = ["uniform", "validation_weighted", "manual"]


def matching_configs(kd_type: str, teacher_models: list[str]) -> list[Path]:
    expected = tuple(sorted(teacher_models))
    matches = []
    for path in sorted((CONFIG_ROOT / kd_type).glob("*.json")):
        config = load_multi_kd_config(path)
        actual = tuple(teacher.model_name for teacher in config.teachers)
        if actual == expected:
            matches.append(path)
    return matches


def teacher_runs(model_name: str) -> pd.DataFrame:
    if not TEACHER_RESULTS.exists():
        return pd.DataFrame()
    frame = pd.read_csv(TEACHER_RESULTS)
    frame = frame[
        (frame["dataset_name"].astype(str) == "plantpathology2021")
        & (frame["model_name"].astype(str) == model_name)
    ].copy()
    frame["checkpoint_exists"] = frame["checkpoint_path"].map(
        lambda value: (PROJECT_ROOT / str(value)).exists()
    )
    return frame[frame["checkpoint_exists"]].sort_values(
        "best_val_macro_f1", ascending=False
    )


def run_label(row: pd.Series) -> str:
    return (
        f"{row['model_name']} | val_f1={float(row['best_val_macro_f1']):.4f} | "
        f"test_f1={float(row.get('test_macro_f1', 0)):.4f} | {row['mlflow_run_id']}"
    )


@st.cache_data(show_spinner=False)
def checkpoint_classes(path_text: str) -> list[str]:
    checkpoint = torch.load(PROJECT_ROOT / path_text, map_location="cpu", mmap=True)
    return list(checkpoint.get("class_names", []))


@st.cache_data(show_spinner=False)
def checkpoint_model_compatible(model_name: str, path_text: str) -> bool:
    checkpoint = torch.load(PROJECT_ROOT / path_text, map_location="cpu", mmap=True)
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        return False
    model = create_model(model_name, len(expected_classes()), pretrained=False)
    try:
        model.load_state_dict(state)
        with torch.no_grad():
            output = model(torch.zeros(1, 3, 64, 64))
    except (RuntimeError, ValueError):
        return False
    return output.shape == (1, len(expected_classes()))


@st.cache_data(show_spinner=False)
def expected_classes() -> list[str]:
    return list(json.loads(SPLIT_FILE.read_text(encoding="utf-8"))["classes"])


@st.cache_data(show_spinner=False)
def model_profile(model_name: str, image_size: int) -> dict[str, float | int]:
    model = create_model(model_name, 6, pretrained=False)
    return {
        **get_model_summary_dict(model),
        **get_model_compute_dict(model, image_size),
    }


@st.cache_data(show_spinner=False)
def feature_shape_profile(teacher_name: str, image_size: int) -> dict[str, list[int]]:
    student = create_model("mobilenet_v3_small", 6, pretrained=False)
    teacher = create_model(teacher_name, 6, pretrained=False)
    shapes = infer_feature_shapes(
        student,
        teacher,
        FEATURE_LAYER_REGISTRY["mobilenet_v3_small"],
        FEATURE_LAYER_REGISTRY[teacher_name],
        image_size,
        torch.device("cpu"),
    )
    return {"student": list(shapes.student), "teacher": list(shapes.teacher)}


def queue_job(config, *, dry_run: bool):
    job_id = uuid.uuid4().hex[:12]
    payload = config.to_dict()
    payload["training"]["dry_run"] = dry_run
    snapshot = save_job_config_snapshot(payload, job_id)
    command = build_multi_kd_command(snapshot, dry_run=dry_run)
    teacher_models = [teacher.model_name for teacher in config.teachers]
    teacher_run_ids = [teacher.run_id for teacher in config.teachers]
    weights = [teacher.resolved_weight for teacher in config.teachers]
    return start_job(
        command,
        {
            "job_id": job_id,
            "stage": "multi_teacher_knowledge_distillation",
            "dataset": config.experiment.dataset_name,
            "model": config.student.model_name,
            "student_model": config.student.model_name,
            "teacher_models": teacher_models,
            "teacher_run_ids": teacher_run_ids,
            "teacher_count": len(teacher_models),
            "kd_type": config.experiment.kd_type,
            "aggregation": config.distillation.aggregation,
            "resolved_teacher_weights": weights,
            "config": snapshot,
            "dry_run": dry_run,
        },
    )


st.set_page_config(page_title="Multi-Teacher Knowledge Distillation", layout="wide")
st.title("Multi-Teacher Knowledge Distillation")
st.caption("Plant Pathology 2021 · MobileNetV3-Small · 2–3 farklı CNN teacher")

st.subheader("Model ve Yöntem Seçimi")
first = st.columns(4)
with first[0]:
    dataset = st.selectbox("Dataset", ["plantpathology2021"], disabled=True)
with first[1]:
    student_model = st.selectbox(
        "Student model", ["mobilenet_v3_small"], disabled=True
    )
with first[2]:
    teacher_count = st.selectbox("Teacher sayısı", [2, 3])
with first[3]:
    kd_type = st.selectbox("KD türü", KD_TYPES)

selected_models: list[str] = []
model_columns = st.columns(int(teacher_count))
for index, column in enumerate(model_columns):
    available = [model for model in TEACHER_MODELS if model not in selected_models]
    with column:
        selected = st.selectbox(
            f"Teacher {index + 1}",
            available,
            key=f"multi_teacher_model_{index}",
        )
    selected_models.append(selected)

selected_rows: list[pd.Series] = []
st.subheader("Teacher Run Seçimi")
for index, model_name in enumerate(selected_models):
    runs = teacher_runs(model_name)
    if runs.empty:
        st.error(f"{model_name} için uygun tamamlanmış run bulunamadı.")
        st.stop()
    run_index = st.selectbox(
        f"{model_name} run",
        runs.index.tolist(),
        format_func=lambda value, frame=runs: run_label(frame.loc[value]),
        key=f"multi_teacher_run_{index}",
    )
    selected_rows.append(runs.loc[run_index])

aggregation = st.selectbox("Aggregation", AGGREGATIONS)
manual_values: dict[str, float] = {}
if aggregation == "manual":
    weight_columns = st.columns(len(selected_models))
    for model_name, column in zip(selected_models, weight_columns):
        with column:
            manual_values[model_name] = st.number_input(
                f"{model_name} ağırlığı", min_value=0.0, value=1.0
            )

teachers = [
    TeacherConfig(
        model_name=model_name,
        feature_layer=FEATURE_LAYER_REGISTRY[model_name],
        run_id=str(row["mlflow_run_id"]),
        checkpoint_path=str(row["checkpoint_path"]),
        best_val_macro_f1=float(row["best_val_macro_f1"]),
        manual_weight=manual_values.get(model_name),
    )
    for model_name, row in zip(selected_models, selected_rows)
]
try:
    weights = resolve_teacher_weights(aggregation, teachers)
except ValueError as exc:
    st.error(f"Teacher ağırlıkları geçersiz: {exc}")
    st.stop()
teachers = [
    TeacherConfig(**{**teacher.__dict__, "resolved_weight": weight})
    for teacher, weight in zip(teachers, weights)
]
st.dataframe(
    pd.DataFrame(
        [
            {
                "teacher": teacher.model_name,
                "run_id": teacher.run_id,
                "val_macro_f1": teacher.best_val_macro_f1,
                "normalized_weight": weight,
            }
            for teacher, weight in zip(teachers, weights)
        ]
    ),
    hide_index=True,
    width="stretch",
)

st.subheader("Config Seçimi")
source = st.radio("Config kaynağı", ["Hazır config", "JSON yükle"], horizontal=True)
payload: dict[str, Any] | None = None
if source == "Hazır config":
    paths = matching_configs(kd_type, selected_models)
    if not paths:
        st.error("Seçilen teacher kombinasyonu ve KD türü için config bulunamadı.")
        st.stop()
    selected_path = st.selectbox(
        "Config dosyası",
        paths,
        format_func=lambda path: path.name,
        key=f"multi_config_{kd_type}_{'+'.join(sorted(selected_models))}",
    )
    payload = json.loads(selected_path.read_text(encoding="utf-8"))
else:
    uploaded = st.file_uploader("Multi-teacher KD JSON", type=["json"])
    if uploaded:
        try:
            payload = json.loads(uploaded.getvalue().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            st.error(f"JSON okunamadı: {exc}")
if payload is None:
    st.info("Geçerli bir config seçin veya yükleyin.")
    st.stop()
try:
    defaults = load_multi_kd_config(payload)
except ValueError as exc:
    st.error(f"Config doğrulanamadı: {exc}")
    st.stop()
if tuple(teacher.model_name for teacher in defaults.teachers) != tuple(
    sorted(selected_models)
) or defaults.experiment.kd_type != kd_type:
    st.error("Config seçilen teacher kombinasyonu ve KD türüyle eşleşmiyor.")
    st.stop()

st.subheader("Eğitim Parametreleri")
left, right = st.columns(2)
with left:
    epochs = st.number_input("Epochs", 1, 200, defaults.training.epochs)
    batch_size = st.number_input("Batch size", 2, 256, defaults.training.batch_size)
    image_size = st.number_input("Image size", 32, 1024, defaults.training.image_size)
    learning_rate = st.number_input(
        "Learning rate",
        0.000001,
        1.0,
        defaults.training.learning_rate,
        format="%.6f",
    )
    weight_decay = st.number_input(
        "Weight decay", 0.0, 1.0, defaults.training.weight_decay, format="%.6f"
    )
with right:
    optimizer = st.selectbox("Optimizer", ["adamw", "adam", "sgd"])
    scheduler = st.selectbox(
        "Scheduler", ["cosine", "step", "reduce_on_plateau", "none"]
    )
    monitor = st.selectbox(
        "Monitor metric", ["val_macro_f1", "val_accuracy", "val_loss"]
    )
    patience = st.number_input(
        "Early stopping patience", 1, 100, defaults.training.early_stopping_patience
    )
    num_workers = st.number_input("Num workers", 0, 32, defaults.training.num_workers)
    device = st.selectbox("Device", ["auto", "cuda", "cpu"])

distillation_overrides: dict[str, Any] = {"aggregation": aggregation}
st.subheader("Distillation Parametreleri")
if kd_type == "logit_based":
    distillation_overrides["temperature"] = st.select_slider(
        "Temperature",
        [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0],
        value=defaults.distillation.temperature,
    )
    distillation_overrides["alpha"] = st.slider(
        "KD alpha", 0.0, 1.0, defaults.distillation.alpha, 0.05
    )
elif kd_type == "feature_based":
    distillation_overrides["feature_loss"] = st.selectbox(
        "Feature loss", ["mse", "l1", "cosine"]
    )
    distillation_overrides["feature_loss_weight"] = st.number_input(
        "Feature loss weight", 0.0, 100.0, defaults.distillation.feature_loss_weight
    )
else:
    distillation_overrides["relation_type"] = st.selectbox(
        "Relation type", ["distance_and_angle", "distance", "angle", "correlation"]
    )
    distillation_overrides["distance_loss_weight"] = st.number_input(
        "Distance weight", 0.0, 100.0, defaults.distillation.distance_loss_weight
    )
    distillation_overrides["angle_loss_weight"] = st.number_input(
        "Angle weight", 0.0, 100.0, defaults.distillation.angle_loss_weight
    )
    if batch_size < 8:
        st.warning("Relation KD için batch size < 8 az sayıda ilişki üretir.")

resolved = merge_multi_kd_config(
    defaults,
    {
        "teachers": [teacher.__dict__ for teacher in teachers],
        "training": {
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "image_size": int(image_size),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "optimizer": optimizer,
            "scheduler": scheduler,
            "monitor_metric": monitor,
            "early_stopping_patience": int(patience),
            "num_workers": int(num_workers),
            "device": device,
        },
        "distillation": distillation_overrides,
    },
)
validate_multi_kd_config(resolved, runtime=True)
resolved_weights = [float(teacher.resolved_weight or 0) for teacher in resolved.teachers]

st.subheader("Uyumluluk ve Loss Önizlemesi")
classes = expected_classes()
checks = {
    "En az iki farklı teacher": len(set(selected_models)) >= 2,
    "Tüm checkpointler mevcut": all(
        (PROJECT_ROOT / str(teacher.checkpoint_path)).exists()
        for teacher in resolved.teachers
    ),
    "Class sıraları uyumlu": all(
        checkpoint_classes(str(teacher.checkpoint_path)) == classes
        for teacher in resolved.teachers
    ),
    "Output logit boyutları uyumlu": all(
        checkpoint_model_compatible(
            teacher.model_name, str(teacher.checkpoint_path)
        )
        for teacher in resolved.teachers
    ),
    "Feature katmanları kayıtlı": all(
        teacher.model_name in FEATURE_LAYER_REGISTRY for teacher in resolved.teachers
    ),
}
for label, valid in checks.items():
    st.write(f"{'✓' if valid else '✗'} {label}")
terms = " + ".join(
    f"{weight:.3f} × {teacher.model_name}"
    for teacher, weight in zip(resolved.teachers, resolved_weights)
)
if kd_type == "logit_based":
    formula = (
        f"Total = {1-resolved.distillation.alpha:.2f} × CE + "
        f"{resolved.distillation.alpha:.2f} × T² × ({terms}) KL"
    )
else:
    formula = f"Total = CE + weighted {kd_type.replace('_based', '')} loss ({terms})"
st.code(formula, language="text")

with st.spinner("Model ve feature maliyetleri hesaplanıyor..."):
    profiles = [
        {"role": "Student", "model": student_model, **model_profile(student_model, int(image_size))}
    ]
    for teacher in resolved.teachers:
        row = {
            "role": "Teacher",
            "model": teacher.model_name,
            **model_profile(teacher.model_name, int(image_size)),
        }
        if kd_type != "logit_based":
            row["feature_shapes"] = json.dumps(
                feature_shape_profile(teacher.model_name, int(image_size))
            )
        profiles.append(row)
st.dataframe(pd.DataFrame(profiles), hide_index=True, width="stretch")
st.caption("estimated_flops = 2 × MACs; teacher maliyetleri yalnızca eğitimdedir.")

valid = all(checks.values())
buttons = st.columns(2)
with buttons[0]:
    if st.button("Multi-KD Dry Run Başlat", disabled=not valid):
        job = queue_job(resolved, dry_run=True)
        st.success(f"Dry-run kuyruğa eklendi: {job['job_id']}")
with buttons[1]:
    st.warning("Birden fazla teacher GPU belleği ve eğitim süresini artırır.")
    if st.button(
        "Multi-KD Eğitimini Başlat", type="primary", disabled=not valid
    ):
        job = queue_job(resolved, dry_run=False)
        st.success(f"Eğitim kuyruğa eklendi: {job['job_id']}")
