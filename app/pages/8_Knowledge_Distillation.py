"""Single-teacher knowledge-distillation experiment launcher."""

from __future__ import annotations

import json
import shlex
import sys
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_DIR = PROJECT_ROOT / "app"
SRC_DIR = PROJECT_ROOT / "src"
for import_path in (APP_DIR, SRC_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from experiment_runner import (
    build_kd_command,
    save_job_config_snapshot,
    start_job,
)
from models.model_factory import create_model
from models.model_info import get_model_compute_dict, get_model_summary_dict
from training.kd_config import KDConfig, load_kd_config, merge_kd_config
from training.kd_features import FEATURE_LAYER_REGISTRY


CONFIG_ROOT = PROJECT_ROOT / "configs/knowledge_distillation/plantpathology2021"
TEACHER_RESULTS = PROJECT_ROOT / "results/teachers/cnn/teacher_results.csv"
SPLIT_FILE = PROJECT_ROOT / "splits/plantpathology2021_seed42_70_15_15.json"
KD_LABELS = {
    "logit_based": "Logit / Response-based",
    "feature_based": "Feature-based",
    "relation_based": "Relation-based",
}
DATASETS = ["plantpathology2021"]
STUDENT_MODELS = ["mobilenet_v3_small"]
TEACHER_MODELS = ["densenet201", "resnet50", "regnet_y_8gf"]


def config_paths() -> list[Path]:
    return sorted(CONFIG_ROOT.glob("*/*.json"))


def matching_config_paths(
    dataset_name: str,
    student_model_name: str,
    teacher_model_name: str,
    kd_type: str,
) -> list[Path]:
    matches: list[Path] = []
    for path in config_paths():
        try:
            config = load_kd_config(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        experiment = config.experiment
        if (
            experiment.dataset_name == dataset_name
            and experiment.student_model_name == student_model_name
            and experiment.teacher_model_name == teacher_model_name
            and experiment.kd_type == kd_type
        ):
            matches.append(path)
    return matches


def config_label(path: Path) -> str:
    return f"{path.parent.name} | {path.stem.replace('__', ' → ')}"


def load_uploaded(uploaded) -> dict[str, Any] | None:
    if uploaded is None:
        return None
    try:
        value = json.loads(uploaded.getvalue().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        st.error(f"Yüklenen JSON okunamadı: {exc}")
        return None
    return value if isinstance(value, dict) else None


def teacher_runs(model_name: str) -> pd.DataFrame:
    if not TEACHER_RESULTS.exists():
        return pd.DataFrame()
    frame = pd.read_csv(TEACHER_RESULTS)
    required = {"dataset_name", "model_name", "mlflow_run_id", "checkpoint_path"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
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


def teacher_label(row: pd.Series) -> str:
    return (
        f"{row['model_name']} | val_f1={float(row.get('best_val_macro_f1', 0)):.4f} | "
        f"test_f1={float(row.get('test_macro_f1', 0)):.4f} | {row['mlflow_run_id']}"
    )


@st.cache_data(show_spinner=False)
def model_profile(model_name: str, image_size: int) -> dict[str, float | int]:
    model = create_model(model_name, num_classes=6, pretrained=False)
    return {**get_model_summary_dict(model), **get_model_compute_dict(model, image_size)}


@st.cache_data(show_spinner=False)
def checkpoint_classes(path_text: str) -> list[str]:
    checkpoint = torch.load(PROJECT_ROOT / path_text, map_location="cpu", mmap=True)
    return list(checkpoint.get("class_names", []))


@st.cache_data(show_spinner=False)
def expected_classes() -> list[str]:
    payload = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    return [str(value) for value in payload.get("classes", [])]


def loss_preview(config: KDConfig) -> str:
    kd_type = config.experiment.kd_type
    kd = config.distillation
    if kd_type == "logit_based":
        return (
            f"Total = {1-kd.alpha:.2f} × CrossEntropy + {kd.alpha:.2f} × "
            f"{kd.temperature:.2f}² × KL"
        )
    if kd_type == "feature_based":
        return (
            f"Total = {kd.hard_label_loss_weight:.2f} × CrossEntropy + "
            f"{kd.feature_loss_weight:.2f} × {kd.feature_loss.upper()}(features)"
        )
    return (
        f"Total = {kd.hard_label_loss_weight:.2f} × CrossEntropy + "
        f"{kd.distance_loss_weight:.2f} × Distance + "
        f"{kd.angle_loss_weight:.2f} × Angle"
    )


def queue_job(config: KDConfig, teacher: pd.Series, *, dry_run: bool) -> dict[str, Any]:
    job_id = uuid.uuid4().hex[:12]
    resolved = merge_kd_config(
        config,
        {
            "experiment": {
                "teacher_run_id": str(teacher["mlflow_run_id"]),
                "teacher_checkpoint_path": str(teacher["checkpoint_path"]),
            },
            "training": {"dry_run": dry_run},
        },
    )
    snapshot = save_job_config_snapshot(resolved.to_dict(), job_id)
    command = build_kd_command(
        snapshot,
        str(teacher["mlflow_run_id"]),
        str(teacher["checkpoint_path"]),
        dry_run=dry_run,
    )
    return start_job(
        command,
        {
            "job_id": job_id,
            "stage": "knowledge_distillation",
            "dataset": resolved.experiment.dataset_name,
            "model": resolved.experiment.student_model_name,
            "student_model": resolved.experiment.student_model_name,
            "teacher_model": resolved.experiment.teacher_model_name,
            "teacher_run_id": resolved.experiment.teacher_run_id,
            "kd_type": resolved.experiment.kd_type,
            "config": snapshot,
            "dry_run": dry_run,
        },
    )


st.set_page_config(page_title="Knowledge Distillation", layout="wide")
st.title("Knowledge Distillation")
st.caption("Plant Pathology 2021 için single-teacher KD deneyleri")

st.subheader("Model ve Yöntem Seçimi")
selection = st.columns(4)
with selection[0]:
    dataset_name = st.selectbox("Dataset", DATASETS, disabled=True)
with selection[1]:
    student_model = st.selectbox("Student model", STUDENT_MODELS, disabled=True)
with selection[2]:
    teacher_model = st.selectbox("Teacher model", TEACHER_MODELS)
with selection[3]:
    kd_type = st.selectbox("KD türü", list(KD_LABELS), format_func=KD_LABELS.get)

runs = teacher_runs(teacher_model)
if runs.empty:
    st.error("Bu teacher için checkpoint'i mevcut tamamlanmış Plant Pathology run'ı yok.")
    st.stop()
run_indices = runs.index.tolist()
selected_run_index = st.selectbox(
    "Teacher run",
    run_indices,
    format_func=lambda index: teacher_label(runs.loc[index]),
)
selected_teacher = runs.loc[selected_run_index]

st.subheader("Config Seçimi")
source = st.radio("Config kaynağı", ["Hazır config", "JSON yükle"], horizontal=True)
payload: dict[str, Any] | None = None
if source == "Hazır config":
    paths = matching_config_paths(dataset_name, student_model, teacher_model, kd_type)
    if not paths:
        st.error("Seçilen student, teacher ve KD türü için hazır config bulunamadı.")
        st.stop()
    selected_path = st.selectbox("Config dosyası", paths, format_func=config_label)
    payload = json.loads(selected_path.read_text(encoding="utf-8"))
else:
    payload = load_uploaded(st.file_uploader("KD JSON config", type=["json"]))

if payload is None:
    st.info("Devam etmek için geçerli bir config seçin veya yükleyin.")
    st.stop()
try:
    defaults = load_kd_config(payload)
except ValueError as exc:
    st.error(f"Config doğrulanamadı: {exc}")
    st.stop()

selected_identity = (
    dataset_name,
    student_model,
    teacher_model,
    kd_type,
)
config_identity = (
    defaults.experiment.dataset_name,
    defaults.experiment.student_model_name,
    defaults.experiment.teacher_model_name,
    defaults.experiment.kd_type,
)
if config_identity != selected_identity:
    st.error(
        "Config seçilen dataset, student, teacher ve KD türüyle eşleşmiyor. "
        "Lütfen bu seçime uygun bir config kullanın."
    )
    st.stop()

st.subheader("Ortak Eğitim Parametreleri")
left, right = st.columns(2)
with left:
    epochs = st.number_input("Epochs", 1, 200, defaults.training.epochs)
    batch_size = st.number_input("Batch size", 2, 256, defaults.training.batch_size)
    image_size = st.number_input("Image size", 32, 1024, defaults.training.image_size)
    learning_rate = st.number_input(
        "Learning rate", 0.000001, 1.0, defaults.training.learning_rate, format="%.6f"
    )
    weight_decay = st.number_input(
        "Weight decay", 0.0, 1.0, defaults.training.weight_decay, format="%.6f"
    )
    optimizer = st.selectbox(
        "Optimizer", ["adamw", "adam", "sgd"], index=["adamw", "adam", "sgd"].index(defaults.training.optimizer)
    )
with right:
    scheduler = st.selectbox(
        "Scheduler",
        ["cosine", "step", "reduce_on_plateau", "none"],
        index=["cosine", "step", "reduce_on_plateau", "none"].index(defaults.training.scheduler),
    )
    monitor = st.selectbox(
        "Monitor metric",
        ["val_macro_f1", "val_accuracy", "val_loss"],
        index=["val_macro_f1", "val_accuracy", "val_loss"].index(defaults.training.monitor_metric),
    )
    patience = st.number_input("Early stopping patience", 1, 100, defaults.training.early_stopping_patience)
    num_workers = st.number_input("Num workers", 0, 32, defaults.training.num_workers)
    student_pretrained = st.checkbox("Student pretrained", defaults.training.student_pretrained)
    augmentation = st.checkbox("Data augmentation", defaults.training.data_augmentation)
    device = st.selectbox("Device", ["auto", "cuda", "cpu"], index=["auto", "cuda", "cpu"].index(defaults.training.device))

optimizer_params: dict[str, Any] = {}
if optimizer == "sgd":
    optimizer_params["momentum"] = st.number_input("SGD momentum", 0.0, 0.999, 0.9)
scheduler_params: dict[str, Any] = {}
if scheduler == "step":
    scheduler_params["step_size"] = st.number_input("Step size", 1, 100, 10)
    scheduler_params["gamma"] = st.number_input("Step gamma", 0.01, 1.0, 0.1)
elif scheduler == "reduce_on_plateau":
    scheduler_params["factor"] = st.number_input("Plateau factor", 0.01, 0.99, 0.1)
    scheduler_params["patience"] = st.number_input("Plateau patience", 1, 50, 3)

st.subheader("Distillation Parametreleri")
distillation_overrides: dict[str, Any] = {}
if kd_type == "logit_based":
    distillation_overrides["temperature"] = st.select_slider(
        "Temperature", [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0], value=defaults.distillation.temperature
    )
    distillation_overrides["alpha"] = st.slider("KD alpha", 0.0, 1.0, defaults.distillation.alpha, 0.05)
elif kd_type == "feature_based":
    distillation_overrides["student_layer"] = st.selectbox("Student layer", [FEATURE_LAYER_REGISTRY["mobilenet_v3_small"]])
    distillation_overrides["teacher_layer"] = st.selectbox("Teacher layer", [FEATURE_LAYER_REGISTRY[teacher_model]])
    distillation_overrides["feature_loss"] = st.selectbox("Feature loss", ["mse", "l1", "cosine"])
    distillation_overrides["feature_loss_weight"] = st.number_input("Feature loss weight", 0.0, 100.0, 1.0)
else:
    distillation_overrides["student_layer"] = st.selectbox("Student layer", [FEATURE_LAYER_REGISTRY["mobilenet_v3_small"]])
    distillation_overrides["teacher_layer"] = st.selectbox("Teacher layer", [FEATURE_LAYER_REGISTRY[teacher_model]])
    distillation_overrides["relation_type"] = st.selectbox("Relation type", ["distance_and_angle", "distance", "angle", "correlation"])
    distillation_overrides["distance_loss_weight"] = st.number_input("Distance weight", 0.0, 100.0, 1.0)
    distillation_overrides["angle_loss_weight"] = st.number_input("Angle weight", 0.0, 100.0, 2.0)
    distillation_overrides["feature_normalization"] = st.checkbox("Feature normalization", True)
    if batch_size < 8:
        st.warning("Relation-based KD için batch size < 8 az sayıda örnek ilişkisi üretir.")

try:
    resolved = merge_kd_config(
        defaults,
        {
            "experiment": {"kd_type": kd_type, "teacher_model_name": teacher_model},
            "training": {
                "epochs": int(epochs), "batch_size": int(batch_size), "image_size": int(image_size),
                "learning_rate": float(learning_rate), "weight_decay": float(weight_decay),
                "optimizer": optimizer, "optimizer_params": optimizer_params,
                "scheduler": scheduler, "scheduler_params": scheduler_params,
                "monitor_metric": monitor, "early_stopping_patience": int(patience),
                "num_workers": int(num_workers), "student_pretrained": student_pretrained,
                "data_augmentation": augmentation, "device": device,
            },
            "distillation": distillation_overrides,
        },
    )
except ValueError as exc:
    st.error(str(exc))
    st.stop()

st.subheader("Uyumluluk ve Loss Önizlemesi")
checkpoint_path = str(selected_teacher["checkpoint_path"])
dataset_classes = expected_classes()
teacher_classes = checkpoint_classes(checkpoint_path)
checks = {
    "Dataset uyumlu": selected_teacher["dataset_name"] == resolved.experiment.dataset_name,
    "Teacher modeli uyumlu": selected_teacher["model_name"] == teacher_model,
    "Checkpoint mevcut": (PROJECT_ROOT / checkpoint_path).exists(),
    "Sınıf sayısı uyumlu": int(selected_teacher.get("num_classes", 0)) == len(dataset_classes),
    "Class sırası uyumlu": teacher_classes == dataset_classes,
    "Feature katmanları kayıtlı": teacher_model in FEATURE_LAYER_REGISTRY,
    "Output logit boyutları eşleşiyor": int(selected_teacher.get("num_classes", 0)) == len(dataset_classes),
}
for label, valid in checks.items():
    st.write(f"{'✓' if valid else '✗'} {label}")
st.code(loss_preview(resolved), language="text")

with st.spinner("Model parametreleri ve hesaplama maliyeti hesaplanıyor..."):
    student_profile = model_profile("mobilenet_v3_small", int(image_size))
    teacher_profile = model_profile(teacher_model, int(image_size))
profile_frame = pd.DataFrame(
    [
        {"model": "Student", **student_profile},
        {"model": "Teacher", **teacher_profile},
    ]
)
st.dataframe(profile_frame, hide_index=True, width="stretch")
st.caption("estimated_flops = 2 × MACs tahminidir; donanım gecikmesi değildir.")

st.subheader("Komut ve Başlatma")
st.code(
    "Resolved config snapshot → src/training/train_kd.py → "
    + shlex.quote(str(selected_teacher["mlflow_run_id"])),
    language="powershell",
)
valid = all(checks.values())
buttons = st.columns(2)
with buttons[0]:
    if st.button("KD Dry Run Başlat", disabled=not valid):
        job = queue_job(resolved, selected_teacher, dry_run=True)
        st.success(f"Dry-run kuyruğa eklendi: {job['job_id']} | {job['status']}")
with buttons[1]:
    st.warning("Gerçek KD eğitimi GPU kullanabilir ve uzun sürebilir.")
    confirmed = st.checkbox("Uzun süren eğitimi onaylıyorum")
    if st.button(
        "KD Eğitimini Başlat", type="primary", disabled=not valid or not confirmed
    ):
        job = queue_job(resolved, selected_teacher, dry_run=False)
        st.success(f"KD eğitimi kuyruğa eklendi: {job['job_id']} | {job['status']}")
