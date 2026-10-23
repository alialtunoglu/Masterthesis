"""Lightweight model membership; role and architecture are separate concepts."""

from dataclasses import dataclass

STUDENT_CNN_MODELS = {"mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_b0", "resnet18",
                      "shufflenet_v2_x0_5", "micronet_m0"}
STUDENT_VIT_MODELS = {"deit_tiny_patch16_224", "deit_small_patch16_224"}
CNN_TEACHER_MODELS = {
    "resnet50", "resnet101", "densenet121", "densenet201", "vgg19_bn",
    "efficientnet_b3", "efficientnet_b4", "convnext_tiny", "convnext_base", "regnet_y_8gf",
}
VISION_TRANSFORMER_TEACHER_MODELS = {"swin_v2_t", "maxvit_t", "vit_b_16", "dinov2_vitb14"}
STUDENT_MODELS = STUDENT_CNN_MODELS | STUDENT_VIT_MODELS
SUPPORTED_MODELS = STUDENT_MODELS | CNN_TEACHER_MODELS | VISION_TRANSFORMER_TEACHER_MODELS


def get_model_family(model_name: str) -> str | None:
    name = str(model_name).lower().strip()
    if name in STUDENT_CNN_MODELS | CNN_TEACHER_MODELS:
        return "cnn"
    if name in STUDENT_VIT_MODELS | VISION_TRANSFORMER_TEACHER_MODELS:
        return "vision_transformer"
    return None


@dataclass(frozen=True)
class ModelSpec:
    name: str
    roles: tuple[str, ...]
    family: str
    fixed_input_size: int | None


def get_model_spec(name: str) -> ModelSpec:
    name = str(name).strip().lower()
    family = get_model_family(name)
    if family is None:
        raise ValueError(f"Unsupported model: {name}")
    return ModelSpec(name, ("student",) if name in STUDENT_MODELS else ("teacher",),
                     family, 224 if family == "vision_transformer" or name == "micronet_m0" else None)
