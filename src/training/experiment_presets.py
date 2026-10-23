"""Dataset-independent presets; selections are applied after template values."""
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def list_presets(stage, model_name=None, kd_type=None):
    folder = "student_baseline" if stage == "baseline" else "teacher_baseline" if stage.startswith("teacher_") else stage
    paths = sorted((ROOT / "configs/presets" / folder).glob("*.json"))
    if kd_type:
        paths = [path for path in paths if path.stem == kd_type]
    elif model_name:
        from ..models.model_catalog import get_model_family
        paths = [path for path in paths if path.stem == get_model_family(model_name)]
    return [{"name": path.stem, "path": path.relative_to(ROOT).as_posix(),
             "payload": json.loads(path.read_text(encoding="utf-8"))} for path in paths]


def compose_preset(preset, dataset_ref, selections):
    payload = copy.deepcopy(preset.get("payload", preset))
    if "training" in payload:
        payload.setdefault("experiment", {}).update(selections)
        if dataset_ref:
            payload["experiment"]["dataset_ref"] = dict(dataset_ref)
    else:
        payload.update(selections)
        if dataset_ref:
            payload["dataset_ref"] = dict(dataset_ref)
    return payload
