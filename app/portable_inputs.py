"""Source for mounted custom data and teacher inputs, with no embedded weights."""
import json


def portable_inputs_source(package, teachers, config):
    encoded = json.dumps({"package": package, "teachers": teachers, "config": config}, ensure_ascii=False)
    return f'''\nfrom pathlib import Path
import json, shutil, sys
sys.path.insert(0, str(Path.cwd()))
try:
    from src.training.experiment_config import EXTENSIBLE_DATASET_API
    assert EXTENSIBLE_DATASET_API >= 2
except (ImportError, AssertionError) as error:
    raise RuntimeError("Published repository does not support extensible datasets/KD yet. Publish the new code and regenerate this notebook; no local edits are included automatically.") from error
from src.datasets.dataset_runtime import validate_dataset_package, verify_dataset_content
from src.datasets.dataset_registry import DatasetRef
from src.datasets.dataset_registration import atomic_json, contained_path, sha256_file

# Edit mount paths after attaching a Kaggle Input or mounting Google Drive.
DATASET_MOUNT = ""  # folder containing images/ or train/, val/, test/
TEACHER_CHECKPOINTS = {{}}  # {{"resnet50": "/content/drive/MyDrive/resnet50.pt", ...}}
_portable = json.loads({encoded!r})
_root = Path.cwd().resolve()
_config = _portable["config"]
_package = _portable["package"]

def _write_immutable(path, value):
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != value:
        raise ValueError("Existing portable metadata differs: " + str(path))
    if not path.exists():
        atomic_json(path, value)

if _package:
    _ref = _config.get("experiment", _config)["dataset_ref"]
    validate_dataset_package(_package, _ref)
    if not DATASET_MOUNT:
        raise RuntimeError("Set DATASET_MOUNT to the mounted dataset folder, then rerun this cell.")
    _mount = Path(DATASET_MOUNT).resolve(strict=True)
    _destination = contained_path(_root, _package["descriptor"]["root_relative_path"])
    for _row in _package["manifest"]["samples"]:
        _source = (_mount / _row["relative_path"]).resolve(strict=True)
        if not _source.is_relative_to(_mount) or not _source.is_file() or sha256_file(_source) != _row["content_sha256"]:
            raise ValueError("Mounted dataset image differs: " + _row["relative_path"])
        _target = contained_path(_root, (_destination / _row["relative_path"]).relative_to(_root))
        if not _target.exists():
            _target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(_source, _target)
    _descriptor = dict(_package["descriptor"], default_revision=_ref["revision"], default_split=_ref["split_id"], ready=True)
    _base = _root / "datasets/.registry" / _ref["dataset_id"] / _ref["revision"]
    _write_immutable(_base / "manifest.json", _package["manifest"])
    _write_immutable(_base / "splits" / (_ref["split_id"] + ".json"), _package["split"])
    _write_immutable(_root / "configs/datasets" / (_ref["dataset_id"] + ".json"), _descriptor)
    verify_dataset_content(_root, DatasetRef(**_ref))
    print("Dataset content and fixed split verified.", flush=True)

_inputs = _portable["teachers"]
for _teacher in _inputs:
    _name = _teacher["model_name"]
    if not TEACHER_CHECKPOINTS.get(_name):
        raise RuntimeError("Set TEACHER_CHECKPOINTS for " + _name)
    _source = Path(TEACHER_CHECKPOINTS[_name]).resolve(strict=True)
    if sha256_file(_source) != _teacher["checkpoint_sha256"]:
        raise ValueError("Teacher checkpoint SHA256 mismatch: " + _name)
    _target = _root / "checkpoints/portable_teachers" / (_teacher["checkpoint_sha256"] + ".pt")
    _target.parent.mkdir(parents=True, exist_ok=True)
    if not _target.exists():
        shutil.copy2(_source, _target)
    if sha256_file(_target) != _teacher["checkpoint_sha256"]:
        raise ValueError("Portable checkpoint cache differs: " + _name)
    _teacher["checkpoint_path"] = _target.relative_to(_root).as_posix()
    if "teachers" in _config:
        _selected = next(item for item in _config["teachers"] if item["model_name"] == _name)
        _selected.update(checkpoint_path=_teacher["checkpoint_path"], checkpoint_sha256=_teacher["checkpoint_sha256"])
    else:
        _config["experiment"].update(teacher_checkpoint_path=_teacher["checkpoint_path"], teacher_checkpoint_sha256=_teacher["checkpoint_sha256"])
if _inputs:
    _config["teacher_inputs"] = _inputs
    from src.training.teacher_catalog import validate_teacher_selection
    validate_teacher_selection(_root, _config)
atomic_json(_root / "runs/configs/portable.json", _config)
print("Portable runtime config ready.", flush=True)
'''
