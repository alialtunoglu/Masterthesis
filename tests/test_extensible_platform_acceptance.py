"""Real entry points with tiny networks and temporary registered images only."""
import copy
import importlib
import json
import os
import tempfile
import unittest
import uuid
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import torch
from torch import nn


class TinyClassifier(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 4, 3, stride=4), nn.ReLU())
        self.classifier = nn.Linear(4, classes)

    def forward(self, inputs):
        return self.classifier(self.features(inputs).mean((2, 3)))


class ExtensibleAcceptanceTests(unittest.TestCase):
    def prepare(self, root):
        from test_dataset_runtime import DatasetRuntimeTests
        from src.datasets.dataset_runtime import create_registered_split, resolve_dataset_runtime
        rev = DatasetRuntimeTests().prepare(root)
        ref = create_registered_split(root, rev.dataset_id, rev.revision, dict(train=.7, val=.15, test=.15), 42)
        return ref, resolve_dataset_runtime(root, ref)

    def cli(self, module, root, payload, *, dry=True):
        config = root / f"fixture-{uuid.uuid4().hex}.json"
        config.write_text(json.dumps(payload))
        def model_factory(name, num_classes, pretrained=False):
            self.assertFalse(pretrained, "Acceptance tests must never download weights")
            return TinyClassifier(num_classes)
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ))
            stack.enter_context(patch.object(module, "PROJECT_ROOT", root))
            stack.enter_context(patch.object(module, "create_model", side_effect=model_factory))
            stack.enter_context(patch("sys.argv", [module.__name__, "--config", str(config), *(["--dry-run"] if dry else [])]))
            for name, relative in {"RESULTS_DIR": "results/baseline" if module.__name__.endswith("train_baseline") else "results/" + ("multi_teacher_knowledge_distillation" if module.__name__.endswith("train_multi_kd") else "knowledge_distillation"),
                                   "BASELINE_RUNS_DIR": "results/baseline/runs", "RUNS_DIR": "results/" + ("multi_teacher_knowledge_distillation" if module.__name__.endswith("train_multi_kd") else "knowledge_distillation") + "/runs",
                                   "CHECKPOINTS_DIR": "checkpoints/" + ("multi_teacher_knowledge_distillation" if module.__name__.endswith("train_multi_kd") else "knowledge_distillation"),
                                   "SUMMARY_PATH": "results/" + ("multi_teacher_knowledge_distillation/multi_kd_results.csv" if module.__name__.endswith("train_multi_kd") else "knowledge_distillation/kd_results.csv")}.items():
                if hasattr(module, name):
                    stack.enter_context(patch.object(module, name, root / relative))
            for name in ("setup_mlflow", "log_artifact", "log_metrics", "log_params", "log_text"):
                stack.enter_context(patch.object(module, name))
            finish = stack.enter_context(patch.object(module, "end_run"))
            stack.enter_context(patch.object(module, "start_run", return_value=SimpleNamespace(info=SimpleNamespace(run_id=uuid.uuid4().hex))))
            if hasattr(module, "get_model_compute_dict"):
                stack.enter_context(patch.object(module, "get_model_compute_dict", return_value={"macs": 1, "estimated_flops": 2}))
            module.main()
            finish.assert_called_once_with("FINISHED")

    def test_four_baseline_branches_use_registered_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ref, runtime = self.prepare(root)
            for script, model in (("train_baseline", "mobilenet_v3_small"), ("train_baseline", "deit_tiny_patch16_224"),
                                  ("train_teacher", "resnet50"), ("train_teacher", "swin_v2_t")):
                with self.subTest(model=model):
                    payload = {"dataset_name": "potato", "dataset_ref": ref.to_dict(), "model_name": model,
                               "pretrained": False, "epochs": 1, "warmup_epochs": 0, "batch_size": 3, "num_workers": 0,
                               "device": "cpu", "image_size": 224, "recovery": {"enabled": False}}
                    self.cli(importlib.import_module("src.training." + script), root, payload)
            self.assertFalse(list(root.glob("results/**/teacher_results.csv")))
            self.assertFalse(list(root.glob("checkpoints/**/*.pt")))
            self.assertEqual(len(runtime["class_names"]), 3)

    def test_all_kd_dry_runs_and_one_complete_result(self):
        from src.datasets.dataset_registration import sha256_file
        from src.training.kd_config import load_kd_config
        from src.training.multi_kd_config import load_multi_kd_config
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ref, runtime = self.prepare(root)
            teachers = []
            for name in ("resnet50", "densenet201", "regnet_y_8gf", "convnext_tiny"):
                path = root / "checkpoints" / f"{name}.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model_state_dict": TinyClassifier(3).state_dict(), "class_names": runtime["class_names"]}, path)
                teachers.append({"model_name": name, "run_id": name + "-source", "checkpoint_path": path.relative_to(root).as_posix(),
                                 "checkpoint_sha256": sha256_file(path), "feature_layer": "input:classifier"})
            inputs = [{**item, "dataset_ref": ref.to_dict(), "dataset_name": "potato", "class_names": runtime["class_names"],
                       "image_size": 224, "status": "completed"} for item in teachers]
            train = {"epochs": 1, "batch_size": 3, "image_size": 224, "num_workers": 0, "device": "cpu",
                     "student_pretrained": False, "data_augmentation": False, "scheduler": "none"}
            for multi in (False, True):
                for kind in ("logit_based", "feature_based", "relation_based"):
                    with self.subTest(multi=multi, kind=kind):
                        exp = {"dataset_name": "potato", "dataset_ref": ref.to_dict(), "kd_type": kind}
                        if multi:
                            payload = {"experiment": exp, "student": {"model_name": "deit_tiny_patch16_224", "feature_layer": "input:classifier"},
                                       "teachers": teachers, "training": train, "distillation": {}, "teacher_inputs": inputs, "recovery": {"enabled": False}}
                            payload = load_multi_kd_config(payload).to_dict()
                        else:
                            exp.update(student_model_name="deit_tiny_patch16_224", teacher_model_name=teachers[0]["model_name"],
                                       teacher_run_id=teachers[0]["run_id"], teacher_checkpoint_path=teachers[0]["checkpoint_path"], teacher_checkpoint_sha256=teachers[0]["checkpoint_sha256"])
                            payload = {"experiment": exp, "training": train, "distillation": {"student_layer": "input:classifier", "teacher_layer": "input:classifier"},
                                       "teacher_inputs": inputs[:1], "recovery": {"enabled": False}}
                            payload = load_kd_config(payload).to_dict()
                        module = importlib.import_module("src.training.train_multi_kd" if multi else "src.training.train_kd")
                        self.cli(module, root, payload)
                        self.assertFalse(list(root.glob("results/**/*results.csv")))
                        if not multi and kind == "logit_based":
                            full = copy.deepcopy(payload)
            self.cli(importlib.import_module("src.training.train_kd"), root, full, dry=False)
            frame = pd.read_csv(root / "results/knowledge_distillation/kd_results.csv")
            self.assertEqual(len(frame), 1)
            self.assertEqual(json.loads(frame.iloc[0]["dataset_ref"]), ref.to_dict())
            self.assertTrue((root / frame.iloc[0]["checkpoint_path"]).is_file())
            self.assertTrue((root / frame.iloc[0]["history_path"]).is_file())
            from src.tracking.external_bundle import create_external_bundle
            from app.external_run_import import import_external_bundle
            from src.evaluation.experiment_comparison import load_comparison_runs
            from app.experiment_removal import ExperimentRemovalService, RemovalMode
            import shutil
            row = frame.iloc[0].to_dict()
            outputs = create_external_bundle(root, row, source_commit="a" * 40, output_dir=root / "export")
            target = root / "import-target"
            inbox = target / "runs/import_inbox" / outputs.bundle_id
            inbox.mkdir(parents=True)
            shutil.copy2(outputs.archive, inbox / outputs.archive.name)
            shutil.copy2(outputs.checkpoint, inbox / outputs.checkpoint.name)
            import_external_bundle(target, inbox, register_run=lambda *_: "imported-fixture")
            runs = load_comparison_runs(target)
            self.assertEqual(runs[0]["dataset_ref"], ref.to_dict())
            class Tracking:
                def delete_run(self, *args):
                    return True
                def restore_run(self, *args):
                    pass
            original = (root / "configs/datasets/potato.json").read_bytes()
            ExperimentRemovalService(root, tracking_gateway=Tracking()).remove(row, RemovalMode.ARCHIVE)
            self.assertEqual((root / "configs/datasets/potato.json").read_bytes(), original)
            self.assertTrue((root / "datasets/Potato/images/Healthy/0.png").exists())

    def test_notebook_mount_rehydrates_exact_identity_and_rejects_changed_bytes(self):
        from app.portable_inputs import portable_inputs_source
        from src.datasets.dataset_runtime import dataset_package, resolve_dataset_runtime
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source, target = base / "source", base / "remote"
            ref, runtime = self.prepare(source)
            target.mkdir()
            payload = {"dataset_name": "potato", "dataset_ref": ref.to_dict(), "model_name": "mobilenet_v3_small"}
            code = portable_inputs_source(dataset_package(source, ref), [], payload).replace('DATASET_MOUNT = ""', f'DATASET_MOUNT = {str(source / "datasets/Potato")!r}')
            previous = Path.cwd()
            try:
                os.chdir(target)
                exec(compile(code, "portable", "exec"), {})
                self.assertEqual(resolve_dataset_runtime(target, ref)["dataset_ref"], runtime["dataset_ref"])
                image = next((source / "datasets/Potato/images").rglob("*.png"))
                image.write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "Mounted dataset image differs"):
                    exec(compile(code, "portable", "exec"), {})
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
