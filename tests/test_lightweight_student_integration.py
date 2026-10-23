import ast
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from src.models.model_factory import create_model
from src.training.kd_config import DistillationConfig
from src.training.kd_features import FeatureCapture, default_feature_layer
from src.training.kd_strategies import create_distillation_strategy
from src.training.multi_kd_config import MultiDistillationConfig
from src.training.multi_kd_strategies import create_multi_teacher_strategy

ROOT = Path(__file__).resolve().parents[1]
STUDENTS = ("shufflenet_v2_x0_5", "micronet_m0")


class LightweightIntegrationTests(unittest.TestCase):
    def test_clean_configs_preserve_control_protocol_and_parse_in_baseline_cli(self):
        from unittest.mock import patch
        from src.training import train_baseline
        reference = json.loads((ROOT / "configs/baseline/mobilenetv3_small_plantpathology2021_clean.json").read_text())
        for name in STUDENTS:
            with self.subTest(model=name):
                path = ROOT / f"configs/baseline/{name}_plantpathology2021_clean.json"
                self.assertTrue(path.exists(), str(path))
                payload = json.loads(path.read_text())
                self.assertEqual({k: v for k, v in payload.items() if k != "model_name"},
                                 {k: v for k, v in reference.items() if k != "model_name"})
                with patch.object(sys, "argv", ["train", "--config", str(path)]):
                    config = train_baseline.build_final_config(train_baseline.parse_args())
                self.assertEqual(config["model_name"], name)
                self.assertEqual(config["model_role"], "student")
                self.assertEqual(config["architecture_family"], "cnn")

    def test_new_students_support_all_single_and_multi_teacher_losses(self):
        torch.set_num_threads(2)
        for name in STUDENTS:
            for kind in ("logit_based", "feature_based", "relation_based"):
                for multi in (False, True):
                    with self.subTest(model=name, kind=kind, multi=multi):
                        student = create_model(name, 6, False).train()
                        capture = FeatureCapture(student, default_feature_layer(name))
                        try:
                            logits = student(torch.randn(3, 3, 224, 224))
                            features = capture.require_output()
                            self.assertEqual(features.ndim, 4)
                            teachers = [torch.randn(3, 6), torch.randn(3, 6)]
                            teacher_features = [torch.randn(3, 16, 7, 7), torch.randn(3, 24, 7, 7)]
                            targets = torch.tensor([0, 1, 2])
                            if multi:
                                strategy = create_multi_teacher_strategy(kind, MultiDistillationConfig(),
                                    ["resnet50", "densenet201"], [.5, .5], student_channels=features.shape[1],
                                    teacher_channels=[16, 24], feature_modes=["spatial", "spatial"])
                                loss = strategy.compute(logits, teachers, targets,
                                                        student_features=features, teacher_features=teacher_features)
                            else:
                                strategy = create_distillation_strategy(kind, DistillationConfig(),
                                    student_channels=features.shape[1], teacher_channels=16, feature_mode="spatial")
                                loss = strategy.compute(logits, teachers[0], targets,
                                                        student_features=features, teacher_features=teacher_features[0])
                            loss.total.backward()
                            self.assertTrue(torch.isfinite(loss.total))
                            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in student.parameters()))
                            self.assertTrue(all(p.grad is not None for p in strategy.parameters()))
                        finally:
                            capture.close()

    def test_external_result_roundtrip_preserves_new_student_identity(self):
        import pandas as pd
        from app.external_run_import import import_external_bundle
        from src.tracking.external_bundle import create_external_bundle
        for name in STUDENTS:
            with self.subTest(model=name), tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                source, target = base / "source", base / "target"
                artifact = source / "results/baseline/runs/demo/history.csv"
                artifact.parent.mkdir(parents=True)
                artifact.write_text("epoch,loss\n1,0.2\n")
                split = source / "splits/appleleaf9.json"
                split.parent.mkdir()
                split.write_text('{"classes":["a","b"]}')
                row = dict(stage="baseline", dataset_name="appleleaf9", model_name=name,
                           run_name="demo", test_macro_f1=.8, history_path="results/baseline/runs/demo/history.csv",
                           split_file="splits/appleleaf9.json")
                outputs = create_external_bundle(source, row, source_commit="a" * 40, output_dir=base / "out")
                (target / "splits").mkdir(parents=True)
                shutil.copy2(split, target / "splits/appleleaf9.json")
                inbox = target / "runs/import_inbox" / outputs.bundle_id
                inbox.mkdir(parents=True)
                shutil.copy2(outputs.archive, inbox / outputs.archive.name)
                import_external_bundle(target, inbox, register_run=lambda *_: "fixture-local")
                imported = pd.read_csv(target / "results/baseline/baseline_results.csv").iloc[0]
                self.assertEqual(imported["model_name"], name)
                self.assertEqual(imported["model_role"], "student")
                self.assertEqual(imported["architecture_family"], "cnn")
                self.assertEqual(imported["test_macro_f1"], .8)

    def test_notebook_embeds_selected_student_and_clean_snapshot(self):
        from app.portable_notebook import build_portable_notebook
        from src.datasets.dataset_runtime import dataset_package
        for name in STUDENTS:
            payload = json.loads((ROOT / f"configs/baseline/{name}_plantpathology2021_clean.json").read_text())
            value = build_portable_notebook(repository_url="https://example.invalid/repo.git", source_commit="a" * 40,
                stage="baseline", dataset="plantpathology2021_clean", model=name,
                command=["python", "src/training/train_baseline.py", "--model", name], runtime_config=payload,
                dataset_package=dataset_package(ROOT, payload["dataset_ref"]))
            notebook = json.loads(value)
            cells = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
            for cell in cells:
                compile(cell, "notebook", "exec")
            inputs = next(cell for cell in cells if "_portable = json.loads" in cell)
            assignment = next(node for node in ast.parse(inputs).body if isinstance(node, ast.Assign)
                              and any(isinstance(target, ast.Name) and target.id == "_portable" for target in node.targets))
            scope = {"json": json}
            exec(compile(ast.Module(body=[assignment], type_ignores=[]), "notebook-config", "exec"), scope)
            self.assertEqual(scope["_portable"]["config"], payload)
            self.assertEqual(scope["_portable"]["package"]["descriptor"]["dataset_id"], "plantpathology2021_clean")

    def test_ui_selects_each_clean_config_and_keeps_training_disabled(self):
        code = """
import threading
threading.stack_size(8 * 1024 * 1024)
from streamlit.testing.v1 import AppTest
app = AppTest.from_file('app/streamlit_app.py').run(timeout=60)
app.switch_page('pages/0_New_Experiment.py').run(timeout=60)
def widget(kind, label):
    return next(w for w in getattr(app, kind) if w.label == label)
dataset = widget('selectbox', 'Veri seti')
dataset.select('plantpathology2021_clean').run(timeout=60)
for name in ('shufflenet_v2_x0_5', 'micronet_m0'):
    assert name in widget('selectbox', 'Student model').options
    widget('selectbox', 'Student model').select(name).run(timeout=60)
    assert not app.exception
    assert widget('selectbox', 'Config dosyası').value.endswith(name + '_plantpathology2021_clean.json')
    assert widget('selectbox', 'Scheduler').value == 'cosine'
    assert widget('number_input', 'Early stopping patience').value == 7
    assert widget('button', 'Eğitimi Başlat').disabled
    command = next(w.value for w in app.code if 'src/training/train_baseline.py' in w.value)
    assert '--model ' + name in command
    assert '--dataset plantpathology2021_clean' in command
print('LIGHT_STUDENTS_UI_OK; NO_TRAINING_CLICKED')
"""
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT,
                                capture_output=True, text=True, timeout=100)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("LIGHT_STUDENTS_UI_OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
