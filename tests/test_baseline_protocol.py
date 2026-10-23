from contextlib import ExitStack
import json
from pathlib import Path
import sys
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

class BaselineProtocolTests(unittest.TestCase):
    def setUp(self):
        global baseline
        from training import train_baseline as baseline

    def test_cosine_and_patience_survive_config_validation(self):
        with patch.object(sys, "argv", ["baseline", "--dataset", "plantpathology2021", "--model", "mobilenet_v3_small"]):
            args = baseline.parse_args()
        with patch.object(baseline, "load_config_file", return_value={"scheduler": "cosine", "early_stopping_patience": 7}):
            config = baseline.build_final_config(args)
        self.assertEqual(config["scheduler"], "cosine")
        self.assertEqual(config["early_stopping_patience"], 7)

    def test_main_applies_cosine_and_stops_after_patience(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            root = Path(folder)
            split = root / "split.json"
            split.write_text(json.dumps({"dataset_name": "fixture", "classes": ["healthy", "rust"], "splits": {"train": [], "val": [], "test": []}}))
            config = {**baseline.DEFAULT_CONFIG, "dataset_name": "plantpathology2021", "model_name": "mobilenet_v3_small",
                      "epochs": 6, "batch_size": 2, "image_size": 4, "learning_rate": .0003, "scheduler": "cosine",
                      "early_stopping_patience": 2, "monitor_metric": "val_macro_f1", "device": "cpu"}
            loader = DataLoader(TensorDataset(torch.ones(2, 3, 4, 4), torch.tensor([0, 1])), batch_size=2)
            histories = []
            actual_fit = baseline.fit
            def recording_fit(**kwargs):
                result = actual_fit(**kwargs)
                histories.append(result["history"])
                return result
            evaluation = {"loss": .5, "metrics": {"accuracy": .5, "macro_precision": .5, "macro_recall": .5, "macro_f1": .5,
                           "weighted_precision": .5, "weighted_recall": .5, "weighted_f1": .5}, "y_true": [0, 1], "y_pred": [0, 0]}
            replacements = {"PROJECT_ROOT": root, "RESULTS_DIR": root / "results", "BASELINE_RUNS_DIR": root / "results/runs",
                            "CHECKPOINTS_DIR": root / "checkpoints", "parse_args": lambda: SimpleNamespace(no_recovery=False, resume_from=None, backup_dir=None),
                            "build_final_config": lambda args: config, "resolve_dataset_runtime": lambda *args: {"split_file": str(split)},
                            "create_dataloaders": lambda **kwargs: (loader, loader, loader, ["healthy", "rust"]),
                            "create_model": lambda *args, **kwargs: torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(48, 2)),
                            "fit": recording_fit, "start_run": lambda **kwargs: SimpleNamespace(info=SimpleNamespace(run_id="fixture"))}
            for name, value in replacements.items():
                stack.enter_context(patch.object(baseline, name, value))
            for name in ("setup_mlflow", "end_run", "log_artifact", "log_text", "log_metrics", "log_params", "write_training_artifacts", "append_summary_row"):
                stack.enter_context(patch.object(baseline, name, lambda *args, **kwargs: None))
            stack.enter_context(patch("training.trainer.evaluate", return_value=evaluation))
            stack.enter_context(patch.dict("os.environ", {"TORCH_HOME": str(root / "cache")}))
            baseline.main()
            self.assertEqual(len(histories[0]), 3)
            rates = [row["learning_rate"] for row in histories[0]]
            self.assertLess(rates[-1], rates[0])
            self.assertGreater(rates[-1], 0)

    def test_legacy_scheduler_default_is_unchanged(self):
        with patch.object(sys, "argv", ["baseline", "--dataset", "plantpathology2021", "--model", "mobilenet_v3_small"]):
            config = baseline.build_final_config(baseline.parse_args())
        self.assertEqual(config["scheduler"], "none")
        self.assertIsNone(config.get("early_stopping_patience"))

    def test_invalid_patience_and_scheduler_params_are_rejected(self):
        with patch.object(sys, "argv", ["baseline", "--dataset", "plantpathology2021", "--model", "mobilenet_v3_small"]):
            args = baseline.parse_args()
        for payload in ({"early_stopping_patience": -1}, {"scheduler_params": []}):
            with self.subTest(payload=payload), patch.object(baseline, "load_config_file", return_value=payload):
                with self.assertRaises(ValueError):
                    baseline.build_final_config(args)

    def test_command_builder_protocol_flags_roundtrip(self):
        from app.experiment_runner import build_baseline_command
        command = build_baseline_command("plantpathology2021", "mobilenet_v3_small", None, 30, 32, 224, .0003, .0001, True, False,
                                         scheduler="cosine", early_stopping_patience=7)
        with patch.object(sys, "argv", command[1:]):
            config = baseline.build_final_config(baseline.parse_args())
        self.assertEqual((config["scheduler"], config["early_stopping_patience"]), ("cosine", 7))

    def test_new_run_names_leave_room_for_windows_recovery_files(self):
        from tracking.experiment_naming import build_unique_run_name
        root = Path(__file__).resolve().parents[1]
        for stage, prefix, model in (("baseline", "results/baseline/runs", "mobilenet_v3_small"),
                                     ("teacher_cnn", "results/teachers/cnn/runs", "regnet_y_8gf")):
            run = build_unique_run_name(stage, 42, "cuda", False)
            path = root / prefix / "plantpathology2021_clean" / model / run / "recovery/.pending-12345678" / ("manifest.json." + "a" * 32 + ".tmp")
            self.assertLess(len(str(path)), 260)
            self.assertNotEqual(run, build_unique_run_name(stage, 42, "cuda", False))

    def test_launcher_protocol_values_reach_the_effective_command(self):
        # Isolate AppTest's process-wide navigation state from CLI trainer imports.
        code = """
from streamlit.testing.v1 import AppTest
app = AppTest.from_file('app/streamlit_app.py').run(timeout=60)
app.switch_page('pages/2_Baseline_Experiments.py').run(timeout=60)
assert not app.exception
scheduler = next(w for w in app.selectbox if w.label == 'Scheduler')
scheduler.set_value('cosine').run(timeout=60)
next(w for w in app.number_input if w.label == 'Early stopping patience').set_value(7).run(timeout=60)
assert not app.exception
command = next(item.value for item in app.code if 'src/training/train_baseline.py' in item.value)
assert '--scheduler cosine' in command
assert '--early-stopping-patience 7' in command
assert next(w for w in app.button if w.label == 'Eğitimi Başlat').disabled
print('UI_PROTOCOL_OK')
"""
        result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("UI_PROTOCOL_OK", result.stdout)
