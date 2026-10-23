import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))


class DynamicBaselineTests(unittest.TestCase):
    def test_enqueue_snapshots_resolved_data_ref(self):
        from test_dataset_runtime import DatasetRuntimeTests
        from src.datasets.dataset_runtime import create_registered_split
        import app.experiment_runner as runner
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rev = DatasetRuntimeTests().prepare(root)
            ref = create_registered_split(root, "potato", rev.revision, dict(train=.7, val=.15, test=.15), 42)
            source = root / "preset.json"
            source.write_text(json.dumps(dict(stage="baseline", dataset_name="potato", model_name="resnet18")))
            with patch.object(runner, "RUNS_DIR", root / "runs"), patch.object(runner, "JOBS_DIR", root / "runs/jobs"), patch.object(runner, "LOGS_DIR", root / "runs/logs"), patch.object(runner, "get_project_root", return_value=root):
                job = runner.enqueue_job(["python", "src/training/train_baseline.py", "--config", str(source)], dict(dataset="potato", model="resnet18"))
            stored = json.loads((root / job["config"]).read_text())
            self.assertEqual(stored["dataset_ref"], ref.to_dict())
            create_registered_split(root, "potato", rev.revision, dict(train=.7, val=.15, test=.15), 43)
            self.assertEqual(json.loads((root / job["config"]).read_text())["dataset_ref"], ref.to_dict())

    def test_both_cli_configs_pin_custom_revision(self):
        from test_dataset_runtime import DatasetRuntimeTests
        from src.datasets.dataset_runtime import create_registered_split
        from src.training import train_baseline, train_teacher
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rev = DatasetRuntimeTests().prepare(root)
            ref = create_registered_split(root, "potato", rev.revision, dict(train=.7, val=.15, test=.15), 42)
            for module, model in ((train_baseline, "mobilenet_v3_small"), (train_teacher, "resnet50")):
                path = root / "config.json"
                path.write_text(json.dumps(dict(dataset_name="potato", model_name=model, epochs=4, pretrained=False)))
                with patch.object(module, "PROJECT_ROOT", root), patch.object(sys, "argv", ["train", "--config", str(path)]):
                    config = module.build_final_config(module.parse_args())
                self.assertEqual(config["dataset_ref"], ref.to_dict())

    def test_content_identity_in_recovery_and_loader(self):
        from test_dataset_runtime import DatasetRuntimeTests
        from src.datasets.dataset_runtime import create_registered_split, resolve_dataset_runtime
        from src.datasets.dataloaders import create_dataloaders
        from src.training.recovery import recovery_metadata
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rev = DatasetRuntimeTests().prepare(root)
            ref = create_registered_split(root, "potato", rev.revision, dict(train=.7, val=.15, test=.15), 42)
            runtime = resolve_dataset_runtime(root, ref)
            loaders = create_dataloaders(runtime["split_file"], 16, 2, 0, project_root=root)
            self.assertEqual(tuple(next(iter(loaders[0]))[0].shape), (2, 3, 16, 16))
            metadata = recovery_metadata(dict(stage="baseline", dataset_name="potato", dataset_ref=ref.to_dict()), runtime["split_file"], root)
            self.assertEqual(metadata["dataset_ref"], ref.to_dict())
            self.assertEqual(metadata["provenance_level"], "content_v2")


if __name__ == "__main__":
    unittest.main()
