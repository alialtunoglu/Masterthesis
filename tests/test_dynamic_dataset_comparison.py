import tempfile
import unittest
from pathlib import Path

from src.evaluation.experiment_comparison import build_comparison


class DatasetComparisonTests(unittest.TestCase):
    def test_content_or_split_identity_cannot_share_a_baseline(self):
        from test_experiment_comparison import _run
        ref = {"dataset_id": "potato", "revision": "a" * 64, "split_id": "b" * 64}
        baseline = _run("base", "baseline", .8, dataset_ref=ref, identity_version=2)
        different = _run("kd", "single_kd", .9, dataset_ref={**ref, "split_id": "c" * 64}, identity_version=2)
        report = build_comparison([baseline, different], ["kd"], None)
        self.assertIsNone(report["provenance"]["baseline_id"])
        self.assertIsNone(report["rows"][0]["delta_macro_f1_pp"])
        self.assertIn("dataset_ref", report["provenance"]["protocols"]["kd"])

    def test_run_removal_cannot_target_a_dataset_registry(self):
        from app.experiment_removal import ExperimentRemovalService, ExperimentRemovalError
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "datasets/.registry/potato/manifest.json"
            raw.parent.mkdir(parents=True)
            raw.write_text("preserve")
            with self.assertRaises(ExperimentRemovalError):
                ExperimentRemovalService(root)._artifact_paths({"run_dir": "datasets/.registry/potato"})
            self.assertEqual(raw.read_text(), "preserve")
            (root / "results").mkdir()
            (root / "checkpoints").mkdir()
            for row in ({"run_dir": "results"}, {"checkpoint_path": "checkpoints"}):
                with self.assertRaises(ExperimentRemovalError):
                    ExperimentRemovalService(root)._artifact_paths(row)
            ancestor = root / "results/teachers/cnn/runs/potato/resnet50/legitimate-run"
            ancestor.mkdir(parents=True)
            with self.assertRaises(ExperimentRemovalError):
                ExperimentRemovalService(root)._artifact_paths({"run_dir": "results/teachers/cnn/runs/potato", "run_name": "potato"})


if __name__ == "__main__":
    unittest.main()
