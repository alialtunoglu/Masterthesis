import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))


class ExperimentTaxonomyTests(unittest.TestCase):
    def test_legacy_baseline_is_explicit_student_cnn(self):
        from src.tracking.experiment_taxonomy import enrich_experiment_metadata
        job = {"job_id": "old", "command": ["python", "src/training/train_baseline.py"], "model": "resnet18"}
        enriched = enrich_experiment_metadata(job)
        self.assertEqual({key: enriched[key] for key in ("stage", "training_method", "model_role", "architecture_family")},
                         {"stage": "baseline", "training_method": "baseline", "model_role": "student", "architecture_family": "cnn"})
        self.assertNotIn("stage", job)

    def test_teacher_role_and_vit_family_are_independent(self):
        from src.tracking.experiment_taxonomy import enrich_experiment_metadata
        for stage, model, role, family in [
            ("teacher_cnn", "resnet50", "teacher", "cnn"),
            ("teacher_vision_transformer", "swin_v2_t", "teacher", "vision_transformer"),
            ("baseline", "deit_tiny_patch16_224", "student", "vision_transformer"),
            ("knowledge_distillation", "mobilenet_v3_small", "student", "cnn"),
            ("multi_teacher_knowledge_distillation", "mobilenet_v3_small", "student", "cnn"),
        ]:
            with self.subTest(stage=stage, model=model):
                enriched = enrich_experiment_metadata({"stage": stage, "model": model})
                self.assertEqual(enriched["stage"], stage)
                self.assertEqual((enriched["model_role"], enriched["architecture_family"]), (role, family))

    def test_unknown_or_conflicting_records_are_not_guessed(self):
        from src.tracking.experiment_taxonomy import enrich_experiment_metadata
        self.assertNotIn("stage", enrich_experiment_metadata({"command": ["python", "custom.py"], "model": "resnet18"}))
        self.assertNotIn("stage", enrich_experiment_metadata({"command": ["python", "custom.py", "train_baseline.py"], "model": "resnet18"}))
        with self.assertRaises(ValueError):
            enrich_experiment_metadata({"stage": "teacher_cnn", "command": ["python", "train_baseline.py"], "model": "resnet18"})

    def test_enqueue_persists_classification(self):
        import experiment_runner as runner
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(runner, "RUNS_DIR", root / "runs"), patch.object(runner, "JOBS_DIR", root / "runs/jobs"), \
                 patch.object(runner, "LOGS_DIR", root / "runs/logs"), patch.object(runner, "get_project_root", return_value=root):
                job = runner.enqueue_job(["python", "src/training/train_baseline.py"], {"model": "deit_small_patch16_224"})
            stored = json.loads((root / "runs/jobs" / f"{job['job_id']}.json").read_text())
            self.assertEqual((stored["stage"], stored["model_role"], stored["architecture_family"]),
                             ("baseline", "student", "vision_transformer"))


if __name__ == "__main__":
    unittest.main()
