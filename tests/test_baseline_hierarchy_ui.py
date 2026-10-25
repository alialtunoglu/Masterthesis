import unittest
import sys
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))


def by_label(widgets, label):
    return next(widget for widget in widgets if widget.label == label)


class BaselineHierarchyUITests(unittest.TestCase):
    def test_conflicting_legacy_job_does_not_crash_overview_or_monitor(self):
        from app.experiment_ui import job_identity

        job = {"job_id": "conflict", "stage": "teacher_cnn", "model": "resnet18",
               "dataset": "plantvillage", "status": "finished",
               "command": ["python", "src/training/train_baseline.py"],
               "log_path": "runs/logs/missing-test.log"}
        self.assertIn("Sınıflandırılamayan deney", job_identity(job))
        self.assertEqual(job["stage"], "teacher_cnn")
        with patch("app.experiment_runner.list_jobs", return_value=[job]), patch("app.experiment_runner.get_recovery_status", return_value={}):
            app = AppTest.from_file("app/streamlit_app.py").run(timeout=60)
            self.assertFalse(app.exception)
            app.switch_page("pages/3_Job_Monitor.py").run(timeout=60)
            self.assertFalse(app.exception)
            self.assertTrue(app.warning)

    def test_student_vit_is_a_baseline_with_compatible_presets(self):
        app = AppTest.from_file("app/streamlit_app.py").run(timeout=60)
        app.switch_page("pages/0_New_Experiment.py").run(timeout=60)
        self.assertEqual(by_label(app.selectbox, "Eğitim türü").value, "Baseline")
        by_label(app.selectbox, "Mimari ailesi").select("ViT").run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(by_label(app.selectbox, "Model rolü").value, "Student")
        self.assertEqual(set(by_label(app.selectbox, "Student model").options), {"deit_small_patch16_224", "deit_tiny_patch16_224"})
        self.assertIn("configs/baseline/vision_transformers", by_label(app.selectbox, "Config dosyası").value)
        self.assertNotIn("Image size", [widget.label for widget in app.number_input])
        self.assertNotIn("Mixed precision (AMP)", [widget.label for widget in app.checkbox])
        self.assertTrue(by_label(app.button, "Eğitimi Başlat").disabled)

    def test_legacy_student_jobs_are_filterable_as_baseline_student_cnn(self):
        job = {"job_id": "old", "model": "resnet18", "dataset": "plantvillage", "status": "finished",
               "command": ["python", "src/training/train_baseline.py"], "log_path": "runs/logs/missing-test.log"}
        with patch("app.experiment_runner.list_jobs", return_value=[job]), patch("app.experiment_runner.get_recovery_status", return_value={}):
            app = AppTest.from_file("app/streamlit_app.py").run(timeout=60)
            app.switch_page("pages/3_Job_Monitor.py").run(timeout=60)
            self.assertFalse(app.exception)
            self.assertFalse(by_label(app.button, "Baseline").disabled)
            by_label(app.button, "Baseline").click().run(timeout=60)
            by_label(app.selectbox, "Model rolü").select("Student").run(timeout=60)
            by_label(app.selectbox, "Mimari ailesi").select("CNN").run(timeout=60)
            self.assertIn("Student baseline", by_label(app.selectbox, "Deney seç").options[0])


if __name__ == "__main__":
    unittest.main()
