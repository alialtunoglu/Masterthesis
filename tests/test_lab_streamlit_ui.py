"""Native Streamlit GPU/monitor controls; no process launch or training."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import gpu_view

CARDS = [{"id": str(index), "name": "NVIDIA fixture", "memory_used_mb": 100,
          "memory_total_mb": 16000, "utilization": 2} for index in range(2)]


class LabStreamlitUITests(unittest.TestCase):
    def test_gpu_selection_returns_physical_gpu_and_explains_queue_behavior(self):
        with patch.object(gpu_view, "gpu_inventory", return_value=CARDS):
            app = AppTest.from_string('from gpu_view import render_gpu_selection\nimport streamlit as st\nst.write(render_gpu_selection("fixture"))').run()
            selector = next(item for item in app.selectbox if item.label == "Çalıştırılacak GPU")
            selector.set_value("1").run()
        self.assertFalse(app.exception)
        self.assertTrue(any(item.value == "1" for item in app.markdown))
        self.assertTrue(any("kuyrukta" in item.value for item in app.caption))

    def test_notebook_job_monitor_shows_gpu_and_safe_controls_not_force_kill(self):
        job = {"job_id": "lab-fixture", "stage": "knowledge_distillation", "dataset": "fixture",
               "model": "shufflenet_v2_x0_5", "student_model": "shufflenet_v2_x0_5", "teacher_model": "resnet50",
               "kd_type": "feature_based", "execution_origin": "lab_notebook", "gpu_id": "1", "dry_run": False,
               "status": "running", "process_identity_verified": True, "log_path": "fixture.log"}
        with patch("experiment_runner.list_jobs", return_value=[job]), \
                patch("experiment_runner.read_log_tail", return_value="EPOCH_SUMMARY epoch=3/30 val_macro_f1=0.8"), \
                patch("experiment_runner.get_recovery_status", return_value={"metadata": {"recovery": {"enabled": True}}}), \
                patch.object(gpu_view, "gpu_inventory", return_value=CARDS):
            app = AppTest.from_file("app/pages/3_Job_Monitor.py").run(timeout=60)
        self.assertFalse(app.exception)
        self.assertTrue(any("GPU 1" in item.value and "Notebook" in item.value for item in app.caption))
        self.assertFalse(any(item.label == "Zorla Durdur" for item in app.button))
        self.assertTrue(any(item.label == "Epoch Sonunda Kaydet ve Durdur" for item in app.button))

    def test_running_job_does_not_offer_result_export(self):
        with patch("experiment_runner.list_jobs", return_value=[{"job_id": "fixture", "stage": "baseline",
                "dataset": "fixture", "model": "resnet18", "status": "running", "gpu_id": "0"}]), \
                patch.object(gpu_view, "gpu_inventory", return_value=CARDS):
            app = AppTest.from_file("app/pages/3_Job_Monitor.py").run(timeout=60)
        self.assertFalse(app.exception)
        self.assertFalse(any(item.label == "Taşınacak klasörü hazırla ve göster" for item in app.button))

    def test_completed_job_exports_only_after_explicit_click(self):
        job = {"job_id": "completed-fixture", "stage": "baseline", "dataset": "fixture", "model": "resnet18",
               "status": "finished", "gpu_id": "0", "mlflow_run_id": "source", "log_path": "fixture.log"}
        transfer = {"folder": "/fixture/runs/transfer/completed-fixture", "inbox_folder": "runs/import_inbox/completed-fixture"}
        with patch("experiment_runner.list_jobs", return_value=[job]), \
                patch.object(gpu_view, "gpu_inventory", return_value=CARDS), \
                patch("streamlit.page_link"), \
                patch("run_transfer.prepare_run_transfer", return_value=transfer) as export:
            app = AppTest.from_file("app/pages/3_Job_Monitor.py").run(timeout=60)
            export.assert_not_called()
            next(item for item in app.button if item.label == "Taşınacak klasörü hazırla ve göster").click().run(timeout=60)
            self.assertEqual(export.call_count, 1)
        self.assertFalse(app.exception)
        self.assertTrue(any("runs/import_inbox/completed-fixture" in item.value for item in app.code))

    def test_unknown_notebook_status_warns_and_cannot_force_stop(self):
        job = {"job_id": "lab-unknown", "stage": "baseline", "dataset": "fixture", "model": "resnet18",
               "status": "unknown", "execution_origin": "lab_notebook", "monitor_warning": "Kimlik doğrulanamadı"}
        with patch("experiment_runner.list_jobs", return_value=[job]), \
                patch.object(gpu_view, "gpu_inventory", return_value=CARDS):
            app = AppTest.from_file("app/pages/3_Job_Monitor.py").run(timeout=60)
        self.assertFalse(app.exception)
        self.assertTrue(any("Kimlik doğrulanamadı" in item.value for item in app.warning))
        self.assertFalse(any(item.label == "Zorla Durdur" for item in app.button))

    def test_overview_does_not_hide_uncertain_notebook_attempt(self):
        with patch("experiment_runner.list_jobs", return_value=[{"job_id": "lab-unknown", "model": "resnet18",
                "status": "unknown", "execution_origin": "lab_notebook"}]), \
                patch.object(gpu_view, "gpu_inventory", return_value=CARDS):
            app = AppTest.from_file("app/streamlit_app.py").run(timeout=60)
        self.assertFalse(app.exception)
        self.assertTrue(any("1 notebook" in item.value for item in app.warning))


if __name__ == "__main__":
    unittest.main()
