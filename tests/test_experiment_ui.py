"""Task-oriented UI contracts, without starting a training process."""

import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))


class ExperimentUITests(unittest.TestCase):
    def test_overview_queue_removal_preserves_job_and_artifacts(self):
        import experiment_runner as runner

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = root / "jobs"
            jobs.mkdir()
            job_path = jobs / "waiting-fixture.json"
            job_path.write_text(json.dumps({"job_id": "waiting-fixture", "status": "queued",
                                            "model": "resnet18"}), encoding="utf-8")
            artifact = root / "config.json"
            artifact.write_text('{"epochs":30}', encoding="utf-8")
            with patch.object(runner, "JOBS_DIR", jobs), patch.object(runner, "LOGS_DIR", root / "logs"), \
                    patch.object(runner, "get_queue_worker_status", return_value={"running": True}):
                app = AppTest.from_file("app/streamlit_app.py").run(timeout=60)
                remove = next((button for button in app.button
                               if button.key == "overview_cancel_waiting-fixture"), None)
                self.assertIsNotNone(remove, "Queued jobs need a confirmed removal action on Overview")
                self.assertEqual(json.loads(job_path.read_text())["status"], "queued")
                remove.click().run(timeout=60)
                self.assertFalse(app.exception)
                self.assertEqual(json.loads(job_path.read_text())["status"], "cancelled")
                self.assertTrue(artifact.is_file())
                self.assertTrue(any("kuyruktan çıkarıldı" in message.value for message in app.success))
                self.assertFalse(any(button.key == "overview_cancel_waiting-fixture" for button in app.button))

    def test_overview_recent_jobs_are_collapsed(self):
        with patch("experiment_runner.list_jobs", return_value=[
            {"job_id": "finished-fixture", "status": "finished", "model": "resnet18"}
        ]):
            app = AppTest.from_file("app/streamlit_app.py").run(timeout=60)
        self.assertFalse(app.exception)
        recent = next((group for group in app.expander if group.label == "Son deneyler (1)"), None)
        self.assertIsNotNone(recent, "Recent jobs should not compete with active work")
        self.assertFalse(recent.proto.expanded)
        self.assertTrue(any("finished-fixture" in item.value for item in recent.caption))

    def test_overview_emphasizes_running_job_and_discloses_queued_jobs_separately(self):
        jobs = [{"job_id": "waiting-fixture", "status": "queued", "model": "resnet18"},
                {"job_id": "running-fixture", "status": "running", "model": "mobilenet_v3_small"}]
        with patch("experiment_runner.list_jobs", return_value=jobs), \
                patch("experiment_runner.get_queue_worker_status", return_value={"running": True}):
            app = AppTest.from_file("app/streamlit_app.py").run(timeout=60)
        self.assertFalse(app.exception)
        follow = next((button for button in app.button if button.label == "Eğitimi izle"), None)
        self.assertIsNotNone(follow, "Running job needs a distinct primary follow action")
        self.assertEqual(follow.key, "overview_running-fixture")
        self.assertEqual(follow.proto.type, "primary")
        pending = next(group for group in app.expander if group.label == "Kuyrukta bekleyenler (1)")
        self.assertTrue(pending.proto.expanded)
        self.assertTrue(any("waiting-fixture" in item.value for item in pending.caption))
        self.assertFalse(any("running-fixture" in item.value for item in pending.caption))
        self.assertTrue(any("Çalışıyor" in item.value for item in app.markdown))
        self.assertTrue(any("Kuyrukta" in item.value for item in pending.markdown))
        self.assertFalse(any(button.key == "overview_cancel_running-fixture" for button in app.button))

    def test_overview_does_not_call_stop_requested_job_running(self):
        job = {"job_id": "stopping-fixture", "status": "stop_requested", "model": "resnet18"}
        with patch("experiment_runner.list_jobs", return_value=[job]):
            app = AppTest.from_file("app/streamlit_app.py").run(timeout=60)
        self.assertFalse(app.exception)
        self.assertTrue(any("Durdurma bekleniyor" in item.value for item in app.markdown))
        self.assertFalse(any("Çalışıyor" in item.value for item in app.markdown))

    def test_overview_shows_running_before_stopping_and_queued_jobs(self):
        jobs = [{"job_id": job_id, "status": status, "model": "resnet18"}
                for job_id, status in [("queue-a", "queued"), ("stopping", "stop_requested"),
                                       ("queue-b", "queued"), ("running-a", "running"),
                                       ("running-b", "running"), ("done", "finished")]]
        with patch("experiment_runner.list_jobs", return_value=jobs), \
                patch("experiment_runner.get_queue_worker_status", return_value={"running": True}):
            app = AppTest.from_file("app/streamlit_app.py").run(timeout=60)
        self.assertFalse(app.exception)
        displayed = [caption.value.split("Job ID: ")[1] for caption in app.caption
                     if "Job ID: " in caption.value]
        self.assertEqual(displayed, ["running-a", "running-b", "stopping", "queue-a", "queue-b", "done"])

    def test_overview_recent_jobs_use_time_not_random_id_order(self):
        jobs = [{"job_id": "zz-old", "queued_time": "2026-01-01T00:00:00", "status": "finished", "model": "resnet18"},
                {"job_id": "aa-new", "queued_time": "2026-10-01T00:00:00", "status": "finished", "model": "resnet18"}]
        with patch("experiment_runner.list_jobs", return_value=jobs):
            app = AppTest.from_file("app/streamlit_app.py").run(timeout=60)
        captions = [caption.value for caption in app.caption if "Job ID:" in caption.value]
        self.assertIn("aa-new", captions[0])

    def test_launch_actions_precede_optional_advanced_details(self):
        app = AppTest.from_file("app/streamlit_app.py").run(timeout=60)
        app.switch_page("pages/2_Baseline_Experiments.py").run(timeout=60)
        elements = list(app.main)
        launch = next(index for index, element in enumerate(elements)
                      if type(element).__name__ == "Button" and element.label == "Eğitimi Başlat")
        advanced = next(index for index, element in enumerate(elements)
                        if type(element).__name__ == "Expander" and element.label == "Gelişmiş eğitim ayarları")
        self.assertLess(launch, advanced)

    def test_monitor_secondary_filters_are_in_a_collapsed_group(self):
        job = {"job_id": "filter-fixture", "status": "finished", "stage": "baseline",
               "dataset": "plantvillage", "model": "resnet18", "log_path": "runs/logs/ui-fixture-missing.log"}
        with patch("experiment_runner.list_jobs", return_value=[job]), \
             patch("experiment_runner.get_recovery_status", return_value={}):
            app = AppTest.from_file("app/streamlit_app.py").run(timeout=60)
            app.switch_page("pages/3_Job_Monitor.py").run(timeout=60)
        groups = [group for group in app.expander
                  if {widget.label for widget in group.selectbox} >= {"Durum", "Veri seti"}
                  and {button.label for button in group.button} >= {"Baseline", "Single-teacher KD", "Multi-teacher KD"}]
        self.assertEqual(len(groups), 1)
        self.assertFalse(groups[0].proto.expanded)

    def test_launch_handoff_opens_the_queued_job_without_starting_a_process(self):
        job = {"job_id": "new-queued-fixture", "status": "queued", "dataset": "appleleaf9",
               "model": "mobilenet_v3_small", "stage": "baseline", "log_path": "runs/logs/ui-fixture-missing.log"}
        with patch("standard_launcher.start_job", return_value=job) as start, \
             patch("standard_launcher.save_job_config_snapshot", return_value="runs/configs/ui-fixture.json"), \
             patch("experiment_runner.list_jobs", return_value=[job]), \
             patch("experiment_runner.get_recovery_status", return_value={}):
            app = AppTest.from_file("app/streamlit_app.py").run(timeout=60)
            app.switch_page("pages/2_Baseline_Experiments.py").run(timeout=60)
            next(item for item in app.checkbox if item.label == "Uzun süren eğitimi onaylıyorum").check().run(timeout=60)
            next(item for item in app.button if item.label == "Eğitimi Başlat").click().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["job_monitor_selected_job_id"], "new-queued-fixture")
        self.assertEqual(app.title[0].value, "Deneyler")
        self.assertFalse(start.call_args.args[1]["dry_run"])

    def test_compact_results_put_model_and_scores_before_identifiers(self):
        from experiment_ui import compact_results

        frame = pd.DataFrame([{"run_name": "long-id", "model_name": "resnet50", "test_accuracy": .9,
                               "test_macro_f1": .8, "mlflow_run_id": "id"}])
        self.assertEqual(list(compact_results(frame)), ["model_name", "test_macro_f1", "test_accuracy"])
        self.assertIn("run_name", frame)

    def test_student_options_exclude_teacher_only_models(self):
        from comparison_view import student_options

        runs = [{"role": "teacher", "student": "densenet201"},
                {"role": "baseline", "student": "mobilenet_v3_small"},
                {"role": "single_kd", "student": "mobilenet_v3_large"}]
        self.assertEqual(student_options(runs), ["mobilenet_v3_large", "mobilenet_v3_small"])

    def test_queue_handoff_keeps_the_explicit_job_selected(self):
        from experiment_ui import open_queued_job

        with patch("experiment_ui.st.session_state", {}), patch("experiment_ui.st.switch_page") as switch:
            open_queued_job({"job_id": "queued-id"})
            from experiment_ui import st
            self.assertEqual(st.session_state["job_monitor_selected_job_id"], "queued-id")
            self.assertFalse(st.session_state["monitor_follow_running"])
            switch.assert_called_once_with("pages/3_Job_Monitor.py")

    def test_baseline_preset_only_contains_selected_dataset_and_model(self):
        app = AppTest.from_file("app/streamlit_app.py").run(timeout=60)
        app.switch_page("pages/2_Baseline_Experiments.py").run(timeout=60)
        self.assertFalse(app.exception)
        dataset, model, config = app.selectbox[:3]
        self.assertEqual(dataset.label, "Veri seti")
        self.assertEqual(model.label, "Student model")
        for option in config.options:
            if option:
                payload = json.loads(Path(option).read_text(encoding="utf-8"))
                self.assertEqual((payload["dataset_name"], payload["model_name"]), (dataset.value, model.value))
        dataset.select("plantpathology2021").run(timeout=60)
        model = app.selectbox[1]
        model.select("resnet18").run(timeout=60)
        self.assertFalse(app.exception)
        self.assertTrue(all("plantpathology2021" in option and "resnet18" in option
                            for option in app.selectbox[2].options if option))

    def test_finished_job_has_no_stop_or_cancel_controls(self):
        job = {"job_id": "finished-fixture", "status": "finished", "stage": "baseline",
               "dataset": "plantvillage", "model": "resnet18", "log_path": "runs/logs/ui-fixture-missing.log"}
        with patch("experiment_runner.list_jobs", return_value=[job]), \
             patch("experiment_runner.get_recovery_status", return_value={}):
            app = AppTest.from_file("app/streamlit_app.py").run(timeout=60)
            app.switch_page("pages/3_Job_Monitor.py").run(timeout=60)
        self.assertFalse(app.exception)
        self.assertNotIn("Zorla Durdur", [button.label for button in app.button])
        self.assertNotIn("Queued Job'ı İptal Et", [button.label for button in app.button])

    def test_baseline_monitor_does_not_render_missing_distillation_metrics(self):
        job = {"job_id": "metrics-fixture", "status": "finished", "stage": "baseline",
               "dataset": "plantvillage", "model": "resnet18", "log_path": "runs/logs/ui-fixture-missing.log"}
        with patch("experiment_runner.list_jobs", return_value=[job]), \
             patch("experiment_runner.get_recovery_status", return_value={}), \
             patch("experiment_runner.read_log_tail", return_value="EPOCH_SUMMARY epoch=1/5 train_loss=0.5 val_macro_f1=0.8"):
            app = AppTest.from_file("app/streamlit_app.py").run(timeout=60)
            app.switch_page("pages/3_Job_Monitor.py").run(timeout=60)
        self.assertFalse(app.exception)
        labels = [metric.label for metric in app.metric]
        self.assertNotIn("feature_loss", labels)
        self.assertNotIn("teacher_student_agreement", labels)

    def test_new_experiment_dispatches_without_duplicate_titles(self):
        app = AppTest.from_file("app/streamlit_app.py").run(timeout=60)
        app.switch_page("pages/0_New_Experiment.py").run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual([title.value for title in app.title], ["Yeni deney"])
        self.assertIn("Eğitimi Başlat", [button.label for button in app.button])

    def test_results_only_show_the_selected_dataset(self):
        frame = pd.DataFrame([
            {"dataset_name": "appleleaf9", "model_name": "resnet18", "test_macro_f1": .8, "test_accuracy": .85},
            {"dataset_name": "plantvillage", "model_name": "resnet18", "test_macro_f1": .99, "test_accuracy": .99},
        ])
        with patch("result_loader.load_csv_if_exists", return_value=frame):
            app = AppTest.from_file("app/pages/4_Results_Explorer.py").run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(app.dataframe[0].value["test_macro_f1"].tolist(), [.8])


if __name__ == "__main__":
    unittest.main()
