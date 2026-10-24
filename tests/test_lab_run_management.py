"""Notebook adoption and manual transfers are tested with isolated fixtures."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import lab_runs
import run_transfer
import experiment_runner as runner


def attempt_fixture(root, *, status="running", dry=False):
    attempt = root / "lab_state/01_fixture__abcdef123456"
    attempt.mkdir(parents=True)
    (attempt / "state.json").write_text(json.dumps({"experiment": "01_fixture", "attempt": "abcdef123456",
        "runner_pid": 12345, "gpu": 1, "dry_run": dry, "status": status, "started_at": 1700000000}))
    (attempt / "config.json").write_text(json.dumps({"experiment": {"stage": "knowledge_distillation",
        "dataset_name": "fixture", "student_model_name": "shufflenet_v2_x0_5", "teacher_model_name": "resnet50",
        "kd_type": "logit_based"}, "training": {"epochs": 30}}))
    (attempt / "training.log").write_text('JOB_RUN_CONTEXT {"mlflow_run_id":"source","run_name":"one"}\nEPOCH_SUMMARY epoch=2/30 val_macro_f1=0.75\n')
    return attempt


class LabRunManagementTests(unittest.TestCase):
    def test_runner_identity_checks_actual_command_arguments_not_just_pid(self):
        from unittest.mock import Mock
        import psutil
        state = {"runner_pid": 123, "experiment": "fixture", "attempt": "abcdef123456"}
        process = Mock()
        process.is_running.return_value = True
        process.status.return_value = "running"
        process.cmdline.return_value = ["python", "/lab/run_lab_experiment.py", "--experiment", "fixture", "--attempt", "abcdef123456"]
        with patch.object(lab_runs.psutil, "Process", return_value=process):
            self.assertTrue(lab_runs.runner_identity(state))
            process.cmdline.return_value[-1] = "different"
            self.assertFalse(lab_runs.runner_identity(state))
        with patch.object(lab_runs.psutil, "Process", side_effect=psutil.AccessDenied(123)):
            self.assertIsNone(lab_runs.runner_identity(state))

    def test_running_notebook_is_read_only_virtual_job(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempt = attempt_fixture(root)
            before = (attempt / "state.json").read_bytes()
            with patch.object(lab_runs, "runner_identity", return_value=True):
                jobs = lab_runs.list_lab_jobs(root)
            self.assertEqual(len(jobs), 1)
            job = jobs[0]
            self.assertEqual((job["gpu_id"], job["stage"], job["teacher_model"]), ("1", "knowledge_distillation", "resnet50"))
            self.assertEqual(job["mlflow_run_id"], "source")
            self.assertFalse(job["dry_run"])
            self.assertEqual(job["execution_origin"], "lab_notebook")
            self.assertEqual(before, (attempt / "state.json").read_bytes())

    def test_reused_pid_and_unreadable_identity_are_not_reported_running(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempt_fixture(root)
            for identity, expected in ((False, "interrupted"), (None, "unknown")):
                with patch.object(lab_runs, "runner_identity", return_value=identity):
                    self.assertEqual(lab_runs.list_lab_jobs(root)[0]["status"], expected)

    def test_malformed_state_is_visible_without_crashing_other_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempt = attempt_fixture(root)
            (attempt / "state.json").write_text("{")
            job = lab_runs.list_lab_jobs(root)[0]
            self.assertEqual(job["status"], "unknown")
            self.assertIn("state.json", job["monitor_warning"])

    def test_malformed_nested_config_does_not_break_the_common_job_list(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempt = attempt_fixture(root)
            (attempt / "config.json").write_text(json.dumps({"experiment": [], "student": [], "teachers": None, "distillation": []}))
            with patch.object(lab_runs, "runner_identity", return_value=True):
                job = lab_runs.list_lab_jobs(root)[0]
            self.assertEqual(job["status"], "unknown")
            self.assertIn("config", job["monitor_warning"])

    def test_finished_dry_run_remains_a_dry_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempt_fixture(root, status="dry_run_finished", dry=True)
            job = lab_runs.list_lab_jobs(root)[0]
            self.assertEqual(job["status"], "finished")
            self.assertTrue(job["dry_run"])
            self.assertFalse(run_transfer.can_export(job))

    def test_safe_stop_uses_notebook_stop_file_without_creating_native_job(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempt = attempt_fixture(root)
            with patch.object(lab_runs, "runner_identity", return_value=True):
                job = lab_runs.list_lab_jobs(root)[0]
                with patch.object(runner, "get_project_root", return_value=root), \
                        patch.object(runner, "get_recovery_status", return_value={"metadata": {"recovery": {"enabled": True}}}):
                    runner.request_epoch_stop(job)
            self.assertTrue((attempt / "stop.request").is_file())
            self.assertFalse((root / "runs/jobs").exists())

    def test_notebook_stop_rejects_a_symbolic_link(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempt = attempt_fixture(root)
            with patch.object(lab_runs, "runner_identity", return_value=True):
                job = lab_runs.list_lab_jobs(root)[0]
                with patch.object(Path, "is_symlink", autospec=True, side_effect=lambda path: path.name == "stop.request"):
                    with self.assertRaises(ValueError):
                        lab_runs.request_notebook_stop(root, job)
            self.assertFalse((attempt / "stop.request").exists())

    def test_export_matches_exact_run_and_can_be_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = root / "results/baseline/baseline_results.csv"
            summary.parent.mkdir(parents=True)
            summary.write_text("stage,dataset_name,model_name,run_name,mlflow_run_id,checkpoint_path\nbaseline,fixture,resnet18,one,source,checkpoints/one.pt\n")
            (root / "checkpoints").mkdir()
            (root / "checkpoints/one.pt").write_bytes(b"test-only-checkpoint")
            job = {"job_id": "fixture", "status": "finished", "stage": "baseline", "mlflow_run_id": "source", "source_commit": "a" * 40}
            first = run_transfer.prepare_run_transfer(root, job)
            second = run_transfer.prepare_run_transfer(root, job)
            self.assertEqual(first, second)
            self.assertTrue(Path(first["archive"]).is_file())
            self.assertEqual(Path(first["checkpoint"]).read_bytes(), b"test-only-checkpoint")
            Path(first["checkpoint"]).write_bytes(b"corrupted")
            with self.assertRaises(ValueError):
                run_transfer.prepare_run_transfer(root, job)

    def test_failed_or_dry_run_cannot_export(self):
        for job in ({"status": "running"}, {"status": "failed"}, {"status": "finished", "dry_run": True}):
            self.assertFalse(run_transfer.can_export(job))

    def test_transfer_rejects_tampered_manifest(self):
        import zipfile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "transfer"
            folder.mkdir()
            (folder / "checkpoint.pt").write_bytes(b"fixture")
            with zipfile.ZipFile(folder / "result.zip", "w") as archive:
                archive.writestr("manifest.json", json.dumps({"source_mlflow_run_id": "wanted", "files": [],
                    "checkpoint": {"filename": "checkpoint.pt", "size": 7, "sha256": "a"*64}, "fingerprint": "bad"}))
            with self.assertRaisesRegex(ValueError, "manifest"):
                run_transfer.verify_transfer_folder(root, folder, "wanted")

    def test_process_source_commit_is_persisted_from_startup_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "training.log"
            log.write_text('JOB_PROCESS_CONTEXT {"source_commit":"' + "a" * 40 + '"}\nJOB_RUN_CONTEXT {"mlflow_run_id":"source"}\n')
            job = {"gpu_id": "0", "log_path": str(log)}
            self.assertTrue(runner._sync_job_context(job))
            self.assertEqual(job["source_commit"], "a" * 40)

    def test_portable_lab_teacher_metadata_is_preserved_for_strict_validation(self):
        import kd_launcher
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "lab_notebooks").mkdir()
            (root / "checkpoints").mkdir()
            (root / "checkpoints/teacher.pt").write_bytes(b"fixture")
            entry = {"model_name": "resnet50", "run_id": "original", "checkpoint_path": "checkpoints/teacher.pt",
                     "checkpoint_sha256": "a" * 64, "dataset_name": "fixture", "dataset_ref": {"revision": "same"},
                     "image_size": 224, "status": "completed", "class_names": ["a", "b"]}
            (root / "lab_notebooks/lab_manifest.json").write_text(json.dumps({"teachers": [entry]}))
            with patch.object(kd_launcher, "find_compatible_teacher_runs", return_value=[]):
                candidates = kd_launcher.teacher_candidates.__wrapped__(root, "fixture", {"revision": "same"}, "shufflenet_v2_x0_5", 224)
            self.assertEqual(candidates[0]["mlflow_run_id"], "original")
            self.assertEqual(kd_launcher._teacher_input(candidates[0])["portable_input"], entry)


if __name__ == "__main__":
    unittest.main()
