import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import experiment_runner as runner
from experiment_removal import _active_resume_claims


class RecoveryJobTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        patches = [
            mock.patch.object(runner, "RUNS_DIR", self.root / "runs"),
            mock.patch.object(runner, "JOBS_DIR", self.root / "runs/jobs"),
            mock.patch.object(runner, "LOGS_DIR", self.root / "runs/logs"),
            mock.patch.object(runner, "get_project_root", return_value=self.root),
        ]
        for patcher in patches:
            patcher.start(); self.addCleanup(patcher.stop)

    def test_epoch_stop_writes_job_specific_flag(self):
        job = {"job_id": "abc", "status": "running"}
        with mock.patch("experiment_runner.get_recovery_status", return_value={
            "metadata": {"recovery": {"enabled": True}}
        }):
            updated = runner.request_epoch_stop(job)
        self.assertEqual(updated["status"], "stop_requested")
        self.assertTrue((runner.JOBS_DIR / "abc.stop").is_file())

    def test_epoch_stop_rejects_job_without_recovery(self):
        job = {"job_id": "abc", "status": "running"}
        with mock.patch("experiment_runner.get_recovery_status", return_value={
            "metadata": {"recovery": {"enabled": False}}
        }):
            with self.assertRaisesRegex(ValueError, "recovery is enabled"):
                runner.request_epoch_stop(job)
        self.assertEqual(job["status"], "running")
        self.assertFalse((runner.JOBS_DIR / "abc.stop").exists())

    def test_graceful_stop_log_is_not_failure(self):
        self.assertEqual(runner.completed_status("TRAINING_STOPPED after epoch"), "stopped")

    @mock.patch("mlflow.tracking.MlflowClient")
    def test_force_stop_reconciles_mlflow_as_killed(self, client):
        log = runner.LOGS_DIR / "abc.log"
        log.parent.mkdir(parents=True)
        log.write_text(
            'JOB_RUN_CONTEXT {"mlflow_run_id":"run","tracking_uri":"sqlite:///x.db"}\n'
            + "\n".join(f"epoch output {line}" for line in range(6000)),
            encoding="utf-8",
        )
        job = {"log_path": str(log)}
        runner._reconcile_mlflow_status(job, "KILLED")
        client.return_value.set_terminated.assert_called_once_with("run", status="KILLED")
        self.assertEqual(job["mlflow_run_id"], "run")

    @mock.patch("experiment_runner.subprocess.Popen")
    def test_worker_passes_stop_file_and_src_first(self, popen):
        popen.return_value.pid = 10
        job = {"job_id": "abc", "command": ["python", "train.py"], "status": "queued"}
        runner._start_queued_job(job)
        environment = popen.call_args.kwargs["env"]
        self.assertEqual(environment["MASTERTHESIS_STOP_FILE"], str((runner.JOBS_DIR / "abc.stop").resolve()))
        self.assertEqual(environment["PYTHONPATH"].split(runner.os.pathsep)[0], str((runner.get_project_root() / "src").resolve()))

    @mock.patch("experiment_runner.subprocess.Popen")
    @mock.patch("experiment_runner._validate_resume_generation", side_effect=ValueError("Resume mismatch: teacher_hashes"))
    def test_resume_mismatch_is_rejected_before_process_start(self, _validate, popen):
        job = {"job_id": "abc", "command": ["python", "train.py"], "status": "queued",
               "resume_from": str(self.root / "epoch-000001"), "resume_experiment_id": "exp"}
        with self.assertRaisesRegex(ValueError, "teacher_hashes"):
            runner._start_queued_job(job)
        popen.assert_not_called()

    @mock.patch("experiment_runner.subprocess.Popen")
    @mock.patch("experiment_runner._validate_resume_generation")
    def test_changed_resume_job_snapshot_is_rejected_before_process_start(self, validate, popen):
        validate.return_value = {"metadata": {"experiment_id": "exp", "config": {"epochs": 2}}}
        config = self.root / "runs/configs/abc.json"
        config.parent.mkdir(parents=True)
        config.write_text('{"epochs":3}', encoding="utf-8")
        job = {"job_id": "abc", "command": ["python", "train.py"], "status": "queued",
               "config": str(config), "resume_from": str(self.root / "epoch-000001"),
               "resume_experiment_id": "exp"}
        with self.assertRaisesRegex(ValueError, "config changed"):
            runner._start_queued_job(job)
        popen.assert_not_called()

    @mock.patch("experiment_runner._start_queued_job", side_effect=ValueError("Resume mismatch"))
    def test_worker_marks_invalid_queued_resume_failed_without_crashing(self, _start):
        job = {"job_id": "abc", "command": ["python", "train.py"], "status": "queued", "queued_time": "1"}
        with mock.patch("experiment_runner.load_dashboard_settings", return_value={"max_parallel_jobs": 1}):
            runner.promote_queued_jobs([job])
        self.assertEqual(job["status"], "failed")
        self.assertIn("Resume mismatch", job["start_error"])

    def test_recovery_events_are_parsed_without_crashing_on_bad_lines(self):
        text = 'RECOVERY_CONTEXT {"status_path":"x","experiment_id":"e","attempt_id":"a"}\nRECOVERY_STATUS nope'
        self.assertEqual(runner.parse_recovery_events(text)["experiment_id"], "e")

    def test_active_resume_claims_its_source_generation(self):
        generation = self.root / "results/run/recovery/attempt/epoch-000001"
        generation.mkdir(parents=True)
        runner.JOBS_DIR.mkdir(parents=True, exist_ok=True)
        (runner.JOBS_DIR / "child.json").write_text(json.dumps({
            "job_id": "child", "status": "queued", "resume_from": str(generation)
        }), encoding="utf-8")
        self.assertEqual(_active_resume_claims(self.root, [self.root / "results/run"]), ["child"])

    def test_enqueue_locks_config_file_to_job_snapshot(self):
        source = self.root / "config.json"
        source.write_text('{"dataset_name":"appleleaf9"}', encoding="utf-8")
        job = runner.enqueue_job(["python", "train.py", "--config", str(source)], {"stage": "baseline"})
        snapshot = self.root / job["config"]
        self.assertTrue(snapshot.is_file())
        self.assertEqual(job["command"][3], job["config"])
        source.write_text('{"dataset_name":"changed"}', encoding="utf-8")
        self.assertEqual(json.loads(snapshot.read_text(encoding="utf-8"))["dataset_name"], "appleleaf9")

    @mock.patch("experiment_runner._validate_resume_generation")
    def test_resume_queues_locked_original_config_and_stage_cli(self, validate):
        validate.return_value = {"metadata": {"experiment_id": "exp", "attempt_id": "old",
            "stage": "baseline", "config": {"dataset": "d", "model": "m"}}}
        job = runner.enqueue_resume_job(self.root / "epoch-000001")
        self.assertIn("src/training/train_baseline.py", job["command"])
        self.assertEqual(job["command"][-2:], ["--resume-from", str(self.root / "epoch-000001")])
        config = json.loads((self.root / job["config"]).read_text(encoding="utf-8"))
        self.assertEqual(config, {"dataset": "d", "model": "m"})
