"""GPU admission tests use fake inventory and OS file locks, never CUDA."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import app.experiment_runner as runner
import app.gpu_runtime as gpu
class GpuSchedulingTests(unittest.TestCase):
    def test_inventory_is_read_without_initializing_torch(self):
        with patch.object(gpu.subprocess, "check_output", return_value="0, NVIDIA A, 100, 16000, 12\n1, NVIDIA B, 200, 16000, 5\n"):
            cards = gpu.discover_gpus()
        self.assertEqual([card["id"] for card in cards], ["0", "1"])
        self.assertEqual(cards[1]["memory_used_mb"], 200)
        with patch.object(gpu.subprocess, "check_output", side_effect=FileNotFoundError):
            self.assertEqual(gpu.discover_gpus(), [])
        with patch.object(gpu.subprocess, "check_output", side_effect=subprocess.CalledProcessError(1, "nvidia-smi")):
            with self.assertRaises(RuntimeError):
                gpu.discover_gpus()

    def test_gpu_lock_contends_and_releases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertTrue(gpu.gpu_is_available(root, "1"))
            with gpu.gpu_lock(root, "1"):
                self.assertFalse(gpu.gpu_is_available(root, "1"))
                with self.assertRaises(gpu.GpuBusyError):
                    with gpu.gpu_lock(root, "1"):
                        self.fail("Second owner entered GPU lock")
            self.assertTrue(gpu.gpu_is_available(root, "1"))

    def test_auto_uses_other_gpu_and_explicit_busy_does_not_spill(self):
        cards = [{"id": "0"}, {"id": "1"}]
        active = [{"status": "running", "gpu_id": "0"}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(gpu.choose_gpu("auto", cards, active, root), "1")
            with self.assertRaises(gpu.GpuBusyError):
                gpu.choose_gpu("0", cards, active, root)
            with self.assertRaisesRegex(ValueError, "GPU"):
                gpu.choose_gpu("2", cards, active, root)
            self.assertIsNone(gpu.choose_gpu("auto", [], [], root))

    def test_notebook_lock_blocks_admission_without_a_job_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with gpu.gpu_lock(root, "0"):
                self.assertEqual(gpu.choose_gpu("auto", [{"id": "0"}, {"id": "1"}], [], root), "1")

    def test_legacy_unknown_gpu_owner_is_conservative(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(gpu.GpuBusyError):
                gpu.choose_gpu("auto", [{"id": "0"}], [{"status": "running"}], Path(directory))

    def test_busy_gpu_stays_queued_and_other_gpu_can_start(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = [{"job_id": "busy", "status": "queued", "gpu_requested": "0", "queued_time": "1"},
                    {"job_id": "free", "status": "queued", "gpu_requested": "1", "queued_time": "2"}]
            started = []
            def start(job):
                started.append((job["job_id"], job["gpu_id"]))
                job["status"] = "running"
            with patch.object(runner, "get_project_root", return_value=root), \
                    patch.object(runner, "JOBS_DIR", root / "runs/jobs"), \
                    patch.object(runner, "LOGS_DIR", root / "runs/logs"), \
                    patch.object(runner, "load_dashboard_settings", return_value={"max_parallel_jobs": 2}), \
                    patch.object(runner, "discover_gpus", return_value=[{"id": "0"}, {"id": "1"}]), \
                    patch.object(runner, "_start_queued_job", side_effect=start):
                with gpu.gpu_lock(root, "0"):
                    runner.promote_queued_jobs(jobs)
            self.assertEqual(started, [("free", "1")])
            self.assertEqual(jobs[0]["status"], "queued")

    def test_nonzero_wrapper_exit_is_failure_even_without_python_traceback(self):
        self.assertEqual(runner.completed_status('JOB_EXIT {"returncode": 137}'), "failed")

    def test_explicit_gpu_is_passed_to_child_environment(self):
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = {"job_id": "fixture", "command": [sys.executable, "trainer.py"], "gpu_id": "1"}
            with patch.object(runner, "get_project_root", return_value=root), \
                    patch.object(runner, "JOBS_DIR", root / "runs/jobs"), \
                    patch.object(runner, "LOGS_DIR", root / "runs/logs"), \
                    patch.object(runner.subprocess, "Popen", return_value=Mock(pid=123)) as spawn:
                runner._start_queued_job(job)
            self.assertEqual(spawn.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "1")
            self.assertEqual(spawn.call_args.kwargs["env"]["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
            self.assertIn("--gpu", spawn.call_args.args[0])

    def test_process_wrapper_is_importable_and_reports_exact_exit_code(self):
        root = Path(__file__).resolve().parents[1]
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        result = subprocess.run([sys.executable, "app/job_process.py", "--gpu", "cpu", "--",
                                 sys.executable, "-c", "raise SystemExit(7)"], cwd=root,
                                env=environment, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertIn('JOB_EXIT {"returncode": 7}', result.stdout)
        self.assertIn("JOB_PROCESS_CONTEXT", result.stdout)


if __name__ == "__main__":
    unittest.main()
