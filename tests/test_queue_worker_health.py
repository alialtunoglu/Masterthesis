"""Worker startup/diagnostics checks; never promote the real experiment queue."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import ExitStack
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
import experiment_runner as runner


class QueueWorkerHealthTests(unittest.TestCase):
    def isolated_paths(self, root):
        stack = ExitStack()
        for name, value in {
            "RUNS_DIR": root / "runs", "JOBS_DIR": root / "runs/jobs", "LOGS_DIR": root / "runs/logs",
            "QUEUE_WORKER_STATE_PATH": root / "runs/queue_worker.json",
            "QUEUE_WORKER_HEARTBEAT_PATH": root / "runs/queue_worker_heartbeat.json",
            "QUEUE_WORKER_LOG_PATH": root / "runs/logs/queue_worker.log",
        }.items():
            stack.enter_context(patch.object(runner, name, value))
        stack.enter_context(patch.object(runner, "get_project_root", return_value=root))
        return stack

    def test_direct_worker_help_imports_without_inherited_pythonpath(self):
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        result = subprocess.run([sys.executable, "-B", "app/queue_worker.py", "--help"], cwd=ROOT,
                                env=environment, capture_output=True, text=True, timeout=45)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("--poll-seconds", result.stdout)

    def test_dead_worker_is_failed_instead_of_persisted_running(self):
        with tempfile.TemporaryDirectory() as directory, self.isolated_paths(Path(directory)):
            runner.RUNS_DIR.mkdir()
            runner.QUEUE_WORKER_STATE_PATH.write_text(json.dumps({"process_id": 123, "status": "running"}))
            with patch.object(runner, "is_process_running", return_value=False):
                status = runner.get_queue_worker_status()
            self.assertFalse(status["running"])
            self.assertEqual(status["status"], "failed")

    def test_alive_worker_requires_its_own_recent_heartbeat(self):
        now = datetime.now()
        with tempfile.TemporaryDirectory() as directory, self.isolated_paths(Path(directory)):
            runner.RUNS_DIR.mkdir()
            runner.QUEUE_WORKER_STATE_PATH.write_text(json.dumps({
                "process_id": 123, "status": "starting", "start_time": now.isoformat()}))
            with patch.object(runner, "is_process_running", return_value=True):
                pending = runner.get_queue_worker_status()
                self.assertFalse(pending["running"])
                self.assertTrue(pending["process_alive"])
                self.assertEqual(pending["status"], "starting")
                for pid, timestamp, expected in (
                    (999, now.isoformat(), False),
                    (123, (now - timedelta(minutes=2)).isoformat(), False),
                    (123, datetime.now().isoformat(), True),
                ):
                    runner.QUEUE_WORKER_HEARTBEAT_PATH.write_text(json.dumps({
                        "process_id": pid, "timestamp": timestamp, "status": "running"}))
                    self.assertEqual(runner.get_queue_worker_status()["running"], expected)

    def test_start_requests_launch_without_claiming_readiness_and_passes_import_paths(self):
        with tempfile.TemporaryDirectory() as directory, self.isolated_paths(Path(directory)):
            fake = Mock(pid=123)
            with patch.object(runner, "is_process_running", side_effect=lambda pid: bool(pid)), \
                    patch.object(runner.subprocess, "Popen", return_value=fake) as spawn:
                # No state yet means no existing worker, even though PID probing is mocked alive.
                result = runner.start_queue_worker()
                self.assertTrue(result["started"])
                self.assertFalse(result["running"])
                self.assertEqual(result["status"], "starting")
                command = spawn.call_args.args[0]
                self.assertIn("-u", command)
                paths = spawn.call_args.kwargs["env"]["PYTHONPATH"].split(os.pathsep)
                self.assertIn(str(Path(directory)), paths)
                self.assertIn(str(Path(directory) / "src"), paths)
                again = runner.start_queue_worker()
                self.assertFalse(again["started"])
                self.assertEqual(spawn.call_count, 1)

    def test_spawn_failure_returns_error_and_keeps_log(self):
        with tempfile.TemporaryDirectory() as directory, self.isolated_paths(Path(directory)):
            with patch.object(runner.subprocess, "Popen", side_effect=OSError("fixture spawn failure")):
                result = runner.start_queue_worker()
            self.assertFalse(result["started"])
            self.assertEqual(result["status"], "failed")
            self.assertIn("fixture spawn failure", runner.QUEUE_WORKER_LOG_PATH.read_text())

    def test_real_worker_runs_only_an_isolated_empty_queue(self):
        processes = []
        real_spawn = subprocess.Popen

        def spawn(*args, **kwargs):
            process = real_spawn(*args, **kwargs)
            processes.append(process)
            return process

        with tempfile.TemporaryDirectory() as directory, self.isolated_paths(Path(directory)), \
                patch.object(runner.subprocess, "Popen", side_effect=spawn):
            root = Path(directory)
            (root / "app").mkdir()
            # Only this temporary queue is visible to the child; never touch real jobs.
            wrapper = f"""
import sys
from pathlib import Path
sys.path.insert(0, {str(ROOT / 'app')!r})
import experiment_runner as runner
root = Path({str(root)!r})
runner.get_project_root = lambda: root
runner.RUNS_DIR = root / 'runs'
runner.JOBS_DIR = root / 'runs/jobs'
runner.LOGS_DIR = root / 'runs/logs'
runner.QUEUE_WORKER_STATE_PATH = root / 'runs/queue_worker.json'
runner.QUEUE_WORKER_HEARTBEAT_PATH = root / 'runs/queue_worker_heartbeat.json'
runner.QUEUE_WORKER_LOG_PATH = root / 'runs/logs/queue_worker.log'
runner.SETTINGS_PATH = root / 'app/dashboard_settings.json'
import queue_worker
queue_worker.main()
"""
            (root / "app/queue_worker.py").write_text(wrapper, encoding="utf-8")
            try:
                launched = runner.start_queue_worker()
                self.assertTrue(launched["started"])
                deadline = time.monotonic() + 25
                while time.monotonic() < deadline:
                    status = runner.get_queue_worker_status()
                    if status["running"] and "Checked 0 jobs" in status["heartbeat"].get("message", ""):
                        break
                    time.sleep(.1)
                self.assertTrue(status["running"], runner.read_log_tail(runner.QUEUE_WORKER_LOG_PATH))
                self.assertIn("Checked 0 jobs", status["heartbeat"]["message"])
                self.assertEqual(list(runner.JOBS_DIR.glob("*.json")), [])
            finally:
                runner.stop_queue_worker()
                for process in processes:
                    process.wait(timeout=10)
            self.assertEqual(runner.get_queue_worker_status()["status"], "stopped")

    def test_worker_loop_failure_records_failed_heartbeat(self):
        import queue_worker
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state, heartbeat = root / "worker.json", root / "heartbeat.json"
            with patch.object(queue_worker, "QUEUE_WORKER_STATE_PATH", state), \
                    patch.object(queue_worker, "QUEUE_WORKER_HEARTBEAT_PATH", heartbeat), \
                    patch.object(queue_worker, "get_project_root", return_value=root), \
                    patch.object(queue_worker, "list_jobs", side_effect=RuntimeError("fixture queue error")), \
                    patch.object(sys, "argv", ["worker"]):
                with self.assertRaisesRegex(RuntimeError, "fixture queue error"):
                    queue_worker.main()
            self.assertEqual(json.loads(state.read_text())["status"], "failed")
            signal = json.loads(heartbeat.read_text())
            self.assertEqual(signal["status"], "failed")
            self.assertEqual(signal["message"], "fixture queue error")
            self.assertEqual(signal["process_id"], os.getpid())

    def test_settings_exposes_worker_traceback_and_does_not_start_training(self):
        code = """
import threading
threading.stack_size(8 * 1024 * 1024)
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
state = {'running': False, 'process_alive': False, 'status': 'failed', 'heartbeat': {},
         'log_path': 'runs/logs/queue_worker.log'}
with patch('experiment_runner.get_queue_worker_status', return_value=state), \
     patch('experiment_runner.read_log_tail', return_value="ModuleNotFoundError: No module named 'src'"), \
     patch('experiment_runner.start_queue_worker', side_effect=AssertionError('unexpected launch')):
    app = AppTest.from_file('app/streamlit_app.py').run(timeout=60)
    app.switch_page('pages/10_Settings.py').run(timeout=60)
    assert not app.exception
    assert any('ModuleNotFoundError' in item.value for item in app.code)
    assert app.error
    assert any(w.label == 'Durumu ve logu yenile' for w in app.button)
print('WORKER_LOG_UI_OK; NO_TRAINING_STARTED')
"""
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT,
                                capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("WORKER_LOG_UI_OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
