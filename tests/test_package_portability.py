"""Installed entry points must not depend on the caller's working directory or PYTHONPATH."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


class PackagePortabilityTests(unittest.TestCase):
    def test_cpu_cuda_selection_preserves_errors(self):
        from src.utils.device import select_device
        with patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(select_device("auto").type, "cpu")
            with self.assertRaises(RuntimeError):
                select_device("cuda")
        with patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(select_device("auto").type, "cuda")
            self.assertEqual(select_device("cpu").type, "cpu")
        with self.assertRaises(ValueError):
            select_device("mps")

    def test_workspace_override_and_mlflow_relative_uri(self):
        from src.utils.project import get_project_root
        from src.tracking.mlflow_tracker import resolve_tracking_uri
        with tempfile.TemporaryDirectory(prefix="project with spaces ") as directory:
            root = Path(directory).resolve()
            with patch.dict(os.environ, {"MASTERTHESIS_PROJECT_ROOT": str(root)}):
                self.assertEqual(get_project_root(), root)
                self.assertEqual(resolve_tracking_uri("sqlite:///mlflow.db"), "sqlite:///" + (root / "mlflow.db").as_posix())
            self.assertEqual(resolve_tracking_uri("sqlite:///:memory:"), "sqlite:///:memory:")
            self.assertEqual(resolve_tracking_uri("https://tracking.example"), "https://tracking.example")
            for uri in ("sqlite:////home/research/mlflow.db", "sqlite:///C:/research/mlflow.db"):
                self.assertEqual(resolve_tracking_uri(uri), uri)

    def test_installed_packages_import_in_isolated_process(self):
        with tempfile.TemporaryDirectory(prefix="workspace with spaces ") as directory:
            result = subprocess.run(
                [sys.executable, "-B", "-I", "-c", "import src.training.kd_config; import app.experiment_runner"],
                cwd=directory, capture_output=True, text=True, timeout=45,
            )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_module_cli_help_without_pythonpath_from_other_directory(self):
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        with tempfile.TemporaryDirectory(prefix="workspace with spaces ") as directory:
            for module in ("src.training.train_baseline", "src.training.train_teacher",
                           "src.training.train_kd", "src.training.train_multi_kd", "app.queue_worker"):
                result = subprocess.run([sys.executable, "-B", "-m", module, "--help"],
                                        cwd=directory, env=env, capture_output=True, text=True, timeout=45)
                self.assertEqual(result.returncode, 0, module + "\n" + result.stderr)


if __name__ == "__main__":
    unittest.main()
