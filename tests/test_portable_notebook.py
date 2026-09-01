import ast
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from app import portable_notebook
from app.portable_notebook import (
    _output_resolver_source,
    build_portable_notebook,
    cached_export_status,
    repository_export_status,
)


def _resolver():
    namespace: dict = {}
    exec(_output_resolver_source(), namespace)
    return namespace["resolve_bundle_output"]


class _MountRecorder:
    def __init__(self, error: Exception | None = None):
        self.calls: list[str] = []
        self.error = error

    def __call__(self, mountpoint: str) -> None:
        self.calls.append(mountpoint)
        if self.error is not None:
            raise self.error


def _code_cells(payload: bytes) -> list[str]:
    notebook = json.loads(payload)
    return [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]


def _cell_index(cells: list[str], needle: str) -> int:
    for index, source in enumerate(cells):
        if needle in source:
            return index
    raise AssertionError(f"No code cell contains {needle!r}")


def _notebook_source(payload: bytes) -> str:
    notebook = json.loads(payload)
    return "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


class PortableNotebookTests(unittest.TestCase):
    def test_notebook_is_valid_and_uses_existing_cli(self):
        payload = build_portable_notebook(
            repository_url="https://github.com/example/project.git",
            source_commit="abc123",
            stage="teacher_vision_transformer",
            dataset="appleleaf9",
            model="swin_v2_t",
            command=["python", "src/training/train_teacher.py", "--config", "config.json"],
        )
        notebook = json.loads(payload)
        source = "\n".join(
            "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
        )
        self.assertEqual(notebook["nbformat"], 4)
        self.assertIn("src/training/train_teacher.py", source)
        self.assertIn("JasonYangCode/AppleLeaf9", source)
        self.assertIn("package_external_run.py", source)

    def test_plant_pathology_notebook_supports_kaggle_colab_and_local_auth(self):
        source = _notebook_source(
            build_portable_notebook(
                repository_url="https://github.com/example/project.git",
                source_commit="abc123",
                stage="teacher",
                dataset="plantpathology2021",
                model="resnet50",
                command=["python", "src/training/train_teacher.py", "--config", "config.json"],
            )
        )

        self.assertIn("/kaggle/input/plant-pathology-2021-fgvc8", source)
        self.assertIn("google.colab", source)
        self.assertIn("files.upload()", source)
        self.assertIn("KAGGLE_API_TOKEN", source)
        self.assertIn(".kaggle/access_token", source)
        self.assertIn(".kaggle/kaggle.json", source)
        self.assertIn("train.csv", source)
        self.assertIn("train_images", source)
        self.assertIn("Add Input", source)
        self.assertNotIn("KAGGLE_KEY =", source)

    def test_plant_pathology_notebook_reuses_existing_data_before_download(self):
        source = _notebook_source(
            build_portable_notebook(
                repository_url="https://github.com/example/project.git",
                source_commit="abc123",
                stage="teacher",
                dataset="plantpathology2021",
                model="resnet50",
                command=["python", "src/training/train_teacher.py", "--config", "config.json"],
            )
        )

        ready_check = "all(path.exists() for path in required)"
        download = "'competitions', 'download'"
        self.assertIn(ready_check, source)
        self.assertIn(download, source)
        self.assertLess(source.index(ready_check), source.index(download))

    def test_colab_is_not_detected_from_kaggle_input_directory(self):
        source = _notebook_source(
            build_portable_notebook(
                repository_url="https://github.com/example/project.git",
                source_commit="abc123",
                stage="teacher",
                dataset="plantpathology2021",
                model="resnet50",
                command=["python", "src/training/train_teacher.py", "--config", "config.json"],
            )
        )

        self.assertNotIn("if Path('/kaggle/input').exists()", source)
        self.assertIn("KAGGLE_KERNEL_RUN_TYPE", source)

    def test_training_prioritizes_project_src_and_streams_subprocess_output(self):
        source = _notebook_source(
            build_portable_notebook(
                repository_url="https://github.com/example/project.git",
                source_commit="abc123",
                stage="teacher_vision_transformer",
                dataset="plantpathology2021",
                model="swin_v2_t",
                command=["python", "src/training/train_teacher.py", "--config", "config.json"],
            )
        )

        self.assertIn("environment['PYTHONPATH']", source)
        self.assertIn("str(Path('src').resolve())", source)
        self.assertIn("environment['PYTHONUNBUFFERED'] = '1'", source)
        self.assertIn("subprocess.Popen(", source)
        self.assertIn("stderr=subprocess.STDOUT", source)
        self.assertIn("for line in process.stdout:", source)
        self.assertIn("print(line, end='', flush=True)", source)

    def test_export_is_disabled_without_origin(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            import subprocess

            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            status = repository_export_status(root)
            self.assertFalse(status.ready)
            self.assertIn("origin", status.message)

    def test_export_is_disabled_when_git_status_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            import subprocess

            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(
                ["git", "remote", "add", "origin", "https://github.com/example/project.git"],
                cwd=root,
                check=True,
            )
            (root / ".git" / "index").write_bytes(b"not-a-git-index")

            status = repository_export_status(root)

            self.assertFalse(status.ready)
            self.assertTrue(status.message)


class BundleOutputResolverTests(unittest.TestCase):
    def test_kaggle_session_writes_to_persisted_working_directory(self):
        mount = _MountRecorder()

        directory, message = _resolver()(
            {"KAGGLE_KERNEL_RUN_TYPE": "Interactive"}, mount
        )

        self.assertEqual(directory, "/kaggle/working")
        self.assertEqual(mount.calls, [])
        self.assertIn("/kaggle/working", message)

    def test_plain_local_session_writes_to_current_directory(self):
        mount = _MountRecorder()

        directory, message = _resolver()({}, mount)

        self.assertEqual(directory, ".")
        self.assertEqual(mount.calls, [])
        self.assertTrue(message)

    def test_colab_session_mounts_drive_and_writes_under_my_drive(self):
        mount = _MountRecorder()
        with tempfile.TemporaryDirectory() as directory:
            drive_root = Path(directory)

            resolved, message = _resolver()(
                {"COLAB_RELEASE_TAG": "release-2026"}, mount, drive_root
            )

            expected = drive_root / "MyDrive" / "MasterThesis" / "bundles"
            self.assertEqual(mount.calls, ["/content/drive"])
            self.assertEqual(Path(resolved), expected)
            self.assertTrue(expected.is_dir())
            self.assertIn("Drive", message)

    def test_colab_falls_back_to_content_and_warns_when_mount_fails(self):
        mount = _MountRecorder(error=RuntimeError("mount rejected"))
        with tempfile.TemporaryDirectory() as directory:
            drive_root = Path(directory)

            resolved, message = _resolver()(
                {"COLAB_RELEASE_TAG": "release-2026"}, mount, drive_root
            )

            self.assertEqual(mount.calls, ["/content/drive"])
            self.assertEqual(resolved, "/content")
            self.assertIn("UYARI", message)
            self.assertIn("mount rejected", message)
            self.assertFalse((drive_root / "MyDrive").exists())


class NotebookOutputWiringTests(unittest.TestCase):
    def _cells(self) -> list[str]:
        return _code_cells(
            build_portable_notebook(
                repository_url="https://github.com/example/project.git",
                source_commit="abc123",
                stage="teacher_cnn",
                dataset="appleleaf9",
                model="resnet50",
                command=["python", "src/training/train_teacher.py", "--config", "config.json"],
            )
        )

    def test_package_cell_writes_to_resolved_output_instead_of_content(self):
        cells = self._cells()
        package = cells[_cell_index(cells, "package_external_run.py")]

        self.assertIn("BUNDLE_OUTPUT", package)
        self.assertNotIn("'/content' if", package)
        self.assertNotIn("__import__('os')", package)

    def test_package_cell_resolves_the_destination_itself(self):
        """Training runs unattended for hours and Colab reconnects meanwhile.

        Holding the destination only in kernel memory lost it to a NameError
        after a real 14-epoch run, so the cell has to stand on its own.
        """
        cells = self._cells()
        package = cells[_cell_index(cells, "package_external_run.py")]
        tree = ast.parse(package)
        assigned = {
            target.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        } | {
            element.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Tuple)
            for element in target.elts
            if isinstance(element, ast.Name)
        }
        self.assertIn("BUNDLE_OUTPUT", assigned)

    def test_package_cell_recovers_the_working_directory_after_a_restart(self):
        """A kernel restart loses the chdir but leaves the clone on disk."""
        cells = self._cells()
        package = cells[_cell_index(cells, "package_external_run.py")]
        calls: list[list[str]] = []
        with tempfile.TemporaryDirectory() as directory:
            clone = Path(directory) / "MasterThesis" / "scripts"
            clone.mkdir(parents=True)
            (clone / "package_external_run.py").write_text("", encoding="utf-8")
            previous = os.getcwd()
            os.chdir(directory)
            try:
                with mock.patch(
                    "subprocess.run", lambda *a, **k: calls.append(list(a[0]))
                ):
                    exec(compile(package, "<package>", "exec"), {"__name__": "__main__"})
                self.assertEqual(Path(os.getcwd()).name, "MasterThesis")
            finally:
                # Windows cannot remove the temp tree while it is the cwd.
                os.chdir(previous)
        self.assertEqual(len(calls), 1)

    def test_package_cell_run_standalone_targets_drive(self):
        cells = self._cells()
        package = cells[_cell_index(cells, "package_external_run.py")]
        calls: list[list[str]] = []
        colab = types.ModuleType("google.colab")
        colab.drive = types.SimpleNamespace(mount=lambda mountpoint: None)
        google = types.ModuleType("google")
        google.colab = colab

        with mock.patch.dict(sys.modules, {"google": google, "google.colab": colab}), \
                mock.patch.dict(os.environ, {"COLAB_RELEASE_TAG": "release"}), \
                mock.patch.object(Path, "mkdir", lambda *a, **k: None), \
                mock.patch("subprocess.run", lambda *a, **k: calls.append(list(a[0]))):
            exec(compile(package, "<package>", "exec"), {"__name__": "__main__"})

        self.assertEqual(len(calls), 1)
        command = calls[0]
        expected = str(Path("/content/drive") / "MyDrive" / "MasterThesis" / "bundles")
        self.assertEqual(command[command.index("--output") + 1], expected)

    def test_drive_is_mounted_before_dataset_and_training_cells(self):
        cells = self._cells()

        resolve = _cell_index(cells, "resolve_bundle_output(os.environ")
        dataset = _cell_index(cells, "JasonYangCode/AppleLeaf9")
        training = _cell_index(cells, "subprocess.Popen(")

        self.assertLess(resolve, dataset)
        self.assertLess(resolve, training)


class GeneratedCellSyntaxTests(unittest.TestCase):
    def test_every_generated_code_cell_compiles_for_every_dataset(self):
        for dataset in ("appleleaf9", "plantvillage", "plantpathology2021"):
            cells = _code_cells(
                build_portable_notebook(
                    repository_url="https://github.com/example/project.git",
                    source_commit="abc123",
                    stage="teacher_cnn",
                    dataset=dataset,
                    model="resnet50",
                    command=[
                        "python",
                        "src/training/train_teacher.py",
                        "--config",
                        "config.json",
                    ],
                )
            )
            self.assertTrue(cells)
            for index, source in enumerate(cells):
                with self.subTest(dataset=dataset, cell=index):
                    compile(source, f"<{dataset}:cell{index}>", "exec")


class CachedExportStatusTests(unittest.TestCase):
    """Streamlit reruns the whole script on every widget change.

    The status probe shells out to git four times and hits the network twice,
    so an uncached call makes each keystroke cost most of a second.
    """

    def setUp(self):
        cached_export_status.clear()
        self.addCleanup(cached_export_status.clear)

    def test_repeated_renders_probe_the_repository_only_once(self):
        calls = []
        original = portable_notebook.repository_export_status

        def counting(root):
            calls.append(root)
            return original(root)

        portable_notebook.repository_export_status = counting
        self.addCleanup(
            setattr, portable_notebook, "repository_export_status", original
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for _ in range(3):
                cached_export_status(root)
        self.assertEqual(len(calls), 1)

    def test_clearing_the_cache_probes_again(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = cached_export_status(root)
            cached_export_status.clear()
            second = cached_export_status(root)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
