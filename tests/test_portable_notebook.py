import json
import tempfile
import unittest
from pathlib import Path

from app.portable_notebook import build_portable_notebook, repository_export_status


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

    def test_export_is_disabled_without_origin(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            import subprocess

            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            status = repository_export_status(root)
            self.assertFalse(status.ready)
            self.assertIn("origin", status.message)


if __name__ == "__main__":
    unittest.main()
