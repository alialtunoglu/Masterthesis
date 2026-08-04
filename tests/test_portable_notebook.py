import json
import tempfile
import unittest
from pathlib import Path

from app.portable_notebook import build_portable_notebook, repository_export_status


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
