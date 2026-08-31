import tempfile
import unittest
from pathlib import Path

from scripts.reset_experiments import (
    CRITICAL_DIRS,
    existing_output_paths,
    recreate_empty_dirs,
)


def _project_tree(root: Path, *outputs: str) -> None:
    for name in CRITICAL_DIRS:
        (root / name).mkdir(parents=True, exist_ok=True)
    for relative in outputs:
        (root / relative).mkdir(parents=True, exist_ok=True)


class ResetCoverageTests(unittest.TestCase):
    def test_reset_collects_current_knowledge_distillation_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _project_tree(
                root,
                "results/knowledge_distillation/runs",
                "results/multi_teacher_knowledge_distillation/runs",
            )

            relatives = {
                path.relative_to(root).as_posix() for path in existing_output_paths(root)
            }

            self.assertIn("results/knowledge_distillation", relatives)
            self.assertIn("results/multi_teacher_knowledge_distillation", relatives)

    def test_recreated_layout_matches_the_directories_training_writes_to(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _project_tree(root)

            recreate_empty_dirs(root)

            self.assertTrue((root / "results/knowledge_distillation").is_dir())
            self.assertTrue((root / "results/multi_teacher_knowledge_distillation").is_dir())

    def test_stale_result_directories_are_not_recreated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _project_tree(root)

            recreate_empty_dirs(root)

            for stale in ("results/kd_single", "results/kd_multi", "results/vit_teachers"):
                self.assertFalse((root / stale).exists(), stale)


if __name__ == "__main__":
    unittest.main()
