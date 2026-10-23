"""Dynamic datasets and presets must reach every launcher without source edits."""
import sys
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))


class DynamicLauncherTests(unittest.TestCase):
    def test_registration_publishes_into_all_six_launcher_routes_without_restart(self):
        from contextlib import ExitStack
        from test_dataset_registration import make_dataset
        import dataset_view
        import standard_launcher
        import kd_launcher
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_dataset(root)
            shutil.copytree(ROOT / "configs/presets", root / "configs/presets")
            with ExitStack() as stack:
                stack.enter_context(patch.object(dataset_view, "get_project_root", return_value=root))
                stack.enter_context(patch.object(standard_launcher, "get_project_root", return_value=root))
                stack.enter_context(patch.object(kd_launcher, "get_project_root", return_value=root))
                app = AppTest.from_file(str(ROOT / "app/streamlit_app.py")).run(timeout=60)
                app.switch_page("pages/12_Datasets.py").run(timeout=60)
                next(item for item in app.button if item.label == "Görüntüleri kontrol et").click().run(timeout=60)
                self.assertFalse(app.exception)
                next(item for item in app.button if item.label == "Kaydet ve sabit bölmeyi yayımla").click().run(timeout=60)
                self.assertFalse(app.exception)
                self.assertTrue((root / "configs/datasets/potato.json").exists())
                for page in ("2_Baseline_Experiments.py", "11_Student_ViT_Baseline.py", "5_CNN_Teacher_Experiments.py",
                             "6_Vision_Transformer_Teacher_Experiments.py", "8_Knowledge_Distillation.py", "9_Multi_Teacher_Knowledge_Distillation.py"):
                    with self.subTest(page=page):
                        app.switch_page("pages/" + page).run(timeout=60)
                        self.assertFalse(app.exception)
                        dataset = next(item for item in app.selectbox if item.label == "Veri seti")
                        self.assertEqual(dataset.value, "potato")
                        if "Distillation" in page:
                            self.assertTrue(next(item for item in app.button if "Eğitimini Başlat" in item.label).disabled)

    def test_registered_dataset_is_selectable_in_all_launchers(self):
        from test_dataset_runtime import DatasetRuntimeTests
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            from src.datasets.dataset_runtime import create_registered_split
            revision = DatasetRuntimeTests().prepare(root)
            create_registered_split(root, revision.dataset_id, revision.revision, dict(train=.7, val=.15, test=.15), 42)
            with patch("dataset_view.get_project_root", return_value=root):
                app = AppTest.from_string(
                    "from dataset_view import select_dataset\n"
                    "value = select_dataset(key='fixture')\n"
                    "import streamlit as st\nst.write(value[0])"
                ).run(timeout=30)
            self.assertFalse(app.exception)
            self.assertIn("potato", app.selectbox[0].options)

    def test_generic_presets_compose_pinned_selection(self):
        from src.training.experiment_presets import list_presets, compose_preset
        ref = {"dataset_id": "potato", "revision": "a" * 64, "split_id": "b" * 64}
        presets = list_presets("knowledge_distillation", kd_type="feature_based")
        payload = compose_preset(presets[0], ref, {
            "dataset_name": "potato", "student_model_name": "deit_tiny_patch16_224",
            "teacher_model_name": "swin_v2_t", "kd_type": "feature_based",
        })
        self.assertEqual(payload["experiment"]["dataset_ref"], ref)
        self.assertIsNone(payload["distillation"].get("student_layer"))

    def test_dataset_page_is_registered(self):
        app = AppTest.from_file(str(ROOT / "app/streamlit_app.py")).run(timeout=60)
        app.switch_page("pages/12_Datasets.py").run(timeout=60)
        self.assertFalse(app.exception)
        self.assertIn("Veri Setleri", [title.value for title in app.title])


if __name__ == "__main__":
    unittest.main()
