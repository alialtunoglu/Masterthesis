import unittest

from streamlit.testing.v1 import AppTest


class StreamlitVitPageTests(unittest.TestCase):
    def test_vit_page_selection_order_and_config(self):
        app = AppTest.from_file(
            "app/pages/9_Vision_Transformer_Teacher_Experiments.py"
        ).run(timeout=30)
        self.assertFalse(app.exception)
        self.assertEqual(app.selectbox[0].label, "Dataset")
        self.assertEqual(app.selectbox[1].label, "Vision Transformer teacher model")
        self.assertEqual(app.selectbox[2].label, "Config dosyası")
        self.assertIn("swin_v2_t.json", app.selectbox[2].value)

    def test_results_explorer_exposes_external_import(self):
        app = AppTest.from_file("app/pages/4_Results_Explorer.py").run(timeout=60)
        self.assertFalse(app.exception)
        self.assertIn("External Run Import", [tab.label for tab in app.tabs])


if __name__ == "__main__":
    unittest.main()
