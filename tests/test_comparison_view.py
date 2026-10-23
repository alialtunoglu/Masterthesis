import unittest

from app.comparison_view import report_downloads, table_frames
from src.evaluation.experiment_comparison import build_comparison
from streamlit.testing.v1 import AppTest


class ComparisonViewTests(unittest.TestCase):
    def test_results_explorer_exposes_comparison_tab_without_errors(self):
        app = AppTest.from_file("app/streamlit_app.py").run(timeout=60)
        app.switch_page("pages/4_Results_Explorer.py").run(timeout=60)
        self.assertFalse(app.exception)
        self.assertIn("Karşılaştırma ve Rapor", [tab.label for tab in app.tabs])

    def test_view_uses_the_same_report_for_tables_and_downloads(self):
        runs = [{"run_id": "b", "role": "baseline", "dataset": "d", "model": "m",
                 "student": "m", "seed": 1, "test_macro_f1": .8, "val_macro_f1": .8,
                 "eligible": True, "compatibility": "legacy_unknown", "class_metrics": []}]
        report = build_comparison(runs, ["b"], "b")
        main, classes = table_frames(report)
        downloads = report_downloads(report)
        self.assertEqual(main.iloc[0]["run_id"], "b")
        self.assertTrue(classes.empty)
        self.assertEqual(set(downloads), {"Markdown", "CSV", "LaTeX", "JSON"})
        self.assertIn(b"b", downloads["CSV"])
