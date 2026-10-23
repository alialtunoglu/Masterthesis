import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.evaluation.experiment_comparison import (
    build_comparison,
    export_comparison,
    load_comparison_runs,
)


def _run(run_id, role, f1, val=.8, **extra):
    return {
        "run_id": run_id, "role": role, "dataset": "leaves", "model": "tiny",
        "student": "tiny", "seed": 42, "test_macro_f1": f1,
        "val_macro_f1": val, "test_accuracy": f1, "eligible": True,
        "compatibility": "verified", "class_mapping": ["a"],
        "dataset_fingerprint": "dataset", "split_hash": "split",
        "eval_fingerprint": "eval", "training_protocol": {"epochs": 2}, **extra,
    }


class ExperimentComparisonTests(unittest.TestCase):
    def test_delta_is_percentage_points_and_auto_baseline_uses_validation(self):
        runs = [_run("b-low", "baseline", .9, .7), _run("b", "baseline", .8, .9),
                _run("kd", "single_kd", .83, teacher_run_ids=["teacher-1"])]
        report = build_comparison(runs, ["b", "kd"], None)
        self.assertEqual(report["provenance"]["baseline_id"], "b")
        self.assertEqual(report["rows"][1]["delta_macro_f1_pp"], 3.0)
        self.assertEqual(report["rows"][1]["teacher_run_ids"], ["teacher-1"])


    def test_loader_filters_smoke_and_reads_artifact_without_guessing_ensemble(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        path = root / "results/multi_teacher_knowledge_distillation"
        path.mkdir(parents=True)
        per_class = path / "classes.csv"
        pd.DataFrame([{"class_id": 0, "class_name": "a", "precision": .7,
                       "recall": .8, "f1": .75, "support": 4}]).to_csv(per_class, index=False)
        pd.DataFrame([
        {"mlflow_run_id": "ok", "run_name": "ok", "stage": "multi_teacher_knowledge_distillation",
         "dataset_name": "leaves", "student_model_name": "tiny", "seed": 42,
         "best_val_macro_f1": .8, "test_macro_f1": .81,
         "teacher_run_ids": '["t1", "t2"]', "teacher_models": '["A", "B"]',
         "per_class_metrics_path": str(per_class.relative_to(root)), "dry_run": False},
        {"mlflow_run_id": "smoke", "run_name": "smoke", "stage": "multi_teacher_knowledge_distillation",
         "dataset_name": "leaves", "student_model_name": "tiny", "seed": 42,
         "test_macro_f1": .2, "max_test_batches": 1},
        ]).to_csv(path / "multi_kd_results.csv", index=False)
        runs = load_comparison_runs(root)
        self.assertEqual([r["run_id"] for r in runs], ["ok"])
        self.assertIsNone(runs[0]["ensemble_test_macro_f1"])
        self.assertEqual(runs[0]["class_metrics"][0]["class_name"], "a")
        self.assertEqual(runs[0]["compatibility"], "legacy_unknown")

    def test_loader_finds_run_metadata_beside_history_and_filters_nested_dry_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "results/baseline/runs/demo"
            run_dir.mkdir(parents=True)
            history = run_dir / "history.csv"
            history.write_text("epoch,loss\n1,0.2\n", encoding="utf-8")
            metadata = {
                "attempt_id": "a", "experiment_id": "e", "status": "FINISHED",
                "dataset_name": "leaves", "dataset_fingerprint": "dataset",
                "split_fingerprint": "split", "classes": ["a"],
                "eval_transform": {"size": 8}, "training_protocol": {"epochs": 2},
                "config": {"training": {"dry_run": False}},
            }
            (run_dir / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            output = root / "results/baseline/baseline_results.csv"
            output.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{"mlflow_run_id": "a", "dataset_name": "leaves", "model_name": "tiny",
                           "seed": 42, "test_macro_f1": .8,
                           "history_path": str(history.relative_to(root))}]).to_csv(output, index=False)
            loaded = load_comparison_runs(root)
            self.assertEqual(loaded[0]["experiment_id"], "e")
            self.assertEqual(loaded[0]["compatibility"], "verified")

            metadata["config"]["training"] = {"dry_run": True}
            (run_dir / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            self.assertEqual(load_comparison_runs(root), [])

    def test_multi_teacher_ensemble_uses_only_recorded_metrics(self):
        multi = _run("multi", "multi_kd", .84, teacher_models=["A", "B"],
                     teacher_run_ids=["ta", "tb"], ensemble_test_macro_f1=.91,
                     ensemble_test_accuracy=.92)
        report = build_comparison([_run("base", "baseline", .8), multi], ["multi"], "base")
        ensemble = next(row for row in report["rows"] if row["role"] == "teacher_ensemble")
        self.assertEqual(ensemble["test_macro_f1"], .91)
        self.assertEqual(ensemble["params"], None)


    def test_exports_share_report_and_escape_latex(self):
        report = build_comparison([_run("b_1", "baseline", .8, model="x&y")], ["b_1"], "b_1")
        self.assertIn("b_1", export_comparison(report, "markdown").decode())
        self.assertIn("b_1", export_comparison(report, "csv").decode())
        self.assertIn("b\\_1", export_comparison(report, "latex").decode())
        manifest = json.loads(export_comparison(report, "json"))
        self.assertEqual(manifest["report_schema_version"], 1)
        self.assertTrue(manifest["selection_policy"])
