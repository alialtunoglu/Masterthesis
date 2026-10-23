"""Dependent filter behavior against real Streamlit state; no jobs are started."""

import json
import sys
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))


JOBS = [
    dict(job_id="apple", dataset="appleleaf9", training_method="baseline", model_role="student", architecture_family="cnn", model="resnet18", status="finished"),
    dict(job_id="base", dataset="plantpathology2021", training_method="baseline", model_role="student", architecture_family="cnn", model="resnet18", status="finished"),
    dict(job_id="teacher", dataset="plantpathology2021", training_method="baseline", model_role="teacher", architecture_family="cnn", model="resnet50", status="finished"),
    dict(job_id="single", dataset="plantpathology2021", training_method="single_teacher_kd", student_model="mobilenet_v3_small", teacher_model="resnet50", kd_type="logit_based", status="finished"),
    dict(job_id="single-feature", dataset="plantpathology2021", training_method="single_teacher_kd", model="mobilenet_v3_small", teacher_model="densenet201", kd_type="feature_based", status="stopped"),
    dict(job_id="pair", dataset="plantpathology2021", training_method="multi_teacher_kd", model="mobilenet_v3_small", teacher_models=["resnet50", "densenet201"], teacher_count=2, kd_type="logit_based", aggregation="uniform", status="finished"),
    dict(job_id="reversed", dataset="plantpathology2021", training_method="multi_teacher_kd", model="mobilenet_v3_small", teacher_models=["densenet201", "resnet50"], teacher_count=2, kd_type="logit_based", aggregation="manual", status="failed"),
    dict(job_id="triple", dataset="plantpathology2021", training_method="multi_teacher_kd", model="mobilenet_v3_small", teacher_models=["regnet_y_8gf", "resnet50", "densenet201"], teacher_count=99, kd_type="logit_based", aggregation="uniform", status="stopped"),
]


def widget(widgets, label):
    return next(item for item in widgets if item.label == label)


class JobFilterTests(unittest.TestCase):
    def create_app(self, jobs=None):
        script = ("import streamlit as st\nfrom job_filters import render_job_filters\n"
                  f"jobs = {repr(JOBS if jobs is None else jobs)}\n"
                  "st.json([job['job_id'] for job in render_job_filters(jobs)])\n")
        return AppTest.from_string(script).run(timeout=30)

    def ids(self, app):
        self.assertFalse(app.exception)
        return json.loads(app.json[0].value)

    def test_dataset_disables_unavailable_methods(self):
        app = self.create_app()
        widget(app.selectbox, "Veri seti").select("appleleaf9").run()
        self.assertFalse(widget(app.button, "Baseline").disabled)
        self.assertTrue(widget(app.button, "Single-teacher KD").disabled)
        self.assertTrue(widget(app.button, "Multi-teacher KD").disabled)
        self.assertEqual(self.ids(app), ["apple"])

    def test_baseline_filters_role_then_family_then_model(self):
        app = self.create_app()
        widget(app.button, "Baseline").click().run()
        widget(app.selectbox, "Model rolü").select("Teacher").run()
        self.assertEqual(widget(app.selectbox, "Model").options, ["Tümü", "resnet50"])
        self.assertNotIn("Student model", [item.label for item in app.selectbox])
        self.assertEqual(self.ids(app), ["teacher"])

    def test_single_teacher_fields_and_options_follow_each_choice(self):
        app = self.create_app()
        widget(app.button, "Single-teacher KD").click().run()
        self.assertNotIn("Model rolü", [item.label for item in app.selectbox])
        self.assertNotIn("Mimari ailesi", [item.label for item in app.selectbox])
        widget(app.selectbox, "Teacher model").select("resnet50").run()
        self.assertEqual(widget(app.selectbox, "KD türü").options, ["Tümü", "Logit-based"])
        self.assertEqual(widget(app.selectbox, "Durum").options, ["Tümü", "Tamamlandı"])
        self.assertEqual(self.ids(app), ["single"])

    def test_multi_teacher_exact_unordered_combination_and_count(self):
        app = self.create_app()
        widget(app.button, "Multi-teacher KD").click().run()
        self.assertNotIn("Model rolü", [item.label for item in app.selectbox])
        self.assertNotIn("Mimari ailesi", [item.label for item in app.selectbox])
        self.assertEqual(widget(app.selectbox, "Teacher sayısı").options, ["Tümü", "2", "3"])
        widget(app.selectbox, "Teacher kombinasyonu").select("densenet201 + resnet50").run()
        self.assertEqual(self.ids(app), ["pair", "reversed"])
        self.assertEqual(widget(app.selectbox, "Aggregation").options, ["Tümü", "Manual", "Uniform"])
        widget(app.selectbox, "Teacher sayısı").select("3").run()
        self.assertEqual(widget(app.selectbox, "Teacher kombinasyonu").value, "all")
        self.assertEqual(self.ids(app), ["triple"])

    def test_dataset_change_clears_invalid_method_teacher_and_status(self):
        app = self.create_app()
        widget(app.button, "Single-teacher KD").click().run()
        widget(app.selectbox, "Teacher model").select("densenet201").run()
        widget(app.selectbox, "Durum").select("Durduruldu").run()
        widget(app.selectbox, "Veri seti").select("appleleaf9").run()
        self.assertEqual(widget(app.selectbox, "Durum").value, "all")
        self.assertNotIn("Teacher model", [item.label for item in app.selectbox])
        self.assertEqual(self.ids(app), ["apple"])
        widget(app.selectbox, "Veri seti").select("plantpathology2021").run()
        widget(app.button, "Single-teacher KD").click().run()
        self.assertEqual(widget(app.selectbox, "Teacher model").value, "all")

    def test_clear_restores_all_jobs_and_search(self):
        app = self.create_app()
        widget(app.button, "Multi-teacher KD").click().run()
        widget(app.text_input, "Deney ara").input("triple").run()
        self.assertEqual(self.ids(app), ["triple"])
        widget(app.button, "Filtreleri temizle").click().run()
        self.assertEqual(widget(app.text_input, "Deney ara").value, "")
        self.assertEqual(self.ids(app), [job["job_id"] for job in JOBS])

    def test_unknown_records_and_empty_list_remain_safe(self):
        app = self.create_app([dict(job_id="unknown", dataset="custom", status="failed")])
        self.assertEqual(self.ids(app), ["unknown"])
        self.assertTrue(widget(app.button, "Baseline").disabled)
        self.assertEqual(self.ids(self.create_app([])), [])

    def test_missing_or_invalid_teacher_count_is_not_a_selectable_count(self):
        jobs = [dict(job_id="incomplete", training_method="multi_teacher_kd", teacher_count="unknown", status="failed")]
        app = self.create_app(jobs)
        widget(app.button, "Multi-teacher KD").click().run()
        self.assertEqual(self.ids(app), ["incomplete"])
        self.assertEqual(widget(app.selectbox, "Teacher sayısı").options, ["Tümü"])
        self.assertTrue(widget(app.selectbox, "Teacher sayısı").disabled)

    def test_real_monitor_uses_conditional_filters(self):
        from unittest.mock import patch

        jobs = [dict(JOBS[5], stage="multi_teacher_knowledge_distillation", log_path="runs/logs/missing-filter-test.log")]
        with patch("experiment_runner.list_jobs", return_value=jobs), patch("experiment_runner.get_recovery_status", return_value={}):
            app = AppTest.from_file("app/streamlit_app.py").run(timeout=30)
            app.switch_page("pages/3_Job_Monitor.py").run(timeout=30)
            widget(app.button, "Multi-teacher KD").click().run()
            self.assertFalse(app.exception)
            self.assertNotIn("Model rolü", [item.label for item in app.selectbox])
            self.assertIn("Teacher kombinasyonu", [item.label for item in app.selectbox])
            self.assertEqual(widget(app.selectbox, "Deney seç").value, "pair")


if __name__ == "__main__":
    unittest.main()
