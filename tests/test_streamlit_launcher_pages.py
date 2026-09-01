"""UX contract tests shared by the three experiment launcher pages."""

import unittest

from streamlit.testing.v1 import AppTest


LAUNCHERS = (
    ("app/pages/2_Baseline_Experiments.py", "Eğitimi Başlat"),
    ("app/pages/5_CNN_Teacher_Experiments.py", "Teacher Eğitimi Başlat"),
    (
        "app/pages/6_Vision_Transformer_Teacher_Experiments.py",
        "ViT Eğitimi Başlat",
    ),
)

CONFIRM_LABEL = "Uzun süren eğitimi onaylıyorum"

# The KD pages launch the same class of long GPU job from two explicit
# buttons, so they need the confirmation gate but not the dry-run guard.
DISTILLERS = (
    ("app/pages/8_Knowledge_Distillation.py", "KD Eğitimini Başlat"),
    (
        "app/pages/9_Multi_Teacher_Knowledge_Distillation.py",
        "Multi-KD Eğitimini Başlat",
    ),
)


def _run(path: str) -> AppTest:
    """Enter the page the way a user does, so st.page_link resolves."""
    app = AppTest.from_file("app/streamlit_app.py").run(timeout=60)
    app.switch_page(path.removeprefix("app/")).run(timeout=60)
    assert not app.exception, f"{path} raised {app.exception}"
    return app


def _widget(collection, label):
    for widget in collection:
        if widget.label == label:
            return widget
    raise AssertionError(f"No widget labelled {label!r}")


def _index_of(app: AppTest, type_name: str, label: str) -> int:
    for index, element in enumerate(app.main):
        if type(element).__name__ != type_name:
            continue
        try:
            if element.label == label:
                return index
        except Exception:  # pragma: no cover - elements without a label
            continue
    raise AssertionError(f"No {type_name} labelled {label!r}")


class TrainingLaunchGuardTests(unittest.TestCase):
    """A long GPU job must never be one stray click away."""

    def test_training_button_is_disabled_until_the_user_confirms(self):
        for path, launch_label in LAUNCHERS:
            with self.subTest(page=path):
                app = _run(path)
                self.assertTrue(
                    _widget(app.button, launch_label).disabled,
                    "training must start disabled",
                )

    def test_confirming_enables_the_training_button(self):
        for path, launch_label in LAUNCHERS:
            with self.subTest(page=path):
                app = _run(path)
                _widget(app.checkbox, CONFIRM_LABEL).set_value(True).run(timeout=60)
                self.assertFalse(_widget(app.button, launch_label).disabled)

    def test_dry_run_keeps_the_training_button_disabled(self):
        """Otherwise both buttons queue the same dry run under different names."""
        for path, launch_label in LAUNCHERS:
            with self.subTest(page=path):
                app = _run(path)
                _widget(app.checkbox, CONFIRM_LABEL).set_value(True).run(timeout=60)
                _widget(app.checkbox, "Dry run").set_value(True).run(timeout=60)
                self.assertTrue(
                    _widget(app.button, launch_label).disabled,
                    "dry run must not be launchable as real training",
                )


class DistillationLaunchGuardTests(unittest.TestCase):
    def test_distillation_training_also_requires_confirmation(self):
        for path, launch_label in DISTILLERS:
            with self.subTest(page=path):
                app = _run(path)
                self.assertIn(
                    CONFIRM_LABEL,
                    [checkbox.label for checkbox in app.checkbox],
                )
                self.assertTrue(_widget(app.button, launch_label).disabled)


class LauncherLayoutTests(unittest.TestCase):
    def test_notebook_export_is_offered_after_the_launch_buttons(self):
        for path, launch_label in LAUNCHERS:
            with self.subTest(page=path):
                app = _run(path)
                download = _index_of(app, "DownloadButton", "Portable .ipynb indir")
                launch = _index_of(app, "Button", launch_label)
                self.assertGreater(
                    download,
                    launch,
                    "the primary action must come before the export option",
                )

    def test_every_launcher_links_to_the_job_monitor(self):
        for path, _ in LAUNCHERS:
            with self.subTest(page=path):
                app = _run(path)
                targets = [
                    element.proto.page
                    for element in app.main
                    if getattr(element, "type", None) == "page_link"
                ]
                self.assertIn(
                    "Job_Monitor",
                    targets,
                    "launchers should hand monitoring over to the Job Monitor",
                )


class VitImageSizeTests(unittest.TestCase):
    def test_fixed_image_size_is_not_rendered_as_a_dead_input(self):
        app = _run("app/pages/6_Vision_Transformer_Teacher_Experiments.py")
        labels = [widget.label for widget in app.number_input]
        self.assertNotIn("Image size", labels)


if __name__ == "__main__":
    unittest.main()
