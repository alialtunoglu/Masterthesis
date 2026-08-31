"""Cross-page consistency contracts for the Streamlit dashboard."""

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from app import external_run_import
from src.tracking.mlflow_tracker import DEFAULT_EXPERIMENT_NAME


PAGES_DIR = Path("app/pages")


def _sidebar_order() -> list[str]:
    """Streamlit orders the sidebar by the numeric filename prefix."""
    return [path.name for path in sorted(PAGES_DIR.glob("*.py"))]


def _home_link_order() -> list[str]:
    app = AppTest.from_file("app/streamlit_app.py").run(timeout=60)
    assert not app.exception, app.exception
    return [
        element.proto.page_script_hash
        for element in app.main
        if getattr(element, "type", None) == "page_link"
    ]


class NavigationOrderTests(unittest.TestCase):
    def test_home_page_lists_every_page(self):
        app = AppTest.from_file("app/streamlit_app.py").run(timeout=60)
        self.assertFalse(app.exception)
        linked = [
            element.proto.page
            for element in app.main
            if getattr(element, "type", None) == "page_link"
        ]
        self.assertEqual(len(linked), len(_sidebar_order()))

    def test_home_page_order_matches_the_sidebar_order(self):
        """Two navigations disagreeing is worse than either order alone."""
        app = AppTest.from_file("app/streamlit_app.py").run(timeout=60)
        self.assertFalse(app.exception)
        linked = [
            element.proto.page
            for element in app.main
            if getattr(element, "type", None) == "page_link"
        ]
        # st.page_link reports the page name without its numeric prefix.
        expected = [
            name.split("_", 1)[1].removesuffix(".py") for name in _sidebar_order()
        ]
        self.assertEqual(linked, expected)


class ExperimentNamingTests(unittest.TestCase):
    def test_imported_baseline_runs_land_in_the_tracked_experiment(self):
        """A plural typo silently created a second, empty experiment."""
        self.assertEqual(
            external_run_import.EXPERIMENT_NAMES["baseline"],
            DEFAULT_EXPERIMENT_NAME,
        )


class DeprecatedApiTests(unittest.TestCase):
    def test_no_page_uses_the_deprecated_container_width_flag(self):
        """Streamlit 1.59 prints a visible deprecation notice for it."""
        offenders = [
            path.name
            for path in sorted(PAGES_DIR.glob("*.py"))
            if "use_container_width" in path.read_text(encoding="utf-8")
        ]
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
