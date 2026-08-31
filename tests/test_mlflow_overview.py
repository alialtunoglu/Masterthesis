"""The MLflow helper page should report what is really in the store."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from app.result_loader import mlflow_experiment_overview


class FakeExperiment:
    def __init__(self, experiment_id: str, name: str):
        self.experiment_id = experiment_id
        self.name = name


class FakeMlflowClient:
    """Mirrors the slice of MlflowClient the overview relies on."""

    def __init__(self, experiments, runs_by_id, error: Exception | None = None):
        self._experiments = experiments
        self._runs = runs_by_id
        self._error = error
        self.tracking_uri = None

    def search_experiments(self):
        if self._error is not None:
            raise self._error
        return self._experiments

    def search_runs(self, experiment_ids, max_results=None):
        return self._runs.get(experiment_ids[0], [])


def _factory(client):
    def make(tracking_uri):
        client.tracking_uri = tracking_uri
        return client

    return make


class MlflowExperimentOverviewTests(unittest.TestCase):
    def test_reports_each_experiment_with_its_run_count(self):
        client = FakeMlflowClient(
            [FakeExperiment("1", "MasterThesis-Baseline")],
            {"1": ["run", "run", "run"]},
        )
        overview = mlflow_experiment_overview(
            "sqlite:///mlflow.db", client_factory=_factory(client)
        )
        self.assertEqual(
            overview, [{"experiment": "MasterThesis-Baseline", "runs": 3}]
        )
        self.assertEqual(client.tracking_uri, "sqlite:///mlflow.db")

    def test_results_are_sorted_by_name(self):
        client = FakeMlflowClient(
            [FakeExperiment("2", "B-Second"), FakeExperiment("1", "A-First")],
            {"1": ["run"], "2": ["run"]},
        )
        overview = mlflow_experiment_overview("uri", client_factory=_factory(client))
        self.assertEqual([row["experiment"] for row in overview], ["A-First", "B-Second"])

    def test_the_empty_default_experiment_is_hidden(self):
        client = FakeMlflowClient(
            [FakeExperiment("0", "Default"), FakeExperiment("1", "Real")],
            {"0": [], "1": ["run"]},
        )
        overview = mlflow_experiment_overview("uri", client_factory=_factory(client))
        self.assertEqual([row["experiment"] for row in overview], ["Real"])

    def test_a_used_default_experiment_is_kept(self):
        client = FakeMlflowClient([FakeExperiment("0", "Default")], {"0": ["run"]})
        overview = mlflow_experiment_overview("uri", client_factory=_factory(client))
        self.assertEqual([row["experiment"] for row in overview], ["Default"])

    def test_an_unreachable_store_yields_no_rows_instead_of_raising(self):
        client = FakeMlflowClient([], {}, error=RuntimeError("no such table"))
        self.assertEqual(
            mlflow_experiment_overview("uri", client_factory=_factory(client)), []
        )


if __name__ == "__main__":
    unittest.main()
