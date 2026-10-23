import tempfile
import unittest
from pathlib import Path
import pandas as pd
from src.utils.io import upsert_result_row


class ResultStoreTests(unittest.TestCase):
    def test_retry_replaces_same_run_without_dropping_other_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'results.csv'
            upsert_result_row({'mlflow_run_id': 'a', 'score': .5}, path)
            upsert_result_row({'mlflow_run_id': 'b', 'score': .6}, path)
            upsert_result_row({'mlflow_run_id': 'a', 'score': .7}, path)
            frame = pd.read_csv(path)
            self.assertEqual(len(frame), 2)
            self.assertAlmostEqual(frame.set_index('mlflow_run_id').loc['a', 'score'], .7)
