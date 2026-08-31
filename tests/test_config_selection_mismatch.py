"""The config picker and the dataset/model pickers can disagree silently."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from app.ui_utils import config_selection_mismatches


class ConfigSelectionMismatchTests(unittest.TestCase):
    def test_no_config_selected_is_never_a_mismatch(self):
        self.assertEqual(config_selection_mismatches({}, "appleleaf9", "resnet18"), [])

    def test_matching_selection_reports_nothing(self):
        defaults = {"dataset_name": "appleleaf9", "model_name": "resnet18"}
        self.assertEqual(
            config_selection_mismatches(defaults, "appleleaf9", "resnet18"), []
        )

    def test_dataset_override_is_reported(self):
        defaults = {"dataset_name": "appleleaf9", "model_name": "resnet18"}
        mismatches = config_selection_mismatches(defaults, "plantvillage", "resnet18")
        self.assertEqual(len(mismatches), 1)
        self.assertIn("appleleaf9", mismatches[0])
        self.assertIn("plantvillage", mismatches[0])

    def test_model_override_is_reported(self):
        defaults = {"dataset_name": "appleleaf9", "model_name": "resnet18"}
        mismatches = config_selection_mismatches(
            defaults, "appleleaf9", "mobilenet_v3_small"
        )
        self.assertEqual(len(mismatches), 1)
        self.assertIn("mobilenet_v3_small", mismatches[0])

    def test_both_overrides_are_reported(self):
        defaults = {"dataset_name": "appleleaf9", "model_name": "resnet18"}
        self.assertEqual(
            len(config_selection_mismatches(defaults, "plantvillage", "efficientnet_b0")),
            2,
        )

    def test_a_config_without_those_keys_is_not_flagged(self):
        self.assertEqual(
            config_selection_mismatches({"epochs": 5}, "appleleaf9", "resnet18"), []
        )


if __name__ == "__main__":
    unittest.main()
