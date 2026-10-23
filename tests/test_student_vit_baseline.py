import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


class StudentViTBaselineTests(unittest.TestCase):
    def test_deit_students_have_requested_output_count(self):
        from src.models.model_factory import create_model, get_model_type, STUDENT_MODELS
        for name in ("deit_tiny_patch16_224", "deit_small_patch16_224"):
            with self.subTest(model=name):
                self.assertIn(name, STUDENT_MODELS)
                model = create_model(name, 6, pretrained=False).eval()
                with torch.no_grad():
                    self.assertEqual(tuple(model(torch.zeros(1, 3, 224, 224)).shape), (1, 6))
                self.assertEqual(get_model_type(name), "baseline_student")

    def test_six_configs_resolve_to_student_vit_not_teacher(self):
        from src.training.train_baseline import build_final_config, parse_args
        paths = list(Path("configs/baseline/vision_transformers").glob("*/*.json"))
        self.assertEqual(len(paths), 6)
        for path in paths:
            with patch.object(sys, "argv", ["train_baseline", "--config", str(path)]):
                config = build_final_config(parse_args())
            self.assertEqual((config["stage"], config["training_method"], config["model_role"], config["architecture_family"]),
                             ("baseline", "baseline", "student", "vision_transformer"))
            self.assertEqual(config["image_size"], 224)

    def test_teacher_cannot_silently_become_a_student_baseline(self):
        from src.training.train_baseline import build_final_config, parse_args
        with patch.object(sys, "argv", ["train_baseline", "--dataset", "appleleaf9", "--model", "resnet50"]):
            with self.assertRaises((ValueError, SystemExit)):
                build_final_config(parse_args())

    def test_student_vit_can_be_imported_from_notebook(self):
        from app.external_run_import import MODELS
        self.assertIn("deit_tiny_patch16_224", MODELS["baseline"])
        self.assertIn("deit_small_patch16_224", MODELS["baseline"])


if __name__ == "__main__":
    unittest.main()
