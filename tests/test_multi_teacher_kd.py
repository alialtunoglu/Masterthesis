from __future__ import annotations

import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from training.multi_kd_aggregation import resolve_teacher_weights
from training.multi_kd_config import (
    TeacherConfig,
    load_multi_kd_config,
    validate_multi_kd_config,
)
from training.multi_kd_strategies import create_multi_teacher_strategy
from training.multi_kd_trainer import train_multi_teacher_epoch
from training.train_multi_kd import build_paths, build_run_name


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = (
    ROOT / "configs/multi_teacher_knowledge_distillation/plantpathology2021"
)


class MultiTeacherConfigTests(unittest.TestCase):
    def test_all_twelve_templates_are_valid(self):
        paths = sorted(CONFIG_ROOT.glob("*/*.json"))
        self.assertEqual(12, len(paths))
        combinations = set()
        for path in paths:
            config = load_multi_kd_config(path)
            key = (
                config.experiment.kd_type,
                tuple(teacher.model_name for teacher in config.teachers),
            )
            combinations.add(key)
            self.assertIn(len(config.teachers), {2, 3})
        self.assertEqual(12, len(combinations))

    def test_duplicate_teacher_models_are_rejected(self):
        config = load_multi_kd_config(next(CONFIG_ROOT.glob("*/*.json"))).to_dict()
        config["teachers"][1]["model_name"] = config["teachers"][0]["model_name"]
        with self.assertRaisesRegex(ValueError, "unique"):
            load_multi_kd_config(config)

    def test_runtime_fields_are_required(self):
        config = load_multi_kd_config(next(CONFIG_ROOT.glob("*/*.json")))
        with self.assertRaisesRegex(ValueError, "run_id"):
            validate_multi_kd_config(config, runtime=True)

    def test_three_teacher_artifact_paths_fit_windows_limit(self):
        path = next(
            p
            for p in CONFIG_ROOT.glob("*/*.json")
            if len(load_multi_kd_config(p).teachers) == 3
        )
        config = load_multi_kd_config(path)
        run_name = build_run_name(config, torch.device("cuda"))
        longest = max(
            len(str(value.resolve())) for value in build_paths(config, run_name).values()
        )
        self.assertLess(longest, 260)


class AggregationTests(unittest.TestCase):
    def setUp(self):
        self.teachers = (
            TeacherConfig("a", "x", best_val_macro_f1=0.8, manual_weight=1),
            TeacherConfig("b", "y", best_val_macro_f1=0.2, manual_weight=3),
        )

    def test_uniform_weights(self):
        self.assertEqual((0.5, 0.5), resolve_teacher_weights("uniform", self.teachers))

    def test_validation_weights(self):
        self.assertEqual(
            (0.8, 0.2),
            resolve_teacher_weights("validation_weighted", self.teachers),
        )

    def test_manual_weights_are_normalized(self):
        self.assertEqual(
            (0.25, 0.75), resolve_teacher_weights("manual", self.teachers)
        )


class MultiTeacherStrategyTests(unittest.TestCase):
    def _config(self, kd_type):
        return load_multi_kd_config(
            next((CONFIG_ROOT / kd_type).glob("*.json"))
        )

    def test_logit_strategy_runs_full_epoch_for_three_teachers(self):
        config = self._config("logit_based")
        names = ("a", "b", "c")
        strategy = create_multi_teacher_strategy(
            "logit_based", config.distillation, names, (0.2, 0.3, 0.5)
        )
        student = torch.nn.Linear(4, 6)
        teachers = [torch.nn.Linear(4, 6) for _ in names]
        loader = DataLoader(
            TensorDataset(torch.randn(6, 4), torch.tensor([0, 1, 2, 3, 4, 5])),
            batch_size=3,
        )
        metrics = train_multi_teacher_epoch(
            student,
            teachers,
            names,
            strategy,
            loader,
            torch.optim.SGD(student.parameters(), lr=0.01),
            torch.device("cpu"),
        )
        self.assertIn("kd_loss", metrics)
        self.assertIn("agreement_a", metrics)
        self.assertIn("ensemble_student_agreement", metrics)
        self.assertTrue(all(p.grad is None for model in teachers for p in model.parameters()))

    def test_feature_strategy_trains_independent_adapters(self):
        config = self._config("feature_based")
        strategy = create_multi_teacher_strategy(
            "feature_based",
            config.distillation,
            ("a", "b"),
            (0.5, 0.5),
            student_channels=4,
            teacher_channels=(6, 8),
        )
        output = strategy.compute(
            torch.randn(3, 6, requires_grad=True),
            [torch.randn(3, 6), torch.randn(3, 6)],
            torch.tensor([0, 1, 2]),
            student_features=torch.randn(3, 4, 4, 4, requires_grad=True),
            teacher_features=[torch.randn(3, 6, 2, 2), torch.randn(3, 8, 3, 3)],
        )
        output.total.backward()
        self.assertTrue(
            all(adapter.projection.weight.grad is not None for adapter in strategy.adapters.values())
        )

    def test_relation_strategy_accepts_different_feature_widths(self):
        config = self._config("relation_based")
        strategy = create_multi_teacher_strategy(
            "relation_based", config.distillation, ("a", "b"), (0.5, 0.5)
        )
        output = strategy.compute(
            torch.randn(3, 6, requires_grad=True),
            [torch.randn(3, 6), torch.randn(3, 6)],
            torch.tensor([0, 1, 2]),
            student_features=torch.randn(3, 4, 2, 2, requires_grad=True),
            teacher_features=[torch.randn(3, 7, 2, 2), torch.randn(3, 9, 2, 2)],
        )
        self.assertTrue(torch.isfinite(output.total))
        self.assertIn("relation_loss", output.components)


if __name__ == "__main__":
    unittest.main()
