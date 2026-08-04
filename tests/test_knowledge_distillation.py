from __future__ import annotations

import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from models.model_factory import create_model
from training.kd_config import load_kd_config
from training.kd_features import FEATURE_LAYER_REGISTRY, infer_feature_shapes
from training.kd_strategies import create_distillation_strategy
from training.kd_trainer import freeze_teacher, train_kd_epoch
from training.optimization import create_optimizer, create_scheduler, is_improvement


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class KDConfigTests(unittest.TestCase):
    def test_all_nine_templates_are_valid_and_scoped(self) -> None:
        paths = sorted(
            (PROJECT_ROOT / "configs/knowledge_distillation/plantpathology2021").glob(
                "*/*.json"
            )
        )
        self.assertEqual(9, len(paths))
        combinations = set()
        for path in paths:
            config = load_kd_config(path)
            combinations.add(
                (config.experiment.kd_type, config.experiment.teacher_model_name)
            )
            self.assertEqual("plantpathology2021", config.experiment.dataset_name)
            self.assertEqual("mobilenet_v3_small", config.experiment.student_model_name)
            self.assertFalse(config.distillation.include_logit_distillation)
        self.assertEqual(9, len(combinations))

    def test_invalid_dataset_is_rejected(self) -> None:
        config = load_kd_config(
            PROJECT_ROOT
            / "configs/knowledge_distillation/plantpathology2021/logit_based/"
            "mobilenet_v3_small__resnet50.json"
        ).to_dict()
        config["experiment"]["dataset_name"] = "plantvillage"
        with self.assertRaisesRegex(ValueError, "dataset_name"):
            load_kd_config(config)

    def test_unimplemented_combined_logit_mode_is_rejected(self) -> None:
        config = load_kd_config(
            PROJECT_ROOT
            / "configs/knowledge_distillation/plantpathology2021/feature_based/"
            "mobilenet_v3_small__resnet50.json"
        ).to_dict()
        config["distillation"]["include_logit_distillation"] = True
        with self.assertRaisesRegex(ValueError, "combined logit"):
            load_kd_config(config)


class KDStrategyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = load_kd_config(
            PROJECT_ROOT
            / "configs/knowledge_distillation/plantpathology2021/logit_based/"
            "mobilenet_v3_small__resnet50.json"
        )
        self.student_logits = torch.randn(4, 6, requires_grad=True)
        self.teacher_logits = torch.randn(4, 6)
        self.targets = torch.tensor([0, 1, 2, 3])

    def test_logit_strategy_produces_ce_and_kd_gradients(self) -> None:
        strategy = create_distillation_strategy(
            "logit_based", self.base.distillation
        )
        output = strategy.compute(
            self.student_logits, self.teacher_logits, self.targets
        )
        output.total.backward()
        self.assertIn("ce_loss", output.components)
        self.assertIn("kd_loss", output.components)
        self.assertIsNotNone(self.student_logits.grad)

    def test_logit_strategy_runs_through_training_epoch_interface(self) -> None:
        student = torch.nn.Linear(4, 6)
        teacher = torch.nn.Linear(4, 6)
        strategy = create_distillation_strategy(
            "logit_based", self.base.distillation
        )
        loader = DataLoader(
            TensorDataset(torch.randn(4, 4), torch.tensor([0, 1, 2, 3])),
            batch_size=2,
        )
        metrics = train_kd_epoch(
            student=student,
            teacher=teacher,
            strategy=strategy,
            dataloader=loader,
            optimizer=torch.optim.SGD(student.parameters(), lr=0.01),
            device=torch.device("cpu"),
            student_capture=None,
            teacher_capture=None,
            max_batches=None,
        )
        self.assertIn("kd_loss", metrics)
        self.assertIn("teacher_student_agreement", metrics)

    def test_feature_strategy_trains_adapter_and_student(self) -> None:
        config = load_kd_config(
            PROJECT_ROOT
            / "configs/knowledge_distillation/plantpathology2021/feature_based/"
            "mobilenet_v3_small__resnet50.json"
        )
        strategy = create_distillation_strategy(
            "feature_based", config.distillation, student_channels=8, teacher_channels=12
        )
        student_features = torch.randn(4, 8, 4, 4, requires_grad=True)
        teacher_features = torch.randn(4, 12, 2, 2)
        output = strategy.compute(
            self.student_logits,
            self.teacher_logits,
            self.targets,
            student_features,
            teacher_features,
        )
        output.total.backward()
        self.assertIsNotNone(student_features.grad)
        self.assertIsNotNone(strategy.adapter.projection.weight.grad)

    def test_relation_strategy_handles_different_feature_widths(self) -> None:
        config = load_kd_config(
            PROJECT_ROOT
            / "configs/knowledge_distillation/plantpathology2021/relation_based/"
            "mobilenet_v3_small__resnet50.json"
        )
        strategy = create_distillation_strategy("relation_based", config.distillation)
        output = strategy.compute(
            self.student_logits,
            self.teacher_logits,
            self.targets,
            torch.randn(4, 8, 2, 2, requires_grad=True),
            torch.randn(4, 12, 2, 2),
        )
        self.assertTrue(torch.isfinite(output.total))
        self.assertIn("relation_loss", output.components)

    def test_teacher_is_frozen(self) -> None:
        teacher = torch.nn.Linear(4, 2)
        freeze_teacher(teacher)
        self.assertFalse(teacher.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in teacher.parameters()))


class KDFeatureAndOptimizationTests(unittest.TestCase):
    def test_validated_feature_registry_shapes(self) -> None:
        student = create_model("mobilenet_v3_small", 6, pretrained=False)
        for teacher_name in ("resnet50", "densenet201", "regnet_y_8gf"):
            with self.subTest(teacher=teacher_name):
                teacher = create_model(teacher_name, 6, pretrained=False)
                shapes = infer_feature_shapes(
                    student,
                    teacher,
                    FEATURE_LAYER_REGISTRY["mobilenet_v3_small"],
                    FEATURE_LAYER_REGISTRY[teacher_name],
                    64,
                    torch.device("cpu"),
                )
                self.assertEqual(4, len(shapes.student))
                self.assertEqual(4, len(shapes.teacher))
                del teacher

    def test_all_optimizer_and_scheduler_variants_construct(self) -> None:
        for optimizer_name in ("adamw", "adam", "sgd"):
            for scheduler_name in ("cosine", "step", "reduce_on_plateau", "none"):
                config_dict = self._base_config_dict()
                config_dict["training"]["optimizer"] = optimizer_name
                config_dict["training"]["scheduler"] = scheduler_name
                config = load_kd_config(config_dict)
                parameter = torch.nn.Parameter(torch.ones(1))
                optimizer = create_optimizer([parameter], config.training)
                create_scheduler(optimizer, config.training)

    def test_monitor_directions(self) -> None:
        self.assertTrue(is_improvement(0.2, 0.3, "val_loss"))
        self.assertTrue(is_improvement(0.9, 0.8, "val_macro_f1"))
        self.assertFalse(is_improvement(0.7, 0.8, "val_accuracy"))

    @staticmethod
    def _base_config_dict() -> dict:
        return load_kd_config(
            PROJECT_ROOT
            / "configs/knowledge_distillation/plantpathology2021/logit_based/"
            "mobilenet_v3_small__resnet50.json"
        ).to_dict()


if __name__ == "__main__":
    unittest.main()
