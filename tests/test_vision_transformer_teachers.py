import unittest
import sys
import json
import gc
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from app.experiment_runner import build_teacher_command
from src.models.model_factory import (
    VISION_TRANSFORMER_TEACHER_MODELS,
    create_model,
    get_model_type,
)
from src.training.trainer import train_one_epoch
from src.training.train_teacher import build_final_config, parse_args


class VisionTransformerTeacherTests(unittest.TestCase):
    def test_registry_and_model_type(self):
        self.assertEqual(
            VISION_TRANSFORMER_TEACHER_MODELS,
            {"swin_v2_t", "maxvit_t", "vit_b_16", "dinov2_vitb14"},
        )
        for model_name in VISION_TRANSFORMER_TEACHER_MODELS:
            self.assertEqual(get_model_type(model_name), "vision_transformer_teacher")

    def test_teacher_command_accepts_vit_training_options(self):
        command = build_teacher_command(
            dataset="plantpathology2021",
            model="swin_v2_t",
            config=None,
            epochs=30,
            batch_size=16,
            image_size=224,
            learning_rate=1e-4,
            weight_decay=0.05,
            pretrained=True,
            dry_run=False,
            gradient_accumulation_steps=2,
            label_smoothing=0.1,
            mixed_precision=True,
            gradient_clip_norm=1.0,
            warmup_epochs=3,
            scheduler_eta_min=1e-6,
        )
        self.assertIn("--mixed-precision", command)
        self.assertEqual(command[command.index("--gradient-accumulation-steps") + 1], "2")
        self.assertEqual(command[command.index("--warmup-epochs") + 1], "3")

    def test_vit_models_produce_requested_class_count(self):
        for model_name in VISION_TRANSFORMER_TEACHER_MODELS:
            with self.subTest(model=model_name):
                model = create_model(model_name, num_classes=6, pretrained=False)
                model.eval()
                with torch.no_grad():
                    output = model(torch.zeros(1, 3, 224, 224))
                self.assertEqual(tuple(output.shape), (1, 6))
                del output, model
                gc.collect()

    def test_gradient_accumulation_steps_at_window_and_tail(self):
        model = torch.nn.Linear(2, 2)
        expected_model = torch.nn.Linear(2, 2)
        expected_model.load_state_dict(model.state_dict())
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        original_step = optimizer.step
        calls = []

        def counted_step(*args, **kwargs):
            calls.append(True)
            return original_step(*args, **kwargs)

        optimizer.step = counted_step
        batches = [(torch.ones(1, 2), torch.zeros(1, dtype=torch.long)) for _ in range(3)]
        train_one_epoch(
            model,
            batches,
            torch.nn.CrossEntropyLoss(),
            optimizer,
            torch.device("cpu"),
            gradient_accumulation_steps=2,
        )
        self.assertEqual(len(calls), 2)
        expected_optimizer = torch.optim.SGD(expected_model.parameters(), lr=0.1)
        expected_optimizer.zero_grad()
        first_window = sum(
            torch.nn.functional.cross_entropy(expected_model(inputs), targets)
            for inputs, targets in batches[:2]
        ) / 2
        first_window.backward()
        expected_optimizer.step()
        expected_optimizer.zero_grad()
        torch.nn.functional.cross_entropy(
            expected_model(batches[2][0]), batches[2][1]
        ).backward()
        expected_optimizer.step()
        for actual, expected in zip(model.parameters(), expected_model.parameters()):
            self.assertTrue(torch.allclose(actual, expected))

    def test_all_twelve_vit_configs_resolve_to_vit_stage(self):
        root = Path("configs/teachers/vision_transformers")
        paths = sorted(root.glob("*/*.json"))
        self.assertEqual(len(paths), 12)
        for config_path in paths:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            with patch.object(sys, "argv", ["train_teacher", "--config", str(config_path)]):
                config = build_final_config(parse_args())
            self.assertEqual(config["stage"], "teacher_vision_transformer")
            self.assertEqual(config["teacher_family"], "vision_transformer")
            self.assertEqual(config["model_name"], raw["model_name"])


if __name__ == "__main__":
    unittest.main()
