"""Small student contracts, without data training or network downloads."""

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.models.model_catalog import get_model_spec
from src.models.model_factory import create_model

NEW_STUDENTS = ("shufflenet_v2_x0_5", "micronet_m0")


class LightweightStudentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_new_students_are_smaller_six_class_models_with_checkpoint_roundtrip(self):
        reference = create_model("mobilenet_v3_small", 6, False)
        reference_count = sum(p.numel() for p in reference.parameters())
        for name in NEW_STUDENTS:
            with self.subTest(model=name):
                spec = get_model_spec(name)
                self.assertEqual(spec.roles, ("student",))
                self.assertEqual(spec.family, "cnn")
                model = create_model(name, 6, False).eval()
                self.assertLess(sum(p.numel() for p in model.parameters()), reference_count)
                sample = torch.randn(1, 3, 224, 224)
                with torch.no_grad():
                    expected = model(sample)
                self.assertEqual(tuple(expected.shape), (1, 6))
                self.assertTrue(torch.isfinite(expected).all())
                buffer = io.BytesIO()
                torch.save(model.state_dict(), buffer)
                buffer.seek(0)
                restored = create_model(name, 6, False).eval()
                restored.load_state_dict(torch.load(buffer, weights_only=True))
                with torch.no_grad():
                    torch.testing.assert_close(expected, restored(sample))

    def test_singleton_training_batch_keeps_finite_gradients(self):
        for name in NEW_STUDENTS:
            with self.subTest(model=name):
                model = create_model(name, 6, False).train()
                loss = torch.nn.functional.cross_entropy(model(torch.randn(1, 3, 224, 224)), torch.tensor([2]))
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()))

    def test_micronet_pretrained_loads_backbone_strictly_before_replacing_head(self):
        from src.models._micronet.network import MicroNet
        from src.models.micronet import m0_config
        official = MicroNet(m0_config()).state_dict()
        first = next(iter(official))
        official[first].fill_(0.125)
        fixture = {"module." + key: value for key, value in official.items()}
        with patch("src.models.micronet._load_pretrained_state", return_value=fixture):
            model = create_model("micronet_m0", 6, True)
        torch.testing.assert_close(model.state_dict()[first], official[first])
        self.assertIsInstance(model.classifier[-1], torch.nn.Linear)
        self.assertEqual(model.classifier[-1].out_features, 6)
        fixture.pop("module." + first)
        with patch("src.models.micronet._load_pretrained_state", return_value=fixture):
            with self.assertRaises(RuntimeError):
                create_model("micronet_m0", 6, True)
        with patch("src.models.micronet._load_pretrained_state", side_effect=OSError("offline")):
            with self.assertRaisesRegex(OSError, "offline"):
                create_model("micronet_m0", 6, True)

    def test_micronet_corrupt_cached_weights_are_rejected_before_deserialization(self):
        from src.models.micronet import _load_pretrained_state, WEIGHTS_FILENAME
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "checkpoints" / WEIGHTS_FILENAME
            path.parent.mkdir()
            path.write_bytes(b"corrupt checkpoint")
            with patch("torch.hub.get_dir", return_value=folder):
                with self.assertRaisesRegex(RuntimeError, "SHA256"):
                    _load_pretrained_state()

    def test_micronet_requires_224_input(self):
        self.assertEqual(get_model_spec("micronet_m0").fixed_input_size, 224)
        with self.assertRaisesRegex(ValueError, "224"):
            create_model("micronet_m0", 6, False)(torch.zeros(1, 3, 192, 192))


if __name__ == "__main__":
    unittest.main()
