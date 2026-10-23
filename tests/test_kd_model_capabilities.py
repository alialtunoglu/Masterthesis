import gc
import unittest
import torch


class KDModelCapabilityTests(unittest.TestCase):
    def test_failed_probe_cleans_up_hooks(self):
        from src.training.kd_features import infer_feature_shapes
        model = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 1))
        with self.assertRaises(ValueError):
            infer_feature_shapes(model, model, "0", "missing", 8, torch.device("cpu"))
        self.assertEqual(len(model[0]._forward_hooks), 0)

    def test_all_registered_models_expose_validated_features_without_downloads(self):
        from src.models.model_factory import create_model
        from src.models.model_catalog import SUPPORTED_MODELS
        from src.training.kd_features import FeatureCapture, default_feature_layer
        torch.set_num_threads(2)
        for name in sorted(SUPPORTED_MODELS):
            with self.subTest(model=name):
                model = create_model(name, 3, pretrained=False).eval()
                capture = FeatureCapture(model, default_feature_layer(name))
                try:
                    with torch.no_grad():
                        output = model(torch.zeros(1, 3, 224, 224))
                    self.assertEqual(tuple(output.shape), (1, 3))
                    self.assertIn(capture.require_output().ndim, (2, 4))
                finally:
                    capture.close()
                    del model, capture
                    gc.collect()

    def test_layout_pooling_is_explicit(self):
        from src.training.kd_features import extract_feature_vector
        tokens = torch.tensor([[[9., 8.], [1., 3.], [5., 7.]]])
        self.assertEqual(extract_feature_vector(tokens, "NLC", "cls").tolist(), [[9., 8.]])
        self.assertEqual(extract_feature_vector(tokens, "NLC", "patch_mean").tolist(), [[3., 5.]])
        nhwc = torch.ones(2, 3, 4, 5)
        self.assertEqual(tuple(extract_feature_vector(nhwc, "NHWC").shape), (2, 5))

    def test_vector_feature_adapter_and_four_teacher_gradient_flow(self):
        from src.training.kd_config import DistillationConfig
        from src.training.kd_strategies import create_distillation_strategy
        from src.training.multi_kd_config import MultiDistillationConfig
        from src.training.multi_kd_strategies import create_multi_teacher_strategy
        features = torch.randn(3, 8, requires_grad=True)
        targets = torch.tensor([0, 1, 2])
        student_logits = torch.randn(3, 3, requires_grad=True)
        teacher_features = torch.randn(3, 12, 2, 2)
        strategy = create_distillation_strategy("feature_based", DistillationConfig(), student_channels=8, teacher_channels=12, feature_mode="pooled_vector")
        loss = strategy.compute(student_logits, torch.randn(3, 3), targets, features, teacher_features)
        loss.total.backward()
        self.assertIsNotNone(features.grad)
        self.assertTrue(all(parameter.grad is not None for parameter in strategy.parameters()))
        multi = create_multi_teacher_strategy("feature_based", MultiDistillationConfig(), ["a", "b", "c", "d"], [.25] * 4,
                                               student_channels=8, teacher_channels=[12] * 4, feature_modes=["pooled_vector"] * 4)
        multi.compute(student_logits.detach().requires_grad_(), [torch.randn(3, 3)] * 4, targets,
                      features.detach().requires_grad_(), [teacher_features] * 4).total.backward()
        self.assertEqual(len(multi.adapters), 4)


if __name__ == "__main__":
    unittest.main()
