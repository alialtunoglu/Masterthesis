import tempfile
import unittest
from pathlib import Path


class DynamicExperimentConfigTests(unittest.TestCase):
    def test_custom_data_ref_pinned_and_model_role_validated(self):
        from test_dataset_runtime import DatasetRuntimeTests
        from src.datasets.dataset_runtime import create_registered_split
        from src.training.experiment_config import resolve_experiment_config
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rev = DatasetRuntimeTests().prepare(root)
            ref = create_registered_split(root, "potato", rev.revision, dict(train=.7, val=.15, test=.15), 42)
            result = resolve_experiment_config(root, {"stage": "baseline", "dataset_name": "potato", "model_name": "deit_tiny_patch16_224", "image_size": 224})
            self.assertEqual(result["dataset_ref"], ref.to_dict())
            with self.assertRaises(ValueError):
                resolve_experiment_config(root, {**result, "model_name": "resnet50"})
            with self.assertRaises(ValueError):
                resolve_experiment_config(root, {**result, "image_size": 256})

    def test_kd_accepts_catalog_models_and_logit_needs_no_feature_node(self):
        from src.training.kd_config import load_kd_config
        from src.training.multi_kd_config import load_multi_kd_config
        ref = dict(dataset_id="potato", revision="a" * 64, split_id="b" * 64)
        payload = {"experiment": {"dataset_name": "potato", "dataset_ref": ref, "student_model_name": "deit_tiny_patch16_224", "teacher_model_name": "swin_v2_t"},
                   "training": {}, "distillation": {}}
        config = load_kd_config(payload)
        self.assertIsNone(config.distillation.student_layer)
        teachers = [{"model_name": name} for name in ("resnet50", "densenet201", "swin_v2_t", "vit_b_16")]
        multi = load_multi_kd_config({"experiment": {"dataset_name": "potato", "dataset_ref": ref}, "student": {"model_name": "deit_small_patch16_224"},
                                      "teachers": teachers, "training": {}, "distillation": {}})
        self.assertEqual(len(multi.teachers), 4)


if __name__ == "__main__":
    unittest.main()
