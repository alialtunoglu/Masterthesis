import json
import tempfile
import unittest
from pathlib import Path
import pandas as pd


class TeacherCatalogTests(unittest.TestCase):
    def test_only_matching_complete_content_verified_teachers_are_returned(self):
        from test_dataset_runtime import DatasetRuntimeTests
        from src.datasets.dataset_runtime import create_registered_split, resolve_dataset_runtime
        from src.datasets.dataset_registration import sha256_file
        from src.training.teacher_catalog import find_compatible_teacher_runs, validate_teacher_selection
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rev = DatasetRuntimeTests().prepare(root)
            ref = create_registered_split(root, "potato", rev.revision, dict(train=.7, val=.15, test=.15), 42)
            runtime = resolve_dataset_runtime(root, ref)
            checkpoint = root / "checkpoints/teachers/cnn/model.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint-fixture")
            config = root / "results/teachers/cnn/run/config.json"
            config.parent.mkdir(parents=True)
            config.write_text(json.dumps({**runtime, "image_size": 224, "dry_run": False}))
            row = dict(dataset_name="potato", model_name="resnet50", mlflow_run_id="teacher1", checkpoint_path=checkpoint.relative_to(root).as_posix(),
                       config_path=config.relative_to(root).as_posix(), checkpoint_sha256=sha256_file(checkpoint), best_val_macro_f1=.9)
            summary = root / "results/teachers/cnn/teacher_results.csv"
            pd.DataFrame([row]).to_csv(summary, index=False)
            candidates = find_compatible_teacher_runs(root, ref, "deit_tiny_patch16_224", "logit_based")
            self.assertEqual([item["mlflow_run_id"] for item in candidates], ["teacher1"])
            payload = {"experiment": dict(stage="knowledge_distillation", dataset_name="potato", dataset_ref=ref.to_dict(), student_model_name="deit_tiny_patch16_224",
                                            teacher_model_name="resnet50", teacher_run_id="teacher1", teacher_checkpoint_path=row["checkpoint_path"]), "training": {"image_size": 224}, "distillation": {}}
            validate_teacher_selection(root, payload)
            metadata = json.loads(config.read_text())
            metadata["split_fingerprint"] = "different-split"
            config.write_text(json.dumps(metadata))
            self.assertEqual(find_compatible_teacher_runs(root, ref, "deit_tiny_patch16_224", "logit_based"), [])
            config.write_text(json.dumps({**runtime, "image_size": 224, "dry_run": False}))
            partial = dict(row, max_train_batches=1)
            pd.DataFrame([partial]).to_csv(summary, index=False)
            self.assertEqual(find_compatible_teacher_runs(root, ref, "deit_tiny_patch16_224", "logit_based"), [])
            pd.DataFrame([dict(row, dataset_ref=json.dumps({**ref.to_dict(), "split_id": "c" * 64}))]).to_csv(summary, index=False)
            self.assertEqual(find_compatible_teacher_runs(root, ref, "deit_tiny_patch16_224", "logit_based"), [])
            pd.DataFrame([row]).to_csv(summary, index=False)
            wrong = dict(ref.to_dict(), split_id="a" * 64)
            config.write_text(json.dumps({**runtime, "dataset_ref": wrong, "image_size": 224}))
            self.assertEqual(find_compatible_teacher_runs(root, ref, "deit_tiny_patch16_224", "logit_based"), [])
            config.write_text(json.dumps({**runtime, "image_size": 224}))
            checkpoint.write_bytes(b"changed-checkpoint")
            with self.assertRaises(ValueError):
                validate_teacher_selection(root, payload)


if __name__ == "__main__":
    unittest.main()
