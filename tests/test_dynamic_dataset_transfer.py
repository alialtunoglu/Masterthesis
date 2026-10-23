import json
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path

from app.portable_notebook import build_portable_notebook
from app.external_run_import import import_external_bundle
from src.tracking.external_bundle import create_external_bundle


class DynamicTransferTests(unittest.TestCase):
    def test_custom_notebook_carries_metadata_not_images_or_weights(self):
        package = {"descriptor": {"dataset_id": "potato", "root_relative_path": "datasets/Potato"},
                   "manifest": {"samples": [], "classes": ["Healthy", "Rust"]}, "split": {"classes": ["Healthy", "Rust"]}}
        value = build_portable_notebook(repository_url="https://example.invalid/repo.git", source_commit="a" * 40,
                                        stage="knowledge_distillation", dataset="potato", model="deit_tiny_patch16_224",
                                        command=["python", "src/training/train_kd.py", "--config", "old.json"],
                                        dataset_package=package, runtime_config={"experiment": {}, "training": {}}, teacher_inputs=[])
        notebook = json.loads(value)
        source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
        self.assertIn("DATASET_MOUNT", source)
        self.assertIn("EXTENSIBLE_DATASET_API", source)
        self.assertIn("verify_dataset_content", source)
        self.assertNotIn("competitions', 'download'", source)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), "notebook", "exec")

    def test_v2_custom_result_import_does_not_register_raw_dataset(self):
        from test_dataset_runtime import DatasetRuntimeTests
        from src.datasets.dataset_runtime import create_registered_split, resolve_dataset_runtime
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source, target = base / "source", base / "target"
            rev = DatasetRuntimeTests().prepare(source)
            ref = create_registered_split(source, rev.dataset_id, rev.revision, dict(train=.7, val=.15, test=.15), 42)
            runtime = resolve_dataset_runtime(source, ref)
            artifact = source / "results/knowledge_distillation/runs/potato/logit_based/demo/config.json"
            artifact.parent.mkdir(parents=True)
            artifact.write_text(json.dumps({"experiment": {"dataset_ref": ref.to_dict(), "dataset_name": "potato", "stage": "knowledge_distillation",
                                                            "student_model_name": "mobilenet_v3_small", "teacher_model_name": "resnet50"}, "class_names": runtime["class_names"]}))
            row = {"stage": "knowledge_distillation", "dataset_name": "potato", "student_model_name": "mobilenet_v3_small",
                   "kd_type": "logit_based",
                   "teacher_model_name": "resnet50", "run_name": "demo", "mlflow_run_id": "remote", "test_macro_f1": .8,
                   "config_path": artifact.relative_to(source).as_posix(), "split_file": str(Path(runtime["split_file"]).relative_to(source)),
                   "dataset_ref": json.dumps(ref.to_dict())}
            outputs = create_external_bundle(source, row, source_commit="a" * 40, output_dir=base / "out")
            with zipfile.ZipFile(outputs.archive) as archive:
                self.assertEqual(json.loads(archive.read("manifest.json"))["schema_version"], 2)
            inbox = target / "runs/import_inbox" / outputs.bundle_id
            inbox.mkdir(parents=True)
            shutil.copy2(outputs.archive, inbox / outputs.archive.name)
            receipt = import_external_bundle(target, inbox, register_run=lambda *_: "local")
            self.assertEqual(receipt["mlflow_run_id"], "local")
            self.assertFalse((target / "configs/datasets/potato.json").exists())
            self.assertFalse((target / "datasets").exists())

            # A valid manifest must not bless a config claiming a different dataset version.
            bad = json.loads(artifact.read_text())
            bad["experiment"]["dataset_ref"]["revision"] = "c" * 64
            artifact.write_text(json.dumps(bad))
            outputs = create_external_bundle(source, row, source_commit="a" * 40, output_dir=base / "bad-out")
            target = base / "bad-target"
            inbox = target / "runs/import_inbox" / outputs.bundle_id
            inbox.mkdir(parents=True)
            shutil.copy2(outputs.archive, inbox / outputs.archive.name)
            from app.external_run_import import ExternalRunImportError
            from unittest.mock import Mock
            register = Mock(return_value="must-not-register")
            with self.assertRaises(ExternalRunImportError):
                import_external_bundle(target, inbox, register_run=register)
            register.assert_not_called()


if __name__ == "__main__":
    unittest.main()
