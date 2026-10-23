import tempfile
import unittest
from pathlib import Path

from PIL import Image
from test_dataset_registration import make_dataset


class DatasetRuntimeTests(unittest.TestCase):
    def test_tampered_split_assignment_is_rejected(self):
        from src.datasets.dataset_runtime import create_registered_split, resolve_dataset_runtime
        import json
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rev = self.prepare(root)
            ref = create_registered_split(root, "potato", rev.revision, dict(train=.7, val=.15, test=.15), 42)
            path = Path(resolve_dataset_runtime(root, ref)["split_file"])
            payload = json.loads(path.read_text())
            payload["splits"]["train"][0]["label_id"] = 999
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                resolve_dataset_runtime(root, ref)

    def prepare(self, root, presplit=False):
        from src.datasets.dataset_registration import scan_dataset, register_dataset_revision
        from src.datasets.dataset_registry import DatasetDefinition
        path = make_dataset(root, presplit=presplit)
        layout = "presplit" if presplit else "imagefolder"
        scan = scan_dataset(root, path, layout)
        return register_dataset_revision(root, DatasetDefinition("potato", "Potato", "datasets/Potato", layout), scan)

    def test_same_seed_split_is_idempotent_and_all_classes_present(self):
        from src.datasets.dataset_runtime import create_registered_split, resolve_dataset_runtime, verify_dataset_content
        from src.datasets.dataset_registry import list_datasets
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            revision = self.prepare(root)
            ratios = dict(train=.7, val=.15, test=.15)
            ref = create_registered_split(root, revision.dataset_id, revision.revision, ratios, 42)
            self.assertEqual(ref, create_registered_split(root, revision.dataset_id, revision.revision, ratios, 42))
            runtime = resolve_dataset_runtime(root, ref, None)
            import json
            payload = json.loads(Path(runtime["split_file"]).read_text())
            self.assertEqual([len(payload["splits"][part]) for part in ("train", "val", "test")], [15, 3, 3])
            self.assertTrue(list_datasets(root, include_legacy=False)[0].ready)
            verify_dataset_content(root, ref)
            Image.new("RGB", (8, 8), (255, 255, 255)).save(root / "datasets/Potato/images/Healthy/0.png")
            with self.assertRaises(ValueError):
                verify_dataset_content(root, ref)

    def test_presplit_is_not_randomized_and_invalid_ratios_rejected(self):
        from src.datasets.dataset_runtime import create_registered_split, resolve_dataset_runtime
        import json
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            revision = self.prepare(root, True)
            with self.assertRaises(ValueError):
                create_registered_split(root, revision.dataset_id, revision.revision, dict(train=.9, val=.2, test=.1), 42)
            ref = create_registered_split(root, revision.dataset_id, revision.revision, dict(train=.7, val=.15, test=.15), 42)
            payload = json.loads(Path(resolve_dataset_runtime(root, ref, None)["split_file"]).read_text())
            self.assertTrue(all("/val/" in row["path"] for row in payload["splits"]["val"]))

    def test_insufficient_class_does_not_publish_ready_default(self):
        from src.datasets.dataset_runtime import create_registered_split
        from src.datasets.dataset_registration import scan_dataset, register_dataset_revision
        from src.datasets.dataset_registry import DatasetDefinition, list_datasets
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = make_dataset(root, count=2)
            scan = scan_dataset(root, path, "imagefolder")
            revision = register_dataset_revision(root, DatasetDefinition("potato", "Potato", "datasets/Potato", "imagefolder"), scan)
            with self.assertRaises(ValueError):
                create_registered_split(root, revision.dataset_id, revision.revision, dict(train=.7, val=.15, test=.15), 42)
            self.assertFalse(list_datasets(root, include_legacy=False)[0].ready)


if __name__ == "__main__":
    unittest.main()
