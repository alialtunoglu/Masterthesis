import unittest
from src.tracking.run_metadata import dataset_fingerprint, split_fingerprint, protocol_fingerprint


class MetadataTests(unittest.TestCase):
    def test_dataset_fingerprint_ignores_split_membership_but_not_labels(self):
        first = {"classes": ["a"], "splits": {"train": [{"path": "datasets/x/a.jpg", "label_id": 0}], "test": []}}
        moved = {"classes": ["a"], "splits": {"train": [], "test": [{"path": "datasets/x/a.jpg", "label_id": 0}]}}
        changed = {"classes": ["a"], "splits": {"train": [], "test": [{"path": "datasets/x/a.jpg", "label_id": 1}]}}
        self.assertEqual(dataset_fingerprint(first), dataset_fingerprint(moved))
        self.assertNotEqual(dataset_fingerprint(first), dataset_fingerprint(changed))

    def test_portable_paths_but_not_changed_labels(self):
        def split(path, label=0):
            return {"dataset_name": "test", "classes": ["a", "b"],
                    "splits": {"train": [{"path": path, "label_id": label}]}}
        a = split(r"C:\repo\datasets\test\a.jpg")
        b = split("/content/repo/datasets/test/a.jpg")
        self.assertEqual(split_fingerprint(a), split_fingerprint(b))
        self.assertNotEqual(split_fingerprint(a), split_fingerprint(split(b["splits"]["train"][0]["path"], 1)))

    def test_protocol_order_is_stable(self):
        self.assertEqual(protocol_fingerprint({"a": 1, "b": 2}), protocol_fingerprint({"b": 2, "a": 1}))
