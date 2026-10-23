import hashlib
import json
from pathlib import Path
import shutil
import os
import subprocess
import tempfile
import unittest

from PIL import Image


class CleanPlantPathologyTests(unittest.TestCase):
    def fixture(self, root):
        images = root / "datasets/PlantPathology2021/train_images"
        images.mkdir(parents=True)
        rows = []
        for label_id, label in enumerate(("healthy", "rust")):
            for index in range(10):
                path = images / f"{label}-{index}.png"
                Image.new("RGB", (8, 8), (index * 20, label_id * 150, 7)).save(path)
                rows.append({"path": path.relative_to(root).as_posix(), "label": label, "label_id": label_id})
        duplicate = images / "wrong-rust.png"
        shutil.copyfile(images / "healthy-0.png", duplicate)
        rows.append({"path": duplicate.relative_to(root).as_posix(), "label": "rust", "label_id": 1})
        source = root / "splits/plantpathology2021_seed42_70_15_15.json"
        source.parent.mkdir()
        source.write_text(json.dumps({"classes": ["healthy", "rust"], "splits": {"train": rows, "val": [], "test": []}}))
        return source, images

    def test_conflicting_group_is_excluded_and_sources_unchanged(self):
        from scripts.prepare_clean_plantpathology import prepare_clean_dataset
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, images = self.fixture(root)
            before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in [source, *images.iterdir()]}
            result = prepare_clean_dataset(root)
            self.assertEqual(result["kept_count"], 19)
            self.assertEqual(result["excluded_count"], 2)
            self.assertEqual({r["label"] for r in result["excluded"]}, {"healthy", "rust"})
            for path, digest in before.items():
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
            ref = result["dataset_ref"]
            split = json.loads((root / f"datasets/.registry/{ref['dataset_id']}/{ref['revision']}/splits/{ref['split_id']}.json").read_text())
            hashes = [{row["content_sha256"] for row in rows} for rows in split["splits"].values()]
            self.assertTrue(all(not hashes[a] & hashes[b] for a, b in ((0, 1), (0, 2), (1, 2))))
            self.assertEqual(split["classes"], ["healthy", "rust"])
            self.assertEqual(prepare_clean_dataset(root)["dataset_ref"], ref)

    def test_modified_destination_is_not_overwritten(self):
        from scripts.prepare_clean_plantpathology import prepare_clean_dataset
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            prepare_clean_dataset(root)
            target = root / "datasets/PlantPathology2021Clean/images/rust/rust-1.png"
            target.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed|differs|mismatch"):
                prepare_clean_dataset(root)
            self.assertEqual(target.read_bytes(), b"changed")

    def test_same_label_duplicate_is_retained_once(self):
        from scripts.prepare_clean_plantpathology import prepare_clean_dataset
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, images = self.fixture(root)
            duplicate = images / "extra-rust.png"
            shutil.copyfile(images / "rust-0.png", duplicate)
            payload = json.loads(source.read_text())
            payload["splits"]["train"].append({"path": duplicate.relative_to(root).as_posix(), "label": "rust", "label_id": 1})
            source.write_text(json.dumps(payload))
            result = prepare_clean_dataset(root)
            self.assertEqual(result["kept_count"], 19)
            self.assertEqual(result["excluded_count"], 3)
            self.assertIn("duplicate_same_label", {r["reason"] for r in result["excluded"]})

    def test_invalid_source_label_ids_fail_before_publication(self):
        from scripts.prepare_clean_plantpathology import prepare_clean_dataset
        for label_id in (-1, True, 2):
            with self.subTest(label_id=label_id), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                source, _ = self.fixture(root)
                payload = json.loads(source.read_text())
                payload["splits"]["train"][0].update(label="rust", label_id=label_id)
                source.write_text(json.dumps(payload))
                with self.assertRaises(ValueError):
                    prepare_clean_dataset(root)
                self.assertFalse((root / "configs/datasets/plantpathology2021_clean.json").exists())

    def test_nested_destination_link_cannot_write_outside_clean_folder(self):
        from scripts.prepare_clean_plantpathology import prepare_clean_dataset
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            prepare_clean_dataset(root)
            linked = root / "datasets/PlantPathology2021Clean/images/rust"
            outside = root / "outside"
            shutil.move(str(linked), outside)
            shutil.move(str(outside / "rust-1.png"), root / "parked.png")
            if os.name == "nt":
                subprocess.run(["cmd", "/c", "mklink", "/J", str(linked), str(outside)], check=True, capture_output=True)
            else:
                linked.symlink_to(outside, target_is_directory=True)
            before = {p.name for p in outside.iterdir()}
            with self.assertRaises(ValueError):
                prepare_clean_dataset(root)
            self.assertEqual({p.name for p in outside.iterdir()}, before)
