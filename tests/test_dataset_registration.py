import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image


def make_dataset(root, name="Potato", count=7, presplit=False):
    candidate = root / "datasets" / name
    for label_index, label in enumerate(("Healthy", "Early_Blight", "Late_Blight")):
        for index in range(count):
            part = ("train" if index < count - 2 else "val" if index == count - 2 else "test") if presplit else "images"
            path = candidate / part / label / f"{index}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8), (label_index * 50, index * 10, 123)).save(path)
    return candidate


class DatasetRegistrationTests(unittest.TestCase):
    def test_top_level_images_junction_cannot_escape_root(self):
        from src.datasets.dataset_registration import scan_dataset
        import os
        import subprocess
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            outside = make_dataset(base / "outside")
            root = base / "project"
            candidate = root / "datasets/Potato"
            candidate.mkdir(parents=True)
            link = candidate / "images"
            if os.name == "nt":
                subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(outside / "images")], check=True, capture_output=True)
            else:
                link.symlink_to(outside / "images", target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "links|escapes"):
                scan_dataset(root, candidate, "imagefolder")

    def test_truncated_jpeg_must_be_decoded_before_registration(self):
        from src.datasets.dataset_registration import scan_dataset
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = make_dataset(root)
            image = candidate / "images/Healthy/truncated.jpg"
            Image.new("RGB", (32, 32), (111, 45, 12)).save(image)
            image.write_bytes(image.read_bytes()[:-2])
            with self.assertRaisesRegex(ValueError, "Unreadable image"):
                scan_dataset(root, candidate, "imagefolder")

    def test_empty_class_and_linked_content_are_rejected(self):
        from src.datasets.dataset_registration import scan_dataset
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = make_dataset(root)
            empty = candidate / "images/Empty"
            empty.mkdir()
            with self.assertRaisesRegex(ValueError, "Empty class"):
                scan_dataset(root, candidate, "imagefolder")
            empty.rmdir()
            import os
            linked = candidate / "images/Healthy/linked.png"
            try:
                os.symlink(candidate / "images/Healthy/0.png", linked)
            except OSError:
                return  # Windows may deny symlinks without Developer Mode.
            with self.assertRaisesRegex(ValueError, "links"):
                scan_dataset(root, candidate, "imagefolder")

    def test_concurrent_same_id_cannot_bind_two_roots(self):
        from concurrent.futures import ThreadPoolExecutor
        from src.datasets.dataset_registration import scan_dataset, register_dataset_revision
        from src.datasets.dataset_registry import DatasetDefinition
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [make_dataset(root, name=name) for name in ("Potato", "Other")]
            scans = [scan_dataset(root, path, "imagefolder") for path in paths]
            def register(index):
                try:
                    register_dataset_revision(root, DatasetDefinition("potato", "Potato", paths[index].relative_to(root).as_posix(), "imagefolder"), scans[index])
                    return "registered"
                except ValueError:
                    return "rejected"
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(register, (0, 1)))
            self.assertCountEqual(results, ["registered", "rejected"])

    def test_discover_register_idempotent_and_not_ready_until_split(self):
        from src.datasets.dataset_registration import discover_datasets, scan_dataset, register_dataset_revision
        from src.datasets.dataset_registry import DatasetDefinition, list_datasets
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            candidate = make_dataset(root)
            (root / "datasets/.registry").mkdir()
            self.assertEqual(discover_datasets(root), [candidate])
            scan = scan_dataset(root, candidate, "imagefolder")
            self.assertEqual(scan.classes, ["Early_Blight", "Healthy", "Late_Blight"])
            self.assertEqual(len(scan.samples), 21)
            definition = DatasetDefinition("potato", "Potato", "datasets/Potato", "imagefolder")
            ref = register_dataset_revision(root, definition, scan)
            self.assertEqual(ref, register_dataset_revision(root, definition, scan))
            self.assertFalse(list_datasets(root, include_legacy=False)[0].ready)
            self.assertEqual(json.loads((root / "configs/datasets/potato.json").read_text())["default_split"], None)

    def test_duplicates_corruption_empty_and_case_collision_rejected(self):
        from src.datasets.dataset_registration import scan_dataset
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            candidate = make_dataset(root)
            source = candidate / "images/Healthy/0.png"
            duplicate = candidate / "images/Early_Blight/duplicate.png"
            duplicate.write_bytes(source.read_bytes())
            with self.assertRaisesRegex(ValueError, "[Dd]uplicate"):
                scan_dataset(root, candidate, "imagefolder")
            duplicate.unlink()
            source.write_bytes(b"broken")
            with self.assertRaises(ValueError):
                scan_dataset(root, candidate, "imagefolder")

    def test_revision_ignores_location_but_detects_changed_bytes(self):
        from src.datasets.dataset_registration import scan_dataset
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first = make_dataset(root)
            second = make_dataset(root, "OtherName")
            a = scan_dataset(root, first, "imagefolder")
            b = scan_dataset(root, second, "imagefolder")
            self.assertEqual(a.revision, b.revision)
            Image.new("RGB", (8, 8), (255, 255, 255)).save(second / "images/Healthy/0.png")
            self.assertNotEqual(a.revision, scan_dataset(root, second, "imagefolder").revision)

    def test_root_escape_and_stale_registration_rejected(self):
        from src.datasets.dataset_registration import scan_dataset, register_dataset_revision
        from src.datasets.dataset_registry import DatasetDefinition
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            candidate = make_dataset(root)
            with self.assertRaises(ValueError):
                scan_dataset(root, root.parent, "imagefolder")
            scan = scan_dataset(root, candidate, "imagefolder")
            Image.new("RGB", (8, 8), (255, 255, 255)).save(candidate / "images/Healthy/0.png")
            with self.assertRaises(ValueError):
                register_dataset_revision(root, DatasetDefinition("potato", "Potato", "datasets/Potato", "imagefolder"), scan)


if __name__ == "__main__":
    unittest.main()
