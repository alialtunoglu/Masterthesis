import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch
from unittest import mock

from src.training.recovery import RecoverySession, write_generation, load_generation, mirror_generation, validate_resume, recovery_metadata


class RecoveryTests(unittest.TestCase):
    def test_recovery_loader_disables_persistent_workers_and_owns_generator(self):
        from src.datasets.dataloaders import create_dataloaders

        class Dummy(torch.utils.data.TensorDataset):
            classes = ["a"]

        dataset = Dummy(torch.zeros(4, 3, 2, 2), torch.zeros(4, dtype=torch.long))
        with mock.patch("src.datasets.dataloaders.SplitJsonImageDataset", return_value=dataset):
            train, validation, test, _ = create_dataloaders(Path("split.json"), 2, 2, 2, recovery_seed=42)
        self.assertFalse(train.persistent_workers)
        self.assertIsNotNone(train.worker_init_fn)
        self.assertIsNotNone(train.generator)
        self.assertNotEqual(train.generator.initial_seed(), validation.generator.initial_seed())
        self.assertNotEqual(validation.generator.initial_seed(), test.generator.initial_seed())

    def test_kd_recovery_settings_survive_config_round_trip(self):
        from training.kd_config import load_kd_config, merge_kd_config
        config = load_kd_config({'experiment': {}, 'training': {}, 'distillation': {},
                                 'recovery': {'enabled': False, 'backup_dir': 'backup'}})
        self.assertFalse(config.to_dict()['recovery']['enabled'])
        cleared = merge_kd_config(config, {"recovery": {"enabled": True, "backup_dir": None}})
        self.assertTrue(cleared.recovery.enabled)
        self.assertIsNone(cleared.recovery.backup_dir)

    def test_multi_kd_recovery_override_is_not_dropped(self):
        from training.multi_kd_config import load_multi_kd_config, merge_multi_kd_config
        path = next(Path("configs/multi_teacher_knowledge_distillation").glob("**/*.json"))
        config = load_multi_kd_config(path)
        updated = merge_multi_kd_config(config, {"recovery": {"enabled": False, "backup_dir": None}})
        self.assertFalse(updated.recovery.enabled)

    def test_metadata_ignores_transport_but_rejects_hyperparameter_change(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split = root / 'split.json'
            split.write_text(json.dumps({'dataset_name': 'test', 'classes': ['a'],
                                         'splits': {'train': [{'path': 'datasets/a.jpg', 'label_id': 0}]}}))
            a = recovery_metadata({'dataset_name': 'test', 'learning_rate': .01}, split, root)
            self.assertEqual(a["dataset_name"], "test")
            b = recovery_metadata({'dataset_name': 'test', 'learning_rate': .01, 'device': 'cuda'}, split, root)
            validate_resume(a, b)
            c = recovery_metadata({'dataset_name': 'test', 'learning_rate': .02}, split, root)
            with self.assertRaises(ValueError):
                validate_resume(a, c)

    def test_teacher_dependencies_keep_distinct_runs_of_same_architecture(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split = root / "split.json"
            split.write_text(json.dumps({"dataset_name": "test", "classes": ["a"], "splits": {"train": []}}))
            first, second = root / "first.pt", root / "second.pt"
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            metadata = recovery_metadata({
                "experiment": {"dataset_name": "test"},
                "teachers": [
                    {"model_name": "resnet50", "run_id": "run-a", "checkpoint_path": "first.pt"},
                    {"model_name": "resnet50", "run_id": "run-b", "checkpoint_path": "second.pt"},
                ],
                "training": {},
            }, split, root)
            self.assertEqual(len(metadata["teacher_hashes"]), 2)
            self.assertEqual([item["run_id"] for item in metadata["teacher_hashes"]], ["run-a", "run-b"])
            self.assertNotEqual(metadata["teacher_hashes"][0]["sha256"], metadata["teacher_hashes"][1]["sha256"])

    def test_post_backup_duration_is_available_to_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = torch.nn.Linear(1, 1)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(torch.zeros(1, 1), torch.zeros(1, dtype=torch.long)),
                generator=torch.Generator().manual_seed(1),
            )
            source = RecoverySession(root / "source/recovery", {"protocol": "same"}, backup_dir=root / "backup")
            source.save(1, {"history": [], "best_epoch": None}, model, optimizer, None, None, None, loader, root / "best.pt")
            self.assertGreater(source.backup_seconds, 0.0)
            resumed = RecoverySession(root / "resumed/recovery", {"protocol": "same"}, resume_from=source.last_generation)
            resumed.restore(model, optimizer, None, None, None, loader, root / "resumed-best.pt")
            self.assertAlmostEqual(resumed.backup_seconds, source.backup_seconds)

            manifest = json.loads((source.last_generation / "manifest.json").read_text())
            self.assertIn("backup_status.json", manifest["files"])
            timing = source.last_generation / "backup_status.json"
            timing.write_text('{"epoch":1,"backup_seconds_total":999}')
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                load_generation(source.last_generation)
            timing.write_text('{"epoch":1,"backup_seconds_total":NaN}')
            manifest["files"][timing.name] = hashlib.sha256(timing.read_bytes()).hexdigest()
            (source.last_generation / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "backup timing"):
                load_generation(source.last_generation)

    def test_round_trip_and_retention(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for epoch in range(1, 4):
                path = write_generation(root, {"schema_version": 1, "epoch": epoch,
                                               "model": {"w": torch.tensor([epoch])}})
            self.assertEqual(load_generation(path)["epoch"], 3)
            self.assertEqual(len(list(root.glob("epoch-*"))), 2)

    def test_corrupt_snapshot_rejected_and_previous_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = write_generation(root, {"schema_version": 1, "epoch": 1})
            second = write_generation(root, {"schema_version": 1, "epoch": 2})
            (second / "state.pt").write_bytes(b"broken")
            with self.assertRaises(ValueError):
                load_generation(second)
            self.assertEqual(load_generation(first)["epoch"], 1)

    def test_mirror_is_independent_and_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = write_generation(root / "local", {"schema_version": 1, "epoch": 7})
            copy = mirror_generation(source, root / "backup")
            self.assertEqual(load_generation(copy)["epoch"], 7)
            (source / "state.pt").write_bytes(b"broken")
            self.assertEqual(load_generation(copy)["epoch"], 7)

    def test_protocol_difference_is_not_ignored(self):
        with self.assertRaisesRegex(ValueError, "protocol"):
            validate_resume({"protocol": "a"}, {"protocol": "b"})
