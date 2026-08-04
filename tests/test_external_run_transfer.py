import json
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path

import pandas as pd

from app.external_run_import import ExternalRunImportError, import_external_bundle
from app.experiment_removal import ExperimentRemovalService, RemovalMode
from src.tracking.external_bundle import create_external_bundle


class ExternalRunTransferTests(unittest.TestCase):
    def test_round_trip_imports_metrics_and_optional_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            target = base / "target"
            artifact = source / "results/baseline/runs/demo/history.csv"
            checkpoint = source / "checkpoints/demo.pt"
            artifact.parent.mkdir(parents=True)
            checkpoint.parent.mkdir(parents=True)
            split = source / "splits/appleleaf9.json"
            split.parent.mkdir(parents=True)
            artifact.write_text("epoch,loss\n1,0.2\n", encoding="utf-8")
            checkpoint.write_bytes(b"checkpoint")
            split.write_text('{"classes": ["a", "b"]}', encoding="utf-8")
            row = {
                "run_name": "demo",
                "mlflow_run_id": "remote-run",
                "stage": "baseline",
                "dataset_name": "appleleaf9",
                "model_name": "mobilenet_v3_small",
                "test_accuracy": 0.9,
                "test_macro_f1": 0.8,
                "history_path": "results/baseline/runs/demo/history.csv",
                "checkpoint_path": "checkpoints/demo.pt",
                "split_file": "splits/appleleaf9.json",
            }
            outputs = create_external_bundle(
                source, row, source_commit="a" * 40, output_dir=base / "out"
            )
            target_split = target / "splits/appleleaf9.json"
            target_split.parent.mkdir(parents=True)
            shutil.copy2(split, target_split)
            inbox = target / "runs/import_inbox" / outputs.bundle_id
            inbox.mkdir(parents=True)
            shutil.copy2(outputs.archive, inbox / outputs.archive.name)
            shutil.copy2(outputs.checkpoint, inbox / outputs.checkpoint.name)

            receipt = import_external_bundle(
                target,
                inbox,
                register_run=lambda manifest, paths: "local-run",
            )
            self.assertEqual(receipt["mlflow_run_id"], "local-run")
            imported = pd.read_csv(target / "results/baseline/baseline_results.csv")
            self.assertEqual(imported.iloc[-1]["external_bundle_id"], outputs.bundle_id)
            self.assertEqual(
                imported.iloc[-1]["import_receipt_path"],
                f"runs/imported/{outputs.bundle_id}.json",
            )
            self.assertTrue((target / "checkpoints/demo.pt").exists())
            second = import_external_bundle(
                target, inbox, register_run=lambda *_: self.fail("registered twice")
            )
            self.assertEqual(second, receipt)
            self.assertEqual(len(pd.read_csv(target / "results/baseline/baseline_results.csv")), 1)

            class TrackingGateway:
                def delete_run(self, run_id, tracking_uri):
                    return True

                def restore_run(self, run_id, tracking_uri):
                    return None

            imported_row = pd.read_csv(
                target / "results/baseline/baseline_results.csv"
            ).iloc[-1].to_dict()
            ExperimentRemovalService(
                target, tracking_gateway=TrackingGateway()
            ).remove(imported_row, RemovalMode.DELETE)
            self.assertFalse(inbox.exists())
            self.assertFalse((target / f"runs/imported/{outputs.bundle_id}.json").exists())
            self.assertFalse((target / "checkpoints/demo.pt").exists())

    def test_rejects_archive_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inbox = root / "runs/import_inbox/evil"
            inbox.mkdir(parents=True)
            archive = inbox / "evil.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("../outside.txt", "bad")
                bundle.writestr("manifest.json", json.dumps({"bundle_id": "evil"}))
            with self.assertRaises(ExternalRunImportError):
                import_external_bundle(root, inbox, register_run=lambda *_: "unused")
            self.assertFalse((root / "outside.txt").exists())

    def test_rejects_unexpected_inbox_file(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            artifact = source / "results/baseline/runs/demo/history.csv"
            artifact.parent.mkdir(parents=True)
            split = source / "splits/appleleaf9.json"
            split.parent.mkdir(parents=True)
            artifact.write_text("epoch,loss\n1,0.2\n", encoding="utf-8")
            split.write_text('{"classes": ["a", "b"]}', encoding="utf-8")
            outputs = create_external_bundle(
                source,
                {
                    "run_name": "demo",
                    "stage": "baseline",
                    "dataset_name": "appleleaf9",
                    "model_name": "mobilenet_v3_small",
                    "history_path": "results/baseline/runs/demo/history.csv",
                    "split_file": "splits/appleleaf9.json",
                },
                source_commit="a" * 40,
                output_dir=base / "out",
            )
            target = base / "target"
            target_split = target / "splits/appleleaf9.json"
            target_split.parent.mkdir(parents=True)
            shutil.copy2(split, target_split)
            inbox = target / "runs/import_inbox" / outputs.bundle_id
            inbox.mkdir(parents=True)
            shutil.copy2(outputs.archive, inbox / outputs.archive.name)
            (inbox / "unexpected.txt").write_text("no", encoding="utf-8")
            with self.assertRaises(ExternalRunImportError):
                import_external_bundle(target, inbox, register_run=lambda *_: "unused")


if __name__ == "__main__":
    unittest.main()
