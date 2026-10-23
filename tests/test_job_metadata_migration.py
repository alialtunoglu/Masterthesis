import json
import tempfile
import unittest
from pathlib import Path


class JobMetadataMigrationTests(unittest.TestCase):
    def test_dry_run_then_backed_up_idempotent_update(self):
        from scripts.backfill_job_metadata import backfill_job_metadata
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "runs/jobs/legacy.json"
            path.parent.mkdir(parents=True)
            original = b'{"job_id":"legacy","model":"resnet18","status":"finished","command":["python","src/training/train_baseline.py"],"metric":0.91}'
            path.write_bytes(original)
            dry = backfill_job_metadata(root)
            self.assertEqual(dry["candidate_count"], 1)
            self.assertEqual(path.read_bytes(), original)
            self.assertFalse((root / "runs/metadata_backups").exists())
            report = backfill_job_metadata(root, apply=True)
            updated = json.loads(path.read_text())
            self.assertEqual(updated["stage"], "baseline")
            self.assertEqual(updated["model_role"], "student")
            self.assertEqual(updated["architecture_family"], "cnn")
            for key, value in json.loads(original).items():
                self.assertEqual(updated[key], value)
            self.assertEqual((Path(report["backup_dir"]) / "jobs/legacy.json").read_bytes(), original)
            self.assertEqual(backfill_job_metadata(root, apply=True)["candidate_count"], 0)

    def test_unknown_conflicting_and_active_jobs_are_not_modified(self):
        from scripts.backfill_job_metadata import backfill_job_metadata
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            jobs = root / "runs/jobs"
            jobs.mkdir(parents=True)
            records = {
                "unknown": {"command": ["python", "custom.py"], "model": "resnet18"},
                "conflicting": {"stage": "teacher_cnn", "command": ["python", "train_baseline.py"], "model": "resnet18"},
                "active": {"stage": "baseline", "model": "resnet18", "status": "running"},
            }
            for name, record in records.items():
                (jobs / f"{name}.json").write_text(json.dumps(record))
            report = backfill_job_metadata(root, apply=True)
            self.assertEqual(report["updated_count"], 0)
            self.assertEqual(len(report["skipped"]), 3)
            for name, record in records.items():
                self.assertEqual(json.loads((jobs / f"{name}.json").read_text()), record)


if __name__ == "__main__":
    unittest.main()
