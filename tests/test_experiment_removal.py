from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.experiment_removal import (
    ExperimentActiveError,
    ExperimentRemovalError,
    ExperimentRemovalService,
    InterruptedJobRemovalService,
    RemovalMode,
    find_jobs_for_run,
)


class FakeRunTrackingGateway:
    def __init__(self, *, fail_delete: bool = False) -> None:
        self.fail_delete = fail_delete
        self.deleted: list[tuple[str, str]] = []
        self.restored: list[tuple[str, str]] = []

    def delete_run(self, run_id: str, tracking_uri: str) -> bool:
        if self.fail_delete:
            raise RuntimeError("tracking backend unavailable")
        self.deleted.append((run_id, tracking_uri))
        return True

    def restore_run(self, run_id: str, tracking_uri: str) -> None:
        self.restored.append((run_id, tracking_uri))


class ExperimentRemovalServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.gateway = FakeRunTrackingGateway()
        self.service = ExperimentRemovalService(self.root, tracking_gateway=self.gateway)

        self.run_name = "appleleaf9_mobilenet_seed42_20260717"
        self.run_id = "run-123"
        self.tracking_uri = "sqlite:///mlflow.db"
        self.run_dir = self.root / "results/baseline/runs" / self.run_name
        self.checkpoint = self.root / "checkpoints/appleleaf9/mobilenet" / f"{self.run_name}.pt"
        self.run_dir.mkdir(parents=True)
        self.checkpoint.parent.mkdir(parents=True)
        (self.run_dir / "history.csv").write_text("epoch,loss\n1,0.1\n", encoding="utf-8")
        self.checkpoint.write_bytes(b"checkpoint")

        self.row = {
            "run_name": self.run_name,
            "mlflow_run_id": self.run_id,
            "tracking_uri": self.tracking_uri,
            "run_dir": self.run_dir.relative_to(self.root).as_posix(),
            "checkpoint_path": self.checkpoint.relative_to(self.root).as_posix(),
        }

        self.raw_summary = self.root / "results/baseline/baseline_results.csv"
        self.clean_summary = self.root / "results/baseline/student_baseline_clean_results.csv"
        self._write_csv(self.raw_summary, [self.row, self._other_row()])
        self._write_csv(self.clean_summary, [self.row, self._other_row()])

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_archive_removes_exact_run_and_preserves_recovery_data(self) -> None:
        self._create_finished_job()

        report = self.service.remove(self.row, RemovalMode.ARCHIVE)

        self.assertFalse(self.run_dir.exists())
        self.assertFalse(self.checkpoint.exists())
        self.assertEqual(["other-run"], self._run_names(self.raw_summary))
        self.assertEqual(["other-run"], self._run_names(self.clean_summary))
        self.assertEqual([(self.run_id, self.tracking_uri)], self.gateway.deleted)
        self.assertIsNotNone(report.archive_path)
        assert report.archive_path is not None
        self.assertTrue(report.archive_path.exists())
        self.assertTrue(
            (report.archive_path / "files" / self.run_dir.relative_to(self.root)).exists()
        )
        self.assertTrue(
            (report.archive_path / "csv_backups" / self.raw_summary.relative_to(self.root)).exists()
        )
        self.assertFalse((self.root / "runs/jobs/job-1.json").exists())
        self.assertFalse((self.root / "runs/logs/job-1.log").exists())

    def test_delete_removes_staging_archive_after_success(self) -> None:
        report = self.service.remove(self.row, RemovalMode.DELETE)

        self.assertIsNone(report.archive_path)
        self.assertFalse(self.run_dir.exists())
        self.assertFalse(self.checkpoint.exists())
        self.assertEqual(["other-run"], self._run_names(self.raw_summary))
        staging_root = self.root / "_archive/.experiment-removal-staging"
        self.assertFalse(staging_root.exists() and any(staging_root.iterdir()))

    def test_legacy_baseline_row_infers_run_directory_from_artifact_paths(self) -> None:
        (self.run_dir / "config.json").write_text("{}", encoding="utf-8")
        legacy_row = {
            key: value for key, value in self.row.items() if key != "run_dir"
        }
        legacy_row["history_path"] = (self.run_dir / "history.csv").relative_to(
            self.root
        ).as_posix()

        report = self.service.remove(legacy_row, RemovalMode.ARCHIVE)

        self.assertFalse(self.run_dir.exists())
        assert report.archive_path is not None
        archived_run_dir = report.archive_path / "files" / self.run_dir.relative_to(self.root)
        self.assertTrue((archived_run_dir / "config.json").exists())

    def test_active_matching_job_blocks_removal_before_any_mutation(self) -> None:
        self._create_job(status="running", include_run_id_in_metadata=True)

        with self.assertRaises(ExperimentActiveError):
            self.service.remove(self.row, RemovalMode.ARCHIVE)

        self.assertTrue(self.run_dir.exists())
        self.assertTrue(self.checkpoint.exists())
        self.assertEqual([self.run_name, "other-run"], self._run_names(self.raw_summary))
        self.assertEqual([], self.gateway.deleted)

    def test_external_artifact_path_is_rejected(self) -> None:
        external = self.root.parent / "do-not-delete.txt"
        external.write_text("important", encoding="utf-8")
        unsafe_row = {**self.row, "checkpoint_path": str(external)}

        with self.assertRaises(ExperimentRemovalError):
            self.service.remove(unsafe_row, RemovalMode.ARCHIVE)

        self.assertTrue(external.exists())
        self.assertTrue(self.run_dir.exists())
        self.assertEqual([], self.gateway.deleted)
        external.unlink()

    def test_tracking_failure_rolls_back_files_and_csv_changes(self) -> None:
        gateway = FakeRunTrackingGateway(fail_delete=True)
        service = ExperimentRemovalService(self.root, tracking_gateway=gateway)

        with self.assertRaises(ExperimentRemovalError):
            service.remove(self.row, RemovalMode.ARCHIVE)

        self.assertTrue(self.run_dir.exists())
        self.assertTrue(self.checkpoint.exists())
        self.assertEqual([self.run_name, "other-run"], self._run_names(self.raw_summary))
        self.assertEqual([self.run_name, "other-run"], self._run_names(self.clean_summary))

    def test_partial_move_failure_rolls_back_completed_moves(self) -> None:
        real_move = __import__("shutil").move
        move_calls = 0

        def fail_second_move(source: str, destination: str) -> str:
            nonlocal move_calls
            move_calls += 1
            if move_calls == 2:
                raise OSError("simulated disk failure")
            return real_move(source, destination)

        with patch("app.experiment_removal.shutil.move", side_effect=fail_second_move):
            with self.assertRaises(ExperimentRemovalError):
                self.service.remove(self.row, RemovalMode.ARCHIVE)

        self.assertTrue(self.run_dir.exists())
        self.assertTrue(self.checkpoint.exists())
        self.assertEqual([self.run_name, "other-run"], self._run_names(self.raw_summary))
        self.assertEqual([], self.gateway.deleted)

    def test_archive_commit_failure_restores_tracking_and_local_state(self) -> None:
        with patch.object(
            self.service, "_commit_staging", side_effect=OSError("archive unavailable")
        ):
            with self.assertRaises(ExperimentRemovalError):
                self.service.remove(self.row, RemovalMode.ARCHIVE)

        self.assertTrue(self.run_dir.exists())
        self.assertTrue(self.checkpoint.exists())
        self.assertEqual([self.run_name, "other-run"], self._run_names(self.raw_summary))
        self.assertEqual([(self.run_id, self.tracking_uri)], self.gateway.deleted)
        self.assertEqual([(self.run_id, self.tracking_uri)], self.gateway.restored)

    def test_find_jobs_for_run_returns_exact_log_match_with_status(self) -> None:
        self._create_finished_job()

        matches = find_jobs_for_run(self.root, self.run_id, self.run_name)

        self.assertEqual(1, len(matches))
        self.assertEqual("job-1", matches[0].job_id)
        self.assertEqual("finished", matches[0].status)
        self.assertEqual(self.root / "runs/logs/job-1.log", matches[0].log_path)

    def test_find_jobs_for_run_does_not_guess_from_dataset_or_model(self) -> None:
        jobs_dir = self.root / "runs/jobs"
        jobs_dir.mkdir(parents=True)
        unrelated_job = {
            "job_id": "job-unrelated",
            "status": "finished",
            "dataset": "appleleaf9",
            "model": "mobilenet",
        }
        (jobs_dir / "job-unrelated.json").write_text(
            json.dumps(unrelated_job), encoding="utf-8"
        )

        matches = find_jobs_for_run(self.root, self.run_id, self.run_name)

        self.assertEqual([], matches)

    def test_interrupted_job_removal_deletes_exact_context_artifacts(self) -> None:
        job_id = "stopped-job"
        partial_run = self.root / "results/teachers/cnn/runs/partial-run"
        partial_checkpoint = self.root / "checkpoints/teachers/cnn/partial.pt"
        partial_run.mkdir(parents=True)
        partial_checkpoint.parent.mkdir(parents=True)
        (partial_run / "config.json").write_text("{}", encoding="utf-8")
        partial_checkpoint.write_bytes(b"partial")
        config_snapshot = self.root / f"runs/configs/{job_id}.json"
        config_snapshot.parent.mkdir(parents=True)
        config_snapshot.write_text("{}", encoding="utf-8")
        self._create_interrupted_job(
            job_id,
            {
                "run_name": "partial-run",
                "mlflow_run_id": "partial-run-id",
                "tracking_uri": self.tracking_uri,
                "artifact_paths": {
                    "run_dir": partial_run.relative_to(self.root).as_posix(),
                    "checkpoint": partial_checkpoint.relative_to(self.root).as_posix(),
                },
            },
        )
        service = InterruptedJobRemovalService(
            self.root, tracking_gateway=self.gateway
        )

        report = service.remove_job(
            {
                "job_id": job_id,
                "status": "stopped",
                "log_path": f"runs/logs/{job_id}.log",
            },
            RemovalMode.DELETE,
        )

        self.assertFalse((self.root / f"runs/jobs/{job_id}.json").exists())
        self.assertFalse((self.root / f"runs/logs/{job_id}.log").exists())
        self.assertFalse(config_snapshot.exists())
        self.assertFalse(partial_run.exists())
        self.assertFalse(partial_checkpoint.exists())
        self.assertEqual([("partial-run-id", self.tracking_uri)], self.gateway.deleted)
        self.assertEqual(5, report.removed_paths)

    def test_interrupted_job_removal_preserves_completed_run_artifacts(self) -> None:
        job_id = "shared-job"
        self._create_interrupted_job(
            job_id,
            {
                "run_name": self.run_name,
                "artifact_paths": {
                    "run_dir": self.run_dir.relative_to(self.root).as_posix(),
                    "checkpoint": self.checkpoint.relative_to(self.root).as_posix(),
                },
            },
        )
        service = InterruptedJobRemovalService(
            self.root, tracking_gateway=self.gateway
        )

        report = service.remove_job(
            {
                "job_id": job_id,
                "status": "failed",
                "log_path": f"runs/logs/{job_id}.log",
            },
            RemovalMode.ARCHIVE,
        )

        self.assertTrue(self.run_dir.exists())
        self.assertTrue(self.checkpoint.exists())
        self.assertTrue(any("artifact korundu" in warning for warning in report.warnings))

    def _create_interrupted_job(
        self, job_id: str, context: dict[str, object]
    ) -> None:
        jobs_dir = self.root / "runs/jobs"
        logs_dir = self.root / "runs/logs"
        jobs_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        job = {
            "job_id": job_id,
            "status": "stopped",
            "log_path": f"runs/logs/{job_id}.log",
        }
        (jobs_dir / f"{job_id}.json").write_text(json.dumps(job), encoding="utf-8")
        (logs_dir / f"{job_id}.log").write_text(
            "JOB_RUN_CONTEXT " + json.dumps(context) + "\n",
            encoding="utf-8",
        )

    def _other_row(self) -> dict[str, str]:
        return {
            "run_name": "other-run",
            "mlflow_run_id": "run-456",
            "tracking_uri": self.tracking_uri,
            "run_dir": "results/baseline/runs/other-run",
            "checkpoint_path": "checkpoints/other-run.pt",
        }

    def _create_finished_job(self) -> None:
        self._create_job(status="finished", include_run_id_in_metadata=False)
        log_path = self.root / "runs/logs/job-1.log"
        log_path.write_text(
            f"Baseline run complete. MLflow run_id={self.run_id}\n", encoding="utf-8"
        )

    def _create_job(self, *, status: str, include_run_id_in_metadata: bool) -> None:
        jobs_dir = self.root / "runs/jobs"
        logs_dir = self.root / "runs/logs"
        jobs_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        job = {
            "job_id": "job-1",
            "status": status,
            "log_path": "runs/logs/job-1.log",
        }
        if include_run_id_in_metadata:
            job["mlflow_run_id"] = self.run_id
        (jobs_dir / "job-1.json").write_text(json.dumps(job), encoding="utf-8")
        (logs_dir / "job-1.log").touch()

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _run_names(path: Path) -> list[str]:
        with path.open(newline="", encoding="utf-8") as handle:
            return [row["run_name"] for row in csv.DictReader(handle)]


if __name__ == "__main__":
    unittest.main()
