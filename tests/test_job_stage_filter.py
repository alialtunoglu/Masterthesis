"""Each launcher page should show its own jobs, not every job on disk."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from app.experiment_runner import jobs_for_stage


LEGACY_BASELINE_JOB = {"job_id": "legacy", "command": ["python", "src/training/train_baseline.py"]}


class JobsForStageTests(unittest.TestCase):
    def test_only_the_requested_stage_is_returned(self):
        jobs = [
            {"job_id": "a", "stage": "teacher_cnn"},
            {"job_id": "b", "stage": "knowledge_distillation"},
            {"job_id": "c", "stage": "teacher_cnn"},
        ]
        self.assertEqual(
            [job["job_id"] for job in jobs_for_stage("teacher_cnn", jobs)],
            ["a", "c"],
        )

    def test_baseline_keeps_jobs_queued_before_the_stage_key_existed(self):
        """The baseline page never wrote a stage, so those rows carry none."""
        jobs = [LEGACY_BASELINE_JOB, {"job_id": "kd", "stage": "knowledge_distillation"}]
        self.assertEqual(
            [job["job_id"] for job in jobs_for_stage("baseline", jobs)],
            ["legacy"],
        )

    def test_a_stageless_job_is_not_claimed_by_other_stages(self):
        self.assertEqual(jobs_for_stage("teacher_cnn", [LEGACY_BASELINE_JOB]), [])

    def test_reads_from_disk_when_no_jobs_are_passed(self):
        self.assertIsInstance(jobs_for_stage("baseline"), list)


if __name__ == "__main__":
    unittest.main()
