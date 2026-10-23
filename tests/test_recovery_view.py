import unittest

from app.recovery_view import add_recovery_flags


class RecoveryViewTests(unittest.TestCase):
    def test_enabled_backup_is_added_without_mutating_source(self):
        source = ["python", "train.py"]
        command = add_recovery_flags(source, enabled=True, backup_dir="D:/backups")
        self.assertEqual(command[-2:], ["--backup-dir", "D:/backups"])
        self.assertEqual(source, ["python", "train.py"])

    def test_disabled_recovery_uses_explicit_cli_flag(self):
        self.assertEqual(
            add_recovery_flags(["python", "train.py"], enabled=False, backup_dir="ignored"),
            ["python", "train.py", "--no-recovery"],
        )


if __name__ == "__main__":
    unittest.main()
