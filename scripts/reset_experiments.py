"""Reset generated experiment outputs while preserving datasets and source code."""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path
from typing import NamedTuple


CRITICAL_DIRS = ["datasets", "splits", "src", "configs", "docs", ".agent"]

# Directories the training entry points actually write to.
CURRENT_RESULT_DIRS = [
    "results/baseline",
    "results/teachers",
    "results/knowledge_distillation",
    "results/multi_teacher_knowledge_distillation",
    "results/quantization",
]

# Directories used by earlier naming schemes. They are still cleaned so a reset
# leaves nothing behind, but they are never recreated.
LEGACY_RESULT_DIRS = [
    "results/vit_teachers",
    "results/kd_single",
    "results/kd_multi",
    "results/experiments",
]

OUTPUT_PATHS = [
    "mlruns",
    "mlflow.db",
    "mlruns.db",
    "checkpoints",
    "runs/jobs",
    "runs/logs",
    *CURRENT_RESULT_DIRS,
    *LEGACY_RESULT_DIRS,
]

RECREATE_DIRS = [
    *CURRENT_RESULT_DIRS,
    "checkpoints",
    "runs/jobs",
    "runs/logs",
]


class ResetResult(NamedTuple):
    affected: list[str]
    skipped: list[str]


def find_project_root() -> Path:
    """Find the project root from this script location."""
    return Path(__file__).resolve().parents[1]


def validate_project_root(project_root: Path) -> None:
    """Fail fast if the script is not running inside the expected project."""
    missing = [name for name in CRITICAL_DIRS if not (project_root / name).exists()]
    if missing:
        raise RuntimeError(
            "Project root validation failed. Missing critical paths: "
            + ", ".join(missing)
            + f"\nResolved root: {project_root}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Archive or delete generated experiment outputs."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--archive",
        action="store_true",
        default=True,
        help="Move generated outputs under _archive. This is the default.",
    )
    mode.add_argument(
        "--delete",
        action="store_true",
        help="Delete generated outputs without archiving.",
    )
    parser.add_argument("--yes", action="store_true", help="Run without confirmation prompt.")
    parser.add_argument(
        "--keep-dataset-analysis",
        action="store_true",
        default=True,
        help="Preserve results/dataset_analysis. Enabled by default.",
    )
    return parser.parse_args()


def confirm_or_exit(args: argparse.Namespace, existing: list[Path], project_root: Path) -> None:
    """Ask for confirmation unless --yes was provided."""
    if args.yes:
        return
    mode = "delete" if args.delete else "archive"
    print(f"Project root: {project_root}")
    print(f"Mode: {mode}")
    print("Existing experiment output paths:")
    for path in existing:
        print(f"  - {path.relative_to(project_root).as_posix()}")
    answer = input("Continue? Type 'yes' to proceed: ").strip().lower()
    if answer != "yes":
        raise SystemExit("Aborted by user.")


def archive_path(project_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return project_root / "_archive" / f"experiment_reset_{timestamp}"


def move_to_archive(
    paths: list[Path],
    project_root: Path,
    destination_root: Path,
) -> ResetResult:
    """Move output paths into the archive root, preserving relative paths."""
    moved: list[str] = []
    skipped: list[str] = []
    destination_root.mkdir(parents=True, exist_ok=True)
    for source in paths:
        relative = source.relative_to(project_root)
        target = destination_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(source), str(target))
        except PermissionError as exc:
            skipped.append(f"{relative.as_posix()} ({exc})")
            if target.exists() and source.exists() and source.is_file():
                target.unlink(missing_ok=True)
            continue
        moved.append(relative.as_posix())
    return ResetResult(moved, skipped)


def delete_paths(paths: list[Path], project_root: Path) -> ResetResult:
    """Delete output paths."""
    deleted: list[str] = []
    skipped: list[str] = []
    for path in paths:
        relative = path.relative_to(project_root)
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        except PermissionError as exc:
            skipped.append(f"{relative.as_posix()} ({exc})")
            continue
        deleted.append(relative.as_posix())
    return ResetResult(deleted, skipped)


def existing_output_paths(project_root: Path) -> list[Path]:
    """Return the generated output paths that currently exist under the project."""
    return [
        project_root / relative
        for relative in OUTPUT_PATHS
        if (project_root / relative).exists()
    ]


def recreate_empty_dirs(project_root: Path) -> list[str]:
    """Recreate the empty output folder structure required for future experiments."""
    recreated: list[str] = []
    for relative in RECREATE_DIRS:
        path = project_root / relative
        path.mkdir(parents=True, exist_ok=True)
        recreated.append(relative)
    return recreated


def main() -> None:
    args = parse_args()
    project_root = find_project_root()
    validate_project_root(project_root)

    existing_paths = existing_output_paths(project_root)
    missing_paths = [relative for relative in OUTPUT_PATHS if not (project_root / relative).exists()]

    print(f"Project root: {project_root}")
    print("Will preserve: datasets, splits, src, configs, docs, .agent")
    if args.keep_dataset_analysis:
        print("Will preserve: results/dataset_analysis")

    if existing_paths:
        print("Existing experiment output paths:")
        for path in existing_paths:
            print(f"  - {path.relative_to(project_root).as_posix()}")
    else:
        print("No experiment output paths found.")

    if missing_paths:
        print("Missing output paths:")
        for relative in missing_paths:
            print(f"  - {relative}")

    confirm_or_exit(args, existing_paths, project_root)

    archive_root: Path | None = None
    result: ResetResult
    if args.delete:
        result = delete_paths(existing_paths, project_root)
        print("Deleted paths:")
    else:
        archive_root = archive_path(project_root)
        result = move_to_archive(existing_paths, project_root, archive_root)
        print(f"Archive root: {archive_root}")
        print("Archived paths:")
    for relative in result.affected:
        print(f"  - {relative}")

    if result.skipped:
        print("Skipped locked paths:")
        for relative in result.skipped:
            print(f"  - {relative}")
        print("Close MLflow UI, Streamlit jobs, or any process using these files, then rerun reset.")

    recreated = recreate_empty_dirs(project_root)
    print("Recreated empty directories:")
    for relative in recreated:
        print(f"  - {relative}")


if __name__ == "__main__":
    main()
