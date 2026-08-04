"""Safely remove one completed experiment from local results and MLflow."""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Protocol


ARTIFACT_COLUMNS = (
    "run_dir",
    "checkpoint_path",
    "config_path",
    "history_path",
    "per_class_metrics_path",
    "confusion_matrix_csv_path",
    "confusion_matrix_png_path",
    "learning_curves_path",
    "import_receipt_path",
    "external_bundle_dir",
)
ACTIVE_JOB_STATUSES = frozenset({"queued", "running"})
INTERRUPTED_JOB_STATUSES = frozenset({"cancelled", "failed", "stopped"})
JOB_RUN_CONTEXT_PREFIX = "JOB_RUN_CONTEXT "


class RemovalMode(str, Enum):
    """Supported local artifact removal modes."""

    ARCHIVE = "archive"
    DELETE = "delete"


class ExperimentRemovalError(RuntimeError):
    """Raised when an experiment cannot be removed safely."""


class ExperimentActiveError(ExperimentRemovalError):
    """Raised when the selected experiment still has an active job."""


class RunTrackingGateway(Protocol):
    """Abstraction for the external run tracking lifecycle."""

    def delete_run(self, run_id: str, tracking_uri: str) -> bool:
        """Soft-delete a run and return whether its state changed."""

    def restore_run(self, run_id: str, tracking_uri: str) -> None:
        """Restore a previously deleted run during rollback."""


class MlflowRunTrackingGateway:
    """MLflow implementation kept behind a small, testable boundary."""

    def delete_run(self, run_id: str, tracking_uri: str) -> bool:
        from mlflow import MlflowClient

        client = MlflowClient(tracking_uri=tracking_uri)
        run = client.get_run(run_id)
        if run.info.lifecycle_stage == "deleted":
            return False
        client.delete_run(run_id)
        return True

    def restore_run(self, run_id: str, tracking_uri: str) -> None:
        from mlflow import MlflowClient

        MlflowClient(tracking_uri=tracking_uri).restore_run(run_id)


@dataclass(frozen=True)
class ExperimentIdentity:
    run_name: str
    run_id: str
    tracking_uri: str


@dataclass(frozen=True)
class RemovalReport:
    identity: ExperimentIdentity
    mode: RemovalMode
    removed_csv_rows: int
    removed_paths: int
    tracking_run_deleted: bool
    archive_path: Path | None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class InterruptedJobRemovalReport:
    job_id: str
    status: str
    removed_paths: int
    tracking_run_deleted: bool
    archive_path: Path | None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class RunJobMatch:
    """A job record linked to one experiment by exact run identity."""

    job_id: str
    status: str
    job_path: Path
    log_path: Path | None


def find_jobs_for_run(
    project_root: str | Path,
    run_id: str,
    run_name: str = "",
    *,
    jobs: list[Mapping[str, Any]] | None = None,
) -> list[RunJobMatch]:
    """Find jobs by exact metadata or an exact MLflow run ID in their log."""
    root = Path(project_root).resolve()
    jobs_dir = root / "runs/jobs"
    records = jobs if jobs is not None else _load_job_records(jobs_dir)
    matches: list[RunJobMatch] = []

    for job in records:
        job_id = _text(job.get("job_id"))
        if not job_id:
            continue
        job_path = (jobs_dir / f"{job_id}.json").resolve()
        log_path = _resolve_job_log_path(root, job.get("log_path"))
        metadata_match = (
            _text(job.get("mlflow_run_id")) == run_id
            or bool(run_name and _text(job.get("run_name")) == run_name)
        )
        log_match = bool(log_path and _file_contains(log_path, run_id))
        if metadata_match or log_match:
            matches.append(
                RunJobMatch(
                    job_id=job_id,
                    status=_text(job.get("status")) or "unknown",
                    job_path=job_path,
                    log_path=log_path,
                )
            )

    return matches


def _load_job_records(jobs_dir: Path) -> list[Mapping[str, Any]]:
    if not jobs_dir.exists():
        return []
    records: list[Mapping[str, Any]] = []
    for job_path in sorted(jobs_dir.glob("*.json"), reverse=True):
        try:
            records.append(json.loads(job_path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
    return records


def _resolve_job_log_path(root: Path, value: Any) -> Path | None:
    log_value = _text(value)
    if not log_value:
        return None
    path = Path(log_value)
    if not path.is_absolute():
        path = root / path
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError:
        return None
    if not relative.parts or relative.parts[0].lower() != "runs":
        return None
    return resolved


def _file_contains(path: Path, needle: str) -> bool:
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            return any(needle in line for line in handle)
    except OSError:
        return False


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text_value = str(value).strip()
    return "" if text_value.lower() == "nan" else text_value


@dataclass(frozen=True)
class _CsvMutation:
    path: Path
    fieldnames: tuple[str, ...]
    kept_rows: tuple[dict[str, str], ...]
    removed_rows: tuple[dict[str, str], ...]


class ExperimentRemovalService:
    """Coordinate reversible filesystem changes with MLflow run deletion."""

    def __init__(
        self,
        project_root: str | Path,
        *,
        tracking_gateway: RunTrackingGateway | None = None,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.tracking_gateway = tracking_gateway or MlflowRunTrackingGateway()

    def remove(self, row: Mapping[str, Any], mode: RemovalMode) -> RemovalReport:
        """Remove exactly one completed run and roll back on pre-commit failures."""
        identity = self._build_identity(row)
        csv_mutations = self._find_csv_mutations(identity)
        artifact_paths = self._artifact_paths(row)
        job_paths, active_job_ids = self._matching_job_paths(identity)
        if active_job_ids:
            joined_ids = ", ".join(active_job_ids)
            raise ExperimentActiveError(
                f"Deney aktif job ile bağlantılı ({joined_ids}); önce job'ı durdurun."
            )

        removable_paths = self._without_nested_paths([*artifact_paths, *job_paths])
        staging_path, archive_path = self._removal_paths(identity, mode)
        moved_paths: list[tuple[Path, Path]] = []
        csv_backups: list[tuple[Path, Path]] = []
        tracking_deleted = False
        committed_archive: Path | None = None
        warnings: tuple[str, ...] = ()

        try:
            staging_path.mkdir(parents=True, exist_ok=False)
            self._apply_csv_mutations(csv_mutations, staging_path, csv_backups)
            self._move_paths_to_staging(removable_paths, staging_path, moved_paths)
            tracking_deleted = self.tracking_gateway.delete_run(
                identity.run_id, identity.tracking_uri
            )
            self._write_manifest(
                staging_path,
                identity,
                mode,
                csv_mutations,
                [source for source, _ in moved_paths],
            )
            committed_archive, warnings = self._commit_staging(
                staging_path, archive_path, mode
            )
        except Exception as exc:
            rollback_errors = self._rollback(moved_paths, csv_backups)
            if tracking_deleted:
                try:
                    self.tracking_gateway.restore_run(identity.run_id, identity.tracking_uri)
                except Exception as restore_exc:  # pragma: no cover - defensive reporting
                    rollback_errors.append(f"MLflow restore failed: {restore_exc}")
            shutil.rmtree(staging_path, ignore_errors=True)
            rollback_note = (
                f" Rollback errors: {'; '.join(rollback_errors)}" if rollback_errors else ""
            )
            raise ExperimentRemovalError(f"Deney kaldırılamadı: {exc}.{rollback_note}") from exc

        return RemovalReport(
            identity=identity,
            mode=mode,
            removed_csv_rows=sum(len(mutation.removed_rows) for mutation in csv_mutations),
            removed_paths=len(moved_paths),
            tracking_run_deleted=tracking_deleted,
            archive_path=committed_archive,
            warnings=warnings,
        )

    def _build_identity(self, row: Mapping[str, Any]) -> ExperimentIdentity:
        run_name = self._required_text(row.get("run_name"), "run_name")
        run_id = self._required_text(row.get("mlflow_run_id"), "mlflow_run_id")
        tracking_uri = self._optional_text(row.get("tracking_uri"))
        if not tracking_uri:
            tracking_uri = (self.project_root / "mlruns").as_uri()
        return ExperimentIdentity(run_name, run_id, tracking_uri)

    def _find_csv_mutations(self, identity: ExperimentIdentity) -> list[_CsvMutation]:
        results_root = self.project_root / "results"
        if not results_root.exists():
            return []

        mutations: list[_CsvMutation] = []
        for path in sorted(results_root.rglob("*.csv")):
            try:
                with path.open(newline="", encoding="utf-8-sig") as handle:
                    reader = csv.DictReader(handle)
                    if not reader.fieldnames or "run_name" not in reader.fieldnames:
                        continue
                    rows = list(reader)
            except (OSError, csv.Error) as exc:
                raise ExperimentRemovalError(f"CSV okunamadı: {path}: {exc}") from exc

            removed = tuple(row for row in rows if self._matches_identity(row, identity))
            if not removed:
                continue
            kept = tuple(row for row in rows if not self._matches_identity(row, identity))
            mutations.append(_CsvMutation(path, tuple(reader.fieldnames), kept, removed))
        return mutations

    @staticmethod
    def _matches_identity(row: Mapping[str, Any], identity: ExperimentIdentity) -> bool:
        row_run_id = ExperimentRemovalService._optional_text(row.get("mlflow_run_id"))
        if row_run_id:
            return row_run_id == identity.run_id
        return ExperimentRemovalService._optional_text(row.get("run_name")) == identity.run_name

    def _artifact_paths(self, row: Mapping[str, Any]) -> list[Path]:
        paths: list[Path] = []
        run_artifact_paths: list[Path] = []
        has_explicit_run_dir = bool(self._optional_text(row.get("run_dir")))
        for column in ARTIFACT_COLUMNS:
            value = self._optional_text(row.get(column))
            if not value:
                continue
            path = Path(value)
            if not path.is_absolute():
                path = self.project_root / path
            allowed_roots = (
                ("runs",)
                if column in {"import_receipt_path", "external_bundle_dir"}
                else ("results", "checkpoints")
            )
            validated = self._validate_project_path(path, allowed_roots=allowed_roots)
            paths.append(validated)
            if column not in {
                "run_dir",
                "checkpoint_path",
                "import_receipt_path",
                "external_bundle_dir",
            }:
                run_artifact_paths.append(validated)

        if not has_explicit_run_dir:
            inferred_run_dir = self._infer_run_dir(run_artifact_paths)
            if inferred_run_dir:
                paths.append(inferred_run_dir)
        return [path for path in paths if path.exists()]

    def _infer_run_dir(self, artifact_paths: list[Path]) -> Path | None:
        """Infer legacy baseline run_dir only when every artifact shares one parent."""
        if not artifact_paths:
            return None
        parents = {path.parent for path in artifact_paths}
        if len(parents) != 1:
            return None
        candidate = parents.pop()
        validated = self._validate_project_path(candidate, allowed_roots=("results",))
        relative_parts = validated.relative_to(self.project_root).parts
        if "runs" not in {part.lower() for part in relative_parts}:
            return None
        return validated

    def _matching_job_paths(
        self, identity: ExperimentIdentity
    ) -> tuple[list[Path], list[str]]:
        matched_paths: list[Path] = []
        active_job_ids: list[str] = []
        matches = find_jobs_for_run(self.project_root, identity.run_id, identity.run_name)
        for match in matches:
            if match.status in ACTIVE_JOB_STATUSES:
                active_job_ids.append(match.job_id)
                continue
            if match.job_path.exists():
                matched_paths.append(match.job_path)
            if match.log_path and match.log_path.exists():
                matched_paths.append(match.log_path)
        return matched_paths, active_job_ids

    def _validate_project_path(self, path: Path, *, allowed_roots: tuple[str, ...]) -> Path:
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(self.project_root)
        except ValueError as exc:
            raise ExperimentRemovalError(f"Proje dışındaki yol kaldırılamaz: {path}") from exc
        if not relative.parts or relative.parts[0].lower() not in allowed_roots:
            raise ExperimentRemovalError(f"İzin verilmeyen deney yolu: {path}")
        return resolved

    @staticmethod
    def _without_nested_paths(paths: list[Path]) -> list[Path]:
        selected: list[Path] = []
        for path in sorted(set(paths), key=lambda item: len(item.parts)):
            if any(path == parent or path.is_relative_to(parent) for parent in selected):
                continue
            selected.append(path)
        return selected

    def _removal_paths(
        self, identity: ExperimentIdentity, mode: RemovalMode
    ) -> tuple[Path, Path | None]:
        token = uuid.uuid4().hex
        staging = self.project_root / "_archive/.experiment-removal-staging" / token
        if mode is RemovalMode.DELETE:
            return staging, None
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", identity.run_name).strip("._") or "run"
        archive = self.project_root / "_archive/experiment_removals" / f"{timestamp}_{slug}"
        return staging, archive

    def _apply_csv_mutations(
        self,
        mutations: list[_CsvMutation],
        staging_path: Path,
        backups: list[tuple[Path, Path]],
    ) -> None:
        for mutation in mutations:
            relative = mutation.path.relative_to(self.project_root)
            backup = staging_path / "csv_backups" / relative
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(mutation.path, backup)
            backups.append((mutation.path, backup))
            self._write_csv_atomically(mutation)

    @staticmethod
    def _write_csv_atomically(mutation: _CsvMutation) -> None:
        temporary = mutation.path.with_name(f".{mutation.path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=mutation.fieldnames)
                writer.writeheader()
                writer.writerows(mutation.kept_rows)
            temporary.replace(mutation.path)
        finally:
            temporary.unlink(missing_ok=True)

    def _move_paths_to_staging(
        self,
        paths: list[Path],
        staging_path: Path,
        moved: list[tuple[Path, Path]],
    ) -> None:
        for source in paths:
            destination = staging_path / "files" / source.relative_to(self.project_root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
            moved.append((source, destination))

    @staticmethod
    def _write_manifest(
        staging_path: Path,
        identity: ExperimentIdentity,
        mode: RemovalMode,
        mutations: list[_CsvMutation],
        moved_paths: list[Path],
    ) -> None:
        manifest = {
            "run_name": identity.run_name,
            "mlflow_run_id": identity.run_id,
            "tracking_uri": identity.tracking_uri,
            "mode": mode.value,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "csv_rows_removed": {
                str(mutation.path): len(mutation.removed_rows) for mutation in mutations
            },
            "moved_paths": [str(path) for path in moved_paths],
        }
        (staging_path / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @staticmethod
    def _rollback(
        moved_paths: list[tuple[Path, Path]], csv_backups: list[tuple[Path, Path]]
    ) -> list[str]:
        errors: list[str] = []
        for original, staged in reversed(moved_paths):
            try:
                original.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(staged), str(original))
            except OSError as exc:
                errors.append(f"{original}: {exc}")
        for original, backup in reversed(csv_backups):
            try:
                shutil.copy2(backup, original)
            except OSError as exc:
                errors.append(f"{original}: {exc}")
        return errors

    @staticmethod
    def _commit_staging(
        staging_path: Path, archive_path: Path | None, mode: RemovalMode
    ) -> tuple[Path | None, tuple[str, ...]]:
        if mode is RemovalMode.DELETE:
            try:
                shutil.rmtree(staging_path)
            except OSError as exc:
                warning = (
                    "Deney arayüzlerden kaldırıldı ancak bazı yerel dosyalar geçici "
                    f"temizleme klasöründe kaldı: {staging_path} ({exc})"
                )
                return None, (warning,)
            parent = staging_path.parent
            try:
                if parent.exists() and not any(parent.iterdir()):
                    parent.rmdir()
            except OSError:
                pass
            return None, ()
        assert archive_path is not None
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        staging_path.replace(archive_path)
        staging_parent = staging_path.parent
        try:
            if staging_parent.exists() and not any(staging_parent.iterdir()):
                staging_parent.rmdir()
        except OSError:
            pass
        return archive_path, ()

    @staticmethod
    def _required_text(value: Any, field_name: str) -> str:
        text = ExperimentRemovalService._optional_text(value)
        if not text:
            raise ExperimentRemovalError(f"Deney kaydında {field_name} eksik.")
        return text

    @staticmethod
    def _optional_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float) and math.isnan(value):
            return ""
        text = str(value).strip()
        return "" if text.lower() == "nan" else text


class InterruptedJobRemovalService(ExperimentRemovalService):
    """Remove an interrupted job and only artifacts carrying its exact identity."""

    def remove_job(
        self,
        job: Mapping[str, Any],
        mode: RemovalMode,
    ) -> InterruptedJobRemovalReport:
        job_id = self._required_text(job.get("job_id"), "job_id")
        status = self._required_text(job.get("status"), "status").lower()
        if status not in INTERRUPTED_JOB_STATUSES:
            raise ExperimentRemovalError(
                "Yalnızca cancelled, failed veya stopped job kalıntıları kaldırılabilir."
            )

        job_path = self._validate_project_path(
            self.project_root / "runs/jobs" / f"{job_id}.json",
            allowed_roots=("runs",),
        )
        config_snapshot_path = self._validate_project_path(
            self.project_root / "runs/configs" / f"{job_id}.json",
            allowed_roots=("runs",),
        )
        log_path = _resolve_job_log_path(self.project_root, job.get("log_path"))
        context = self._read_run_context(log_path)
        run_id = self._optional_text(job.get("mlflow_run_id")) or self._optional_text(
            context.get("mlflow_run_id")
        )
        run_name = self._optional_text(job.get("run_name")) or self._optional_text(
            context.get("run_name")
        )
        tracking_uri = self._optional_text(job.get("tracking_uri")) or self._optional_text(
            context.get("tracking_uri")
        )
        if not tracking_uri:
            tracking_uri = (self.project_root / "mlruns").as_uri()

        artifact_paths = self._context_artifact_paths(context)
        protected_paths = self._completed_result_paths()
        removable_artifacts = [
            path for path in artifact_paths if not self._is_claimed(path, protected_paths)
        ]
        skipped_artifacts = [
            path for path in artifact_paths if self._is_claimed(path, protected_paths)
        ]
        removable_paths = [
            path
            for path in (job_path, log_path, config_snapshot_path)
            if path and path.exists()
        ]
        removable_paths.extend(path for path in removable_artifacts if path.exists())
        removable_paths = self._without_nested_paths(removable_paths)

        identity = ExperimentIdentity(run_name or job_id, run_id or job_id, tracking_uri)
        staging_path, archive_path = self._removal_paths(identity, mode)
        moved_paths: list[tuple[Path, Path]] = []
        tracking_deleted = False
        warnings = [
            f"Tamamlanmış bir run tarafından kullanılan artifact korundu: {path}"
            for path in skipped_artifacts
        ]
        if not run_id:
            warnings.append(
                "Bu eski job için MLflow run ID kaydedilmemiş; MLflow kaydı otomatik "
                "olarak eşleştirilip silinemedi."
            )

        try:
            staging_path.mkdir(parents=True, exist_ok=False)
            self._move_paths_to_staging(removable_paths, staging_path, moved_paths)
            if run_id:
                tracking_deleted = self.tracking_gateway.delete_run(run_id, tracking_uri)
            self._write_interrupted_manifest(
                staging_path,
                job,
                mode,
                [source for source, _ in moved_paths],
                run_id,
            )
            committed_archive, commit_warnings = self._commit_staging(
                staging_path, archive_path, mode
            )
            warnings.extend(commit_warnings)
        except Exception as exc:
            rollback_errors = self._rollback(moved_paths, [])
            if tracking_deleted and run_id:
                try:
                    self.tracking_gateway.restore_run(run_id, tracking_uri)
                except Exception as restore_exc:  # pragma: no cover
                    rollback_errors.append(f"MLflow restore failed: {restore_exc}")
            shutil.rmtree(staging_path, ignore_errors=True)
            rollback_note = (
                f" Rollback errors: {'; '.join(rollback_errors)}" if rollback_errors else ""
            )
            raise ExperimentRemovalError(
                f"Job kalıntıları kaldırılamadı: {exc}.{rollback_note}"
            ) from exc

        return InterruptedJobRemovalReport(
            job_id=job_id,
            status=status,
            removed_paths=len(moved_paths),
            tracking_run_deleted=tracking_deleted,
            archive_path=committed_archive,
            warnings=tuple(warnings),
        )

    def _read_run_context(self, log_path: Path | None) -> Mapping[str, Any]:
        if not log_path or not log_path.exists():
            return {}
        try:
            for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
                if not line.startswith(JOB_RUN_CONTEXT_PREFIX):
                    continue
                value = json.loads(line.removeprefix(JOB_RUN_CONTEXT_PREFIX))
                return value if isinstance(value, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}
        return {}

    def _context_artifact_paths(self, context: Mapping[str, Any]) -> list[Path]:
        artifact_values = context.get("artifact_paths", {})
        if not isinstance(artifact_values, Mapping):
            return []
        paths: list[Path] = []
        for value in artifact_values.values():
            text = self._optional_text(value)
            if not text:
                continue
            path = Path(text)
            if not path.is_absolute():
                path = self.project_root / path
            paths.append(
                self._validate_project_path(path, allowed_roots=("results", "checkpoints"))
            )
        return paths

    def _completed_result_paths(self) -> set[Path]:
        claimed: set[Path] = set()
        results_root = self.project_root / "results"
        if not results_root.exists():
            return claimed
        for csv_path in results_root.rglob("*.csv"):
            try:
                with csv_path.open(newline="", encoding="utf-8-sig") as handle:
                    for row in csv.DictReader(handle):
                        for column in ARTIFACT_COLUMNS:
                            value = self._optional_text(row.get(column))
                            if not value:
                                continue
                            path = Path(value)
                            claimed.add(
                                (path if path.is_absolute() else self.project_root / path).resolve()
                            )
            except (OSError, csv.Error):
                continue
        return claimed

    @staticmethod
    def _is_claimed(candidate: Path, claimed: set[Path]) -> bool:
        resolved = candidate.resolve()
        return any(
            resolved == path
            or resolved.is_relative_to(path)
            or path.is_relative_to(resolved)
            for path in claimed
        )

    @staticmethod
    def _write_interrupted_manifest(
        staging_path: Path,
        job: Mapping[str, Any],
        mode: RemovalMode,
        moved_paths: list[Path],
        run_id: str,
    ) -> None:
        manifest = {
            "job_id": job.get("job_id"),
            "status": job.get("status"),
            "mlflow_run_id": run_id,
            "mode": mode.value,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "moved_paths": [str(path) for path in moved_paths],
        }
        (staging_path / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
