"""Read-only, provenance-aware comparison of completed experiment summaries."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


SOURCES = (
    ("baseline", "results/baseline/baseline_results.csv"),
    ("teacher", "results/teachers/cnn/teacher_results.csv"),
    ("teacher", "results/teachers/vision_transformers/teacher_results.csv"),
    ("single_kd", "results/knowledge_distillation/kd_results.csv"),
    ("multi_kd", "results/multi_teacher_knowledge_distillation/multi_kd_results.csv"),
)


def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value)) or str(value).strip().lower() in {"", "nan", "none"}


def _value(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in row and not _missing(row[name]):
            return row[name]
    return None


def _number(value: Any) -> float | int | None:
    if _missing(value):
        return None
    try:
        number = float(value)
        if not math.isfinite(number):
            return None
        return int(number) if number.is_integer() else number
    except (TypeError, ValueError):
        return None


def _list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if _missing(value):
        return []
    try:
        parsed = json.loads(str(value))
        return parsed if isinstance(parsed, list) else []
    except (TypeError, json.JSONDecodeError):
        return []


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _resolve(root: Path, value: Any) -> Path | None:
    if _missing(value):
        return None
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metadata(root: Path, row: dict[str, Any]) -> tuple[dict[str, Any], Path | None]:
    run_dir = _resolve(root, row.get("run_dir"))
    explicit = _resolve(root, row.get("run_metadata_path"))
    history = _resolve(root, row.get("history_path"))
    candidates = [path for path in (
        explicit,
        run_dir / "run_metadata.json" if run_dir else None,
        history.parent / "run_metadata.json" if history else None,
    ) if path]
    config_path = _resolve(root, row.get("config_path"))
    if config_path:
        candidates.append(config_path)
    for path in candidates:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            return data, path
    return {}, None


def _class_metrics(root: Path, row: dict[str, Any]) -> list[dict[str, Any]]:
    path = _resolve(root, row.get("per_class_metrics_path"))
    if not path or not path.is_file():
        return []
    try:
        frame = pd.read_csv(path).sort_values("class_id", kind="stable")
    except (OSError, pd.errors.ParserError, KeyError):
        return []
    return frame.where(pd.notna(frame), None).to_dict("records")


def _eligible(row: dict[str, Any], metadata: dict[str, Any]) -> bool:
    config = metadata.get("config") if isinstance(metadata.get("config"), dict) else metadata
    training = config.get("training", config) if isinstance(config.get("training", config), dict) else {}
    limits = ("max_train_batches", "max_val_batches", "max_test_batches")
    if _bool(_value(metadata, "dry_run") or _value(training, "dry_run") or row.get("dry_run")):
        return False
    if any(not _missing(_value(metadata, name) or _value(training, name) or row.get(name)) for name in limits):
        return False
    status = str(_value(metadata, "status", "phase") or "completed").lower()
    return status in {"completed", "finished", "success"}


def _normalize(root: Path, role: str, source: Path, raw: dict[str, Any]) -> dict[str, Any]:
    metadata, metadata_path = _metadata(root, raw)
    protocol = metadata.get("protocol", {})
    protocol = protocol if isinstance(protocol, dict) else {"hash": protocol}
    classes = _value(metadata, "classes") or protocol.get("classes")
    config = metadata.get("config", metadata)
    exp = config.get("experiment", config)
    dataset_ref = metadata.get("dataset_ref") or exp.get("dataset_ref") or _value(raw, "dataset_ref")
    if isinstance(dataset_ref, str):
        try:
            dataset_ref = json.loads(dataset_ref)
        except ValueError:
            dataset_ref = None
    model = _value(raw, "student_model_name", "model_name")
    run_id = str(_value(raw, "mlflow_run_id") or _value(metadata, "attempt_id") or _value(raw, "run_name"))
    source_files = [source]
    if metadata_path:
        source_files.append(metadata_path)
    per_class_path = _resolve(root, raw.get("per_class_metrics_path"))
    if per_class_path and per_class_path.is_file():
        source_files.append(per_class_path)
    has_protocol = bool(
        _value(metadata, "dataset_fingerprint")
        and _value(metadata, "split_fingerprint")
        and classes
        and _value(metadata, "eval_transform")
    )
    return {
        "run_id": run_id,
        "mlflow_run_id": _value(raw, "mlflow_run_id"),
        "run_name": _value(raw, "run_name"),
        "experiment_id": _value(metadata, "experiment_id") or run_id,
        "attempt_id": _value(metadata, "attempt_id") or run_id,
        "parent_attempt_id": _value(metadata, "parent_attempt_id"),
        "role": role,
        "stage": _value(metadata, "stage") or _value(raw, "stage") or role,
        "dataset": _value(metadata, "dataset_name", "dataset") or _value(raw, "dataset_name"),
        "dataset_fingerprint": _value(metadata, "dataset_fingerprint"),
        "dataset_ref": dataset_ref,
        "identity_version": 2 if dataset_ref else 1,
        "provenance_level": "content_v2" if dataset_ref else "legacy",
        "feature_representation": _value(raw, "feature_representation", "feature_representations") or _value(config, "feature_representation", "feature_representations"),
        "model": model,
        "student": model,
        "teacher_models": _list(_value(raw, "teacher_models")) or ([raw["teacher_model_name"]] if not _missing(raw.get("teacher_model_name")) else []),
        "teacher_run_ids": _list(_value(raw, "teacher_run_ids")) or ([raw["teacher_run_id"]] if not _missing(raw.get("teacher_run_id")) else []),
        "kd_type": _value(raw, "kd_type"),
        "aggregation": _value(raw, "aggregation"),
        "seed": _number(_value(raw, "seed")),
        "val_macro_f1": _number(_value(raw, "best_val_macro_f1", "val_macro_f1")),
        "test_accuracy": _number(_value(raw, "test_accuracy")),
        "test_macro_f1": _number(_value(raw, "test_macro_f1")),
        "ensemble_test_accuracy": _number(_value(raw, "ensemble_test_accuracy")),
        "ensemble_test_macro_f1": _number(_value(raw, "ensemble_test_macro_f1")),
        "params": _number(_value(raw, "student_params", "params")),
        "teacher_inference_params": _number(_value(raw, "teacher_total_params", "teacher_params")),
        "adapter_params": _number(_value(raw, "adapter_params")),
        "macs": _number(_value(raw, "student_macs", "macs")),
        "estimated_flops": _number(_value(raw, "student_estimated_flops", "estimated_flops")),
        "model_size_mb": _number(_value(raw, "student_model_size_mb", "model_size_mb")),
        "latency_ms": _number(_value(raw, "student_inference_latency_ms", "inference_latency_ms")),
        "training_time_seconds": _number(_value(raw, "training_time_seconds", "total_training_time_seconds")),
        "split_hash": _value(metadata, "split_fingerprint") or protocol.get("split_fingerprint"),
        "protocol_hash": protocol.get("hash") or _value(metadata, "protocol_fingerprint"),
        "eval_fingerprint": _value(metadata, "eval_transform") or protocol.get("eval_transform"),
        "training_protocol": _value(metadata, "training_protocol"),
        "hardware": _value(metadata, "hardware"),
        "precision": _value(metadata, "precision"),
        "class_mapping": _list(classes),
        "compatibility": "verified" if has_protocol else "legacy_unknown",
        "eligible": _eligible(raw, metadata),
        "class_metrics": _class_metrics(root, raw),
        "source_files": [{"path": str(path.relative_to(root) if path.is_relative_to(root) else path), "sha256": _sha256(path)} for path in source_files if path.is_file()],
        "source_commit": _value(metadata, "source_commit"),
    }


def load_comparison_runs(root: Path) -> list[dict[str, Any]]:
    """Normalize eligible completed rows from the five existing summaries."""
    normalized: list[dict[str, Any]] = []
    for role, relative in SOURCES:
        source = root / relative
        if not source.is_file():
            continue
        try:
            frame = pd.read_csv(source)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            continue
        normalized.extend(_normalize(root, role, source, row) for row in frame.where(pd.notna(frame), None).to_dict("records"))
    eligible = [run for run in normalized if run["eligible"]]
    # Linear resume attempts are one experiment. Keep terminal branches so the
    # user must choose if an experiment has more than one completed leaf.
    result: list[dict[str, Any]] = []
    by_experiment: dict[str, list[dict[str, Any]]] = {}
    for run in eligible:
        by_experiment.setdefault(str(run["experiment_id"]), []).append(run)
    for attempts in by_experiment.values():
        parent_ids = {str(run.get("parent_attempt_id")) for run in attempts if run.get("parent_attempt_id")}
        leaves = [run for run in attempts if str(run.get("attempt_id")) not in parent_ids]
        for run in leaves or attempts:
            run["resume_branch_ambiguous"] = len(leaves) > 1
            result.append(run)
    return result


def _same_baseline_pool(run: dict[str, Any], target: dict[str, Any]) -> bool:
    return all(run.get(key) == target.get(key) for key in ("dataset", "dataset_ref", "student", "seed"))


def _evaluation_compatibility(a: dict[str, Any], b: dict[str, Any]) -> tuple[bool, str]:
    if a.get("compatibility") != "verified" or b.get("compatibility") != "verified":
        return False, "Protocol metadata is incomplete; compatibility is unknown."
    fields = ("dataset_ref", "dataset_fingerprint", "split_hash", "class_mapping", "eval_fingerprint")
    mismatches = [field for field in fields if a.get(field) != b.get(field)]
    return (not mismatches, "" if not mismatches else "Incompatible evaluation protocol: " + ", ".join(mismatches))


def _controlled_compatibility(a: dict[str, Any], b: dict[str, Any]) -> tuple[bool, str]:
    compatible, warning = _evaluation_compatibility(a, b)
    if not compatible:
        return False, warning
    if not _same_baseline_pool(a, b):
        return False, "Student or seed differs; this is not a controlled baseline/KD comparison."
    if not a.get("training_protocol") or not b.get("training_protocol"):
        return False, "Shared training protocol is incomplete; KD attribution is not verified."
    if a["training_protocol"] != b["training_protocol"]:
        return False, "Shared training settings differ; the performance delta cannot be attributed only to KD."
    return True, ""


def build_comparison(runs: list[dict[str, Any]], selected_ids: list[str], baseline_id: str | None) -> dict[str, Any]:
    """Build one deterministic report without inventing absent measurements."""
    index = {str(run["run_id"]): run for run in runs}
    selected = [index[run_id] for run_id in selected_ids if run_id in index]
    target = next((run for run in selected if run.get("role") in {"single_kd", "multi_kd"}), selected[0] if selected else None)
    candidates = [
        run for run in runs
        if run.get("role") == "baseline"
        and target
        and _same_baseline_pool(run, target)
        and _controlled_compatibility(run, target)[0]
    ]
    if baseline_id not in index:
        baseline_id = min(
            candidates,
            key=lambda run: (
                -(run["val_macro_f1"] if run.get("val_macro_f1") is not None else float("-inf")),
                str(run["run_id"]),
            ),
            default={},
        ).get("run_id")
    baseline = index.get(str(baseline_id)) if baseline_id else None
    if baseline and baseline.get("role") != "baseline":
        baseline = None
        baseline_id = None
    warnings: list[str] = []
    if any(run.get("compatibility") != "verified" for run in selected):
        warnings.append("One or more selected legacy runs lack dataset/protocol evidence; compatibility is unknown.")
    if not baseline and any(run.get("role") in {"single_kd", "multi_kd"} for run in selected):
        warnings.append("No controlled baseline was selected or found; baseline deltas are unavailable.")
    latency_runs = [run for run in selected if run.get("latency_ms") is not None]
    if len(latency_runs) > 1 and (
        any(not run.get("hardware") or not run.get("precision") for run in latency_runs)
        or len({json.dumps({"hardware": run.get("hardware"), "precision": run.get("precision")}, sort_keys=True) for run in latency_runs}) > 1
    ):
        warnings.append("Latency protocol or hardware differs/is unknown; recorded latency values are not directly comparable.")
    if target and target.get("resume_branch_ambiguous"):
        warnings.append("This experiment has multiple completed resume branches; the selected attempt is reported explicitly.")
    expanded = list(selected)
    for target_run in selected:
        if target_run.get("role") != "multi_kd":
            continue
        single_candidates = [run for run in runs if run.get("role") == "single_kd"
                             and _same_baseline_pool(run, target_run) and _evaluation_compatibility(run, target_run)[0]
                             and run.get("kd_type") == target_run.get("kd_type")
                             and set(run.get("teacher_run_ids", [])) and set(run["teacher_run_ids"]).issubset(set(target_run.get("teacher_run_ids", [])))]
        best_single = min(single_candidates, key=lambda run: (-(run.get("val_macro_f1") or 0), str(run["run_id"])), default=None)
        if best_single and best_single not in expanded:
            expanded.append(best_single)
    if baseline and baseline not in expanded:
        expanded.insert(0, baseline)
    requested_teachers = {str(value) for run in selected for value in run.get("teacher_run_ids", [])}
    for run in runs:
        if run.get("role") == "teacher" and (
            str(run.get("mlflow_run_id")) in requested_teachers or str(run.get("run_id")) in requested_teachers
        ) and run not in expanded:
            expanded.append(run)
    rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    baseline_classes = {str(row.get("class_name")): row for row in (baseline or {}).get("class_metrics", [])}
    for run in expanded:
        eval_compatible, warning = _evaluation_compatibility(baseline, run) if baseline and run is not baseline else (True, "")
        if warning and warning not in warnings:
            warnings.append(warning)
        controlled, controlled_warning = _controlled_compatibility(baseline, run) if baseline and run.get("role") in {"single_kd", "multi_kd"} else (True, "")
        if controlled_warning and controlled_warning not in warnings:
            warnings.append(controlled_warning)
        metric = run.get("test_macro_f1")
        delta = round(100 * (metric - baseline["test_macro_f1"]), 12) if eval_compatible and metric is not None and baseline and baseline.get("test_macro_f1") is not None else None
        compression = (
            run["teacher_inference_params"] / run["params"]
            if run.get("role") in {"single_kd", "multi_kd"}
            and run.get("teacher_inference_params") is not None
            and run.get("params") not in {None, 0}
            else None
        )
        rows.append({key: run.get(key) for key in ("run_id", "role", "model", "dataset_ref", "provenance_level", "feature_representation", "teacher_models", "teacher_run_ids", "kd_type", "aggregation", "seed", "val_macro_f1", "test_accuracy", "test_macro_f1", "params", "adapter_params", "macs", "estimated_flops", "model_size_mb", "latency_ms", "training_time_seconds")} | {"delta_macro_f1_pp": delta, "teacher_params_per_student_param": compression, "controlled_kd_comparison": controlled if run.get("role") in {"single_kd", "multi_kd"} else None})
        for item in run.get("class_metrics", []):
            name = str(item.get("class_name"))
            base_f1 = _number(baseline_classes.get(name, {}).get("f1"))
            f1 = _number(item.get("f1"))
            class_rows.append({"run_id": run["run_id"], **item, "delta_f1_pp": round(100 * (f1 - base_f1), 12) if eval_compatible and f1 is not None and base_f1 is not None else None})
        if run.get("role") == "multi_kd" and (run.get("ensemble_test_accuracy") is not None or run.get("ensemble_test_macro_f1") is not None):
            rows.append({
                "run_id": f"{run['run_id']}:ensemble", "role": "teacher_ensemble",
                "model": " + ".join(run.get("teacher_models", [])),
                "teacher_models": run.get("teacher_models", []), "teacher_run_ids": run.get("teacher_run_ids", []),
                "kd_type": run.get("kd_type"), "aggregation": run.get("aggregation"), "seed": run.get("seed"),
                "val_macro_f1": None, "test_accuracy": run.get("ensemble_test_accuracy"),
                "test_macro_f1": run.get("ensemble_test_macro_f1"), "delta_macro_f1_pp": (
                    round(100 * (run["ensemble_test_macro_f1"] - baseline["test_macro_f1"]), 12)
                    if eval_compatible and baseline and run.get("ensemble_test_macro_f1") is not None and baseline.get("test_macro_f1") is not None else None
                ),
                "params": None, "macs": None, "estimated_flops": None, "model_size_mb": None,
                "latency_ms": None, "training_time_seconds": None, "adapter_params": None,
                "teacher_params_per_student_param": None, "controlled_kd_comparison": None,
            })
    provenance = {
        "report_schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selected_ids": [run["run_id"] for run in selected],
        "baseline_id": baseline_id,
        "baseline_candidate_count": len(candidates),
        "selection_policy": "Highest validation macro-F1 among evaluation- and training-compatible runs with matching dataset, student, and seed; run ID breaks ties. Test metrics are never selection criteria.",
        "sources": {run["run_id"]: run.get("source_files", []) for run in expanded},
        "protocols": {run["run_id"]: {key: run.get(key) for key in ("dataset_ref", "identity_version", "provenance_level", "feature_representation", "dataset_fingerprint", "split_hash", "protocol_hash", "eval_fingerprint", "training_protocol", "class_mapping", "hardware", "precision", "source_commit")} for run in expanded},
    }
    return {"rows": rows, "class_rows": class_rows, "warnings": warnings, "provenance": provenance}


def _display(value: Any, column: str = "") -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "N/A"
    if isinstance(value, list):
        return ", ".join(map(str, value)) or "N/A"
    if isinstance(value, (float, int)) and column in {"val_macro_f1", "test_accuracy", "test_macro_f1", "precision", "recall", "f1"}:
        return f"{100 * value:.2f}"
    if isinstance(value, (float, int)) and column.endswith("_pp"):
        return f"{value:.2f}"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No data._"
    columns = list(rows[0])
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines.extend("| " + " | ".join(_display(row.get(column), column).replace("|", "\\|").replace("\n", " ") for column in columns) + " |" for row in rows)
    return "\n".join(lines)


def _latex_escape(value: Any, column: str = "") -> str:
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}"}
    return "".join(replacements.get(character, character) for character in _display(value, column))


def export_comparison(report: dict[str, Any], format: str) -> bytes:
    """Export the exact report object as Markdown, CSV, compact LaTeX, or JSON manifest."""
    format = format.lower()
    if format == "json":
        payload = {**report["provenance"], "warnings": report["warnings"], "rows": report["rows"], "class_rows": report["class_rows"]}
        return json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
    if format == "markdown":
        warnings = "\n".join(f"- {warning}" for warning in report["warnings"]) or "- None"
        deltas = "\n".join(
            f"- {row['run_id']}: {_display(row.get('delta_macro_f1_pp'), 'delta_macro_f1_pp')} percentage points versus baseline."
            for row in report["rows"] if row.get("delta_macro_f1_pp") is not None and row.get("role") != "baseline"
        ) or "- No evaluation-compatible baseline delta is available."
        text = f"# Experiment comparison\n\n## Main results\n\n{_markdown_table(report['rows'])}\n\n## Numerical comparison\n\n{deltas}\n\n## Per-class results\n\n{_markdown_table(report['class_rows'])}\n\n## Limitations\n\n{warnings}\n"
        return text.encode("utf-8")
    if format == "csv":
        output = io.StringIO(newline="")
        rows = [{"table": "main", **row} for row in report["rows"]] + [{"table": "per_class", **row} for row in report["class_rows"]]
        if rows:
            columns = list(dict.fromkeys(key for row in rows for key in row))
            writer = csv.DictWriter(output, fieldnames=columns)
            writer.writeheader(); writer.writerows(rows)
        return output.getvalue().encode("utf-8-sig")
    if format == "latex":
        columns = ("run_id", "role", "model", "test_accuracy", "test_macro_f1", "delta_macro_f1_pp")
        body = "\n".join(" & ".join(_latex_escape(row.get(column), column) for column in columns) + r" \\" for row in report["rows"])
        return ("\\begin{table*}[t]\n\\centering\n\\small\n\\begin{tabular}{lllrrr}\nRun & Role & Model & Accuracy (\\%) & Macro-F1 (\\%) & $\\Delta$ (pp) \\\\\n\\hline\n" + body + "\n\\end{tabular}\n\\caption{Experiment comparison.}\n\\end{table*}\n").encode("utf-8")
    raise ValueError(f"Unsupported export format: {format}")
