# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tracking and query utilities for benchmark batches and runs.

Provides inspection, listing, metric aggregation, comparison, and rerun
resolution for recorded navigation benchmark episodes.

Run directories can be partial or hand-edited, so every reader degrades gracefully: unreadable files are skipped
(logged at debug level) rather than raised.

This module guarantees fast startup and strictly forbids importing heavy
simulation, deep learning, or ROS frameworks (isaaclab, isaacsim, omni, pxr, rclpy, torch).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
import math
from pathlib import Path
from typing import Any

from nav_arena.benchmarks.io import load_json, load_yaml
from nav_arena.benchmarks.manifest import BatchManifest, RunRecord, RunStatus
from nav_arena.benchmarks.spec import EpisodeLimits, RunSpec, VizCfg, validate_spec
from nav_arena.utils.logger import get_logger

logger = get_logger("benchmarks.tracking")

_RUN_MARKERS = ("run_spec.json", "summary.json", "worker.log")
"""Files that identify a directory as a single run."""


##
# Small readers
##


def _is_true(value: Any) -> bool:
    return str(value).strip().lower() in ("true", "1")


def _to_float(value: Any) -> float | None:
    """Convert to float; ``None`` for missing, empty, or unparseable values."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    """Rows of a results.csv; empty if the file is missing or unreadable."""
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error, UnicodeDecodeError) as exc:
        logger.debug("Failed reading %s: %s", path, exc)
        return []


def _load_manifest(path: Path) -> BatchManifest | None:
    """Load a manifest; ``None`` if it is missing or corrupt."""
    try:
        return BatchManifest.load(path)
    except (OSError, ValueError) as exc:
        logger.debug("Failed reading manifest %s: %s", path, exc)
        return None


def _load_dict(loader: Any, path: Path) -> dict[str, Any]:
    """Load JSON/YAML and keep it only if it is a mapping."""
    data = loader(path)
    return data if isinstance(data, dict) else {}


def _is_run_dir(directory: Path, markers: tuple[str, ...] = _RUN_MARKERS) -> bool:
    return directory.is_dir() and any((directory / name).is_file() for name in markers)


def _status_from_cause(terminal_cause: str | None) -> str:
    """Status for a finished run: ``done``, unless it ended as a timeout or failure."""
    if not terminal_cause or terminal_cause not in ("timeout", "failed"):
        return "done"
    return terminal_cause


def _succeeded(record: RunRecord) -> bool:
    return record.terminal_cause == "goal_reached" or record.exit_code == 0


def _infer_batch_status(records: list[RunRecord]) -> str:
    """Batch status from its runs when batch.yaml does not record one."""
    done = sum(1 for r in records if r.status == RunStatus.DONE.value)
    if records and done == len(records):
        return "done"
    if any(r.status == RunStatus.RUNNING.value for r in records):
        return "running"
    if all(r.status == RunStatus.QUEUED.value for r in records):
        return "queued"
    return "partial"


##
# Batch resolution
##


def _is_batch_dir(directory: Path) -> bool:
    if not directory.is_dir():
        return False
    if (directory / "manifest.json").is_file() or (directory / "batch.yaml").is_file():
        return True
    # A directory with results.csv but without run_spec.json is a batch directory
    return (directory / "results.csv").is_file() and not (directory / "run_spec.json").is_file()


def _declares_batch_id(directory: Path, batch_id: str) -> bool:
    """True if the directory's batch.yaml or manifest.json names ``batch_id`` (by id or sweep name)."""
    declared = _load_dict(load_yaml, directory / "batch.yaml")
    if batch_id in (declared.get("batch_id"), declared.get("name")):
        return True
    return _load_dict(load_json, directory / "manifest.json").get("batch_id") == batch_id


def _resolve_batch_dir(batch_id: str, runs_dir: Path) -> Path:
    """Resolve a batch identifier or directory path to an existing Path.

    Args:
        batch_id: Batch identifier or path.
        runs_dir: Root directory holding runs and batches.

    Returns:
        Resolved Path to the batch directory.

    Raises:
        FileNotFoundError: If the batch directory cannot be found.
    """
    for candidate in (Path(batch_id), runs_dir / batch_id):
        if _is_batch_dir(candidate):
            return candidate.resolve()

    if runs_dir.is_dir():
        for directory in runs_dir.iterdir():
            if not directory.is_dir():
                continue
            named = directory.name == batch_id or directory.name.endswith(f"_{batch_id}")
            if (named and _is_batch_dir(directory)) or _declares_batch_id(directory, batch_id):
                return directory.resolve()

    raise FileNotFoundError(f"Batch '{batch_id}' not found under {runs_dir}")


##
# list_runs
##


def _summarize_batch(directory: Path) -> dict[str, Any]:
    """Summary row for a directory holding a manifest and/or batch.yaml."""
    batch_id, status = directory.name, "unknown"
    total = completed = success_count = 0

    declared = _load_dict(load_yaml, directory / "batch.yaml")
    batch_id = declared.get("batch_id", batch_id)
    status = declared.get("status", status)

    manifest = _load_manifest(directory / "manifest.json") if (directory / "manifest.json").is_file() else None
    if manifest is not None:
        total = len(manifest.runs)
        completed = sum(1 for r in manifest.runs if r.status == RunStatus.DONE.value)
        success_count = sum(1 for r in manifest.runs if _succeeded(r))
        if status == "unknown":
            status = _infer_batch_status(manifest.runs)

    rows = _read_csv_rows(directory / "results.csv")
    if rows:  # results.csv gives the most accurate success counts
        total = total or len(rows)
        completed = completed or len(rows)
        success_count = sum(1 for row in rows if _is_true(row.get("success", "")))

    return _batch_row(batch_id, total, completed, success_count, status)


def _summarize_standalone_run(directory: Path) -> dict[str, Any]:
    """Summary row for a single-run directory (no manifest or batch.yaml)."""
    status = "unknown"
    completed = success_count = 0

    summary = _load_dict(load_json, directory / "summary.json")
    if summary:
        terminal_cause = summary.get("terminal_cause")
        success = bool(summary["success"]) if "success" in summary else terminal_cause == "goal_reached"
        completed, success_count = 1, int(success)
        status = _status_from_cause(terminal_cause)
    else:
        rows = _read_csv_rows(directory / "results.csv")
        if rows:
            completed, success_count = 1, int(_is_true(rows[0].get("success", "")))
            status = _status_from_cause(rows[0].get("terminal_cause"))
        elif (directory / "worker.log").is_file():
            status = "finished"
        elif (directory / "run_spec.json").is_file():
            status = "queued"

    return _batch_row(directory.name, 1, completed, success_count, status)


def _batch_row(batch_id: str, total: int, completed: int, success_count: int, status: str) -> dict[str, Any]:
    success_rate = round(success_count / completed * 100.0, 1) if completed > 0 else 0.0
    return {
        "batch_id": batch_id,
        "total": total,
        "total_runs": total,
        "completed": completed,
        "success_rate": success_rate,
        "status": status,
    }


_LISTED_FILES = ("manifest.json", "batch.yaml", "run_spec.json", "summary.json", "results.csv")


def _list_batches(runs_root: Path) -> list[dict[str, Any]]:
    if not runs_root.is_dir():
        return []
    candidates = [d for d in runs_root.iterdir() if _is_run_dir(d, _LISTED_FILES)]
    candidates.sort(key=lambda d: d.stat().st_mtime, reverse=True)  # newest first
    return [
        _summarize_batch(d)
        if (d / "manifest.json").is_file() or (d / "batch.yaml").is_file()
        else _summarize_standalone_run(d)
        for d in candidates
    ]


def _run_row(**fields: Any) -> dict[str, Any]:
    """Row for ``list_runs(batch_id=...)``; ``time_to_goal`` is kept alongside ``time_to_goal_s`` for compatibility."""
    ttg = fields.pop("ttg")
    return {**fields, "time_to_goal": ttg, "time_to_goal_s": ttg}


def _manifest_run_row(record: RunRecord, csv_row: dict[str, Any], batch_dir: Path) -> dict[str, Any]:
    if "success" in csv_row:
        success = _is_true(csv_row["success"])
    else:
        success = _succeeded(record)

    ttg: float | None = None
    summary_path = batch_dir / (record.run_dir or record.id) / "summary.json"
    if csv_row.get("time_to_goal_s"):
        ttg = _to_float(csv_row["time_to_goal_s"])
    elif summary_path.is_file():
        summary = _load_dict(load_json, summary_path)
        ttg = _to_float(summary.get("time_to_goal_s"))
        if "success" in summary:
            success = bool(summary["success"])

    return _run_row(
        run_id=record.id,
        method=record.method,
        robot=record.robot,
        route=record.route if record.route is not None else "custom",
        seed=record.seed,
        status=record.status,
        terminal_cause=record.terminal_cause or csv_row.get("terminal_cause"),
        success=success,
        ttg=ttg,
    )


def _csv_run_row(run_id: str, row: dict[str, Any]) -> dict[str, Any]:
    return _run_row(
        run_id=run_id,
        method=row.get("method", "unknown"),
        robot=row.get("robot", "unknown"),
        route=row.get("route", "unknown"),
        seed=int(row["seed"]) if row.get("seed") else 0,
        status=RunStatus.DONE.value,
        terminal_cause=row.get("terminal_cause"),
        success=_is_true(row.get("success", "")),
        ttg=_to_float(row.get("time_to_goal_s")),
    )


def _list_batch_runs(batch_dir: Path) -> list[dict[str, Any]]:
    csv_by_id = {row["run_id"]: row for row in _read_csv_rows(batch_dir / "results.csv") if row.get("run_id")}
    manifest_path = batch_dir / "manifest.json"
    manifest = _load_manifest(manifest_path) if manifest_path.is_file() else None
    if manifest is not None:
        return [_manifest_run_row(r, csv_by_id.get(r.id, {}), batch_dir) for r in manifest.runs]
    return [_csv_run_row(run_id, row) for run_id, row in csv_by_id.items()]


def list_runs(runs_dir: Path, batch_id: str | None = None) -> list[dict[str, Any]]:
    """List batches in runs_dir, or individual runs within a specific batch.

    Args:
        runs_dir: Directory containing recorded benchmark runs and batches.
        batch_id: Optional batch identifier. If provided, lists runs inside that batch;
                  otherwise lists all batches under runs_dir.

    Returns:
        If batch_id is None:
            List of dictionaries with keys:
            `batch_id`, `total_runs`, `completed`, `success_rate`, `status`.
        If batch_id is provided:
            List of dictionaries with keys:
            `run_id`, `method`, `robot`, `route`, `seed`, `status`,
            `terminal_cause`, `success`, `time_to_goal`.
    """
    runs_root = Path(runs_dir)
    if batch_id is None:
        return _list_batches(runs_root)
    return _list_batch_runs(_resolve_batch_dir(batch_id, runs_root))


##
# show_run
##

_PROGRESS_KEYS = ("total", "queued", "running", "done", "failed", "timeout", "skipped")


def _duration_s(started_at: Any, ended_at: Any) -> float | None:
    if not (started_at and ended_at):
        return None
    try:
        return max(0.0, (datetime.fromisoformat(ended_at) - datetime.fromisoformat(started_at)).total_seconds())
    except (TypeError, ValueError):
        return None


def _show_batch(batch_dir: Path) -> dict[str, Any]:
    declared = _load_dict(load_yaml, batch_dir / "batch.yaml")
    batch_id = declared.get("batch_id", batch_dir.name)
    status = declared.get("status", "unknown")
    started_at = declared.get("started_at", declared.get("created_at"))
    ended_at = declared.get("ended_at")
    spec_summary = declared.get("spec", {})

    progress = dict.fromkeys(_PROGRESS_KEYS, 0)
    manifest_path = batch_dir / "manifest.json"
    manifest = _load_manifest(manifest_path) if manifest_path.is_file() else None
    if manifest is not None:
        progress["total"] = len(manifest.runs)
        for record in manifest.runs:
            key = record.status.lower()
            progress[key] = progress.get(key, 0) + 1
        if status == "unknown":
            status = _infer_batch_status(manifest.runs)
        if not spec_summary and manifest.metadata:
            spec_summary = manifest.metadata

    rows = _read_csv_rows(batch_dir / "results.csv") if (batch_dir / "results.csv").is_file() else None
    if rows is not None:
        success_count = sum(1 for row in rows if _is_true(row.get("success", "")))
    elif manifest is not None:
        success_count = sum(1 for r in manifest.runs if _succeeded(r))
    else:
        success_count = 0

    completed = progress["done"]
    success_rate = round(success_count / completed * 100.0, 1) if completed > 0 else 0.0
    return {
        "type": "batch",
        "batch_id": batch_id,
        "status": status,
        "batch_dir": str(batch_dir),
        "progress": progress,
        "duration": {"started_at": started_at, "ended_at": ended_at, "duration_s": _duration_s(started_at, ended_at)},
        "spec": spec_summary,
        "success_rate": success_rate,
        "results_summary": {
            "total": progress["total"],
            "completed": completed,
            "success": success_count,
            "rate_pct": success_rate,
        },
    }


@dataclass
class _RunLocation:
    """Everything found about one run across a run directory, a batch manifest, and a batch results.csv."""

    run_id: str
    run_dir: Path | None = None
    record: RunRecord | None = None
    csv_row: dict[str, Any] | None = None
    batch_dir: Path | None = None


def _find_direct_run_dir(identifier: str, runs_root: Path) -> tuple[Path | None, Path | None]:
    """A run directory named directly (a path, or a name under runs_root); returns (run_dir, containing batch dir)."""
    candidate = Path(identifier)
    if _is_run_dir(candidate, _RUN_MARKERS + ("results.csv",)):
        run_dir = candidate.resolve()
        batch = run_dir.parent.resolve() if (run_dir.parent / "manifest.json").is_file() else None
        return run_dir, batch
    under_root = runs_root / identifier
    if _is_run_dir(under_root, _RUN_MARKERS + ("results.csv",)):
        return under_root.resolve(), None
    return None, None


def _find_record(manifest: BatchManifest, run_id: str) -> RunRecord | None:
    return next((r for r in manifest.runs if run_id in (r.id, r.run_dir)), None)


def _locate_run(identifier: str, runs_root: Path) -> _RunLocation:
    """Find a run by directory, manifest record, or results.csv row (searching every batch directory)."""
    run_dir, batch_dir = _find_direct_run_dir(identifier, runs_root)
    location = _RunLocation(run_id=run_dir.name if run_dir else identifier, run_dir=run_dir, batch_dir=batch_dir)

    for batch in sorted(runs_root.iterdir()) if runs_root.is_dir() else []:
        if not batch.is_dir():
            continue
        nested = batch / location.run_id
        if _is_run_dir(nested):
            location.run_dir, location.batch_dir = nested.resolve(), batch.resolve()

        manifest = _load_manifest(batch / "manifest.json") if (batch / "manifest.json").is_file() else None
        record = _find_record(manifest, location.run_id) if manifest is not None else None
        if record is not None:
            location.record, location.batch_dir, location.run_id = record, batch.resolve(), record.id
            if location.run_dir is None:
                location.run_dir = (batch / (record.run_dir or record.id)).resolve()

        if location.csv_row is None:
            row = next((r for r in _read_csv_rows(batch / "results.csv") if r.get("run_id") == location.run_id), None)
            if row is not None:
                location.csv_row, location.batch_dir = row, batch.resolve()
                location.run_dir = location.run_dir or (batch / location.run_id).resolve()

        if location.record is not None or (location.run_dir is not None and (location.run_dir / "summary.json").is_file()):
            break
    return location


def _run_settings(location: _RunLocation, summary_dir: Path | None) -> dict[str, Any]:
    """Settings for a run: its run_spec.json/settings.json, filled in from the manifest record and results row."""
    settings: dict[str, Any] = {}
    if summary_dir is not None:
        settings.update(_load_dict(load_json, summary_dir / "run_spec.json"))
        settings.update(_load_dict(load_json, summary_dir / "settings.json"))

    record = location.record
    if record is not None:
        for key in ("method", "robot", "scene", "route"):
            if not settings.get(key):
                settings[key] = getattr(record, key)
        settings.setdefault("seed", record.seed)
        if record.method_params and not settings.get("method_params"):
            settings["method_params"] = record.method_params

    row = location.csv_row
    if row:
        for key in ("method", "robot", "scene", "route"):
            if key not in settings and row.get(key):
                settings[key] = row[key]
        if "seed" not in settings and row.get("seed"):
            try:
                settings["seed"] = int(row["seed"])
            except ValueError:
                pass

    if "policy" in settings and "method" not in settings:
        settings["method"] = settings["policy"]
    return settings


def _terminal_outcome(
    summary: dict[str, Any], record: RunRecord | None, row: dict[str, Any] | None
) -> tuple[str | None, bool, int | None]:
    """(terminal_cause, success, exit_code) from the best available source: summary, manifest, then results.csv."""
    cause = summary.get("terminal_cause") or (record.terminal_cause if record else None) or (row or {}).get("terminal_cause")
    cause = cause or None

    if "success" in summary:
        success = bool(summary["success"])
    elif row and "success" in row:
        success = _is_true(row["success"])
    else:
        success = cause == "goal_reached"

    if record is not None and record.exit_code is not None:
        exit_code: int | None = record.exit_code
    elif cause == "goal_reached":
        exit_code = 0
    elif cause in ("failed", "timeout"):
        exit_code = 1
    else:
        exit_code = 2 if cause else None
    return cause, success, exit_code


def _metric(summary: dict[str, Any], row: dict[str, Any] | None, key: str) -> float | None:
    """A numeric metric from the summary, falling back to the results.csv row when the summary value is falsy."""
    return _to_float(summary.get(key) or (row.get(key) if row else None))


def _step_count(summary: dict[str, Any], row: dict[str, Any] | None, run_dir: Path | None) -> Any:
    steps = summary.get("steps", summary.get("sim_steps"))
    if steps is None and row and (row.get("plans") or row.get("sim_steps")):
        steps = row.get("plans") or row.get("sim_steps")
        try:
            steps = int(steps)
        except ValueError:
            pass
    if steps is None and run_dir is not None and (run_dir / "steps.jsonl").is_file():
        try:
            with (run_dir / "steps.jsonl").open("r", encoding="utf-8") as handle:
                steps = sum(1 for _ in handle)
        except OSError:
            pass
    return steps


def _show_single_run(location: _RunLocation) -> dict[str, Any]:
    run_dir, record, row = location.run_dir, location.record, location.csv_row
    if row is None and location.batch_dir is not None:
        # The batch scan may have stopped early; look the row up in the parent batch directly.
        row = next(
            (r for r in _read_csv_rows(location.batch_dir / "results.csv") if r.get("run_id") == location.run_id), None
        )
    summary = _load_dict(load_json, run_dir / "summary.json") if run_dir else {}
    log_path = run_dir / "worker.log" if run_dir else None
    cause, success, exit_code = _terminal_outcome(summary, record, row)

    if record is not None:
        status = record.status
    elif cause:
        status = _status_from_cause(cause)
    elif log_path is not None and log_path.is_file():
        status = "finished"
    else:
        status = "unknown"

    return {
        "type": "run",
        "run_id": location.run_id,
        "run_dir": str(run_dir) if run_dir is not None else None,
        "status": status,
        "outcome": {"terminal_cause": cause, "success": success, "exit_code": exit_code},
        "metrics": {
            "time_to_goal_s": _metric(summary, row, "time_to_goal_s"),
            "path_length_m": _metric(summary, row, "path_length_m"),
            "mean_inference_ms": _metric(summary, row, "mean_inference_ms"),
            "steps": _step_count(summary, row, run_dir),
            "sim_time_s": _metric(summary, row, "sim_time_s"),
            "final_goal_distance_m": _metric(summary, row, "final_goal_distance_m"),
        },
        "settings": _run_settings(
            _RunLocation(location.run_id, run_dir, record, row, location.batch_dir), run_dir
        ),
        "log_path": str(log_path) if log_path is not None and log_path.is_file() else None,
    }


def show_run(run_or_batch_id: str, runs_dir: Path) -> dict[str, Any]:
    """Retrieve detailed execution metadata for a batch or single episode run.

    Args:
        run_or_batch_id: Run identifier, batch identifier, or path.
        runs_dir: Root directory holding benchmark runs.

    Returns:
        Dictionary containing either batch status or run details.

    Raises:
        FileNotFoundError: If the target cannot be found.
    """
    runs_root = Path(runs_dir)
    try:
        return _show_batch(_resolve_batch_dir(run_or_batch_id, runs_root))
    except FileNotFoundError:
        pass

    location = _locate_run(run_or_batch_id, runs_root)
    if location.run_dir is None and location.record is None and location.csv_row is None:
        raise FileNotFoundError(f"Run or batch '{run_or_batch_id}' not found under {runs_dir}")
    return _show_single_run(location)


##
# compare_batch
##


def _calc_mean_std(values: list[float]) -> tuple[float | None, float | None]:
    """Calculate arithmetic mean and sample standard deviation."""
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) > 1:
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        std = math.sqrt(max(0.0, variance))
    else:
        std = 0.0
    return mean, std


def _floats(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = (_to_float(str(r.get(key) or "").strip()) for r in rows)
    return [v for v in values if v is not None]


def _mean_std_text(values: list[float]) -> str:
    mean, std = _calc_mean_std(values)
    return f"{mean:.2f} ± {std:.2f}" if mean is not None and std is not None else "-"


def _aggregate_group(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [r for r in rows if _is_true(r.get("success", ""))]
    total = len(rows)
    mean_inference, _ = _calc_mean_std(_floats(rows, "mean_inference_ms"))
    return {
        "group": name,
        "total": total,
        "success": len(successes),
        "rate": f"{len(successes) / total * 100.0 if total else 0.0:.1f}%",
        "ttg": _mean_std_text(_floats(successes, "time_to_goal_s")),  # successful runs only
        "path_len": _mean_std_text(_floats(rows, "path_length_m")),
        "collisions": sum(1 for r in rows if str(r.get("terminal_cause") or "").strip().lower() == "collision"),
        "inference": f"{mean_inference:.1f}" if mean_inference is not None else "-",
    }


_TABLE_COLUMNS = (  # (header, row key, right-aligned)
    (None, "group", False),
    ("Total", "total", True),
    ("Success", "success", True),
    ("Rate (%)", "rate", True),
    ("Time to Goal (s)", "ttg", False),
    ("Path Length (m)", "path_len", False),
    ("Collisions", "collisions", True),
    ("Mean Infer (ms)", "inference", True),
)


def _format_table(by: str, table_rows: list[dict[str, Any]]) -> str:
    headers = [header or by.capitalize() for header, _, _ in _TABLE_COLUMNS]
    widths = [
        max([len(headers[i])] + [len(str(row[key])) for row in table_rows])
        for i, (_, key, _) in enumerate(_TABLE_COLUMNS)
    ]
    lines = [
        "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |",
        "|-" + "-|-".join("-" * w for w in widths) + "-|",
    ]
    for row in table_rows:
        cells = [
            str(row[key]).rjust(widths[i]) if right else str(row[key]).ljust(widths[i])
            for i, (_, key, right) in enumerate(_TABLE_COLUMNS)
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the latest row per run_id (retried runs are rewritten); rows without an id are all kept."""
    by_id: dict[str, dict[str, Any]] = {}
    without_id: list[dict[str, Any]] = []
    for row in rows:
        if row.get("run_id"):
            by_id[row["run_id"]] = row
        else:
            without_id.append(row)
    return list(by_id.values()) + without_id


def compare_batch(batch_id: str, runs_dir: Path, by: str = "method") -> str:
    """Aggregate and format metrics from a batch results.csv grouped by dimension.

    Args:
        batch_id: Batch identifier or directory path.
        runs_dir: Root directory holding runs.
        by: Grouping key. Must be one of 'method', 'route', or 'robot'.

    Returns:
        Formatted ASCII/Markdown table of grouped metrics.

    Raises:
        ValueError: If grouping dimension is invalid.
        FileNotFoundError: If batch or results.csv cannot be found.
    """
    by_norm = str(by).strip().lower()
    valid_group_keys = ("method", "route", "robot")
    if by_norm not in valid_group_keys:
        raise ValueError(f"Invalid grouping key '{by}'. Must be one of: {valid_group_keys}")

    csv_path = _resolve_batch_dir(batch_id, Path(runs_dir)) / "results.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"results.csv not found for batch '{batch_id}' at {csv_path}")

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in _dedupe_rows(_read_csv_rows(csv_path)):
        groups.setdefault(row.get(by_norm) or "unknown", []).append(row)
    return _format_table(by, [_aggregate_group(name, groups[name]) for name in sorted(groups)])


##
# find_run_spec_for_rerun
##


def _load_saved_spec(run_dir: Path) -> RunSpec | None:
    spec_file = run_dir / "run_spec.json"
    if not (run_dir.is_dir() and spec_file.is_file()):
        return None
    spec = RunSpec.from_json(spec_file.read_text(encoding="utf-8"))
    validate_spec(spec)
    return spec


def _inline_route(routes: Any, name: str | None) -> dict[str, Any] | None:
    """The sweep's inline route definition called ``name`` (one with its own spawn and goal), if any."""
    for route in routes if isinstance(routes, list) else []:
        if isinstance(route, dict) and route.get("name") == name:
            return route if "spawn" in route and "goal" in route else None
    return None


def _spec_from_record(batch_dir: Path, record: RunRecord, run_dir: Path) -> RunSpec:
    """Rebuild a RunSpec for a run that has no run_spec.json from its manifest record and the batch's batch.yaml."""
    batch_spec = _load_dict(load_yaml, batch_dir / "batch.yaml").get("spec", {})
    options = (batch_spec.get("options") if isinstance(batch_spec, dict) else None) or {}
    inline = _inline_route(batch_spec.get("routes") if isinstance(batch_spec, dict) else None, record.route)

    spec = RunSpec(
        method=record.method,
        robot=record.robot,
        scene=record.scene,
        route=None if inline else record.route,
        spawn=tuple(inline["spawn"]) if inline else None,
        goal=tuple(inline["goal"]) if inline else None,
        spawn_yaw=inline.get("spawn_yaw") if inline else None,
        seed=record.seed,
        method_params=record.method_params,
        limits=EpisodeLimits.from_options(options),
        viz=VizCfg.from_options(options),
        output_dir=run_dir,
    )
    validate_spec(spec)
    return spec


def _rerun_from_batch(batch_dir: Path, run_id: str) -> tuple[Path, RunSpec] | None:
    nested = batch_dir / run_id
    spec = _load_saved_spec(nested)
    if spec is not None:
        return nested.resolve(), spec

    manifest = _load_manifest(batch_dir / "manifest.json") if (batch_dir / "manifest.json").is_file() else None
    record = _find_record(manifest, run_id) if manifest is not None else None
    if record is None:
        return None
    run_dir = batch_dir / (record.run_dir or record.id)
    spec = _load_saved_spec(run_dir) or _spec_from_record(batch_dir, record, run_dir)
    return run_dir.resolve(), spec


def find_run_spec_for_rerun(run_id: str, runs_dir: Path) -> tuple[Path, RunSpec]:
    """Locate a recorded run directory and reconstruct or load its RunSpec.

    Args:
        run_id: Run identifier, batch subfolder name, or direct path.
        runs_dir: Root directory holding benchmark runs.

    Returns:
        Tuple of (run_directory, resolved_run_spec).

    Raises:
        FileNotFoundError: If the run or run spec cannot be found.
    """
    runs_root = Path(runs_dir)
    for candidate in (Path(run_id), runs_root / run_id):
        spec = _load_saved_spec(candidate)
        if spec is not None:
            return candidate.resolve(), spec

    for batch_dir in sorted(runs_root.iterdir()) if runs_root.is_dir() else []:
        if batch_dir.is_dir():
            found = _rerun_from_batch(batch_dir, run_id)
            if found is not None:
                return found

    raise FileNotFoundError(f"Could not find run or spec for '{run_id}' under {runs_dir}")
