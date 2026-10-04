# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tracking and query utilities for benchmark batches and runs.

Provides inspection, listing, metric aggregation, comparison, and rerun
resolution for recorded navigation benchmark episodes.

This module guarantees fast startup and strictly forbids importing heavy
simulation, deep learning, or ROS frameworks (isaaclab, isaacsim, omni, pxr, rclpy, torch).
"""

from __future__ import annotations

import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
from typing import Any

import yaml

from nav_arena.benchmarks.manifest import (
    BatchManifest,
    RunRecord,
    RunStatus,
)
from nav_arena.benchmarks.spec import (
    EpisodeLimits,
    RunSpec,
    VizCfg,
    validate_spec,
)
from nav_arena.utils.logger import get_logger

logger = get_logger("benchmarks.tracking")


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
    def _is_batch_dir(d: Path) -> bool:
        if not d.is_dir():
            return False
        if (d / "manifest.json").is_file() or (d / "batch.yaml").is_file():
            return True
        # A directory with results.csv but without run_spec.json is a batch directory
        if (d / "results.csv").is_file() and not (d / "run_spec.json").is_file():
            return True
        return False

    candidate = Path(batch_id)
    if _is_batch_dir(candidate):
        return candidate.resolve()

    candidate = runs_dir / batch_id
    if _is_batch_dir(candidate):
        return candidate.resolve()

    # Search in runs_dir
    if runs_dir.is_dir():
        for d in runs_dir.iterdir():
            if not d.is_dir():
                continue
            if (d.name == batch_id or d.name.endswith(f"_{batch_id}")) and _is_batch_dir(d):
                return d.resolve()

            # Check batch.yaml metadata
            b_yaml_path = d / "batch.yaml"
            if b_yaml_path.is_file():
                try:
                    data = yaml.safe_load(b_yaml_path.read_text(encoding="utf-8")) or {}
                    if isinstance(data, dict):
                        if data.get("batch_id") == batch_id or data.get("name") == batch_id:
                            return d.resolve()
                except Exception:
                    pass

            # Check manifest.json metadata
            m_path = d / "manifest.json"
            if m_path.is_file():
                try:
                    m_data = json.loads(m_path.read_text(encoding="utf-8")) or {}
                    if isinstance(m_data, dict) and m_data.get("batch_id") == batch_id:
                        return d.resolve()
                except Exception:
                    pass

    raise FileNotFoundError(f"Batch '{batch_id}' not found under {runs_dir}")


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
        # List batches
        if not runs_root.is_dir():
            return []

        batches: list[dict[str, Any]] = []
        batch_dirs = [
            d for d in runs_root.iterdir()
            if d.is_dir() and ((d / "manifest.json").is_file() or (d / "batch.yaml").is_file())
        ]
        # Sort newest first based on directory mtime
        batch_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)

        for b_dir in batch_dirs:
            b_id = b_dir.name
            manifest_path = b_dir / "manifest.json"
            batch_yaml_path = b_dir / "batch.yaml"
            results_csv_path = b_dir / "results.csv"

            status = "unknown"
            total_runs = 0
            completed = 0
            success_count = 0

            # 1. Inspect batch.yaml
            if batch_yaml_path.is_file():
                try:
                    b_yaml = yaml.safe_load(batch_yaml_path.read_text(encoding="utf-8"))
                    if isinstance(b_yaml, dict):
                        b_id = b_yaml.get("batch_id", b_id)
                        status = b_yaml.get("status", status)
                except Exception as exc:
                    logger.debug("Failed reading %s: %s", batch_yaml_path, exc)

            # 2. Inspect manifest.json
            if manifest_path.is_file():
                try:
                    manifest = BatchManifest.load(manifest_path)
                    total_runs = len(manifest.runs)
                    completed = sum(1 for r in manifest.runs if r.status == RunStatus.DONE.value)
                    for r in manifest.runs:
                        if r.terminal_cause == "goal_reached" or r.exit_code == 0:
                            success_count += 1
                    if status == "unknown":
                        if total_runs > 0 and completed == total_runs:
                            status = "done"
                        elif any(r.status == RunStatus.RUNNING.value for r in manifest.runs):
                            status = "running"
                        elif all(r.status == RunStatus.QUEUED.value for r in manifest.runs):
                            status = "queued"
                        else:
                            status = "partial"
                except Exception as exc:
                    logger.debug("Failed reading %s: %s", manifest_path, exc)

            # 3. Inspect results.csv for more accurate success counts
            if results_csv_path.is_file():
                try:
                    with results_csv_path.open("r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        csv_rows = list(reader)
                        if total_runs == 0 and csv_rows:
                            total_runs = len(csv_rows)
                        if completed == 0 and csv_rows:
                            completed = len(csv_rows)
                        if csv_rows:
                            success_count = sum(
                                1 for row in csv_rows
                                if str(row.get("success", "")).strip().lower() in ("true", "1")
                            )
                except Exception as exc:
                    logger.debug("Failed reading %s: %s", results_csv_path, exc)

            success_rate = (
                round((success_count / completed) * 100.0, 1)
                if completed > 0
                else 0.0
            )

            batches.append({
                "batch_id": b_id,
                "total": total_runs,
                "total_runs": total_runs,
                "completed": completed,
                "success_rate": success_rate,
                "status": status,
            })

        return batches

    # Specific batch: list runs within it
    target_dir = _resolve_batch_dir(batch_id, runs_root)
    manifest_path = target_dir / "manifest.json"
    results_csv_path = target_dir / "results.csv"

    csv_data_by_id: dict[str, dict[str, Any]] = {}
    if results_csv_path.is_file():
        try:
            with results_csv_path.open("r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    rid = row.get("run_id")
                    if rid:
                        csv_data_by_id[rid] = row
        except Exception as exc:
            logger.debug("Failed reading %s: %s", results_csv_path, exc)

    runs: list[dict[str, Any]] = []

    if manifest_path.is_file():
        try:
            manifest = BatchManifest.load(manifest_path)
            for r in manifest.runs:
                csv_row = csv_data_by_id.get(r.id, {})

                # Success resolution
                success = False
                if "success" in csv_row:
                    success = str(csv_row["success"]).strip().lower() in ("true", "1")
                elif r.terminal_cause == "goal_reached" or r.exit_code == 0:
                    success = True

                # Time to goal resolution
                ttg: float | None = None
                if "time_to_goal_s" in csv_row and csv_row["time_to_goal_s"]:
                    try:
                        ttg = float(csv_row["time_to_goal_s"])
                    except ValueError:
                        pass
                elif (target_dir / (r.run_dir or r.id) / "summary.json").is_file():
                    try:
                        sum_data = json.loads((target_dir / (r.run_dir or r.id) / "summary.json").read_text(encoding="utf-8"))
                        if "time_to_goal_s" in sum_data and sum_data["time_to_goal_s"] is not None:
                            ttg = float(sum_data["time_to_goal_s"])
                        if "success" in sum_data:
                            success = bool(sum_data["success"])
                    except Exception:
                        pass

                terminal_cause = r.terminal_cause or csv_row.get("terminal_cause")

                runs.append({
                    "run_id": r.id,
                    "method": r.method,
                    "robot": r.robot,
                    "route": r.route if r.route is not None else "custom",
                    "seed": r.seed,
                    "status": r.status,
                    "terminal_cause": terminal_cause,
                    "success": success,
                    "time_to_goal": ttg,
                    "time_to_goal_s": ttg,
                })
            return runs
        except Exception as exc:
            logger.debug("Failed reading manifest: %s", exc)

    # Fallback to results.csv if manifest not found
    for run_id, row in csv_data_by_id.items():
        success = str(row.get("success", "")).strip().lower() in ("true", "1")
        ttg_val: float | None = None
        if row.get("time_to_goal_s"):
            try:
                ttg_val = float(row["time_to_goal_s"])
            except ValueError:
                pass
        runs.append({
            "run_id": run_id,
            "method": row.get("method", "unknown"),
            "robot": row.get("robot", "unknown"),
            "route": row.get("route", "unknown"),
            "seed": int(row.get("seed", 0)) if row.get("seed") else 0,
            "status": RunStatus.DONE.value,
            "terminal_cause": row.get("terminal_cause"),
            "success": success,
            "time_to_goal": ttg_val,
            "time_to_goal_s": ttg_val,
        })

    return runs


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

    # 1. Check if run_or_batch_id refers to a batch
    try:
        batch_dir = _resolve_batch_dir(run_or_batch_id, runs_root)
        is_batch = True
    except FileNotFoundError:
        batch_dir = None
        is_batch = False

    if is_batch and batch_dir is not None:
        manifest_path = batch_dir / "manifest.json"
        batch_yaml_path = batch_dir / "batch.yaml"
        results_csv_path = batch_dir / "results.csv"

        batch_id = batch_dir.name
        status = "unknown"
        started_at = None
        ended_at = None
        spec_summary: dict[str, Any] = {}

        if batch_yaml_path.is_file():
            try:
                b_yaml = yaml.safe_load(batch_yaml_path.read_text(encoding="utf-8"))
                if isinstance(b_yaml, dict):
                    batch_id = b_yaml.get("batch_id", batch_id)
                    status = b_yaml.get("status", status)
                    started_at = b_yaml.get("started_at", b_yaml.get("created_at"))
                    ended_at = b_yaml.get("ended_at")
                    spec_summary = b_yaml.get("spec", {})
            except Exception as exc:
                logger.debug("Failed loading %s: %s", batch_yaml_path, exc)

        progress = {
            "total": 0,
            "queued": 0,
            "running": 0,
            "done": 0,
            "failed": 0,
            "timeout": 0,
            "skipped": 0,
        }

        if manifest_path.is_file():
            try:
                manifest = BatchManifest.load(manifest_path)
                progress["total"] = len(manifest.runs)
                for r in manifest.runs:
                    st = r.status.lower()
                    if st in progress:
                        progress[st] += 1
                    else:
                        progress[st] = 1

                if status == "unknown":
                    if progress["total"] > 0 and progress["done"] == progress["total"]:
                        status = "done"
                    elif progress["running"] > 0:
                        status = "running"
                    elif progress["queued"] == progress["total"]:
                        status = "queued"
                    else:
                        status = "partial"

                if not spec_summary and manifest.metadata:
                    spec_summary = manifest.metadata
            except Exception as exc:
                logger.debug("Failed loading manifest %s: %s", manifest_path, exc)

        # Calculate duration if timestamps are present
        duration_s: float | None = None
        if started_at and ended_at:
            try:
                t0 = datetime.fromisoformat(started_at)
                t1 = datetime.fromisoformat(ended_at)
                duration_s = max(0.0, (t1 - t0).total_seconds())
            except Exception:
                pass

        # Calculate success rate from results.csv or manifest
        success_count = 0
        if results_csv_path.is_file():
            try:
                with results_csv_path.open("r", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        if str(row.get("success", "")).strip().lower() in ("true", "1"):
                            success_count += 1
            except Exception:
                pass
        elif manifest_path.is_file():
            try:
                manifest = BatchManifest.load(manifest_path)
                success_count = sum(
                    1 for r in manifest.runs
                    if r.terminal_cause == "goal_reached" or r.exit_code == 0
                )
            except Exception:
                pass

        completed = progress["done"]
        success_rate = (
            round((success_count / completed) * 100.0, 1)
            if completed > 0
            else 0.0
        )

        return {
            "type": "batch",
            "batch_id": batch_id,
            "status": status,
            "batch_dir": str(batch_dir),
            "progress": progress,
            "duration": {
                "started_at": started_at,
                "ended_at": ended_at,
                "duration_s": duration_s,
            },
            "spec": spec_summary,
            "success_rate": success_rate,
            "results_summary": {
                "total": progress["total"],
                "completed": completed,
                "success": success_count,
                "rate_pct": success_rate,
            },
        }

    # 2. Check if run_or_batch_id refers to a single run
    run_dir: Path | None = None
    target_run_id = run_or_batch_id
    manifest_rec: RunRecord | None = None
    csv_row_data: dict[str, Any] | None = None
    parent_batch_dir: Path | None = None

    # Check direct directory path
    cand_path = Path(run_or_batch_id)
    if cand_path.is_dir() and ((cand_path / "run_spec.json").is_file() or (cand_path / "summary.json").is_file() or (cand_path / "worker.log").is_file() or (cand_path / "results.csv").is_file()):
        run_dir = cand_path.resolve()
        target_run_id = run_dir.name
        if (run_dir.parent / "manifest.json").is_file():
            parent_batch_dir = run_dir.parent.resolve()
    elif (runs_root / run_or_batch_id).is_dir():
        cand = (runs_root / run_or_batch_id).resolve()
        if (cand / "run_spec.json").is_file() or (cand / "summary.json").is_file() or (cand / "worker.log").is_file() or (cand / "results.csv").is_file():
            run_dir = cand
            target_run_id = cand.name

    # Check inside batch directories
    if runs_root.is_dir():
        for d in runs_root.iterdir():
            if not d.is_dir():
                continue

            # 1. Check if folder exists inside batch
            cand = d / target_run_id
            if cand.is_dir() and ((cand / "run_spec.json").is_file() or (cand / "summary.json").is_file() or (cand / "worker.log").is_file()):
                run_dir = cand.resolve()
                parent_batch_dir = d.resolve()

            # 2. Check manifest inside batch
            manifest_path = d / "manifest.json"
            if manifest_path.is_file():
                try:
                    manifest = BatchManifest.load(manifest_path)
                    for r in manifest.runs:
                        if r.id == target_run_id or r.run_dir == target_run_id:
                            manifest_rec = r
                            parent_batch_dir = d.resolve()
                            target_run_id = r.id
                            if run_dir is None:
                                cand_sub = d / (r.run_dir or r.id)
                                run_dir = cand_sub.resolve()
                            break
                except Exception as exc:
                    logger.debug("Failed reading manifest in %s: %s", d, exc)

            # 3. Check results.csv inside batch
            results_csv_p = d / "results.csv"
            if results_csv_p.is_file() and csv_row_data is None:
                try:
                    with results_csv_p.open("r", encoding="utf-8") as f:
                        for row in csv.DictReader(f):
                            if row.get("run_id") == target_run_id:
                                csv_row_data = row
                                parent_batch_dir = d.resolve()
                                if run_dir is None:
                                    run_dir = (d / target_run_id).resolve()
                except Exception as exc:
                    logger.debug("Failed reading results.csv in %s: %s", d, exc)

            if manifest_rec is not None or (run_dir is not None and run_dir.is_dir() and (run_dir / "summary.json").is_file()):
                break

    if run_dir is None and manifest_rec is None and csv_row_data is None:
        raise FileNotFoundError(f"Run or batch '{run_or_batch_id}' not found under {runs_dir}")

    # Inspect run artifacts if directory exists
    summary_path = run_dir / "summary.json" if run_dir is not None else None
    settings_path = run_dir / "settings.json" if run_dir is not None else None
    spec_path = run_dir / "run_spec.json" if run_dir is not None else None
    log_path = run_dir / "worker.log" if run_dir is not None else None

    # Fallback to parent batch results.csv if not found yet
    if csv_row_data is None and parent_batch_dir is not None and (parent_batch_dir / "results.csv").is_file():
        try:
            with (parent_batch_dir / "results.csv").open("r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row.get("run_id") == target_run_id:
                        csv_row_data = row
        except Exception:
            pass

    summary_data: dict[str, Any] = {}
    if summary_path and summary_path.is_file():
        try:
            summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.debug("Failed reading %s: %s", summary_path, exc)

    settings_data: dict[str, Any] = {}
    if spec_path and spec_path.is_file():
        try:
            settings_data.update(json.loads(spec_path.read_text(encoding="utf-8")))
        except Exception as exc:
            logger.debug("Failed reading %s: %s", spec_path, exc)
    if settings_path and settings_path.is_file():
        try:
            settings_data.update(json.loads(settings_path.read_text(encoding="utf-8")))
        except Exception as exc:
            logger.debug("Failed reading %s: %s", settings_path, exc)

    if manifest_rec is not None:
        if not settings_data.get("method"):
            settings_data["method"] = manifest_rec.method
        if not settings_data.get("robot"):
            settings_data["robot"] = manifest_rec.robot
        if not settings_data.get("scene"):
            settings_data["scene"] = manifest_rec.scene
        if not settings_data.get("route"):
            settings_data["route"] = manifest_rec.route
        if "seed" not in settings_data:
            settings_data["seed"] = manifest_rec.seed
        if manifest_rec.method_params and not settings_data.get("method_params"):
            settings_data["method_params"] = manifest_rec.method_params

    if csv_row_data:
        for k in ("method", "robot", "scene", "route"):
            if k not in settings_data and csv_row_data.get(k):
                settings_data[k] = csv_row_data[k]
        if "seed" not in settings_data and csv_row_data.get("seed"):
            try:
                settings_data["seed"] = int(csv_row_data["seed"])
            except ValueError:
                pass

    if "policy" in settings_data and "method" not in settings_data:
        settings_data["method"] = settings_data["policy"]

    # Terminal cause and success
    terminal_cause = None
    if summary_data.get("terminal_cause"):
        terminal_cause = summary_data["terminal_cause"]
    elif manifest_rec and manifest_rec.terminal_cause:
        terminal_cause = manifest_rec.terminal_cause
    elif csv_row_data and csv_row_data.get("terminal_cause"):
        terminal_cause = csv_row_data["terminal_cause"]

    success = False
    if "success" in summary_data:
        success = bool(summary_data["success"])
    elif csv_row_data and "success" in csv_row_data:
        success = str(csv_row_data["success"]).strip().lower() in ("true", "1")
    elif terminal_cause == "goal_reached":
        success = True

    exit_code = None
    if manifest_rec and manifest_rec.exit_code is not None:
        exit_code = manifest_rec.exit_code
    elif terminal_cause == "goal_reached":
        exit_code = 0
    elif terminal_cause:
        exit_code = 2

    # Status resolution
    if manifest_rec is not None:
        status = manifest_rec.status
    elif terminal_cause:
        status = "done" if terminal_cause not in ("timeout", "failed") else terminal_cause
    elif log_path and log_path.is_file():
        status = "finished"
    else:
        status = "unknown"

    ttg = summary_data.get("time_to_goal_s") or (csv_row_data.get("time_to_goal_s") if csv_row_data else None)
    if ttg is not None:
        try:
            ttg = float(ttg)
        except ValueError:
            ttg = None

    path_len = summary_data.get("path_length_m") or (csv_row_data.get("path_length_m") if csv_row_data else None)
    if path_len is not None:
        try:
            path_len = float(path_len)
        except ValueError:
            path_len = None

    mean_infer = summary_data.get("mean_inference_ms") or (csv_row_data.get("mean_inference_ms") if csv_row_data else None)
    if mean_infer is not None:
        try:
            mean_infer = float(mean_infer)
        except ValueError:
            mean_infer = None

    steps = summary_data.get("steps", summary_data.get("sim_steps"))
    if steps is None and csv_row_data and (csv_row_data.get("plans") or csv_row_data.get("sim_steps")):
        steps = csv_row_data.get("plans") or csv_row_data.get("sim_steps")
        try:
            steps = int(steps)
        except ValueError:
            pass
    if steps is None and run_dir and (run_dir / "steps.jsonl").is_file():
        try:
            with (run_dir / "steps.jsonl").open("r", encoding="utf-8") as f:
                steps = sum(1 for _ in f)
        except Exception:
            pass

    final_dist = summary_data.get("final_goal_distance_m") or (csv_row_data.get("final_goal_distance_m") if csv_row_data else None)
    if final_dist is not None:
        try:
            final_dist = float(final_dist)
        except ValueError:
            final_dist = None

    sim_time = summary_data.get("sim_time_s") or (csv_row_data.get("sim_time_s") if csv_row_data else None)
    if sim_time is not None:
        try:
            sim_time = float(sim_time)
        except ValueError:
            sim_time = None

    return {
        "type": "run",
        "run_id": target_run_id,
        "run_dir": str(run_dir) if run_dir is not None else None,
        "status": status,
        "outcome": {
            "terminal_cause": terminal_cause,
            "success": success,
            "exit_code": exit_code,
        },
        "metrics": {
            "time_to_goal_s": ttg,
            "path_length_m": path_len,
            "mean_inference_ms": mean_infer,
            "steps": steps,
            "sim_time_s": sim_time,
            "final_goal_distance_m": final_dist,
        },
        "settings": settings_data,
        "log_path": str(log_path) if log_path and log_path.is_file() else None,
    }


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

    target_dir = _resolve_batch_dir(batch_id, Path(runs_dir))
    csv_path = target_dir / "results.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"results.csv not found for batch '{batch_id}' at {csv_path}")

    # Read rows, deduplicating by run_id (latest row wins for retried runs)
    rows_by_id: dict[str, dict[str, Any]] = {}
    rows_without_id: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rid = r.get("run_id")
            if rid:
                rows_by_id[rid] = r
            else:
                rows_without_id.append(r)
    rows: list[dict[str, Any]] = list(rows_by_id.values()) + rows_without_id

    # Group by key
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        key_val = r.get(by_norm, "unknown")
        if not key_val:
            key_val = "unknown"
        if key_val not in groups:
            groups[key_val] = []
        groups[key_val].append(r)

    # Build aggregation rows
    table_rows: list[dict[str, Any]] = []
    for group_name in sorted(groups.keys()):
        group_rows = groups[group_name]
        total = len(group_rows)
        successes = [
            r for r in group_rows
            if str(r.get("success", "")).strip().lower() in ("true", "1")
        ]
        success_count = len(successes)
        rate = (success_count / total * 100.0) if total > 0 else 0.0

        # TTG over successful runs only
        ttgs: list[float] = []
        for r in successes:
            val_str = str(r.get("time_to_goal_s") or "").strip()
            if val_str:
                try:
                    ttgs.append(float(val_str))
                except ValueError:
                    pass

        mean_ttg, std_ttg = _calc_mean_std(ttgs)
        if mean_ttg is not None and std_ttg is not None:
            ttg_str = f"{mean_ttg:.2f} ± {std_ttg:.2f}"
        else:
            ttg_str = "-"

        # Path length
        path_lengths: list[float] = []
        for r in group_rows:
            pl_str = str(r.get("path_length_m") or "").strip()
            if pl_str:
                try:
                    path_lengths.append(float(pl_str))
                except ValueError:
                    pass

        mean_pl, std_pl = _calc_mean_std(path_lengths)
        if mean_pl is not None and std_pl is not None:
            pl_str = f"{mean_pl:.2f} ± {std_pl:.2f}"
        else:
            pl_str = "-"

        # Collisions
        collisions = sum(
            1 for r in group_rows
            if str(r.get("terminal_cause") or "").strip().lower() == "collision"
        )

        # Inference ms
        inf_times: list[float] = []
        for r in group_rows:
            inf_str = str(r.get("mean_inference_ms") or "").strip()
            if inf_str:
                try:
                    inf_times.append(float(inf_str))
                except ValueError:
                    pass

        mean_inf, _ = _calc_mean_std(inf_times)
        if mean_inf is not None:
            inf_str = f"{mean_inf:.1f}"
        else:
            inf_str = "-"

        table_rows.append({
            "group": group_name,
            "total": total,
            "success": success_count,
            "rate": f"{rate:.1f}%",
            "ttg": ttg_str,
            "path_len": pl_str,
            "collisions": collisions,
            "inference": inf_str,
        })

    # Format Markdown / ASCII table
    header_title = by.capitalize()
    headers = [
        header_title,
        "Total",
        "Success",
        "Rate (%)",
        "Time to Goal (s)",
        "Path Length (m)",
        "Collisions",
        "Mean Infer (ms)",
    ]

    # Compute column widths
    widths = [len(h) for h in headers]
    for r in table_rows:
        widths[0] = max(widths[0], len(str(r["group"])))
        widths[1] = max(widths[1], len(str(r["total"])))
        widths[2] = max(widths[2], len(str(r["success"])))
        widths[3] = max(widths[3], len(str(r["rate"])))
        widths[4] = max(widths[4], len(str(r["ttg"])))
        widths[5] = max(widths[5], len(str(r["path_len"])))
        widths[6] = max(widths[6], len(str(r["collisions"])))
        widths[7] = max(widths[7], len(str(r["inference"])))

    header_line = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    sep_line = "|-" + "-|-".join("-" * widths[i] for i in range(len(headers))) + "-|"

    lines = [header_line, sep_line]
    for r in table_rows:
        row_line = (
            f"| {str(r['group']).ljust(widths[0])} "
            f"| {str(r['total']).rjust(widths[1])} "
            f"| {str(r['success']).rjust(widths[2])} "
            f"| {str(r['rate']).rjust(widths[3])} "
            f"| {str(r['ttg']).ljust(widths[4])} "
            f"| {str(r['path_len']).ljust(widths[5])} "
            f"| {str(r['collisions']).rjust(widths[6])} "
            f"| {str(r['inference']).rjust(widths[7])} |"
        )
        lines.append(row_line)

    return "\n".join(lines)


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

    # 1. Direct path check
    cand_path = Path(run_id)
    if cand_path.is_dir() and (cand_path / "run_spec.json").is_file():
        spec = RunSpec.from_json((cand_path / "run_spec.json").read_text(encoding="utf-8"))
        validate_spec(spec)
        return cand_path.resolve(), spec

    # 2. Check directly under runs_root
    cand = runs_root / run_id
    if cand.is_dir() and (cand / "run_spec.json").is_file():
        spec = RunSpec.from_json((cand / "run_spec.json").read_text(encoding="utf-8"))
        validate_spec(spec)
        return cand.resolve(), spec

    # 3. Check inside batch folders
    if runs_root.is_dir():
        for b_dir in runs_root.iterdir():
            if not b_dir.is_dir():
                continue

            # Check direct subfolder match
            sub_cand = b_dir / run_id
            if sub_cand.is_dir() and (sub_cand / "run_spec.json").is_file():
                spec = RunSpec.from_json((sub_cand / "run_spec.json").read_text(encoding="utf-8"))
                validate_spec(spec)
                return sub_cand.resolve(), spec

            # Check manifest inside batch folder
            manifest_path = b_dir / "manifest.json"
            if manifest_path.is_file():
                try:
                    manifest = BatchManifest.load(manifest_path)
                    for r in manifest.runs:
                        if r.id == run_id or r.run_dir == run_id:
                            target_run_dir = b_dir / (r.run_dir or r.id)
                            spec_file = target_run_dir / "run_spec.json"
                            if spec_file.is_file():
                                spec = RunSpec.from_json(spec_file.read_text(encoding="utf-8"))
                                validate_spec(spec)
                                return target_run_dir.resolve(), spec

                            # Reconstruct RunSpec from RunRecord and batch.yaml
                            batch_yaml_path = b_dir / "batch.yaml"
                            opts: dict[str, Any] = {}
                            b_yaml: dict[str, Any] = {}
                            if batch_yaml_path.is_file():
                                try:
                                    b_yaml = yaml.safe_load(batch_yaml_path.read_text(encoding="utf-8")) or {}
                                    if isinstance(b_yaml, dict):
                                        opts = b_yaml.get("spec", {}).get("options", {}) or {}
                                except Exception:
                                    pass

                            max_steps = int(opts["max_steps"]) if opts.get("max_steps") is not None else 1500
                            goal_tolerance = (
                                float(opts["goal_tolerance"])
                                if opts.get("goal_tolerance") is not None
                                else (float(opts["goal_dist"]) if opts.get("goal_dist") is not None else 0.4)
                            )
                            max_speed = float(opts["max_speed"]) if opts.get("max_speed") is not None else 0.3
                            stall_timeout_s = (
                                float(opts["stall_timeout_s"])
                                if opts.get("stall_timeout_s") is not None
                                else (float(opts["stall_timeout"]) if opts.get("stall_timeout") is not None else 10.0)
                            )
                            limits = EpisodeLimits(
                                max_steps=max_steps,
                                goal_tolerance=goal_tolerance,
                                max_speed=max_speed,
                                stall_timeout_s=stall_timeout_s,
                            )
                            gui = bool(opts["gui"]) if opts.get("gui") is not None else False
                            follow_camera = bool(opts["follow_camera"]) if opts.get("follow_camera") is not None else False
                            goal_overlay = bool(opts["goal_overlay"]) if opts.get("goal_overlay") is not None else True
                            viz = VizCfg(
                                gui=gui,
                                follow_camera=follow_camera,
                                goal_overlay=goal_overlay,
                            )

                            # Check for custom inline route definitions in batch.yaml
                            spawn = None
                            goal = None
                            spawn_yaw = None
                            route_val = r.route
                            routes_cfg = b_yaml.get("spec", {}).get("routes", []) if isinstance(b_yaml, dict) else []
                            for rt in routes_cfg:
                                if isinstance(rt, dict) and rt.get("name") == r.route:
                                    if "spawn" in rt and "goal" in rt:
                                        spawn = tuple(rt["spawn"])
                                        goal = tuple(rt["goal"])
                                        spawn_yaw = rt.get("spawn_yaw")
                                        route_val = None
                                    break

                            spec = RunSpec(
                                method=r.method,
                                robot=r.robot,
                                scene=r.scene,
                                route=route_val,
                                spawn=spawn,
                                goal=goal,
                                spawn_yaw=spawn_yaw,
                                seed=r.seed,
                                method_params=r.method_params,
                                limits=limits,
                                viz=viz,
                                output_dir=target_run_dir,
                            )
                            validate_spec(spec)
                            return target_run_dir.resolve(), spec
                except Exception as exc:
                    logger.debug("Failed reading manifest in %s: %s", b_dir, exc)

    raise FileNotFoundError(f"Could not find run or spec for '{run_id}' under {runs_dir}")
