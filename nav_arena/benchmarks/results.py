# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""results.csv schema and row helpers shared by single runs and batch sweeps.

Importing this module never pulls in simulation, deep learning, or ROS frameworks.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from nav_arena.benchmarks.spec import RunSpec

RESULTS_CSV_COLUMNS: list[str] = [
    "batch_id",
    "run_id",
    "scene",
    "robot",
    "method",
    "method_family",
    "route",
    "seed",
    "terminal_cause",
    "success",
    "time_to_goal_s",
    "sim_time_s",
    "path_length_m",
    "initial_goal_distance_m",
    "final_goal_distance_m",
    "plans",
    "stop_requests",
    "mean_inference_ms",
    "wall_time_s",
    "goal_tolerance",
    "max_speed",
    "spec_hash",
]


def format_results_row(
    spec: RunSpec,
    summary_data: dict[str, Any] | None,
    run_id: str,
    batch_id: str = "",
    route: str | None = None,
    timed_out: bool = False,
    failed: bool = False,
) -> dict[str, Any]:
    """Format one results.csv dictionary row from RunSpec and summary data."""
    if timed_out:
        terminal_cause = "timeout"
        success = False
    elif failed or not summary_data:
        terminal_cause = "failed"
        success = False
    elif summary_data:
        terminal_cause = str(summary_data.get("terminal_cause", "unknown"))
        success = bool(summary_data.get("success", False))
    else:
        terminal_cause = "failed"
        success = False

    data = summary_data or {}
    resolved_route = (
        route if route is not None else (spec.route if spec.route is not None else "custom")
    )

    return {
        "batch_id": batch_id,
        "run_id": run_id,
        "scene": spec.scene,
        "robot": spec.robot,
        "method": spec.method,
        "method_family": spec.method_family,
        "route": resolved_route,
        "seed": spec.seed,
        "terminal_cause": terminal_cause,
        "success": success,
        "time_to_goal_s": data.get("time_to_goal_s", ""),
        "sim_time_s": data.get("sim_time_s", ""),
        "path_length_m": data.get("path_length_m", ""),
        "initial_goal_distance_m": data.get("initial_goal_distance_m", ""),
        "final_goal_distance_m": data.get("final_goal_distance_m", ""),
        "plans": data.get("plans", ""),
        "stop_requests": data.get("stop_requests", ""),
        "mean_inference_ms": data.get("mean_inference_ms", ""),
        "wall_time_s": data.get("wall_time_s", ""),
        "goal_tolerance": spec.limits.goal_tolerance,
        "max_speed": spec.limits.max_speed,
        "spec_hash": spec.spec_hash,
    }


def append_results_row(
    csv_path: Path | str,
    row: dict[str, Any],
) -> None:
    """Append or update a result row in results.csv, creating file and header if needed.

    If a row with the same run_id already exists (e.g. when resuming and retrying a failed
    or timed-out run), that row is replaced in place to prevent duplicate entries.
    """
    path = Path(csv_path)
    if not path.is_file() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=RESULTS_CSV_COLUMNS)
            writer.writeheader()
            writer.writerow(row)
        return

    rows: list[dict[str, Any]] = []
    run_id = row.get("run_id")
    replaced = False
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for existing in reader:
            if run_id and existing.get("run_id") == run_id:
                rows.append(row)
                replaced = True
            else:
                rows.append(existing)

    if not replaced:
        rows.append(row)

    tmp_path = path.with_suffix(".csv.tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)
