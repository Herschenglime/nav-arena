# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sweep matrix planning, batch tracking, and sequential execution for nav_arena.

This module expands Cartesian sweeps across robots, methods, routes, and seeds,
configures batch directories with manifest.json and results.csv, and executes
runs under managed worker subprocesses with pre-flight process guards and Ctrl-C
handling.

Guarantees fast startup and strictly forbids importing heavy simulation, deep
learning, or ROS frameworks (isaaclab, isaacsim, omni, pxr, rclpy, torch).
"""

from __future__ import annotations

import copy
import csv
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, NamedTuple

import yaml

from nav_arena.benchmarks.manifest import (
    BatchManifest,
    RunRecord,
    RunStatus,
    _utcnow_iso,
)
from nav_arena.benchmarks.guard import check_preflight_processes
from nav_arena.benchmarks.launcher import WorkerOutcome, compute_default_timeout, run_worker
from nav_arena.benchmarks.results import RESULTS_CSV_COLUMNS, append_results_row, format_results_row
from nav_arena.benchmarks.spec import (
    VALID_METHODS,
    EpisodeLimits,
    RunSpec,
    VizCfg,
)
from nav_arena.utils.logger import get_logger
from nav_arena.utils.paths import NAVDP_ROOT, PROJECT_ROOT, RUNS_DIR

logger = get_logger("nav_arena.benchmarks.sweep")



def get_git_info(repo_path: Path | str) -> dict[str, Any]:
    """Retrieve commit SHA and dirty status for a git repository."""
    path = Path(repo_path)
    if not (path / ".git").exists() and not path.is_dir():
        return {"commit": None, "dirty": False}
    try:
        res_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            capture_output=True,
            text=True,
            check=True,
        )
        commit = res_sha.stdout.strip()
    except Exception:
        commit = None

    try:
        res_status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=path,
            capture_output=True,
            text=True,
            check=True,
        )
        dirty = bool(res_status.stdout.strip())
    except Exception:
        dirty = False

    return {"commit": commit, "dirty": dirty}


def get_host_info() -> dict[str, str]:
    """Retrieve host and environment metadata."""
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
    }


class PlannedRun(NamedTuple):
    """One expanded sweep entry: its run id, full spec, and the route label recorded in results."""

    run_id: str
    spec: RunSpec
    route_label: str


@dataclass
class SweepSpec:
    """Configuration specification for a multi-run parameter sweep."""

    name: str
    scene: str = "kujiale_0003"
    robots: list[str] = field(default_factory=lambda: ["dingo"])
    methods: list[str] = field(default_factory=list)
    routes: list[str | dict[str, Any]] = field(default_factory=list)
    seeds: list[int] = field(default_factory=lambda: [0])
    options: dict[str, Any] = field(default_factory=dict)
    method_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    timeout_s: float | None = None

    def __post_init__(self) -> None:
        if self.robots is None:
            self.robots = ["dingo"]
        elif isinstance(self.robots, (list, tuple)):
            self.robots = [str(r) for r in self.robots]
        else:
            self.robots = [str(self.robots)]

        if self.methods is None:
            self.methods = []
        elif isinstance(self.methods, (list, tuple)):
            self.methods = [str(m) for m in self.methods]
        else:
            self.methods = [str(self.methods)]

        if self.routes is None:
            self.routes = []
        elif isinstance(self.routes, (list, tuple)):
            cleaned_routes: list[str | dict[str, Any]] = []
            for r in self.routes:
                if isinstance(r, dict):
                    cleaned_routes.append(copy.deepcopy(r))
                else:
                    cleaned_routes.append(str(r))
            self.routes = cleaned_routes
        else:
            self.routes = [str(self.routes)]

        if self.seeds is None:
            self.seeds = [0]
        elif isinstance(self.seeds, (list, tuple)):
            self.seeds = [int(s) for s in self.seeds]
        else:
            self.seeds = [int(self.seeds)]

        if self.options is None:
            self.options = {}
        else:
            self.options = dict(self.options)

        if self.method_params is None:
            self.method_params = {}
        else:
            self.method_params = {str(k): dict(v) for k, v in self.method_params.items()}

        if self.timeout_s is not None:
            self.timeout_s = float(self.timeout_s)

    def to_dict(self) -> dict[str, Any]:
        """Convert SweepSpec to a dictionary."""
        return {
            "name": self.name,
            "scene": self.scene,
            "robots": list(self.robots),
            "methods": list(self.methods),
            "routes": copy.deepcopy(self.routes),
            "seeds": list(self.seeds),
            "options": copy.deepcopy(self.options),
            "method_params": copy.deepcopy(self.method_params),
            "timeout_s": self.timeout_s,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SweepSpec:
        """Construct a SweepSpec from a dictionary."""
        d = dict(data)
        return cls(
            name=str(d["name"]),
            scene=str(d.get("scene") or "kujiale_0003"),
            robots=list(d.get("robots") or ["dingo"]),
            methods=list(d.get("methods") or []),
            routes=list(d.get("routes") or []),
            seeds=list(d.get("seeds") if "seeds" in d else [0]),
            options=dict(d.get("options") or {}),
            method_params=dict(d.get("method_params") or {}),
            timeout_s=float(d["timeout_s"]) if d.get("timeout_s") is not None else None,
        )

    def to_yaml(self) -> str:
        """Serialize SweepSpec to a YAML string."""
        return yaml.dump(self.to_dict(), sort_keys=False)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> SweepSpec:
        """Construct SweepSpec from a YAML string."""
        data = yaml.safe_load(yaml_str)
        if not isinstance(data, dict):
            raise ValueError(f"Expected YAML dictionary, got {type(data).__name__}")
        return cls.from_dict(data)

    @classmethod
    def load(cls, path: Path | str) -> SweepSpec:
        """Load and parse SweepSpec from a YAML or JSON file."""
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Sweep spec file not found: {p}")
        content = p.read_text(encoding="utf-8")
        return cls.from_yaml(content)


def validate_sweep_spec(spec: SweepSpec) -> None:
    """Validate a SweepSpec for consistency and bounds.

    Raises:
        ValueError: If validation fails.
    """
    if not isinstance(spec.name, str) or not spec.name.strip():
        raise ValueError(f"Sweep name must be a non-empty string, got '{spec.name}'")

    if not isinstance(spec.scene, str) or not spec.scene.strip():
        raise ValueError(f"Sweep scene must be a non-empty string, got '{spec.scene}'")

    if not isinstance(spec.robots, list) or not spec.robots:
        raise ValueError("Sweep robots must be a non-empty list of robot names")
    for r in spec.robots:
        if not isinstance(r, str) or not r.strip():
            raise ValueError(f"Each robot in robots must be a non-empty string, got '{r}'")

    if not isinstance(spec.methods, list) or not spec.methods:
        raise ValueError("Sweep methods must be a non-empty list of method names")
    for m in spec.methods:
        if m not in VALID_METHODS:
            raise ValueError(f"Invalid method '{m}' in sweep methods. Expected one of: {VALID_METHODS}")

    if not isinstance(spec.routes, list) or not spec.routes:
        raise ValueError("Sweep routes must be a non-empty list of route names or definitions")
    for r in spec.routes:
        if isinstance(r, str):
            if not r.strip():
                raise ValueError(f"Route name string cannot be empty: '{r}'")
        elif isinstance(r, dict):
            if "name" not in r or not isinstance(r["name"], str) or not r["name"].strip():
                raise ValueError(f"Inline route must specify non-empty 'name', got {r}")
            if "spawn" not in r or not isinstance(r["spawn"], (list, tuple)) or len(r["spawn"]) != 2:
                raise ValueError(f"Inline route '{r.get('name')}' must specify 'spawn' as [x, y]")
            if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in r["spawn"]):
                raise ValueError(f"Inline route '{r.get('name')}' spawn coordinates must be finite numbers")
            if "goal" not in r or not isinstance(r["goal"], (list, tuple)) or len(r["goal"]) != 2:
                raise ValueError(f"Inline route '{r.get('name')}' must specify 'goal' as [x, y]")
            if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in r["goal"]):
                raise ValueError(f"Inline route '{r.get('name')}' goal coordinates must be finite numbers")
            if "spawn_yaw" in r:
                yaw = r["spawn_yaw"]
                if isinstance(yaw, bool) or not isinstance(yaw, (int, float)) or not math.isfinite(yaw):
                    raise ValueError(f"Inline route '{r.get('name')}' spawn_yaw must be a finite number")
        else:
            raise ValueError(f"Route entry must be a string or a dict, got {type(r).__name__}")

    if not isinstance(spec.seeds, list) or not spec.seeds:
        raise ValueError("Sweep seeds must be a non-empty list of integers")
    for s in spec.seeds:
        if isinstance(s, bool) or not isinstance(s, int) or s < 0:
            raise ValueError(f"Seed must be a non-negative integer, got '{s}'")

    if not isinstance(spec.options, dict):
        raise ValueError(f"Sweep options must be a dict, got {type(spec.options).__name__}")

    # Check known options
    if "goal_tolerance" in spec.options or "goal_dist" in spec.options:
        gt = spec.options.get("goal_tolerance", spec.options.get("goal_dist"))
        if isinstance(gt, bool) or not isinstance(gt, (int, float)) or not math.isfinite(gt) or gt <= 0:
            raise ValueError(f"options.goal_tolerance must be a positive number, got {gt}")

    if "max_speed" in spec.options:
        ms = spec.options["max_speed"]
        if isinstance(ms, bool) or not isinstance(ms, (int, float)) or not math.isfinite(ms) or ms <= 0:
            raise ValueError(f"options.max_speed must be a positive number, got {ms}")

    if "max_steps" in spec.options:
        st = spec.options["max_steps"]
        if isinstance(st, bool) or not isinstance(st, int) or st <= 0:
            raise ValueError(f"options.max_steps must be a positive integer, got {st}")

    if "stall_timeout_s" in spec.options or "stall_timeout" in spec.options:
        to = spec.options.get("stall_timeout_s", spec.options.get("stall_timeout"))
        if isinstance(to, bool) or not isinstance(to, (int, float)) or not math.isfinite(to) or to < 0:
            raise ValueError(f"options.stall_timeout_s must be a non-negative number, got {to}")

    if not isinstance(spec.method_params, dict):
        raise ValueError(f"Sweep method_params must be a dict, got {type(spec.method_params).__name__}")
    for m, p in spec.method_params.items():
        if not isinstance(p, dict):
            raise ValueError(f"method_params for method '{m}' must be a dict, got {type(p).__name__}")

    if spec.timeout_s is not None:
        if (
            isinstance(spec.timeout_s, bool)
            or not isinstance(spec.timeout_s, (int, float))
            or not math.isfinite(spec.timeout_s)
            or spec.timeout_s <= 0
        ):
            raise ValueError(f"timeout_s must be a positive number, got {spec.timeout_s}")


def apply_sweep_overrides(
    spec: SweepSpec,
    overrides: dict[str, Any],
    logger: logging.Logger | None = None,
) -> SweepSpec:
    """Apply CLI flag overrides to a SweepSpec copy and log changes.

    Args:
        spec: Base SweepSpec.
        overrides: Mapping of override fields to new values.
        logger: Optional logger for info messages.

    Returns:
        Updated SweepSpec copy.
    """
    updated = copy.deepcopy(spec)

    for key, val in overrides.items():
        if val is None:
            continue

        if key == "name":
            if logger and updated.name != val:
                logger.info("--name overrides spec.name: '%s' -> '%s'", updated.name, val)
            updated.name = str(val)

        elif key == "scene":
            if logger and updated.scene != val:
                logger.warning("--scene overrides spec.scene: '%s' -> '%s'", updated.scene, val)
            updated.scene = str(val)

        elif key in ("robots", "robot"):
            if isinstance(val, str):
                robots_list = [r.strip() for r in val.split(",") if r.strip()]
            else:
                robots_list = [str(r) for r in val]
            if logger and updated.robots != robots_list:
                logger.warning("--robots overrides spec.robots: %s -> %s", updated.robots, robots_list)
            updated.robots = robots_list

        elif key in ("methods", "method"):
            if isinstance(val, str):
                methods_list = [m.strip() for m in val.split(",") if m.strip()]
            else:
                methods_list = [str(m) for m in val]
            if logger and updated.methods != methods_list:
                logger.warning("--methods overrides spec.methods: %s -> %s", updated.methods, methods_list)
            updated.methods = methods_list

        elif key in ("routes", "route"):
            if isinstance(val, str):
                routes_list = [r.strip() for r in val.split(",") if r.strip()]
            else:
                routes_list = list(val)
            if logger and updated.routes != routes_list:
                logger.info("--routes overrides spec.routes: %s -> %s", updated.routes, routes_list)
            updated.routes = routes_list

        elif key in ("seeds", "seed"):
            if isinstance(val, str):
                seeds_list = [int(s.strip()) for s in val.split(",") if s.strip()]
            elif isinstance(val, (list, tuple)):
                seeds_list = [int(s) for s in val]
            else:
                seeds_list = [int(val)]
            if logger and updated.seeds != seeds_list:
                logger.info("--seeds overrides spec.seeds: %s -> %s", updated.seeds, seeds_list)
            updated.seeds = seeds_list

        elif key in ("timeout_s", "timeout"):
            timeout_val = float(val)
            if logger and updated.timeout_s != timeout_val:
                logger.info("--timeout overrides spec.timeout_s: %s -> %s", updated.timeout_s, timeout_val)
            updated.timeout_s = timeout_val

        elif key in ("max_steps", "goal_dist", "goal_tolerance", "max_speed", "stall_timeout", "stall_timeout_s", "gui", "follow_camera", "goal_overlay"):
            opt_key = "goal_tolerance" if key == "goal_dist" else ("stall_timeout_s" if key == "stall_timeout" else key)
            old_val = updated.options.get(opt_key)
            if logger and old_val != val:
                logger.info("--%s overrides spec.options.%s: %s -> %s", key, opt_key, old_val, val)
            updated.options[opt_key] = val

        elif key == "options" and isinstance(val, dict):
            for ok, ov in val.items():
                old_val = updated.options.get(ok)
                if logger and old_val != ov:
                    logger.info("options.%s overridden: %s -> %s", ok, old_val, ov)
                updated.options[ok] = ov

        elif key == "method_params" and isinstance(val, dict):
            for mk, mv in val.items():
                if mk not in updated.method_params:
                    updated.method_params[mk] = {}
                updated.method_params[mk].update(mv)

    return updated


def expand_sweep_matrix(spec: SweepSpec) -> list[PlannedRun]:
    """Generate Cartesian product matrix robots x methods x routes x seeds.

    Args:
        spec: Validated SweepSpec.

    Returns:
        List of PlannedRun entries in deterministic execution order.
        Run ID convention: <NNN>_<method>_<robot>_<route>_s<seed>
    """
    runs: list[PlannedRun] = []
    idx = 1

    # Outer: robots, then methods, then routes, then seeds
    for robot in spec.robots:
        for method in spec.methods:
            for route_entry in spec.routes:
                for seed in spec.seeds:
                    if isinstance(route_entry, str):
                        route_name = route_entry
                        spec_route: str | None = route_entry
                        spawn = None
                        goal = None
                        spawn_yaw = None
                    elif isinstance(route_entry, dict):
                        route_name = str(route_entry["name"])
                        spec_route = None
                        spawn = tuple(float(x) for x in route_entry["spawn"])
                        goal = tuple(float(x) for x in route_entry["goal"])
                        spawn_yaw = float(route_entry["spawn_yaw"]) if "spawn_yaw" in route_entry else None
                    else:
                        raise ValueError(f"Invalid route entry: {route_entry}")

                    run_id = f"{idx:03d}_{method}_{robot}_{route_name}_s{seed}"

                    opts = spec.options or {}
                    limits = EpisodeLimits.from_options(opts)
                    viz = VizCfg.from_options(opts)

                    params = copy.deepcopy(spec.method_params.get(method, {}))

                    run_spec = RunSpec(
                        method=method,
                        robot=robot,
                        scene=spec.scene,
                        route=spec_route,
                        spawn=spawn,
                        goal=goal,
                        spawn_yaw=spawn_yaw,
                        seed=seed,
                        method_params=params,
                        limits=limits,
                        viz=viz,
                    )
                    runs.append(PlannedRun(run_id, run_spec, route_name))
                    idx += 1

    return runs


def setup_batch_directory(
    spec: SweepSpec,
    runs_dir: Path | None = None,
    batch_id: str | None = None,
) -> tuple[Path, BatchManifest, list[PlannedRun]]:
    """Configure batch directory with batch.yaml, manifest.json, and results.csv.

    Args:
        spec: SweepSpec for the batch.
        runs_dir: Optional root runs directory override (default RUNS_DIR).
        batch_id: Optional explicit batch ID.

    Returns:
        Tuple of (batch_dir, manifest, expanded_runs).
    """
    runs_root = runs_dir or RUNS_DIR
    if batch_id is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        candidate_id = f"{timestamp}_{spec.name}"
        if (runs_root / candidate_id).exists():
            sec_str = datetime.now().strftime("%S")
            candidate_id = f"{timestamp}{sec_str}_{spec.name}"
            counter = 1
            while (runs_root / candidate_id).exists():
                candidate_id = f"{timestamp}_{spec.name}_{counter}"
                counter += 1
        batch_id = candidate_id

    batch_dir = runs_root / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    expanded_runs = expand_sweep_matrix(spec)

    # 1. Write batch.yaml
    git_info = get_git_info(PROJECT_ROOT)
    navdp_info = get_git_info(NAVDP_ROOT)
    host_info = get_host_info()
    created_at = _utcnow_iso()

    batch_info = {
        "batch_id": batch_id,
        "name": spec.name,
        "created_at": created_at,
        "started_at": created_at,
        "ended_at": None,
        "status": "running",
        "git": git_info,
        "navdp": navdp_info,
        "host": host_info,
        "spec": spec.to_dict(),
    }
    batch_yaml_path = batch_dir / "batch.yaml"
    batch_yaml_path.write_text(yaml.dump(batch_info, sort_keys=False), encoding="utf-8")

    # 2. Initialize manifest.json with all runs in queued state
    records: list[RunRecord] = []
    for planned in expanded_runs:
        run_id, r_spec = planned.run_id, planned.spec
        records.append(
            RunRecord(
                id=run_id,
                status=RunStatus.QUEUED,
                method=r_spec.method,
                method_family=r_spec.method_family,
                method_params=r_spec.method_params,
                robot=r_spec.robot,
                scene=r_spec.scene,
                route=planned.route_label,
                seed=r_spec.seed,
                run_dir=run_id,
            )
        )

    manifest = BatchManifest(
        batch_id=batch_id,
        runs=records,
        created_at=created_at,
        metadata={"name": spec.name, "scene": spec.scene},
    )
    manifest.save(batch_dir / "manifest.json")

    # 3. Initialize empty results.csv with header
    csv_path = batch_dir / "results.csv"
    if not csv_path.is_file() or csv_path.stat().st_size == 0:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=RESULTS_CSV_COLUMNS)
            writer.writeheader()

    return batch_dir, manifest, expanded_runs


def _update_batch_yaml(batch_dir: Path, **kwargs: Any) -> None:
    """Safely update top-level keys in batch.yaml."""
    yaml_path = batch_dir / "batch.yaml"
    if not yaml_path.is_file():
        return
    try:
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        data.update(kwargs)
        yaml_path.write_text(yaml.dump(data, sort_keys=False), encoding="utf-8")
    except Exception as exc:
        logger.debug("Failed to update batch.yaml: %s", exc)


def resolve_batch_for_resume(
    spec: SweepSpec | None,
    batch_arg: str | Path | None,
    runs_dir: Path | None = None,
) -> Path:
    """Find and validate a batch directory for resuming.

    Args:
        spec: Optional parsed SweepSpec.
        batch_arg: Optional directory path or batch_id string.
        runs_dir: Optional runs root directory.

    Returns:
        Resolved Path to the batch directory containing manifest.json.

    Raises:
        FileNotFoundError: If the batch directory or manifest.json cannot be found.
    """
    runs_root = runs_dir or RUNS_DIR

    # 1. If batch_arg is explicitly a directory or file path
    if batch_arg is not None:
        p = Path(batch_arg)
        if p.is_file() and p.name in ("manifest.json", "batch.yaml"):
            p = p.parent
        if p.is_dir() and (p / "manifest.json").is_file():
            return p.resolve()

        # Check under runs_root
        candidate = runs_root / str(batch_arg)
        if candidate.is_dir() and (candidate / "manifest.json").is_file():
            return candidate.resolve()

        # Check if batch_arg matches suffix under runs_root
        if runs_root.is_dir():
            matching = [
                d for d in runs_root.iterdir()
                if d.is_dir() and d.name.endswith(f"_{batch_arg}") and (d / "manifest.json").is_file()
            ]
            if matching:
                matching.sort(key=lambda d: d.stat().st_mtime, reverse=True)
                return matching[0].resolve()

    # 2. Look for matching spec.name or most recent under runs_root
    if runs_root.is_dir():
        matching = [
            d for d in runs_root.iterdir()
            if d.is_dir() and (d / "manifest.json").is_file()
        ]
        if spec is not None:
            suffix = f"_{spec.name}"
            matching = [d for d in matching if d.name.endswith(suffix)]
        if matching:
            matching.sort(key=lambda d: d.stat().st_mtime, reverse=True)
            return matching[0].resolve()

    raise FileNotFoundError(
        f"Could not locate an existing batch to resume for '{batch_arg or (spec.name if spec else 'unknown')}'. "
        f"Provide the batch directory path or ensure manifest.json exists."
    )


def _prepare_batch(
    spec: SweepSpec,
    batch_dir: Path | None,
    runs_dir: Path | None,
    batch_id: str | None,
    resume: bool,
) -> tuple[Path, BatchManifest, list[PlannedRun]]:
    """Create a new batch directory, or load an existing one and queue its unfinished runs for retry."""
    if resume:
        batch_dir = batch_dir or resolve_batch_for_resume(spec, batch_arg=batch_id, runs_dir=runs_dir)
        manifest_path = batch_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Cannot resume: {manifest_path} not found")
        manifest = BatchManifest.load(manifest_path)
        for record in manifest.runs:
            if record.status not in (RunStatus.DONE.value, RunStatus.SKIPPED.value):
                manifest.reset_run(record.id)
        manifest.save(manifest_path)
        return batch_dir, manifest, expand_sweep_matrix(spec)

    if batch_dir is not None:
        manifest_path = batch_dir / "manifest.json"
        if manifest_path.is_file():
            return batch_dir, BatchManifest.load(manifest_path), expand_sweep_matrix(spec)
        return setup_batch_directory(spec, runs_dir=batch_dir.parent, batch_id=batch_dir.name)
    return setup_batch_directory(spec, runs_dir=runs_dir, batch_id=batch_id)


def _read_summary(summary_path: Path) -> dict[str, Any] | None:
    """Read a worker's ``summary.json``; ``None`` when missing or unreadable."""
    if not summary_path.is_file():
        return None
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Failed to parse %s: %s", summary_path, exc)
        return None


def _record_outcome(
    manifest: BatchManifest,
    batch_dir: Path,
    planned: PlannedRun,
    outcome: WorkerOutcome,
    timeout_s: float,
) -> bool:
    """Record one finished worker in the manifest and results.csv. Returns True for an infrastructure failure."""
    run_id, run_spec = planned.run_id, planned.spec
    summary = _read_summary(batch_dir / run_id / "summary.json")
    row_kwargs = dict(run_id=run_id, batch_id=manifest.batch_id, route=planned.route_label)

    if outcome.timed_out:
        logger.error("Run %s timed out after %.1f seconds", run_id, timeout_s)
        manifest.mark_timeout(run_id, error=f"Run timed out after {timeout_s:.1f}s", run_dir=run_id)
        row = format_results_row(run_spec, None, timed_out=True, **row_kwargs)
        infra_error = True
    elif outcome.exit_code in (0, 2):
        terminal_cause = str(summary.get("terminal_cause", "unknown")) if summary else "unknown"
        manifest.mark_done(run_id, exit_code=outcome.exit_code, terminal_cause=terminal_cause, run_dir=run_id)
        row = format_results_row(run_spec, summary, **row_kwargs)
        logger.info("Finished [%s]: terminal_cause=%s (exit %d)", run_id, terminal_cause, outcome.exit_code)
        infra_error = False
    else:
        logger.error("Run %s failed with worker exit code %d", run_id, outcome.exit_code)
        manifest.mark_failed(
            run_id,
            exit_code=outcome.exit_code,
            error=f"Worker failed with exit code {outcome.exit_code}",
            run_dir=run_id,
        )
        row = format_results_row(run_spec, None, failed=True, **row_kwargs)
        infra_error = True

    append_results_row(batch_dir / "results.csv", row)
    manifest.save(batch_dir / "manifest.json")
    return infra_error


def _execute_planned_run(
    spec: SweepSpec,
    manifest: BatchManifest,
    batch_dir: Path,
    planned: PlannedRun,
    quiet: bool,
    timeout_override: float | None,
) -> bool:
    """Run one worker for ``planned`` and record its outcome. Returns True for an infrastructure failure."""
    run_id, run_spec = planned.run_id, planned.spec
    run_dir = batch_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    run_spec.output_dir = run_dir
    spec_path = run_dir / "run_spec.json"
    spec_path.write_text(run_spec.to_json(), encoding="utf-8")

    stale_summary = run_dir / "summary.json"
    if stale_summary.is_file():
        stale_summary.unlink()

    if timeout_override is not None and timeout_override > 0:
        timeout_s = timeout_override
    elif spec.timeout_s is not None and spec.timeout_s > 0:
        timeout_s = spec.timeout_s
    else:
        timeout_s = compute_default_timeout(run_spec)

    logger.info("Running [%s] (timeout: %.1fs)", run_id, timeout_s)

    def mark_running(pid: int) -> None:
        manifest.mark_running(run_id, pid=pid, run_dir=run_id)
        manifest.save(batch_dir / "manifest.json")

    outcome = run_worker(spec_path, run_dir / "worker.log", timeout_s, quiet=quiet, on_start=mark_running)
    return _record_outcome(manifest, batch_dir, planned, outcome, timeout_s)


def execute_sweep(
    spec: SweepSpec,
    batch_dir: Path | None = None,
    runs_dir: Path | None = None,
    batch_id: str | None = None,
    resume: bool = False,
    force: bool = False,
    quiet: bool = False,
    timeout_override: float | None = None,
) -> int:
    """Execute a batch matrix sweep sequentially with worker subprocesses.

    Args:
        spec: Sweep specification.
        batch_dir: Optional explicit batch directory (used for resume or custom paths).
        runs_dir: Root runs directory.
        batch_id: Custom batch identifier.
        resume: If True, resume an existing batch, skipping completed runs.
        force: Bypass pre-flight conflict check.
        quiet: Suppress live streaming of worker stdout.
        timeout_override: Per-run timeout override in seconds.

    Returns:
        Exit code: 0 if all runs executed without infrastructure errors/timeouts,
        1 if any run timed out, crashed, or was interrupted.
    """
    if not force:
        conflicts = check_preflight_processes()
        if conflicts:
            logger.error("Active simulation or verification processes detected:")
            for pid, cmd in conflicts:
                logger.error("  [PID %d] %s", pid, cmd)
            logger.error("Refusing to start sweep. Use --force to proceed anyway.")
            return 1

    validate_sweep_spec(spec)
    batch_dir, manifest, planned_runs = _prepare_batch(spec, batch_dir, runs_dir, batch_id, resume)
    logger.info("Starting sweep '%s' in %s (%d runs)", spec.name, batch_dir, len(planned_runs))

    active_run_id: str | None = None
    infra_error = False
    try:
        for planned in planned_runs:
            status = manifest.get_run(planned.run_id).status
            if status in (RunStatus.DONE.value, RunStatus.SKIPPED.value):
                logger.info("Skipping run %s (status: %s)", planned.run_id, status)
                continue
            active_run_id = planned.run_id
            infra_error |= _execute_planned_run(spec, manifest, batch_dir, planned, quiet, timeout_override)
            active_run_id = None
    except KeyboardInterrupt:
        logger.warning("Sweep interrupted by user (Ctrl-C).")
        if active_run_id is not None:
            manifest.mark_failed(
                active_run_id, exit_code=1, error="Interrupted by user (SIGINT)", run_dir=active_run_id
            )
            manifest.save(batch_dir / "manifest.json")
        _update_batch_yaml(batch_dir, status="interrupted", ended_at=_utcnow_iso())
        return 1

    final_status = "failed" if infra_error else "completed"
    _update_batch_yaml(batch_dir, status=final_status, ended_at=_utcnow_iso())
    logger.info("Sweep batch '%s' %s", manifest.batch_id, final_status)

    counts = manifest.counts_by_status()
    unfinished = not manifest.is_finished()
    if unfinished or counts.get(RunStatus.FAILED.value, 0) > 0 or counts.get(RunStatus.TIMEOUT.value, 0) > 0:
        return 1
    return 0
