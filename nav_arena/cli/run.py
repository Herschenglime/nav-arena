# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The 'nav_arena run' subcommand: execute one isolated navigation episode."""

from __future__ import annotations

import argparse
import ast
from datetime import datetime
from pathlib import Path
from typing import Any
import uuid

import yaml

from nav_arena.benchmarks.io import load_json
from nav_arena.benchmarks.launcher import WorkerOutcome, compute_default_timeout, run_worker
from nav_arena.benchmarks.results import append_results_row, format_results_row
from nav_arena.benchmarks.spec import (
    VALID_METHODS,
    EpisodeLimits,
    RunSpec,
    VizCfg,
    apply_overrides,
    validate_spec,
)
from nav_arena.benchmarks.sweep import check_preflight_processes
from nav_arena.utils.logger import get_logger
from nav_arena.utils.paths import RUNS_DIR

logger = get_logger("nav_arena.cli")


def generate_run_dir(spec: RunSpec, base_dir: Path | None = None) -> Path:
    """Generate a unique run directory under RUNS_DIR.

    Pattern: <RUNS_DIR>/<YYYYmmdd_HHMMSS>_<method>_<robot>_<route>_<hex4>/
    """
    runs_root = base_dir or RUNS_DIR
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    route_name = spec.route if spec.route else "custom"
    hex4 = uuid.uuid4().hex[:4]
    return runs_root / f"{timestamp}_{spec.method}_{spec.robot}_{route_name}_{hex4}"


def _parse_policy_args(items: list[str]) -> dict[str, Any]:
    """Parse a list of KEY=VALUE strings into a dictionary with literal type casting."""
    parsed: dict[str, Any] = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"--policy-arg expects KEY=VALUE, got '{item}'")
        try:
            parsed[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            parsed[key] = value
    return parsed


def load_spec_file(spec_path: Path | str) -> dict[str, Any]:
    """Load a YAML or JSON run specification file into a raw dictionary."""
    path = Path(spec_path)
    if not path.is_file():
        raise FileNotFoundError(f"Spec file not found: {path}")
    content = path.read_text(encoding="utf-8")
    data = yaml.safe_load(content)
    if not isinstance(data, dict):
        raise ValueError(f"Spec file '{path}' must contain a mapping/dict, got {type(data).__name__}")
    return data


def handle_run(args: argparse.Namespace) -> int:
    """Execute the 'run' subcommand to run an isolated navigation episode."""
    # 1. Pre-flight guard
    if not args.force:
        conflicts = check_preflight_processes()
        if conflicts:
            logger.error("Active simulation or verification processes detected:")
            for pid, cmd in conflicts:
                logger.error("  [PID %d] %s", pid, cmd)
            logger.error("Refusing to start. Use --force to proceed anyway.")
            return 1

    # 2. Build or resolve RunSpec
    if args.spec:
        try:
            raw_dict = load_spec_file(args.spec)
            base_spec = RunSpec.from_dict(raw_dict)
        except Exception as exc:
            logger.error("Failed to load spec from '%s': %s", args.spec, exc)
            return 1

        # Extract explicitly provided CLI flags as overrides
        overrides: dict[str, Any] = {}
        if args.method is not None:
            overrides["method"] = args.method
        if args.robot is not None:
            overrides["robot"] = args.robot
        if args.scene is not None:
            overrides["scene"] = args.scene
        if args.route is not None:
            overrides["route"] = args.route
        if args.spawn is not None:
            overrides["spawn"] = args.spawn
        if args.goal is not None:
            overrides["goal"] = args.goal
        if args.spawn_yaw is not None:
            overrides["spawn_yaw"] = args.spawn_yaw
        if args.seed is not None:
            overrides["seed"] = args.seed
        if args.max_steps is not None:
            overrides["max_steps"] = args.max_steps
        if args.goal_dist is not None:
            overrides["goal_dist"] = args.goal_dist
        if args.max_speed is not None:
            overrides["max_speed"] = args.max_speed
        if args.stall_timeout is not None:
            overrides["stall_timeout"] = args.stall_timeout
        if args.gui is not None:
            overrides["gui"] = args.gui
        if args.follow_camera is not None:
            overrides["follow_camera"] = args.follow_camera
        if args.goal_overlay is not None:
            overrides["goal_overlay"] = args.goal_overlay
        if args.output is not None:
            overrides["output"] = args.output
        if args.policy_arg:
            try:
                overrides["policy_arg"] = args.policy_arg
            except Exception as exc:
                logger.error("Error parsing --policy-arg: %s", exc)
                return 1

        # If custom spawn & goal provided on CLI, clear route if route not explicitly given
        if args.spawn is not None and args.goal is not None and args.route is None:
            overrides["route"] = None
        # If named route provided on CLI, clear custom coordinates if not explicitly given
        elif args.route is not None and args.spawn is None and args.goal is None:
            overrides["spawn"] = None
            overrides["goal"] = None

        try:
            spec = apply_overrides(base_spec, overrides, logger=logger)
        except Exception as exc:
            logger.error("Failed to apply CLI overrides: %s", exc)
            return 1
    else:
        if args.method is None:
            logger.error("Argument --method is required when --spec is not provided.")
            return 1

        policy_args_dict: dict[str, Any] = {}
        if args.policy_arg:
            try:
                policy_args_dict = _parse_policy_args(args.policy_arg)
            except ValueError as exc:
                logger.error("Invalid --policy-arg: %s", exc)
                return 1

        limits = EpisodeLimits(
            max_steps=args.max_steps if args.max_steps is not None else 1500,
            goal_tolerance=args.goal_dist if args.goal_dist is not None else 0.4,
            max_speed=args.max_speed if args.max_speed is not None else 0.3,
            stall_timeout_s=args.stall_timeout if args.stall_timeout is not None else 10.0,
        )
        viz = VizCfg(
            gui=args.gui if args.gui is not None else False,
            follow_camera=args.follow_camera if args.follow_camera is not None else False,
            goal_overlay=args.goal_overlay if args.goal_overlay is not None else True,
        )
        spec = RunSpec(
            method=args.method,
            robot=args.robot if args.robot is not None else "dingo",
            scene=args.scene if args.scene is not None else "kujiale_0003",
            route=args.route,
            spawn=tuple(args.spawn) if args.spawn is not None else None,
            goal=tuple(args.goal) if args.goal is not None else None,
            spawn_yaw=args.spawn_yaw,
            seed=args.seed if args.seed is not None else 0,
            method_params=policy_args_dict,
            limits=limits,
            viz=viz,
            output_dir=Path(args.output) if args.output is not None else None,
        )

    # 3. Validate spec
    try:
        validate_spec(spec)
    except ValueError as exc:
        logger.error("Spec validation error: %s", exc)
        return 1

    # 4. Resolve output directory
    if args.output is not None:
        run_dir = Path(args.output).resolve()
    elif spec.output_dir is not None:
        run_dir = Path(spec.output_dir).resolve()
    else:
        run_dir = generate_run_dir(spec)

    return execute_single_run_process(
        spec=spec,
        run_dir=run_dir,
        timeout=args.timeout,
        quiet=args.quiet,
    )


def execute_single_run_process(
    spec: RunSpec,
    run_dir: Path,
    timeout: float | None = None,
    quiet: bool = False,
    batch_id: str = "",
) -> int:
    """Execute a worker subprocess for a single run and write its artifacts."""
    if timeout is not None and timeout <= 0:
        logger.error("--timeout must be a positive number, got %s", timeout)
        return 1
    timeout_s = float(timeout) if timeout is not None else compute_default_timeout(spec)

    run_dir.mkdir(parents=True, exist_ok=True)
    spec.output_dir = run_dir
    spec_path = run_dir / "run_spec.json"
    spec_path.write_text(spec.to_json(), encoding="utf-8")

    # Drop a stale summary left by an earlier run in the same directory.
    summary_path = run_dir / "summary.json"
    summary_path.unlink(missing_ok=True)

    try:
        outcome = run_worker(spec_path, run_dir / "worker.log", timeout_s, quiet=quiet)
    except KeyboardInterrupt:
        logger.warning("Run interrupted by user (Ctrl-C).")
        outcome = WorkerOutcome(exit_code=1)
    if outcome.timed_out:
        logger.error("Run timed out after %.1f seconds", timeout_s)

    summary_data = None
    if summary_path.is_file() and not outcome.timed_out and outcome.exit_code in (0, 2):
        summary_data = load_json(summary_path)

    failed = outcome.exit_code not in (0, 2) and not outcome.timed_out
    row = format_results_row(
        spec=spec,
        summary_data=summary_data,
        run_id=run_dir.name,
        batch_id=batch_id,
        timed_out=outcome.timed_out,
        failed=failed,
    )
    append_results_row(run_dir / "results.csv", row)

    # 0 goal reached, 2 not reached, 1 failure/timeout
    if outcome.timed_out:
        return 1
    return outcome.exit_code if outcome.exit_code in (0, 2) else 1


def _add_run_subcommand_args(parser: argparse.ArgumentParser) -> None:
    """Register all command-line arguments for the 'run' subcommand."""
    parser.add_argument("--spec", type=Path, default=None, help="Path to RunSpec YAML or JSON file.")
    parser.add_argument(
        "--method",
        choices=VALID_METHODS,
        default=None,
        help="Navigation policy to evaluate.",
    )
    parser.add_argument("--robot", type=str, default=None, help="Embodiment name (default: dingo).")
    parser.add_argument("--scene", type=str, default=None, help="Scene ID or USD path (default: kujiale_0003).")
    parser.add_argument("--route", type=str, default=None, help="Named route for the scene.")
    parser.add_argument(
        "--spawn",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        default=None,
        help="Custom route spawn position in meters.",
    )
    parser.add_argument(
        "--goal",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        default=None,
        help="Custom route goal position in meters.",
    )
    parser.add_argument("--spawn-yaw", type=float, default=None, help="Override spawn heading in radians.")
    parser.add_argument("--seed", "--seeds", type=int, default=None, dest="seed", help="RNG seed for stochastic planners.")
    parser.add_argument(
        "--policy-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra policy config field, e.g. fear_threshold=0.5 (can be repeated).",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum control steps (default: 1500).")
    parser.add_argument("--goal-dist", type=float, default=None, help="Goal tolerance distance in meters (default: 0.4).")
    parser.add_argument("--max-speed", type=float, default=None, help="Path-follower speed limit in m/s (default: 0.3).")
    parser.add_argument(
        "--stall-timeout",
        type=float,
        default=None,
        help="Timeout in seconds without progress before episode is marked stalled (default: 10.0).",
    )
    parser.add_argument(
        "--gui",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Launch interactive Kit GUI viewport (--gui / --no-gui).",
    )
    parser.add_argument(
        "--follow-camera",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable third-person follow camera display (--follow-camera / --no-follow-camera).",
    )
    parser.add_argument(
        "--goal-overlay",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Render viewport goal pin and trajectory overlay (--goal-overlay / --no-goal-overlay).",
    )
    parser.add_argument("--output", type=Path, default=None, help="Custom output directory.")
    parser.add_argument("--timeout", type=float, default=None, help="Process execution timeout override in seconds.")
    parser.add_argument("--force", action="store_true", default=False, help="Bypass pre-flight conflict check.")
    parser.add_argument("--quiet", action="store_true", default=False, help="Suppress worker stdout unless an error occurs.")


def add_run_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the 'run' subcommand."""
    run_parser = subparsers.add_parser(
        "run",
        help="Execute one isolated navigation episode.",
        description="Execute one isolated navigation episode and record artifacts.",
    )
    _add_run_subcommand_args(run_parser)
