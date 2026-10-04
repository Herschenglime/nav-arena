# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command-line entry point for nav_arena.

Provides subcommands for running single benchmark episodes, running batch sweeps,
inspecting tracked runs, and invoking utility tools.

This module guarantees fast startup (<0.1s) and strictly forbids importing heavy
simulation, deep learning, or ROS frameworks (isaaclab, isaacsim, omni, pxr, rclpy, torch).
"""

from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import threading
from typing import Any
import uuid

import yaml

from nav_arena.benchmarks.spec import (
    VALID_METHODS,
    EpisodeLimits,
    RunSpec,
    VizCfg,
    apply_overrides,
    validate_spec,
)
from nav_arena.benchmarks.sweep import (
    RESULTS_CSV_COLUMNS,
    SweepSpec,
    append_results_row,
    apply_sweep_overrides,
    check_preflight_processes,
    compute_default_timeout,
    execute_sweep,
    format_results_row,
    resolve_batch_for_resume,
    validate_sweep_spec,
)
from nav_arena.utils.logger import get_logger
from nav_arena.utils.paths import RUNS_DIR
from nav_arena.utils.process import managed_process

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


def format_results_row(
    spec: RunSpec,
    summary_data: dict[str, Any] | None,
    run_id: str,
    batch_id: str = "",
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

    return {
        "batch_id": batch_id,
        "run_id": run_id,
        "scene": spec.scene,
        "robot": spec.robot,
        "method": spec.method,
        "method_family": spec.method_family,
        "route": spec.route if spec.route is not None else "custom",
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

    spec.output_dir = run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    # 5. Save resolved spec
    run_spec_path = run_dir / "run_spec.json"
    run_spec_path.write_text(spec.to_json(), encoding="utf-8")

    # Clean up any stale summary from a previous run in the same output directory
    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        try:
            summary_path.unlink()
        except OSError:
            pass

    # 6. Calculate execution timeout
    if args.timeout is not None:
        if args.timeout <= 0:
            logger.error("--timeout must be a positive number, got %s", args.timeout)
            return 1
        timeout_s = float(args.timeout)
    else:
        timeout_s = compute_default_timeout(spec)

    # 7. Execute worker subprocess under managed_process
    worker_cmd = [
        sys.executable,
        "-u",
        "-m",
        "nav_arena.benchmarks.worker",
        str(run_spec_path),
    ]

    log_path = run_dir / "worker.log"
    log_lines: list[str] = []
    timed_out = False
    exit_code = 1

    try:
        with open(log_path, "w", encoding="utf-8") as log_file:
            reader_thread: threading.Thread | None = None
            try:
                with managed_process(
                    worker_cmd,
                    timeout=5.0,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                ) as proc:
                    def _stream_output() -> None:
                        if proc.stdout is None:
                            return
                        for line in iter(proc.stdout.readline, ""):
                            log_file.write(line)
                            log_file.flush()
                            log_lines.append(line)
                            if not args.quiet:
                                sys.stdout.write(line)
                                sys.stdout.flush()

                    reader_thread = threading.Thread(target=_stream_output, daemon=True)
                    reader_thread.start()

                    try:
                        exit_code = proc.wait(timeout=timeout_s)
                    except subprocess.TimeoutExpired:
                        timed_out = True
                        logger.error("Run timed out after %.1f seconds", timeout_s)
                        exit_code = 1
            finally:
                if reader_thread is not None:
                    reader_thread.join(timeout=5.0)
    except KeyboardInterrupt:
        logger.warning("Run interrupted by user (Ctrl-C).")
        exit_code = 1
    except Exception as exc:
        logger.error("Failed to execute worker process: %s", exc)
        exit_code = 1

    # If --quiet, dump output on error
    if args.quiet and (exit_code not in (0, 2) or timed_out):
        sys.stderr.write("".join(log_lines))
        sys.stderr.flush()

    # 8. Record summary to results.csv
    summary_data: dict[str, Any] | None = None
    if summary_path.is_file() and not timed_out and exit_code in (0, 2):
        try:
            summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to read summary.json: %s", exc)

    failed = (exit_code not in (0, 2)) and not timed_out
    csv_path = run_dir / "results.csv"
    row = format_results_row(
        spec=spec,
        summary_data=summary_data,
        run_id=run_dir.name,
        batch_id="",
        timed_out=timed_out,
        failed=failed,
    )
    append_results_row(csv_path, row)

    # 9. Return exit code: 0 goal reached, 2 not reached, 1 failure/timeout
    if timed_out:
        return 1
    if exit_code in (0, 2):
        return exit_code
    return 1


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


def create_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser for nav_arena CLI."""
    parser = argparse.ArgumentParser(
        prog="nav_arena",
        description="Unified navigation benchmarking and evaluation framework for Isaac Sim.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", metavar="COMMAND")

    # 1. run
    run_parser = subparsers.add_parser(
        "run",
        help="Execute one isolated navigation episode.",
        description="Execute one isolated navigation episode and record artifacts.",
    )
    _add_run_subcommand_args(run_parser)

def handle_sweep(args: argparse.Namespace) -> int:
    """Execute the 'sweep' subcommand to run a matrix sweep across benchmarks."""
    batch_dir: Path | None = None
    spec: SweepSpec | None = None

    if args.resume:
        # 1. Check if args.spec points to an existing batch directory or batch file
        if args.spec:
            spec_path = Path(args.spec)
            if spec_path.is_file() and spec_path.name in ("manifest.json", "batch.yaml"):
                batch_dir = spec_path.parent.resolve()
            elif spec_path.is_dir() and (spec_path / "manifest.json").is_file():
                batch_dir = spec_path.resolve()
            elif spec_path.is_file():
                try:
                    spec = SweepSpec.load(spec_path)
                except Exception as exc:
                    logger.error("Failed to load sweep spec from '%s': %s", args.spec, exc)
                    return 1
                try:
                    batch_dir = resolve_batch_for_resume(spec, batch_arg=None)
                except FileNotFoundError as exc:
                    logger.error("Resume failed: %s", exc)
                    return 1

        if batch_dir is None:
            try:
                batch_dir = resolve_batch_for_resume(spec, batch_arg=args.spec or args.name)
            except FileNotFoundError as exc:
                logger.error("Resume failed: %s", exc)
                return 1

        # Load spec from batch_dir / "batch.yaml" if not already loaded from YAML file
        batch_yaml = batch_dir / "batch.yaml"
        if batch_yaml.is_file():
            try:
                data = yaml.safe_load(batch_yaml.read_text(encoding="utf-8")) or {}
                if "spec" in data and isinstance(data["spec"], dict):
                    saved_spec = SweepSpec.from_dict(data["spec"])
                    if spec is None:
                        spec = saved_spec
            except Exception as exc:
                logger.warning("Failed to read spec from batch.yaml in %s: %s", batch_dir, exc)

        if spec is None:
            logger.error("Could not resolve sweep specification for resuming.")
            return 1
    else:
        if args.spec:
            try:
                spec = SweepSpec.load(args.spec)
            except Exception as exc:
                logger.error("Failed to load sweep spec from '%s': %s", args.spec, exc)
                return 1
        else:
            scene = args.scene or "kujiale_0003"
            methods = [m.strip() for m in args.methods.split(",")] if args.methods else ["iplanner"]
            routes = [r.strip() for r in args.routes.split(",")] if args.routes else ["hall_straight"]
            seeds = [int(s.strip()) for s in args.seeds.split(",")] if args.seeds else [0]
            robots = [r.strip() for r in args.robots.split(",")] if args.robots else ["dingo"]
            name = args.name or f"sweep_{scene}"
            spec = SweepSpec(
                name=name,
                scene=scene,
                robots=robots,
                methods=methods,
                routes=routes,
                seeds=seeds,
            )

    if spec is None:
        logger.error("Could not resolve sweep specification.")
        return 1

    # Extract CLI overrides
    overrides: dict[str, Any] = {}
    if args.name is not None:
        overrides["name"] = args.name
    if args.scene is not None:
        overrides["scene"] = args.scene
    if args.methods is not None:
        overrides["methods"] = args.methods
    if args.routes is not None:
        overrides["routes"] = args.routes
    if args.seeds is not None:
        overrides["seeds"] = args.seeds
    if args.robots is not None:
        overrides["robots"] = args.robots
    if args.timeout is not None:
        overrides["timeout_s"] = args.timeout
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

    try:
        spec = apply_sweep_overrides(spec, overrides, logger=logger)
    except Exception as exc:
        logger.error("Failed to apply sweep overrides: %s", exc)
        return 1

    try:
        validate_sweep_spec(spec)
    except ValueError as exc:
        logger.error("Sweep validation error: %s", exc)
        return 1

    return execute_sweep(
        spec,
        batch_dir=batch_dir,
        resume=args.resume,
        force=args.force,
        quiet=args.quiet,
        timeout_override=args.timeout,
    )


def _add_sweep_subcommand_args(parser: argparse.ArgumentParser) -> None:
    """Register CLI arguments for the sweep subcommand."""
    parser.add_argument("spec", nargs="?", default=None, help="Sweep specification YAML path or batch directory to resume.")
    parser.add_argument("--scene", type=str, default=None, help="Override scene name.")
    parser.add_argument("--methods", "--method", type=str, default=None, dest="methods", help="Comma-separated list of methods (e.g. iplanner,navdp).")
    parser.add_argument("--routes", "--route", type=str, default=None, dest="routes", help="Comma-separated list of routes.")
    parser.add_argument("--seeds", "--seed", type=str, default=None, dest="seeds", help="Comma-separated list of integer seeds.")
    parser.add_argument("--robots", "--robot", type=str, default=None, dest="robots", help="Comma-separated list of robots.")
    parser.add_argument("--name", type=str, default=None, help="Override sweep name.")
    parser.add_argument("--timeout", type=float, default=None, help="Per-run execution timeout in seconds.")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume an existing sweep batch.")
    parser.add_argument("--force", action="store_true", default=False, help="Bypass pre-flight conflict check.")
    parser.add_argument("--quiet", action="store_true", default=False, help="Suppress worker stdout unless an error occurs.")
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum control steps.")
    parser.add_argument("--goal-dist", type=float, default=None, help="Goal tolerance distance in meters.")
    parser.add_argument("--max-speed", type=float, default=None, help="Path-follower speed limit in m/s.")
    parser.add_argument("--stall-timeout", type=float, default=None, help="Stall timeout in seconds.")
    parser.add_argument("--gui", action=argparse.BooleanOptionalAction, default=None, help="Kit GUI viewport (--gui / --no-gui).")
    parser.add_argument("--follow-camera", action=argparse.BooleanOptionalAction, default=None, help="Follow camera (--follow-camera / --no-follow-camera).")
    parser.add_argument("--goal-overlay", action=argparse.BooleanOptionalAction, default=None, help="Goal overlay (--goal-overlay / --no-goal-overlay).")


def create_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser for nav_arena CLI."""
    parser = argparse.ArgumentParser(
        prog="nav_arena",
        description="Unified navigation benchmarking and evaluation framework for Isaac Sim.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", metavar="COMMAND")

    # 1. run
    run_parser = subparsers.add_parser(
        "run",
        help="Execute one isolated navigation episode.",
        description="Execute one isolated navigation episode and record artifacts.",
    )
    _add_run_subcommand_args(run_parser)

    # 2. sweep (Phase 2B)
    sweep_parser = subparsers.add_parser(
        "sweep",
        help="Execute a batch matrix sweep across methods, robots, routes, and seeds.",
        description="Execute a batch matrix sweep across methods, robots, routes, and seeds.",
    )
    _add_sweep_subcommand_args(sweep_parser)

    # 3. runs (Phase 3)
    runs_parser = subparsers.add_parser(
        "runs",
        help="Inspect, list, compare, or rerun recorded benchmark episodes.",
    )
    runs_sub = runs_parser.add_subparsers(dest="runs_action", metavar="ACTION")
    runs_sub.add_parser("list", help="List batches or runs.")
    runs_sub.add_parser("show", help="Show details for a run or batch.")
    runs_sub.add_parser("compare", help="Compare metrics across methods, routes, or robots.")
    runs_sub.add_parser("rerun", help="Re-execute a previously recorded run.")

    # 4. doctor (Phase 4)
    subparsers.add_parser(
        "doctor",
        help="Inspect simulation environment, weights, checkouts, and caches.",
    )

    # 5. routes (Phase 4)
    routes_parser = subparsers.add_parser(
        "routes",
        help="List and inspect registered PointNav routes.",
    )
    routes_sub = routes_parser.add_subparsers(dest="routes_action", metavar="ACTION")
    routes_sub.add_parser("list", help="List routes for a scene.")
    routes_sub.add_parser("show", help="Show route details or preview.")

    # 6. map (Phase 4)
    map_parser = subparsers.add_parser(
        "map",
        help="Generate or inspect 2D scene occupancy maps.",
    )
    map_sub = map_parser.add_subparsers(dest="map_action", metavar="ACTION")
    map_sub.add_parser("generate", help="Generate 2D occupancy grid.")
    map_sub.add_parser("show", help="Display 2D occupancy map.")

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point returning exit code."""
    parser = create_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    if args.subcommand is None:
        parser.print_help()
        return 0

    if args.subcommand == "run":
        return handle_run(args)

    if args.subcommand == "sweep":
        return handle_sweep(args)

    if args.subcommand in ("runs", "doctor", "routes", "map"):
        logger.error("Subcommand '%s' is not yet implemented.", args.subcommand)
        return 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
