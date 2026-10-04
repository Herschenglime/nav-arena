# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The 'nav_arena sweep' subcommand: execute a batch matrix of runs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from nav_arena.benchmarks.sweep import (
    SweepSpec,
    apply_sweep_overrides,
    execute_sweep,
    resolve_batch_for_resume,
    validate_sweep_spec,
)
from nav_arena.utils.logger import get_logger

logger = get_logger("nav_arena.cli")


def handle_sweep(args: argparse.Namespace) -> int:
    """Execute the 'sweep' subcommand to run a matrix sweep across benchmarks."""
    if args.jobs != 1:
        logger.error("--jobs %d is not supported: runs are sequential (use --jobs 1).", args.jobs)
        return 1
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
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Concurrent runs. Reserved: only 1 (sequential) is supported until parallel runs are investigated.",
    )
    parser.add_argument("--force", action="store_true", default=False, help="Bypass pre-flight conflict check.")
    parser.add_argument("--quiet", action="store_true", default=False, help="Suppress worker stdout unless an error occurs.")
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum control steps.")
    parser.add_argument("--goal-dist", type=float, default=None, help="Goal tolerance distance in meters.")
    parser.add_argument("--max-speed", type=float, default=None, help="Path-follower speed limit in m/s.")
    parser.add_argument("--stall-timeout", type=float, default=None, help="Stall timeout in seconds.")
    parser.add_argument("--gui", action=argparse.BooleanOptionalAction, default=None, help="Kit GUI viewport (--gui / --no-gui).")
    parser.add_argument("--follow-camera", action=argparse.BooleanOptionalAction, default=None, help="Follow camera (--follow-camera / --no-follow-camera).")
    parser.add_argument("--goal-overlay", action=argparse.BooleanOptionalAction, default=None, help="Goal overlay (--goal-overlay / --no-goal-overlay).")


def add_sweep_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the 'sweep' subcommand."""
    sweep_parser = subparsers.add_parser(
        "sweep",
        help="Execute a batch matrix sweep across methods, robots, routes, and seeds.",
        description="Execute a batch matrix sweep across methods, robots, routes, and seeds.",
    )
    _add_sweep_subcommand_args(sweep_parser)
