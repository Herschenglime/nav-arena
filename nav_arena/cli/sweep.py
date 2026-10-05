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
from nav_arena.cli._common import CliError
from nav_arena.utils.logger import get_logger

logger = get_logger("nav_arena.cli")


_LOAD_ERRORS = (OSError, ValueError, KeyError, TypeError, yaml.YAMLError)
"""Exceptions a malformed spec or batch file can raise."""

_SWEEP_FLAG_OVERRIDES = (  # (argparse attribute, apply_sweep_overrides key)
    ("name", "name"),
    ("scene", "scene"),
    ("methods", "methods"),
    ("routes", "routes"),
    ("seeds", "seeds"),
    ("robots", "robots"),
    ("timeout", "timeout_s"),
    ("max_steps", "max_steps"),
    ("goal_dist", "goal_dist"),
    ("max_speed", "max_speed"),
    ("stall_timeout", "stall_timeout"),
    ("gui", "gui"),
    ("follow_camera", "follow_camera"),
    ("goal_overlay", "goal_overlay"),
)


def _split(value: str) -> list[str]:
    return [item.strip() for item in value.split(",")]


def _spec_from_flags(args: argparse.Namespace) -> SweepSpec:
    scene = args.scene or "kujiale_0003"
    return SweepSpec(
        name=args.name or f"sweep_{scene}",
        scene=scene,
        robots=_split(args.robots) if args.robots else ["dingo"],
        methods=_split(args.methods) if args.methods else ["iplanner"],
        routes=_split(args.routes) if args.routes else ["hall_straight"],
        seeds=[int(seed) for seed in _split(args.seeds)] if args.seeds else [0],
    )


def _load_spec_file(path: Any) -> SweepSpec:
    try:
        return SweepSpec.load(path)
    except _LOAD_ERRORS as exc:
        raise CliError(f"Failed to load sweep spec from '{path}': {exc}") from exc


def _saved_spec(batch_dir: Path) -> SweepSpec | None:
    """The sweep spec recorded in a batch's batch.yaml, if readable."""
    batch_yaml = batch_dir / "batch.yaml"
    if not batch_yaml.is_file():
        return None
    try:
        data = yaml.safe_load(batch_yaml.read_text(encoding="utf-8")) or {}
        return SweepSpec.from_dict(data["spec"]) if isinstance(data.get("spec"), dict) else None
    except _LOAD_ERRORS as exc:
        logger.warning("Failed to read spec from batch.yaml in %s: %s", batch_dir, exc)
        return None


def _resume_target(args: argparse.Namespace) -> tuple[Path, SweepSpec]:
    """The batch directory to resume and the spec to resume it with (a given spec file wins over the saved one)."""
    spec: SweepSpec | None = None
    batch_dir: Path | None = None
    if args.spec:
        path = Path(args.spec)
        if path.is_file() and path.name in ("manifest.json", "batch.yaml"):
            batch_dir = path.parent.resolve()
        elif path.is_dir() and (path / "manifest.json").is_file():
            batch_dir = path.resolve()
        elif path.is_file():
            spec = _load_spec_file(path)
    try:
        if batch_dir is None:
            batch_dir = resolve_batch_for_resume(spec, batch_arg=None if spec else (args.spec or args.name))
    except FileNotFoundError as exc:
        raise CliError(f"Resume failed: {exc}") from exc

    spec = spec or _saved_spec(batch_dir)
    if spec is None:
        raise CliError("Could not resolve sweep specification for resuming.")
    return batch_dir, spec


def handle_sweep(args: argparse.Namespace) -> int:
    """Execute the 'sweep' subcommand to run a matrix sweep across benchmarks."""
    if args.jobs != 1:
        logger.error("--jobs %d is not supported: runs are sequential (use --jobs 1).", args.jobs)
        return 1
    try:
        batch_dir: Path | None = None
        if args.resume:
            batch_dir, spec = _resume_target(args)
        else:
            spec = _load_spec_file(args.spec) if args.spec else _spec_from_flags(args)

        overrides = {
            key: getattr(args, attr) for attr, key in _SWEEP_FLAG_OVERRIDES if getattr(args, attr, None) is not None
        }
        try:
            spec = apply_sweep_overrides(spec, overrides, logger=logger)
        except _LOAD_ERRORS as exc:
            raise CliError(f"Failed to apply sweep overrides: {exc}") from exc
        try:
            validate_sweep_spec(spec)
        except ValueError as exc:
            raise CliError(f"Sweep validation error: {exc}") from exc
    except CliError as exc:
        logger.error("%s", exc)
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
