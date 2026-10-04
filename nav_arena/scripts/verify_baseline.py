# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run one in-process navigation baseline on a PointNav route and report time-to-goal, distance remaining, and cause.

This script is a thin CLI shim over :class:`nav_arena.benchmarks.session.RunSession`.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import sys
from typing import Any

from isaaclab.app import AppLauncher

from nav_arena.benchmarks import EpisodeLimits, RunSession, RunSpec, VizCfg
from nav_arena.methods import list_policies
from nav_arena.utils import add_logger_args, configure_logging, get_logger
from nav_arena.utils.cli_args import validate_route_args

logger = get_logger("verify_baseline")


def create_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for verify_baseline."""
    parser = argparse.ArgumentParser(description="Evaluate an in-process navigation baseline on a PointNav route.")
    parser.add_argument("--method", choices=list_policies(), default=None, help="Navigation policy to evaluate.")
    # robot / scene / route are validated after the app boots: the registries import Isaac Lab.
    parser.add_argument("--robot", default="dingo", help="Registered embodiment (e.g. dingo, nova_carter).")
    parser.add_argument("--scene", default="kujiale_0003", help="InteriorAgent scene ID or USD path.")
    parser.add_argument("--route", default=None, help="Named route for the scene (default: hall_straight).")
    parser.add_argument("--spawn", nargs=2, type=float, metavar=("X", "Y"), help="Override route spawn position (m).")
    parser.add_argument("--spawn-yaw", type=float, default=None, help="Override spawn heading (rad); default faces the goal.")
    parser.add_argument("--goal", nargs=2, type=float, metavar=("X", "Y"), help="Override route goal position (m).")
    parser.add_argument("--task", choices=["pointgoal", "imagegoal"], default=None, help="Goal modality (policy default).")
    parser.add_argument("--max-steps", type=int, default=1500, help="Maximum control steps (50 Hz, so 1500 = 30 s).")
    parser.add_argument("--goal-dist", type=float, default=0.4, help="Goal tolerance in meters.")
    parser.add_argument("--max-speed", type=float, default=0.3, help="Path-follower speed limit in m/s.")
    parser.add_argument("--plan-hz", type=float, default=None, help="Replanning rate override (policy default otherwise).")
    parser.add_argument("--planner-device", default=None, help="Torch device for the policy (default: the simulation device).")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for stochastic planners.")
    parser.add_argument(
        "--policy-arg", action="append", default=[], metavar="KEY=VALUE", help="Extra policy config field, e.g. fear_threshold=0.5."
    )
    parser.add_argument(
        "--stall-timeout",
        type=float,
        default=10.0,
        help="End the episode as 'stalled' after this many simulated seconds without movement (0 disables).",
    )
    parser.add_argument(
        "--follow-camera",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Third-person viewport camera behind the robot (default: on when a Kit GUI is requested with --viz kit).",
    )
    parser.add_argument("--follow-distance", type=float, default=1.6, help="Follow camera distance behind the robot (m).")
    parser.add_argument("--follow-height", type=float, default=1.2, help="Follow camera height above the robot (m).")
    parser.add_argument(
        "--goal-overlay",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Viewport-only overlay: goal pin, tolerance ring, and the policy's current path (red while it requests a "
        "stop). Drawn as a UI layer over the viewport, so the policy's cameras cannot see it (default: on with --viz kit).",
    )
    parser.add_argument(
        "--show-goal-marker",
        action="store_true",
        help="Draw Isaac Lab's goal arrow as scene geometry. WARNING: the policy's cameras see it and treat it as an "
        "obstacle, which can make planners stop short of the goal. Prefer --goal-overlay.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Run output directory (default: cache/runs/<timestamp>).")
    add_logger_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse and validate command line arguments."""
    parser = create_parser()
    args_cli = parser.parse_args(argv)
    if args_cli.method is None:
        parser.error("the following arguments are required: --method")
    try:
        validate_route_args(args_cli.route, args_cli.spawn, args_cli.goal)
    except ValueError as error:
        parser.error(str(error))
    return args_cli


def _parse_policy_args(items: list[str]) -> dict[str, Any]:
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


def build_spec_from_cli(args: argparse.Namespace) -> RunSpec:
    """Map CLI arguments to a structured RunSpec."""
    method_params: dict[str, Any] = {}
    if getattr(args, "task", None) is not None:
        method_params["task"] = args.task
    if getattr(args, "plan_hz", None) is not None:
        method_params["plan_hz"] = args.plan_hz
    if getattr(args, "planner_device", None) is not None:
        method_params["device"] = args.planner_device
    method_params.update(_parse_policy_args(getattr(args, "policy_arg", []) or []))

    gui = ("kit" in (getattr(args, "visualizer", None) or [])) and not getattr(args, "headless", False)
    viz = VizCfg(
        gui=gui,
        follow_camera=getattr(args, "follow_camera", None),
        goal_overlay=getattr(args, "goal_overlay", None),
        follow_distance=getattr(args, "follow_distance", 1.6),
        follow_height=getattr(args, "follow_height", 1.2),
        show_goal_marker=bool(getattr(args, "show_goal_marker", False)),
    )
    limits = EpisodeLimits(
        max_steps=getattr(args, "max_steps", 1500),
        goal_tolerance=getattr(args, "goal_dist", 0.4),
        max_speed=getattr(args, "max_speed", 0.3),
        stall_timeout_s=getattr(args, "stall_timeout", 10.0),
    )

    return RunSpec(
        method=args.method,
        robot=getattr(args, "robot", "dingo"),
        scene=getattr(args, "scene", "kujiale_0003"),
        route=getattr(args, "route", None),
        spawn=tuple(args.spawn) if getattr(args, "spawn", None) is not None else None,
        goal=tuple(args.goal) if getattr(args, "goal", None) is not None else None,
        spawn_yaw=getattr(args, "spawn_yaw", None),
        seed=getattr(args, "seed", 0),
        method_params=method_params,
        limits=limits,
        viz=viz,
        output_dir=getattr(args, "output", None),
    )


def run_verification(args: argparse.Namespace, exit_on_finish: bool = True) -> Any:
    """Run baseline verification delegating to RunSession."""
    spec = build_spec_from_cli(args)
    with RunSession(spec, args_cli=args) as session:
        result = session.run_episode()
        if exit_on_finish:
            sys.exit(0 if (result.terminal_cause == "goal_reached" or result.success) else 2)
        return result


def main(argv: list[str] | None = None) -> None:
    args_cli = parse_args(argv)
    configure_logging(args_cli.log_level)
    run_verification(args_cli, exit_on_finish=True)


if __name__ == "__main__":
    main()
