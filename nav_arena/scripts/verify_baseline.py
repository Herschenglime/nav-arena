# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run one in-process navigation baseline on a PointNav route and report time-to-goal, distance remaining, and cause.

This is the integration harness for the learned baselines (iPlanner, ViNT, NavDP, VIPlanner, X-NavDP); it is thin by
design. Episode logic lives in :mod:`nav_arena.methods.in_process.runner`, policies in
:mod:`nav_arena.methods.in_process`. A centralized benchmark runner is expected to supersede this script.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime
import math
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

from nav_arena.core import launch_simulation_app
from nav_arena.methods import list_policies
from nav_arena.utils import RUNS_DIR, add_logger_args, configure_logging, get_logger

logger = get_logger("verify_baseline")

parser = argparse.ArgumentParser(description="Evaluate an in-process navigation baseline on a PointNav route.")
parser.add_argument("--method", choices=list_policies(), required=True, help="Navigation policy to evaluate.")
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
args_cli = parser.parse_args()
# The baselines are camera-driven: RTX rendering must be enabled before the app boots.
args_cli.enable_cameras = True
GUI = "kit" in (getattr(args_cli, "visualizer", None) or [])
OVERLAY = args_cli.goal_overlay if args_cli.goal_overlay is not None else GUI

configure_logging(args_cli.log_level)


def _parse_policy_args(items: list[str]) -> dict:
    parsed = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"--policy-arg expects KEY=VALUE, got '{item}'")
        try:
            parsed[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            parsed[key] = value
    return parsed


def run_verification():
    from nav_arena.methods.in_process import EpisodeCfg, FollowerCfg, get_policy, run_episode
    from nav_arena.scenes import DEFAULT_ROUTE, get_route
    from nav_arena.tasks import PointNavTask, create_point_nav_env_cfg

    if args_cli.spawn and args_cli.goal:
        spawn_xy, goal_xy = tuple(args_cli.spawn), tuple(args_cli.goal)
        spawn_yaw = (
            args_cli.spawn_yaw
            if args_cli.spawn_yaw is not None
            else math.atan2(goal_xy[1] - spawn_xy[1], goal_xy[0] - spawn_xy[0])
        )
        route_name = "custom"
    else:
        route = get_route(args_cli.route or DEFAULT_ROUTE, args_cli.scene)
        spawn_xy, goal_xy = route.spawn_xy, route.goal_xy
        spawn_yaw = args_cli.spawn_yaw if args_cli.spawn_yaw is not None else route.spawn_yaw
        route_name = route.name
    spawn_quat = (0.0, 0.0, math.sin(spawn_yaw / 2.0), math.cos(spawn_yaw / 2.0))

    logger.section(f"BASELINE '{args_cli.method}' ON '{args_cli.robot}': {args_cli.scene}/{route_name}")
    logger.info(f"Spawn {spawn_xy} (yaw {math.degrees(spawn_yaw):.0f} deg) -> goal {goal_xy}")

    gui = GUI
    follow = args_cli.follow_camera if args_cli.follow_camera is not None else gui
    if args_cli.show_goal_marker:
        logger.warning("Goal marker enabled: policies' cameras will see it as an obstacle at the goal.")

    step_dt = 0.02  # dt = 0.01 s, decimation = 2
    env_cfg = create_point_nav_env_cfg(
        scene_id_or_path=args_cli.scene,
        robot_spawn_pos=(spawn_xy[0], spawn_xy[1], 0.25),
        robot_spawn_rot=spawn_quat,
        goal_pos=goal_xy,
        goal_threshold=args_cli.goal_dist,
        episode_length_s=args_cli.max_steps * step_dt + 10.0,
        robot_name=args_cli.robot,
        enable_camera=True,
        enable_goal_camera=args_cli.task == "imagegoal"
        or (args_cli.task is None and args_cli.method == "vint"),
        show_goal_marker=args_cli.show_goal_marker,
        scene_queries=follow,
    )
    task = PointNavTask(cfg=env_cfg)
    viewer = None
    if follow:
        from nav_arena.utils.viewer import ThirdPersonView

        viewer = ThirdPersonView(distance=args_cli.follow_distance, height=args_cli.follow_height)
        logger.info("Third-person follow camera enabled")
    elif gui:
        # The default viewport camera sits outside the building looking at the roof.
        task.sim.set_camera_view(
            eye=[spawn_xy[0] - 2.0, spawn_xy[1] - 2.5, 2.2], target=[spawn_xy[0] + 1.0, spawn_xy[1], 0.4]
        )
    task.reset()
    for _ in range(4):  # RTX output needs a few frames before intrinsics/frames are valid
        task.sim.render()

    cfg_kwargs = dict(device=args_cli.planner_device or str(task.device), seed=args_cli.seed)
    if args_cli.task is not None:
        cfg_kwargs["task"] = args_cli.task
    if args_cli.plan_hz is not None:
        cfg_kwargs["plan_hz"] = args_cli.plan_hz
    cfg_kwargs.update(_parse_policy_args(args_cli.policy_arg))
    logger.info(f"Loading policy '{args_cli.method}' ({cfg_kwargs})...")
    policy = get_policy(args_cli.method, task.get_camera_intrinsics(), **cfg_kwargs)

    output = args_cli.output or (RUNS_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_{args_cli.method}_{args_cli.robot}_{route_name}")
    overlay = None
    if OVERLAY:
        from nav_arena.utils.viewer import DebugOverlay

        overlay = DebugOverlay(goal_xy, tolerance=args_cli.goal_dist)
        logger.info("Viewport goal/plan overlay requested (UI layer; not rendered into the policy's cameras)")
    episode = EpisodeCfg(
        max_steps=args_cli.max_steps,
        follower=FollowerCfg(max_speed=args_cli.max_speed, goal_tolerance=args_cli.goal_dist),
        output_dir=output,
        viewer=viewer,
        overlay=overlay,
        stall_timeout_s=args_cli.stall_timeout if args_cli.stall_timeout > 0 else None,
    )
    result = run_episode(task, policy, episode)

    logger.section("RESULT")
    logger.info(f"Terminal cause:       {result.terminal_cause}")
    ttg = f"{result.time_to_goal_s:.2f} s" if result.time_to_goal_s is not None else "n/a"
    logger.info(f"Time to goal:         {ttg}")
    logger.info(f"Distance remaining:   {result.final_goal_distance_m:.2f} m (started at {result.initial_goal_distance_m:.2f} m)")
    logger.info(f"Path length driven:   {result.path_length_m:.2f} m over {result.sim_time_s:.1f} s sim time")
    logger.info(f"Plans / stop requests: {result.plans} / {result.stop_requests}; {result.mean_inference_ms:.0f} ms per plan")
    logger.info(f"Run artifacts:        {output}")
    logger.check("Goal reached", result.success, f"cause={result.terminal_cause}")
    task.close()
    return result


def main():
    # The exit status must be requested inside the context: SimulationApp.close() ends the process, and
    # launch_simulation_app forwards sys.exit codes (and exceptions as status 1) to it.
    with launch_simulation_app(args_cli, enable_ros2=False, livestream=True):
        result = run_verification()
        sys.exit(0 if result.success else 2)


if __name__ == "__main__":
    main()
