# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simulation session management and episode execution for benchmarks."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass, field
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

from nav_arena.benchmarks.spec import EpisodeLimits, RunSpec, VizCfg, _normalize_for_serialization
from nav_arena.core import launch_simulation_app
from nav_arena.scenes import DEFAULT_ROUTE, get_route
from nav_arena.utils import RUNS_DIR, get_logger

logger = get_logger("benchmarks.session")


@dataclass
class SessionKey:
    """Invariant parameters that define a simulation session (baked into the env or loaded network)."""

    scene: str
    robot: str
    method: str
    method_params: dict[str, Any] = field(default_factory=dict)
    viz: VizCfg = field(default_factory=VizCfg)

    def __post_init__(self) -> None:
        if self.viz is None:
            self.viz = VizCfg()
        elif isinstance(self.viz, dict):
            self.viz = VizCfg(**self.viz)
        if self.method_params is None:
            self.method_params = {}

    @classmethod
    def from_spec(cls, spec: RunSpec) -> SessionKey:
        """Create a SessionKey from a RunSpec."""
        return cls(
            scene=spec.scene,
            robot=spec.robot,
            method=spec.method,
            method_params=copy.deepcopy(spec.method_params),
            viz=copy.deepcopy(spec.viz),
        )

    @property
    def key_hash(self) -> str:
        """SHA-256 hash identifying invariant session configuration."""
        data = {
            "method": self.method,
            "method_params": _normalize_for_serialization(self.method_params),
            "robot": self.robot,
            "scene": self.scene,
            "viz": asdict(self.viz),
        }
        canonical_json = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    def __hash__(self) -> int:
        return hash(self.key_hash)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, SessionKey):
            return False
        return self.key_hash == other.key_hash


@dataclass
class EpisodeSpec:
    """Variant parameters defining a single episode execution within a session."""

    spawn: tuple[float, float]
    goal: tuple[float, float]
    spawn_yaw: float | None = None
    seed: int = 0
    limits: EpisodeLimits = field(default_factory=EpisodeLimits)
    output_dir: Path | None = None
    route: str | None = None

    def __post_init__(self) -> None:
        self.spawn = tuple(float(x) for x in self.spawn)
        self.goal = tuple(float(x) for x in self.goal)
        if self.spawn_yaw is not None:
            self.spawn_yaw = float(self.spawn_yaw)
        self.seed = int(self.seed)
        if self.output_dir is not None and not isinstance(self.output_dir, Path):
            self.output_dir = Path(self.output_dir)
        if self.limits is None:
            self.limits = EpisodeLimits()
        elif isinstance(self.limits, dict):
            self.limits = EpisodeLimits(**self.limits)

    @classmethod
    def from_spec(cls, spec: RunSpec) -> EpisodeSpec:
        """Construct an EpisodeSpec from a RunSpec, resolving route coordinates if needed."""
        if (spec.spawn is None) != (spec.goal is None):
            raise ValueError(
                "Both spawn and goal must be specified together, or neither (to use a named route)."
            )

        if spec.spawn is not None and spec.goal is not None:
            spawn_xy = tuple(float(x) for x in spec.spawn)
            goal_xy = tuple(float(x) for x in spec.goal)
            spawn_yaw = (
                float(spec.spawn_yaw)
                if spec.spawn_yaw is not None
                else math.atan2(goal_xy[1] - spawn_xy[1], goal_xy[0] - spawn_xy[0])
            )
            route_name = spec.route or "custom"
        else:
            route_name = spec.route or DEFAULT_ROUTE
            route_obj = get_route(route_name, spec.scene)
            spawn_xy = route_obj.spawn_xy
            goal_xy = route_obj.goal_xy
            spawn_yaw = float(spec.spawn_yaw) if spec.spawn_yaw is not None else float(route_obj.spawn_yaw)
            route_name = route_obj.name

        return cls(
            spawn=spawn_xy,
            goal=goal_xy,
            spawn_yaw=spawn_yaw,
            seed=spec.seed,
            limits=copy.deepcopy(spec.limits),
            output_dir=spec.output_dir,
            route=route_name,
        )


def print_result_summary(result: Any, output_dir: Path | None = None, custom_logger: Any = None) -> None:
    """Print the standardized benchmark episode result summary."""
    log = custom_logger or logger
    log.section("RESULT")
    log.info(f"Terminal cause:       {result.terminal_cause}")
    ttg = f"{result.time_to_goal_s:.2f} s" if result.time_to_goal_s is not None else "n/a"
    log.info(f"Time to goal:         {ttg}")
    log.info(
        f"Distance remaining:   {result.final_goal_distance_m:.2f} m "
        f"(started at {result.initial_goal_distance_m:.2f} m)"
    )
    log.info(f"Path length driven:   {result.path_length_m:.2f} m over {result.sim_time_s:.1f} s sim time")
    log.info(
        f"Plans / stop requests: {result.plans} / {result.stop_requests}; "
        f"{result.mean_inference_ms:.0f} ms per plan"
    )
    if output_dir is not None:
        log.info(f"Run artifacts:        {output_dir}")
    log.check("Goal reached", result.success, f"cause={result.terminal_cause}")


class RunSession:
    """Encapsulates a booted simulation app, task environment, and policy evaluation lifecycle."""

    def __init__(
        self,
        spec_or_key: RunSpec | SessionKey,
        episode_spec: EpisodeSpec | None = None,
        simulation_app: Any = None,
        args_cli: Any = None,
        render_mode: str | None = None,
    ):
        """Initialize RunSession.

        Args:
            spec_or_key: A RunSpec or a SessionKey defining invariant configuration.
            episode_spec: Optional EpisodeSpec defining initial episode. Inferred from RunSpec if given.
            simulation_app: Optional pre-existing SimulationApp instance (e.g. for testing).
            args_cli: Command-line arguments namespace to pass to AppLauncher.
            render_mode: Optional Gym render mode for PointNavTask (default None).
        """
        if isinstance(spec_or_key, RunSpec):
            self.spec: RunSpec | None = spec_or_key
            self.key = SessionKey.from_spec(spec_or_key)
            self.initial_episode = episode_spec or EpisodeSpec.from_spec(spec_or_key)
            self.method_family = spec_or_key.method_family
        else:
            self.spec = None
            self.key = spec_or_key
            self.initial_episode = episode_spec
            self.method_family = "ros2" if spec_or_key.method == "nav2" else "in_process"

        self._sim_app = simulation_app
        self.args_cli = args_cli
        self.render_mode = render_mode

        self._app_cm: Any | None = None
        self.task: Any | None = None
        self.policy: Any | None = None
        self.viewer: Any | None = None
        self.overlay: Any | None = None
        self.is_open: bool = False

    def _boot_app(self) -> None:
        """Start AppLauncher / SimulationApp unless the caller supplied a running app."""
        if self._sim_app is not None:
            return
        args = copy.deepcopy(self.args_cli) if self.args_cli is not None else argparse.Namespace()
        args.enable_cameras = True
        args.visualizer = ["kit"] if self.key.viz.gui else None
        args.headless = not self.key.viz.gui
        enable_ros2 = self.method_family == "ros2"
        self._app_cm = launch_simulation_app(args, enable_ros2=enable_ros2, livestream=True)
        self._sim_app = self._app_cm.__enter__()

    def _build_task(self, ep: EpisodeSpec, follow: bool) -> None:
        """Construct the PointNav task for this episode's spawn, goal and limits."""
        from nav_arena.embodiments import get_embodiment
        from nav_arena.tasks import PointNavTask, create_point_nav_env_cfg

        step_dt = get_embodiment(self.key.robot).step_dt
        spawn_xy, goal_xy = ep.spawn, ep.goal
        spawn_yaw = (
            ep.spawn_yaw if ep.spawn_yaw is not None else math.atan2(goal_xy[1] - spawn_xy[1], goal_xy[0] - spawn_xy[0])
        )
        route_name = ep.route or "custom"
        logger.section(f"BASELINE '{self.key.method}' ON '{self.key.robot}': {self.key.scene}/{route_name}")
        logger.info(f"Spawn {spawn_xy} (yaw {math.degrees(spawn_yaw):.0f} deg) -> goal {goal_xy}")

        task_param = self.key.method_params.get("task")
        enable_goal_camera = task_param == "imagegoal" or (task_param is None and self.key.method == "vint")
        show_goal_marker = self.key.viz.show_goal_marker
        if show_goal_marker:
            logger.warning("Goal marker enabled: policies' cameras will see it as an obstacle at the goal.")

        env_cfg = create_point_nav_env_cfg(
            scene_id_or_path=self.key.scene,
            robot_spawn_pos=(spawn_xy[0], spawn_xy[1]),
            robot_spawn_rot=(0.0, 0.0, math.sin(spawn_yaw / 2.0), math.cos(spawn_yaw / 2.0)),
            goal_pos=goal_xy,
            goal_threshold=ep.limits.goal_tolerance,
            episode_length_s=ep.limits.max_steps * step_dt + 10.0,
            robot_name=self.key.robot,
            enable_camera=True,
            enable_goal_camera=enable_goal_camera,
            show_goal_marker=show_goal_marker,
            scene_queries=follow,
        )
        env_cfg.seed = ep.seed
        self.task = PointNavTask(cfg=env_cfg, render_mode=self.render_mode)

    def _build_viewer(self, ep: EpisodeSpec, gui: bool, follow: bool) -> None:
        """Create the follow camera, or aim the default viewport camera at the start (GUI only)."""
        self.viewer = None
        if follow:
            from nav_arena.utils.viewer import ThirdPersonView

            self.viewer = ThirdPersonView(distance=self.key.viz.follow_distance, height=self.key.viz.follow_height)
            logger.info("Third-person follow camera enabled")
        elif gui:
            spawn_x, spawn_y = ep.spawn
            self.task.sim.set_camera_view(
                eye=[spawn_x - 2.0, spawn_y - 2.5, 2.2], target=[spawn_x + 1.0, spawn_y, 0.4]
            )

    def _build_policy(self, ep: EpisodeSpec) -> None:
        """Load the navigation policy; its config is the method params plus the device and seed."""
        from nav_arena.methods.in_process import get_policy

        cfg_kwargs: dict[str, Any] = {"device": str(self.task.device), "seed": ep.seed}
        cfg_kwargs.update(self.key.method_params)  # policy config fields only (task, plan_hz, device, ...)
        logger.info(f"Loading policy '{self.key.method}' ({cfg_kwargs})...")
        self.policy = get_policy(self.key.method, self.task.get_camera_intrinsics(), **cfg_kwargs)

    def _build_overlay(self, ep: EpisodeSpec) -> None:
        """Create the viewport goal/plan overlay, a UI layer that the policy's cameras cannot see."""
        from nav_arena.utils.viewer import DebugOverlay

        self.overlay = DebugOverlay(ep.goal, tolerance=ep.limits.goal_tolerance)
        logger.info("Viewport goal/plan overlay requested (UI layer; not rendered into the policy's cameras)")

    def start(self, episode_spec: EpisodeSpec | None = None) -> RunSession:
        """Boot simulator, construct PointNavTask, viewer, overlay, and load policy."""
        if self.is_open:
            return self
        if self.key.method == "nav2":
            raise NotImplementedError("Nav2 backend is reserved and not yet implemented.")

        ep = episode_spec or self.initial_episode
        if ep is None:
            raise ValueError("RunSession requires an EpisodeSpec to initialize environment.")
        self.initial_episode = ep

        self._boot_app()
        try:
            # Everything below imports Isaac Lab modules, which must only happen after the app boots.
            viz = self.key.viz.resolved()
            self._build_task(ep, follow=viz.follow_camera)
            self._build_viewer(ep, gui=viz.gui, follow=viz.follow_camera)
            self.task.reset()
            for _ in range(4):  # RTX output needs a few frames before intrinsics/frames are valid
                self.task.sim.render()
            self._build_policy(ep)
            self.overlay = None
            if viz.goal_overlay:
                self._build_overlay(ep)
            self.is_open = True
            return self
        except BaseException:  # clean up the half-started session (app, env) for any failure, then re-raise
            self.close(*sys.exc_info())
            raise

    def run_episode(self, episode_spec: EpisodeSpec | None = None) -> Any:
        """Run an episode in this session and return the EpisodeResult."""
        ep = episode_spec or self.initial_episode
        if not self.is_open:
            self.start(episode_spec=ep)

        if ep is None:
            raise ValueError("No EpisodeSpec provided.")

        if self.policy is not None:
            self.policy.reseed(ep.seed)

        route_name = ep.route or "custom"
        output = ep.output_dir or (
            RUNS_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_{self.key.method}_{self.key.robot}_{route_name}"
        )

        if self.overlay is not None:
            self.overlay.goal_xy = ep.goal
            self.overlay.tolerance = ep.limits.goal_tolerance
            self.overlay.clear()

        from nav_arena.embodiments import get_embodiment
        from nav_arena.methods.in_process import (
            EpisodeCfg,
            FollowerCfg,
            run_episode as run_in_process_episode,
        )

        embodiment = get_embodiment(self.key.robot)
        episode_cfg = EpisodeCfg(
            max_steps=ep.limits.max_steps,
            follower=FollowerCfg(
                max_speed=ep.limits.max_speed,
                max_lateral_speed=min(embodiment.max_lateral_speed, ep.limits.max_speed),
                goal_tolerance=ep.limits.goal_tolerance,
            ),
            output_dir=output,
            viewer=self.viewer,
            overlay=self.overlay,
            stall_timeout_s=ep.limits.stall_timeout_s if ep.limits.stall_timeout_s > 0 else None,
        )

        result = run_in_process_episode(self.task, self.policy, episode_cfg)
        print_result_summary(result, output)
        return result

    def close(self, exc_type: Any = None, exc_val: Any = None, exc_tb: Any = None) -> None:
        """Tear down task environment and shut down simulation application."""
        if self.task is not None:
            try:
                self.task.close()
            except Exception as exc:  # teardown boundary: a failing env close must not mask the original error
                logger.warning(f"Error closing PointNavTask: {exc}")
            self.task = None

        self.is_open = False

        if self._app_cm is not None:
            app_cm = self._app_cm
            self._app_cm = None
            self._sim_app = None
            app_cm.__exit__(exc_type, exc_val, exc_tb)

    def __enter__(self) -> RunSession:
        return self.start()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close(exc_type, exc_val, exc_tb)
