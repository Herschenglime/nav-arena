# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Closed-loop episode runner for in-process navigation policies.

Plans at the policy's rate and tracks the most recent plan with the shared path follower at the control rate. Episode
logic lives here (not in scripts); the runner talks to the task through a small duck-typed interface so it can be
unit-tested without a simulator (see ``tests/unit/test_in_process_runner.py``).

Ported from the NavDP Isaac Sim integration's demo loop (replanning cadence, depth-validity guard, pose input for
X-NavDP, run recording), re-expressed on top of ``PointNavTask``.

Task interface used: ``reset()``, ``step(action)``, ``step_dt``, ``device``, ``get_robot_pose_w()``,
``get_robot_position_w()``, ``get_robot_quat_w()``, ``get_goal_pose_w()``, ``get_goal_distance()``,
``refresh_camera_frame()``, ``get_terminal_cause()`` and, for image-goal policies, ``render_goal_image(goal_xy)``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from nav_arena.utils import get_logger
from nav_arena.utils.run_dir import recorded_artifacts

from .base import InProcessPolicy, Plan, PolicyObservation
from .controller import FollowerCfg, body_to_world, follow_path, goal_body_input, world_to_body

logger = get_logger("in_process_runner")


@dataclass
class EpisodeCfg:
    """Settings for one closed-loop episode."""

    max_steps: int = 1500
    """Maximum environment (control) steps."""
    follower: FollowerCfg = field(default_factory=FollowerCfg)
    """Path-follower limits; its ``goal_tolerance`` should not exceed the task's goal threshold."""
    warmup_steps: int = 10
    """Zero-velocity steps after reset so physics settles before the first plan."""
    min_valid_depth_fraction: float = 0.05
    """Minimum fraction of finite positive depth pixels; fewer aborts the run (camera orientation/rendering fault)."""
    output_dir: Path | None = None
    """If set, record ``settings.json``, ``steps.jsonl``, ``summary.json`` and periodic RGB/depth snapshots here."""
    snapshot_every_s: float = 5.0
    """Simulated seconds between saved RGB/depth snapshots."""
    goal_image: np.ndarray | None = None
    """RGB goal image for image-goal policies; rendered at the goal pose when omitted."""
    stall_timeout_s: float | None = 10.0
    """End the episode as ``stalled`` if the robot moves less than ``stall_distance_m`` for this many simulated
    seconds (for example a policy that requested a stop and never resumes). ``None`` disables the check."""
    stall_distance_m: float = 0.05
    progress_every_s: float = 5.0
    """Simulated seconds between progress log lines (0 disables)."""
    viewer: Any | None = None
    """Optional display camera with ``update(position, yaw, dt)`` (see :class:`nav_arena.utils.viewer.ThirdPersonView`),
    called every step. It never affects the sensor cameras."""
    overlay: Any | None = None
    """Optional viewport overlay with ``update(path_xy, stop, z)`` and ``clear()`` (see
    :class:`nav_arena.utils.viewer.DebugOverlay`). It draws the goal and each new plan without touching the scene, so the
    sensor cameras never see it."""
    viewer_render_hz: float = 15.0
    """With a viewer, extra display renders per simulated second so the GUI stays smooth and responsive even though
    sensor rendering is only a few Hz."""


@dataclass
class EpisodeResult:
    """Outcome and metrics of one episode."""

    terminal_cause: str
    """``goal_reached``, ``collision``, ``tipped``, ``time_out``, ``stalled`` (no progress; see
    ``EpisodeCfg.stall_timeout_s``), or ``max_steps`` (step budget exhausted)."""
    success: bool
    steps: int
    sim_time_s: float
    time_to_goal_s: float | None
    """Simulated time to reach the goal; None unless the goal was reached."""
    initial_goal_distance_m: float
    final_goal_distance_m: float
    """Distance measured at the start of the final step (the env auto-resets when it terminates)."""
    path_length_m: float
    plans: int
    stop_requests: int
    """Plans in which the policy asked the robot to stop."""
    mean_inference_ms: float
    wall_time_s: float


class _Recorder:
    """Writes run settings, per-plan rows, snapshots, and the summary under ``output_dir``."""

    def __init__(self, output_dir: Path | None, cfg: EpisodeCfg, policy: InProcessPolicy, extra: dict[str, Any]):
        self.dir = Path(output_dir) if output_dir is not None else None
        self._log = None
        self._next_snapshot = 0.0
        if self.dir is None:
            return
        existing = recorded_artifacts(self.dir)
        if existing:
            raise FileExistsError(
                f"{self.dir} already holds a recorded run ({existing[0].name}, ...); choose another output directory "
                "or clear it explicitly."
            )
        self.dir.mkdir(parents=True, exist_ok=True)  # an empty directory (e.g. holding only run_spec.json) is fine
        settings = dict(
            policy=policy.name, policy_cfg=asdict(policy.cfg), plan_hz=policy.plan_hz, episode=_jsonable(cfg), **extra
        )
        (self.dir / "settings.json").write_text(json.dumps(settings, default=str, indent=2))
        self._log = (self.dir / "steps.jsonl").open("w")
        self.snapshot_period = cfg.snapshot_every_s

    def plan_row(self, row: dict[str, Any]) -> None:
        if self._log is not None:
            self._log.write(json.dumps(row, default=_json_default) + "\n")
            self._log.flush()

    def snapshot(self, sim_time: float, rgb: np.ndarray, depth: np.ndarray, step: int) -> None:
        if self.dir is None or sim_time + 1e-9 < self._next_snapshot:
            return
        from PIL import Image

        self._next_snapshot = sim_time + self.snapshot_period
        Image.fromarray(rgb).save(self.dir / f"rgb_{step:06d}.png")
        np.save(self.dir / f"depth_m_{step:06d}.npy", depth)

    def finish(self, result: EpisodeResult) -> None:
        if self._log is not None:
            self._log.close()
        if self.dir is not None:
            (self.dir / "summary.json").write_text(json.dumps(asdict(result), default=_json_default, indent=2))


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return str(value)


def _jsonable(cfg: EpisodeCfg) -> dict[str, Any]:
    data = {f.name: getattr(cfg, f.name) for f in fields(cfg)}
    data["follower"] = asdict(cfg.follower)
    data["goal_image"] = None if cfg.goal_image is None else list(cfg.goal_image.shape)
    data["viewer"] = None if cfg.viewer is None else type(cfg.viewer).__name__
    data["overlay"] = None if cfg.overlay is None else type(cfg.overlay).__name__
    data["output_dir"] = None if cfg.output_dir is None else str(cfg.output_dir)
    return data


def _valid_depth_fraction(depth: np.ndarray) -> float:
    return float(np.mean(np.isfinite(depth) & (depth > 0)))


def run_episode(task: Any, policy: InProcessPolicy, cfg: EpisodeCfg | None = None) -> EpisodeResult:
    """Run one closed-loop navigation episode.

    Args:
        task: A ``PointNavTask`` (or an object providing the interface listed in the module docstring) created with a
            camera enabled.
        policy: The policy to evaluate.
        cfg: Episode settings.

    Returns:
        Metrics and terminal cause.

    Raises:
        RuntimeError: If the camera produces too few valid depth pixels.
    """
    import torch

    cfg = cfg or EpisodeCfg()
    wall_start = time.monotonic()
    step_dt = float(task.step_dt)
    plan_period = max(1, round(1.0 / (policy.plan_hz * step_dt)))

    task.reset()
    zero = torch.zeros((1, 2), device=task.device)
    for _ in range(cfg.warmup_steps):
        task.step(zero)

    goal_x, goal_y, _ = task.get_goal_pose_w()
    goal_xy = np.array([goal_x, goal_y])
    goal_image = cfg.goal_image
    if policy.cfg.task == "imagegoal" and goal_image is None:
        goal_image = task.render_goal_image((goal_x, goal_y))

    recorder = _Recorder(
        cfg.output_dir,
        cfg,
        policy,
        dict(goal_xy=goal_xy.tolist(), plan_period_steps=plan_period, step_dt=step_dt, follower_goal_tolerance=cfg.follower.goal_tolerance),
    )
    if recorder.dir is not None and goal_image is not None:
        from PIL import Image

        Image.fromarray(goal_image).save(recorder.dir / "goal_rgb.png")

    policy.reset()
    if cfg.overlay is not None:
        cfg.overlay.update(np.empty((0, 2)), False, float(task.get_robot_position_w()[2]))
    path_body = np.empty((0, 2))
    path_world = np.empty((0, 2))
    stop = False
    steps = plans = stop_requests = 0
    inference_ms: list[float] = []
    path_length = 0.0
    prev_xy = np.array(task.get_robot_pose_w()[:2])
    initial_distance = distance = float(task.get_goal_distance())
    terminal_cause: str | None = None
    sim_time = 0.0
    v = w = 0.0
    display_period = max(1, round(1.0 / (cfg.viewer_render_hz * step_dt))) if cfg.viewer is not None else 0
    next_progress = 0.0
    stall_ref_xy = prev_xy.copy()
    stall_ref_time = 0.0

    for step in range(cfg.max_steps):
        x, y, yaw = task.get_robot_pose_w()
        position_xy = np.array([x, y])
        distance = float(task.get_goal_distance())
        path_length += float(np.linalg.norm(position_xy - prev_xy))
        prev_xy = position_xy

        if cfg.viewer is not None:
            cfg.viewer.update(task.get_robot_position_w(), yaw, dt=step_dt)
            if step % display_period == 0:
                task.sim.render()

        if cfg.progress_every_s > 0 and sim_time + 1e-9 >= next_progress:
            state = "STOPPED by policy" if stop else f"v={v:.2f} m/s w={w:+.2f} rad/s"
            logger.info(f"t={sim_time:5.1f}s pos=({x:.2f}, {y:.2f}) goal_distance={distance:.2f} m {state}")
            next_progress += cfg.progress_every_s

        if cfg.stall_timeout_s is not None:
            if float(np.linalg.norm(position_xy - stall_ref_xy)) > cfg.stall_distance_m:
                stall_ref_xy, stall_ref_time = position_xy.copy(), sim_time
            elif sim_time - stall_ref_time >= cfg.stall_timeout_s:
                reason = "the policy is requesting a stop" if stop else "the robot is not moving"
                logger.warning(
                    f"Stalled: moved < {cfg.stall_distance_m} m for {cfg.stall_timeout_s:.0f} s at "
                    f"({x:.2f}, {y:.2f}), {distance:.2f} m from the goal; {reason}."
                )
                terminal_cause = "stalled"
                break

        if step % plan_period == 0:
            rgb, depth = task.refresh_camera_frame()
            valid_fraction = _valid_depth_fraction(depth)
            if valid_fraction < cfg.min_valid_depth_fraction:
                raise RuntimeError(
                    f"Depth camera has too few valid pixels ({valid_fraction:.2f} < {cfg.min_valid_depth_fraction}); "
                    "inspect camera orientation and rendering before driving."
                )
            pose_kwargs: dict[str, Any] = {}
            if policy.requires_pose:
                pose_kwargs = dict(
                    robot_position=task.get_robot_position_w()[None], robot_quaternion=task.get_robot_quat_w()[None]
                )
            obs = PolicyObservation(
                rgb=rgb,
                depth=depth,
                goal_body=goal_body_input(goal_xy, position_xy, yaw),
                goal_image=goal_image,
                **pose_kwargs,
            )
            before = time.monotonic()
            plan: Plan = policy.step(obs)
            inference_ms.append((time.monotonic() - before) * 1000.0)
            plans += 1
            if plan.stop and not stop:
                logger.warning(
                    f"{policy.name} requested a stop at ({x:.2f}, {y:.2f}), {distance:.2f} m from the goal: "
                    f"{plan.diagnostics}"
                )
            elif stop and not plan.stop:
                logger.info(f"{policy.name} cleared its stop request at ({x:.2f}, {y:.2f})")
            stop = plan.stop
            stop_requests += int(stop)
            path_world = body_to_world(plan.path[:, :2], np.array([x, y]), yaw)
            if cfg.overlay is not None:
                cfg.overlay.update(path_world, stop, float(task.get_robot_position_w()[2]))
            recorder.plan_row(
                dict(
                    step=step,
                    sim_time=sim_time,
                    position=[x, y],
                    yaw=yaw,
                    goal_distance=distance,
                    stop=stop,
                    diagnostics=plan.diagnostics,
                    valid_depth_fraction=valid_fraction,
                    inference_ms=inference_ms[-1],
                    path_world=path_world,
                )
            )
            recorder.snapshot(sim_time, rgb, depth, step)

        if stop:
            v = w = 0.0
        else:
            v, w = follow_path(world_to_body(path_world, np.array([x, y]), yaw), distance, cfg.follower)
        _, _, terminated, truncated, _ = task.step(torch.tensor([[v, w]], dtype=torch.float32, device=task.device))
        steps += 1
        sim_time += step_dt
        if bool(terminated.any()) or bool(truncated.any()):
            terminal_cause = task.get_terminal_cause() or "time_out"
            break

    if cfg.overlay is not None:
        cfg.overlay.clear()
    if terminal_cause is None:
        terminal_cause = "max_steps"
    success = terminal_cause == "goal_reached"
    result = EpisodeResult(
        terminal_cause=terminal_cause,
        success=success,
        steps=steps,
        sim_time_s=sim_time,
        time_to_goal_s=sim_time if success else None,
        initial_goal_distance_m=initial_distance,
        final_goal_distance_m=distance,
        path_length_m=path_length,
        plans=plans,
        stop_requests=stop_requests,
        mean_inference_ms=float(np.mean(inference_ms)) if inference_ms else 0.0,
        wall_time_s=time.monotonic() - wall_start,
    )
    recorder.finish(result)
    logger.info(
        f"Episode finished: {result.terminal_cause} after {result.sim_time_s:.1f} s sim time "
        f"({result.plans} plans, {result.mean_inference_ms:.0f} ms/plan, {result.path_length_m:.2f} m travelled)"
    )
    return result


__all__ = ["EpisodeCfg", "EpisodeResult", "run_episode"]
