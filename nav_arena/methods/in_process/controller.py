# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Planar frame transforms and the shared differential-drive path follower.

Ported from the NavDP Isaac Sim integration (``isaac_adapter.py``). Pure NumPy: no simulator dependencies.
All positions are meters and yaw is radians. Body axes: X forward, Y left; world is Z-up.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

BODY_TWIST_DIM = 3
"""Width of the command every embodiment accepts: the body-frame twist ``[vx, vy, wz]``."""


def world_to_body(points: np.ndarray, position: np.ndarray, yaw: float) -> np.ndarray:
    """Express world-frame planar points in the body frame of a robot at ``position`` with heading ``yaw``.

    Args:
        points: World points, shape ``[..., >=2]`` (only x, y are used).
        position: Robot world position, at least (x, y).
        yaw: Robot heading in radians.

    Returns:
        Body-frame (x forward, y left) points, shape ``[..., 2]``.
    """
    points = np.asarray(points, dtype=np.float32)[..., :2] - np.asarray(position)[:2]
    c, s = math.cos(yaw), math.sin(yaw)
    return points @ np.array([[c, -s], [s, c]], dtype=np.float32)


def body_to_world(points: np.ndarray, position: np.ndarray, yaw: float) -> np.ndarray:
    """Inverse of :func:`world_to_body`."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.asarray(points)[..., :2] @ np.array([[c, s], [-s, c]]) + np.asarray(position)[:2]


def goal_body_input(goal_world: np.ndarray, position: np.ndarray, yaw: float) -> np.ndarray:
    """Build the ``[1, 3]`` body-frame point-goal input (z = 0) used by the point-goal planners.

    Policies apply their own native goal clamps; none is applied here.
    """
    xy = world_to_body(goal_world, position, yaw)
    return np.array([[xy[0], xy[1], 0.0]], dtype=np.float32)


@dataclass
class FollowerCfg:
    """Path-follower limits."""

    max_speed: float = 0.3
    """Maximum forward speed in m/s."""
    max_yaw_rate: float = 0.8
    """Maximum yaw rate in rad/s."""
    lookahead: float = 0.5
    """Distance along the path to the tracked waypoint in meters."""
    goal_tolerance: float = 0.3
    """Stop within this distance of the goal, in meters."""
    turn_in_place_angle: float = 0.6
    """Rotate in place while the heading error to the target exceeds this many radians."""
    goal_slowdown_gain: float = 0.7
    """Forward speed is capped at ``gain * goal_distance``, ramping down on approach."""


def follow_path(
    path_body: np.ndarray, goal_distance: float, cfg: FollowerCfg | None = None
) -> tuple[float, float]:
    """Track a body-frame path with a lookahead steering law.

    The robot moves between replans, so the already-passed part of the path is trimmed before choosing the
    lookahead target. Invalid, empty, or non-finite paths stop the robot.

    Args:
        path_body: Waypoints in the robot's *current* body frame, shape ``[T, >=2]``.
        goal_distance: Remaining distance to the goal in meters.
        cfg: Follower limits.

    Returns:
        Tuple ``(v, omega)``: forward speed in m/s and yaw rate in rad/s.
    """
    cfg = cfg or FollowerCfg()
    path = np.asarray(path_body, dtype=np.float32)
    if goal_distance <= cfg.goal_tolerance or path.ndim != 2 or len(path) == 0:
        return 0.0, 0.0
    if path.shape[1] < 2 or not np.isfinite(path).all():
        return 0.0, 0.0
    path = path[np.argmin(np.linalg.norm(path[:, :2], axis=1)) :]
    distances = np.linalg.norm(path[:, :2], axis=1)
    candidates = np.flatnonzero(distances >= cfg.lookahead)
    target = path[candidates[0] if len(candidates) else -1, :2]
    if np.linalg.norm(target) < 0.05:
        return 0.0, 0.0
    heading = math.atan2(float(target[1]), float(target[0]))
    if abs(heading) > cfg.turn_in_place_angle:
        return 0.0, float(np.clip(1.5 * heading, -cfg.max_yaw_rate, cfg.max_yaw_rate))
    speed = min(cfg.max_speed, cfg.goal_slowdown_gain * goal_distance) * max(0.0, math.cos(heading))
    curvature = 2.0 * float(target[1]) / max(float(target @ target), 0.01)
    return speed, float(np.clip(speed * curvature, -cfg.max_yaw_rate, cfg.max_yaw_rate))
