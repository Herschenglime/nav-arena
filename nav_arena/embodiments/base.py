# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base specification contract for mobile robot embodiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from isaaclab.assets.articulation import ArticulationCfg
    from .actions import DifferentialDriveActionCfg


@dataclass
class RobotEmbodimentCfg:
    """Base specification contract for mobile robot embodiments.

    Couples physical kinematics, simulation articulation/action configs, REP-105 coordinate frames,
    sensor mounting geometry, physical footprints, and optional simulation stage patches.
    """

    name: str = ""
    articulation_cfg: Any = None
    action_cfg: Any = None
    wheel_radius: float = 0.0
    wheel_base: float = 0.0
    max_linear_speed: float = 2.0
    max_angular_speed: float = 3.0
    base_frame: str = "base_link"
    chassis_frame: str = "chassis_link"
    # USD link (relative to the robot root prim) carrying the chassis colliders; contact and
    # LiDAR sensors attach here. Distinct from the TF frame names above, which the URDF
    # generator synthesizes independently of the USD hierarchy.
    body_link: str = "chassis_link"
    lidar_frame: str = "lidar_link"
    camera_frame: str = "camera_link"
    sensor_height: float = 0.35
    lidar_offset: tuple[float, float, float] = (0.0, 0.0, 0.35)
    camera_offset: tuple[float, float, float] = (0.0, 0.0, 0.30)
    camera_rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    footprint: tuple[tuple[float, float], ...] | list[tuple[float, float]] = ()
    chassis_size: tuple[float, float, float] | None = None
    chassis_offset: tuple[float, float, float] | None = None
    robot_radius: float = 0.30
    robot_height: float = 0.40
    stage_patch_fn: Callable[[Any], None] | None = None
