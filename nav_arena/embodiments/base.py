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

DRIVE_TYPES = ("diff", "holonomic", "quadruped")
"""Supported drive types. Every one takes the same body twist command ``[vx, vy, wz]``."""

_POLICY_DRIVEN = frozenset({"quadruped"})


def is_policy_driven(drive_type: str) -> bool:
    """Whether a learned low-level policy turns the body twist into joint targets (legged robots).

    Policy-driven robots have their own control rate and need a standing start, unlike wheeled drives whose
    command maps directly to wheel velocities.
    """
    return drive_type in _POLICY_DRIVEN


@dataclass
class RobotEmbodimentCfg:
    """Base specification contract for mobile robot embodiments.

    Couples physical kinematics, simulation articulation/action configs, REP-105 coordinate frames,
    sensor mounting geometry, physical footprints, and optional simulation stage patches.
    """

    name: str = ""
    articulation_cfg: Any = None
    action_cfg: Any = None
    drive_type: str = "diff"
    # Wheel geometry. Used by the wheeled drive types; leave at 0 for robots without wheels.
    wheel_radius: float = 0.0
    wheel_base: float = 0.0
    max_linear_speed: float = 2.0
    # Sideways speed limit (m/s). 0 means the robot cannot strafe (differential drive).
    max_lateral_speed: float = 0.0
    max_angular_speed: float = 3.0
    # Physics timing. The control step is sim_dt * decimation (0.02 s = 50 Hz for every robot so far). Learned
    # locomotion policies are tied to the timing they were trained with, so it belongs to the embodiment.
    sim_dt: float = 0.01
    decimation: int = 2
    # Spawn height (m) of the articulation root above the floor; never 0, to avoid depenetration impulses.
    spawn_height: float = 0.25
    base_frame: str = "base_link"
    chassis_frame: str = "chassis_link"
    # USD link (relative to the robot root prim) carrying the chassis colliders; contact and
    # LiDAR sensors attach here. Distinct from the TF frame names above, which the URDF
    # generator synthesizes independently of the USD hierarchy.
    body_link: str = "chassis_link"
    # True when the body link also carries a ground-contacting part (e.g. a caster sphere), so its contact forces
    # include the floor's vertical support force. Collision detection then uses lateral (horizontal) forces only.
    ground_contact_on_body: bool = False
    # Regex (relative to the robot root) of the bodies whose contacts count as collisions. None uses body_link.
    # Legged robots list the base and legs but not the feet.
    contact_bodies: str | None = None
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

    def __post_init__(self) -> None:
        if self.drive_type not in DRIVE_TYPES:
            raise ValueError(f"Unknown drive_type '{self.drive_type}'. Expected one of: {DRIVE_TYPES}")

    @property
    def step_dt(self) -> float:
        """Seconds per control step."""
        return self.sim_dt * self.decimation

    @property
    def contact_body_expr(self) -> str:
        """Body (or regex of bodies) the contact sensor attaches to, relative to the robot root."""
        return self.contact_bodies or self.body_link
