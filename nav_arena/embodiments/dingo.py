# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Clearpath Dingo robot embodiment configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from nav_arena.utils.paths import CACHE_DIR, NAVDP_ROOT, resolve_path

from .actions import DifferentialDriveActionCfg
from .assets import ensure_derived_asset
from .base import RobotEmbodimentCfg
from .registry import register_embodiment

# Full-scale Dingo asset shipped with NavDP; override with NAV_ARENA_DINGO_USD.
DINGO_SOURCE_USD_PATH = resolve_path("NAV_ARENA_DINGO_USD", NAVDP_ROOT / "assets/robots/dingo.usd")


def _resolve_dingo_usd() -> str:
    """Return the derived Dingo USD (ground plane removed, caster friction fixed), building it on first use.

    If the upstream asset is not installed the source path is returned unchanged, so importing this module (and CPU
    unit tests) works without the NavDP checkout; spawning the robot then fails with the missing-file error.
    """
    if not DINGO_SOURCE_USD_PATH.is_file():
        return str(DINGO_SOURCE_USD_PATH)
    return str(ensure_derived_asset("dingo", DINGO_SOURCE_USD_PATH, CACHE_DIR))


DINGO_USD_PATH = _resolve_dingo_usd()
"""Derived asset: the upstream Dingo plus nav_arena's static fixes; see :mod:`nav_arena.embodiments.assets`."""


##
# Articulation Configuration
##

DINGO_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=DINGO_USD_PATH,
        scale=(1.0, 1.0, 1.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.25),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={
            "left_wheel_joint": 0.0,
            "right_wheel_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["left_wheel_joint", "right_wheel_joint"],
            effort_limit=20.0,
            effort_limit_sim=20.0,
            velocity_limit_sim=20.0,
            stiffness=0.0,
            damping=1.0,
        ),
    },
)
"""Configuration for the Clearpath Dingo articulation asset."""


##
# Action Configuration
##

DINGO_ACTION_CFG = DifferentialDriveActionCfg(
    asset_name="robot",
    left_wheel_joint_name="left_wheel_joint",
    right_wheel_joint_name="right_wheel_joint",
    wheel_radius=0.1225,
    wheel_base=0.4523232,
    max_linear_speed=2.0,
    max_angular_speed=3.0,
)
"""Action term configuration for Clearpath Dingo differential drive kinematics."""


##
# Embodiment Container
##

@dataclass
class DingoEmbodimentCfg(RobotEmbodimentCfg):
    """Convenience container coupling the Clearpath Dingo robot and its action term."""

    name: str = "dingo"
    articulation_cfg: ArticulationCfg = field(default_factory=lambda: DINGO_CFG)
    action_cfg: DifferentialDriveActionCfg = field(default_factory=lambda: DINGO_ACTION_CFG)
    wheel_radius: float = 0.1225
    wheel_base: float = 0.4523232
    max_linear_speed: float = 2.0
    max_angular_speed: float = 3.0
    base_frame: str = "base_link"
    chassis_frame: str = "chassis_link"
    # The Dingo USD has a single rigid body: chassis colliders, caster and sensors all live on base_link.
    body_link: str = "base_link"
    # The caster sphere rests on the floor and is part of base_link, so only lateral forces indicate a collision.
    ground_contact_on_body: bool = True
    lidar_frame: str = "lidar_link"
    camera_frame: str = "camera_link"
    sensor_height: float = 0.30
    lidar_offset: tuple[float, float, float] = (0.0, 0.0, 0.30)
    camera_offset: tuple[float, float, float] = (0.0, 0.0, 0.30)
    camera_rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    # Footprint polygon relative to base_link: [[x, y], ...]
    # Clearpath Dingo-D chassis is ~0.686m long, ~0.517m wide
    footprint: tuple[tuple[float, float], ...] = (
        (0.34, 0.26),
        (0.34, -0.26),
        (-0.34, -0.26),
        (-0.34, 0.26),
    )
    chassis_size: tuple[float, float, float] = (0.686, 0.517, 0.25)  # length, width, height (m)
    chassis_offset: tuple[float, float, float] = (0.0, 0.0, 0.125)  # center of box relative to base_link
    robot_radius: float = 0.35  # nominal half-width radius
    robot_height: float = 0.30


# Register default embodiment
register_embodiment("dingo", DingoEmbodimentCfg)
