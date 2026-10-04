# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Clearpath Dingo robot embodiment configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any, Callable

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from .actions import DifferentialDriveActionCfg
from .base import RobotEmbodimentCfg
from .registry import register_embodiment

# Resolve Dingo USD path: prioritize simulation workspace / NavDP
_DEFAULT_DINGO_USD = "/home/robopi/simulation/NavDP/assets/robots/dingo.usd"
DINGO_USD_PATH = os.environ.get("NAV_ARENA_DINGO_USD", _DEFAULT_DINGO_USD)


##
# Stage Patch
##

def dingo_stage_patch(stage: Any) -> None:
    """Apply stage patches for Clearpath Dingo embodiment.

    1. Deactivate embedded GroundPlane to prevent collision conflicts with scene floors.
    2. Set caster wheel friction combine mode to 'min' so zero caster friction is preserved,
       preventing passive caster drag and drive-wheel slippage.
    """
    if stage is None:
        return

    # 1. Direct path check (template or non-cloned /World/Robot)
    ground = stage.GetPrimAtPath("/World/Robot/GroundPlane")
    if ground.IsValid():
        ground.SetActive(False)

    caster = stage.GetPrimAtPath("/World/Robot/PhysicsMaterials/caster_wheel")
    if caster.IsValid():
        try:
            from pxr import PhysxSchema

            PhysxSchema.PhysxMaterialAPI.Apply(caster).CreateFrictionCombineModeAttr("min")
        except (ImportError, AttributeError):
            pass

    # 2. Traverse stage for cloned environments (e.g. /World/envs/env_0/Robot)
    #    or direct root /dingo paths
    try:
        prims = list(stage.Traverse())
    except (AttributeError, TypeError):
        prims = []

    for prim in prims:
        try:
            path_str = prim.GetPath().pathString
        except AttributeError:
            continue
        if path_str == "/World/Robot/GroundPlane":
            continue
        if path_str.endswith("/GroundPlane") and ("Robot" in path_str or "dingo" in path_str):
            prim.SetActive(False)
        elif path_str.endswith("/PhysicsMaterials/caster_wheel") and path_str != "/World/Robot/PhysicsMaterials/caster_wheel":
            try:
                from pxr import PhysxSchema

                PhysxSchema.PhysxMaterialAPI.Apply(prim).CreateFrictionCombineModeAttr("min")
            except (ImportError, AttributeError):
                pass


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
    stage_patch_fn: Callable[[Any], None] | None = dingo_stage_patch


# Register default embodiment
register_embodiment("dingo", DingoEmbodimentCfg)
