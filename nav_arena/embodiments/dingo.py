# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Clearpath Dingo robot embodiment configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from nav_arena.utils.paths import NAVDP_ROOT, resolve_path

from .actions import DifferentialDriveActionCfg
from .base import RobotEmbodimentCfg
from .registry import register_embodiment

# Full-scale Dingo asset shipped with NavDP; override with NAV_ARENA_DINGO_USD.
DINGO_USD_PATH = str(resolve_path("NAV_ARENA_DINGO_USD", NAVDP_ROOT / "assets/robots/dingo.usd"))


##
# Stage Patch
##

_PHYSX_MATERIAL_API = "PhysxMaterialAPI"
_FRICTION_COMBINE_ATTR = "physxMaterial:frictionCombineMode"


def _apply_min_friction_combine(prim: Any) -> None:
    """Author ``PhysxMaterialAPI`` with ``frictionCombineMode = "min"`` on a material prim.

    Written as plain USD (``apiSchemas`` metadata plus the attribute) rather than through
    ``pxr.PhysxSchema``, which only exists inside a running Kit app. The result is identical to
    ``PhysxSchema.PhysxMaterialAPI.Apply(prim).CreateFrictionCombineModeAttr("min")``.
    """
    from pxr import Sdf

    schemas = Sdf.TokenListOp()
    existing = prim.GetMetadata("apiSchemas")
    items = list(existing.GetAddedOrExplicitItems()) if existing else []
    if _PHYSX_MATERIAL_API not in items:
        items.append(_PHYSX_MATERIAL_API)
    schemas.prependedItems = items
    prim.SetMetadata("apiSchemas", schemas)
    prim.CreateAttribute(_FRICTION_COMBINE_ATTR, Sdf.ValueTypeNames.Token).Set("min")


def dingo_stage_patch(stage: Any) -> None:
    """Apply stage patches for the Clearpath Dingo embodiment.

    1. Deactivate the Dingo asset's embedded ``GroundPlane`` so it does not collide with (or
       replace) the scene floor.
    2. Set the caster wheel's friction combine mode to ``min``. The asset models its passive caster
       as a frictionless sphere; averaging that zero friction with the floor's would make the caster
       drag and the drive wheels slip, whereas ``min`` preserves zero friction at the contact.

    Matches every robot instance (``/World/Robot``, cloned ``/World/envs/env_*/Robot``, or a
    standalone ``/dingo`` root). Must run after the robot is spawned and before physics is
    initialized, e.g. from a ``prestartup`` event.

    Args:
        stage: The USD stage holding the spawned Dingo robot(s).

    Raises:
        ValueError: If ``stage`` is None.
        RuntimeError: If no Dingo caster material is found, which means the robot is not spawned
            yet or the asset layout changed.
    """
    if stage is None:
        raise ValueError("dingo_stage_patch requires a USD stage")

    grounds, casters = [], []
    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        segments = path.split("/")[1:]
        if path.endswith("/GroundPlane") and any(seg in ("Robot", "dingo") for seg in segments[:-1]):
            grounds.append(prim)
        elif path.endswith("/PhysicsMaterials/caster_wheel"):
            casters.append(prim)

    if not casters:
        raise RuntimeError(
            "dingo_stage_patch found no '<robot>/PhysicsMaterials/caster_wheel' prim; "
            "spawn the Dingo before patching the stage."
        )
    for prim in grounds:
        prim.SetActive(False)
    for prim in casters:
        _apply_min_friction_combine(prim)


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
