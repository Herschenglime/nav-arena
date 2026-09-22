# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Nova Carter robot embodiment configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from .actions import DifferentialDriveActionCfg

##
# Articulation Configuration
##

NOVA_CARTER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/NVIDIA/NovaCarter/nova_carter.usd",
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
            "joint_wheel_.*": 0.0,
            "joint_caster_.*": 0.0,
            "joint_swing_.*": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["joint_wheel_left", "joint_wheel_right"],
            effort_limit_sim=100.0,
            velocity_limit_sim=20.0,
            stiffness=0.0,
            damping=10.0,
        ),
        "casters": ImplicitActuatorCfg(
            joint_names_expr=["joint_caster_.*", "joint_swing_.*"],
            effort_limit_sim=0.0,
            velocity_limit_sim=50.0,
            stiffness=0.0,
            damping=0.1,
        ),
    },
)
"""Configuration for the NVIDIA Nova Carter articulation asset."""


##
# Action Configuration
##

NOVA_CARTER_ACTION_CFG = DifferentialDriveActionCfg(
    asset_name="robot",
    left_wheel_joint_name="joint_wheel_left",
    right_wheel_joint_name="joint_wheel_right",
    wheel_radius=0.14,
    wheel_base=0.413,
    max_linear_speed=2.0,
    max_angular_speed=3.0,
)
"""Action term configuration for Nova Carter differential drive kinematics."""


@dataclass
class NovaCarterEmbodimentCfg:
    """Convenience container coupling the Nova Carter robot and its action term."""

    articulation_cfg: ArticulationCfg = field(default_factory=lambda: NOVA_CARTER_CFG)
    action_cfg: DifferentialDriveActionCfg = field(default_factory=lambda: NOVA_CARTER_ACTION_CFG)
    name: str = "nova_carter"
    wheel_radius: float = 0.14
    wheel_base: float = 0.413
    chassis_frame: str = "chassis_link"

