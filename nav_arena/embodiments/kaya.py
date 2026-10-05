# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""NVIDIA Kaya holonomic robot embodiment configuration.

Kaya is a 3-omni-wheel holonomic robot shipped with Isaac Sim. It is the smallest well-supported
holonomic asset in the Isaac Sim 6.0 library and fits inside the InteriorAgent home scenes.

Wheel geometry, roller angles and radii are read from the USD at runtime (``isaacmecanumwheel:*``
attributes on each axle joint), so the numbers here are never hand-copied and cannot drift from
the asset. The active joints are the three axle joints; the 30 passive roller joints receive no
actuator targets.

Two variants are registered:

``kaya.mast`` (default)
    Camera on a virtual mast at 0.30 m above the floor (= 0.209 m above ``base_link``). Provides
    a viewpoint comparable to Nova Carter and Dingo so that learned policies trained on one robot
    can be evaluated on another.

``kaya.native``
    The real RealSense D435 pose from the USD: approximately 0.16 m above the floor, pitched
    down ~20 degrees. Reports what Kaya would see in hardware.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from .actions import HolonomicDriveActionCfg
from .base import RobotEmbodimentCfg
from .kaya_variants import KAYA_MAST, KAYA_NATIVE, MAST_CAMERA_OFFSET, ROT_LEVEL  # noqa: F401 (re-exported)
from .registry import register_embodiment

##
# Articulation Configuration
##

#: USD path for the Kaya asset shipped with Isaac Sim 6.0.
KAYA_USD_PATH = f"{ISAAC_NUCLEUS_DIR}/Robots/NVIDIA/Kaya/kaya.usd"

KAYA_ARTICULATION_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=KAYA_USD_PATH,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            # More solver iterations help the 30 free roller joints stay stable.
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
        ),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # base_link rests 0.091 m above the floor; 0.15 m gives comfortable depenetration clearance.
        pos=(0.0, 0.0, 0.15),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={"axle_.*_joint": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        # Only the three driven axle joints get an actuator; the 30 roller joints stay passive.
        # The Kaya USD authors damping=174.5 on the axle joints' PhysicsDriveAPI:angular.
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["axle_0_joint", "axle_1_joint", "axle_2_joint"],
            effort_limit=50.0,
            effort_limit_sim=50.0,
            velocity_limit_sim=30.0,
            stiffness=0.0,
            damping=174.5,
        ),
    },
)
"""ArticulationCfg for the NVIDIA Kaya holonomic robot."""


##
# Action Configuration
##

KAYA_ACTION_CFG = HolonomicDriveActionCfg(
    asset_name="robot",
    # Exact names of the three driven joints; the roller joints are excluded — they stay free.
    wheel_joint_names=("axle_0_joint", "axle_1_joint", "axle_2_joint"),
    # wheel_geometry=None: the action term reads the isaacmecanumwheel:* attrs from the live stage.
    max_linear_speed=0.5,
    max_lateral_speed=0.5,
    max_angular_speed=3.0,
)
"""HolonomicDriveActionCfg for the NVIDIA Kaya (3 omni wheels, geometry read from USD)."""


##
# Embodiment Container
##


@dataclass
class KayaEmbodimentCfg(RobotEmbodimentCfg):
    """Convenience container coupling the NVIDIA Kaya robot and its holonomic action term."""

    name: str = "kaya"
    articulation_cfg: ArticulationCfg = field(default_factory=lambda: KAYA_ARTICULATION_CFG)
    action_cfg: HolonomicDriveActionCfg = field(default_factory=lambda: KAYA_ACTION_CFG)
    drive_type: str = "holonomic"

    # Speed limits — set conservatively for indoor navigation (wheel IK has no saturation issue
    # at these values even at maximum angular rate).
    max_linear_speed: float = 0.5
    max_lateral_speed: float = 0.5
    max_angular_speed: float = 3.0

    # Timing — same control rate as the diff-drive robots (50 Hz).
    sim_dt: float = 0.01
    decimation: int = 2

    # Kaya's base_link origin sits 0.091 m above the resting floor; spawn slightly higher.
    spawn_height: float = 0.15

    # Frame / link names (matching the USD hierarchy).
    base_frame: str = "base_link"
    chassis_frame: str = "base_link"
    body_link: str = "base_link"

    # Include all articulation bodies (chassis, wheels, rollers) in contact sensing with lateral
    # force filtering so collisions on protruding wheels/rollers trigger without floor false positives.
    contact_bodies: str = ".*"
    ground_contact_on_body: bool = True

    # Sensor frame names.
    lidar_frame: str = "lidar_link"
    camera_frame: str = "camera_link"

    # LiDAR at the mast height, 0.30 m above the floor (offsets are relative to base_link, 0.091 m up).
    sensor_height: float = MAST_CAMERA_OFFSET[2]
    lidar_offset: tuple[float, float, float] = MAST_CAMERA_OFFSET

    # kaya.mast (default): virtual camera at 0.30 m above the floor.
    # base_link is 0.091 m above the floor, so the offset is 0.30 - 0.091 = 0.209 m.
    camera_offset: tuple[float, float, float] = MAST_CAMERA_OFFSET
    camera_rot: tuple[float, float, float, float] = ROT_LEVEL

    # Footprint polygon (x, y) relative to base_link. Kaya's visual bounding box is ~0.27 × 0.29 m;
    # the omni-wheel arrangement is roughly circular with radius ~0.15 m.
    footprint: tuple[tuple[float, float], ...] = (
        (0.145, 0.145),
        (0.145, -0.145),
        (-0.145, -0.145),
        (-0.145, 0.145),
    )
    chassis_size: tuple[float, float, float] = (0.29, 0.29, 0.24)  # length, width, height (m)
    chassis_offset: tuple[float, float, float] = (0.0, 0.0, 0.12)  # centre of box above base_link
    robot_radius: float = 0.15
    robot_height: float = 0.24


# Kaya registers itself only to keep the module importable on its own; the variants (and their data) are registered
# by registry.register_default_embodiments from kaya_variants. variants=None keeps them.
register_embodiment("kaya", KayaEmbodimentCfg, drive_type="holonomic")
