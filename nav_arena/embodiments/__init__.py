# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robot embodiment definitions, kinematic math, and embodiment registry."""

from .actions import DifferentialDriveAction, DifferentialDriveActionCfg
from .base import RobotEmbodimentCfg
from .dingo import (
    DINGO_ACTION_CFG,
    DINGO_CFG,
    DingoEmbodimentCfg,
    dingo_stage_patch,
)
from .kinematics import diff_drive_fk, diff_drive_ik
from .nova_carter import (
    NOVA_CARTER_ACTION_CFG,
    NOVA_CARTER_CFG,
    NovaCarterEmbodimentCfg,
)
from .registry import (
    clear_registry,
    get_embodiment,
    list_embodiments,
    register_default_embodiments,
    register_embodiment,
)
from .sensors import SensorSuiteCfg, create_2d_lidar_cfg
from .urdf import generate_minimal_urdf

__all__ = [
    "DifferentialDriveAction",
    "DifferentialDriveActionCfg",
    "diff_drive_fk",
    "diff_drive_ik",
    "RobotEmbodimentCfg",
    "get_embodiment",
    "list_embodiments",
    "register_embodiment",
    "clear_registry",
    "register_default_embodiments",
    "NOVA_CARTER_CFG",
    "NOVA_CARTER_ACTION_CFG",
    "NovaCarterEmbodimentCfg",
    "DINGO_CFG",
    "DINGO_ACTION_CFG",
    "DingoEmbodimentCfg",
    "dingo_stage_patch",
    "SensorSuiteCfg",
    "create_2d_lidar_cfg",
    "generate_minimal_urdf",
]
