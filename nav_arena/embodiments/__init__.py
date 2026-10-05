# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robot embodiment definitions, kinematic math, and embodiment registry."""

from typing import TYPE_CHECKING

from .base import SENSORS, EmbodimentVariant, RobotEmbodimentCfg, is_policy_driven
from .registry import (
    clear_registry,
    default_variant,
    drive_type_of,
    get_embodiment,
    list_embodiments,
    list_variants,
    register_default_embodiments,
    register_embodiment,
    resolve_embodiment_name,
)

# The modules below import torch and Isaac Lab, so they load on first access. This keeps robot names, variants and
# their resolution (everything the CLI needs) free of simulator and ML imports.
_LAZY_EXPORTS = {
    "actions": ("DifferentialDriveAction", "DifferentialDriveActionCfg"),
    "dingo": ("DINGO_ACTION_CFG", "DingoEmbodimentCfg", "create_dingo_articulation_cfg"),
    "go2": ("Go2EmbodimentCfg", "create_go2_action_cfg", "create_go2_articulation_cfg"),
    "kaya": ("KAYA_ACTION_CFG", "KAYA_ARTICULATION_CFG", "KayaEmbodimentCfg", "KAYA_MAST", "KAYA_NATIVE"),
    "kinematics": ("diff_drive_fk", "diff_drive_ik"),
    "nova_carter": ("NOVA_CARTER_ACTION_CFG", "NOVA_CARTER_CFG", "NovaCarterEmbodimentCfg"),
    "sensors": (
        "RGBD_CAMERA_DATA_TYPES",
        "SensorSuiteCfg",
        "create_2d_lidar_cfg",
        "create_embodiment_camera_cfg",
        "create_goal_camera_cfg",
        "create_rgbd_camera_cfg",
        "pinhole_intrinsics",
    ),
    "urdf": ("generate_minimal_urdf",),
}
_LAZY_MODULE_OF = {name: module for module, names in _LAZY_EXPORTS.items() for name in names}

if TYPE_CHECKING:
    from .actions import DifferentialDriveAction, DifferentialDriveActionCfg
    from .dingo import DINGO_ACTION_CFG, DingoEmbodimentCfg, create_dingo_articulation_cfg
    from .go2 import Go2EmbodimentCfg, create_go2_action_cfg, create_go2_articulation_cfg
    from .kaya import KAYA_ACTION_CFG, KAYA_ARTICULATION_CFG, KAYA_MAST, KAYA_NATIVE, KayaEmbodimentCfg
    from .kinematics import diff_drive_fk, diff_drive_ik
    from .nova_carter import NOVA_CARTER_ACTION_CFG, NOVA_CARTER_CFG, NovaCarterEmbodimentCfg
    from .sensors import (
        RGBD_CAMERA_DATA_TYPES,
        SensorSuiteCfg,
        create_2d_lidar_cfg,
        create_embodiment_camera_cfg,
        create_goal_camera_cfg,
        create_rgbd_camera_cfg,
        pinhole_intrinsics,
    )
    from .urdf import generate_minimal_urdf

__all__ = [
    "RobotEmbodimentCfg",
    "EmbodimentVariant",
    "SENSORS",
    "is_policy_driven",
    "default_variant",
    "drive_type_of",
    "list_variants",
    "resolve_embodiment_name",
    "get_embodiment",
    "list_embodiments",
    "register_embodiment",
    "clear_registry",
    "register_default_embodiments",
    *_LAZY_MODULE_OF,
]


def __getattr__(name: str):
    module = _LAZY_MODULE_OF.get(name)
    if module is not None:
        import importlib

        return getattr(importlib.import_module(f".{module}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
