# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""InteriorAgent scene configuration and loader."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg
from isaaclab.utils.configclass import configclass

from nav_arena.embodiments import (
    NOVA_CARTER_CFG,
    create_2d_lidar_cfg,
)

DEFAULT_INTERIOR_AGENT_DIR = "/home/robopi/simulation/data/InteriorAgent"
DEFAULT_INTERIOR_AGENT_SCENE_ID = "kujiale_0003"
DEFAULT_INTERIOR_AGENT_USD = os.path.join(
    DEFAULT_INTERIOR_AGENT_DIR,
    DEFAULT_INTERIOR_AGENT_SCENE_ID,
    f"{DEFAULT_INTERIOR_AGENT_SCENE_ID}.usda",
)


def resolve_interior_agent_usd(
    scene_id_or_path: str = DEFAULT_INTERIOR_AGENT_SCENE_ID,
    base_dir: str = DEFAULT_INTERIOR_AGENT_DIR,
) -> str:
    """Resolve full path to a USD/USDA file for an InteriorAgent scene.

    Args:
        scene_id_or_path: Either a scene directory name (e.g. 'kujiale_0003') or an absolute path to a .usd/.usda file.
        base_dir: Directory containing InteriorAgent scene directories.

    Returns:
        Absolute path to the resolved USD/USDA file.
    """
    if os.path.isabs(scene_id_or_path) and os.path.isfile(scene_id_or_path):
        return scene_id_or_path

    # Check under base_dir/scene_id/scene_id.usda
    candidate_usda = os.path.join(base_dir, scene_id_or_path, f"{scene_id_or_path}.usda")
    if os.path.isfile(candidate_usda):
        return candidate_usda

    candidate_usd = os.path.join(base_dir, scene_id_or_path, f"{scene_id_or_path}.usd")
    if os.path.isfile(candidate_usd):
        return candidate_usd

    # Fallback to direct path under base_dir
    candidate_direct = os.path.join(base_dir, scene_id_or_path)
    if os.path.isfile(candidate_direct):
        return candidate_direct

    raise FileNotFoundError(
        f"Could not find InteriorAgent scene USD for '{scene_id_or_path}' in '{base_dir}'."
    )


@configclass
class InteriorAgentSceneCfg(InteractiveSceneCfg):
    """Configuration for an InteriorAgent interactive environment scene."""

    # Static environment geometry loaded from InteriorAgent USD (global scene)
    scene_asset: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/Scene",
        spawn=sim_utils.UsdFileCfg(
            usd_path=DEFAULT_INTERIOR_AGENT_USD,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
    )

    # Nova Carter Articulation
    # Spawns with clearance at z=0.25 and identity quaternion (x, y, z, w) = (0, 0, 0, 1)
    robot = NOVA_CARTER_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=NOVA_CARTER_CFG.init_state.replace(
            pos=(0.0, -2.0, 0.25),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
    )

    # 2D LiDAR RayCaster attached to robot chassis
    lidar = create_2d_lidar_cfg(
        prim_path="{ENV_REGEX_NS}/Robot/chassis_link",
        use_multi_mesh=True,
        mesh_prim_paths=[
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="/World/Scene",
                merge_prim_meshes=True,
                track_mesh_transforms=False,
            )
        ],
    )


def create_interior_agent_scene_cfg(
    scene_id_or_path: str = DEFAULT_INTERIOR_AGENT_SCENE_ID,
    base_dir: str = DEFAULT_INTERIOR_AGENT_DIR,
    robot_spawn_pos: tuple[float, float, float] = (0.0, -2.0, 0.25),
    robot_spawn_rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    num_envs: int = 1,
    env_spacing: float = 30.0,
) -> InteriorAgentSceneCfg:
    """Create a configured InteriorAgentSceneCfg instance.

    Args:
        scene_id_or_path: Scene identifier (e.g. 'kujiale_0003') or path to USD.
        base_dir: Base directory for dataset scenes.
        robot_spawn_pos: (x, y, z) initial position of the robot in meters.
        robot_spawn_rot: (x, y, z, w) initial quaternion orientation of the robot.
        num_envs: Number of parallel environments.
        env_spacing: Distance between environment origins in meters.

    Returns:
        Configured InteriorAgentSceneCfg.
    """
    usd_path = resolve_interior_agent_usd(scene_id_or_path, base_dir)

    cfg = InteriorAgentSceneCfg(num_envs=num_envs, env_spacing=env_spacing)
    cfg.scene_asset = cfg.scene_asset.replace(
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        )
    )
    cfg.robot = cfg.robot.replace(
        init_state=cfg.robot.init_state.replace(
            pos=robot_spawn_pos,
            rot=robot_spawn_rot,
        )
    )
    return cfg
