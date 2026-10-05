# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""InteriorAgent scene configuration and loader."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg
from isaaclab.utils.configclass import configclass

from nav_arena.embodiments import (
    NOVA_CARTER_CFG,
    create_2d_lidar_cfg,
    create_embodiment_camera_cfg,
    create_goal_camera_cfg,
    get_embodiment,
)
from nav_arena.utils.paths import CACHE_DIR, DATA_DIR

DEFAULT_INTERIOR_AGENT_DIR = str(DATA_DIR / "InteriorAgent")
DEFAULT_INTERIOR_AGENT_SCENE_ID = "kujiale_0003"
DEFAULT_INTERIOR_AGENT_USD = os.path.join(
    DEFAULT_INTERIOR_AGENT_DIR,
    DEFAULT_INTERIOR_AGENT_SCENE_ID,
    f"{DEFAULT_INTERIOR_AGENT_SCENE_ID}.usda",
)
DEFAULT_INTERIOR_AGENT_CACHE_DIR = str(CACHE_DIR / "scenes")

INTERIOR_AGENT_DOOR_PREFIX = "other/door_"



def prepare_interior_agent_stage(stage, disable_doors: bool = True) -> int:
    """Condition InteriorAgent USD stages for static navigation benchmarking.

    InteriorAgent scenes model closed door panels under `/Root/Meshes/other/door_XXXX`
    with collision bodies. For navigation benchmarks without dynamic manipulation,
    deactivating these prims opens all doorways seamlessly for both raycast mapping
    and runtime PhysX + 2D LiDAR.

    Args:
        stage: pxr.Usd.Stage instance to condition.
        disable_doors: If True, deactivates door prims so doorways remain open.

    Returns:
        Number of door prims deactivated.
    """
    if not disable_doors:
        return 0

    count = 0
    for prim in stage.Traverse():
        path_str = prim.GetPath().pathString
        if INTERIOR_AGENT_DOOR_PREFIX in path_str and prim.GetName().startswith("door_"):
            if prim.IsActive():
                prim.SetActive(False)
                count += 1
    return count


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


def get_preprocessed_usd(
    scene_id_or_path: str = DEFAULT_INTERIOR_AGENT_SCENE_ID,
    preprocessor: Callable | None = None,
    variant_name: str = "open_doors",
    base_dir: str = DEFAULT_INTERIOR_AGENT_DIR,
    cache_dir: str = DEFAULT_INTERIOR_AGENT_CACHE_DIR,
    force_regenerate: bool = False,
) -> str:
    """Generate or retrieve a cached USD delta layer conditioned by a preprocessor function.

    Sublayers the original source scene USD into a lightweight delta layer, applies
    the preprocessor callback to modify prim attributes/states (e.g. deactivating doors),
    and saves the result to a cache directory.

    Args:
        scene_id_or_path: Scene directory name or full path to USDA/USD file.
        preprocessor: Callable taking a Usd.Stage to perform non-destructive conditioning.
        variant_name: Descriptive name for the conditioned variant (e.g. 'open_doors').
        base_dir: Base directory containing raw scene datasets.
        cache_dir: Directory where cached delta USD files are stored.
        force_regenerate: If True, regenerates the delta layer even if cached file exists.

    Returns:
        Absolute path to the conditioned USD delta layer file.
    """
    raw_usd = resolve_interior_agent_usd(scene_id_or_path, base_dir)
    scene_id = os.path.basename(os.path.dirname(raw_usd))
    if not scene_id or scene_id == ".":
        scene_id = os.path.splitext(os.path.basename(raw_usd))[0]

    if preprocessor is None:
        return raw_usd

    out_dir = os.path.join(cache_dir, scene_id)
    os.makedirs(out_dir, exist_ok=True)
    delta_usd_path = os.path.join(out_dir, f"{scene_id}_{variant_name}.usda")

    if os.path.isfile(delta_usd_path) and not force_regenerate:
        return delta_usd_path

    # Lazy import to ensure Omniverse / Isaac Sim USD runtime is initialized
    from pxr import Usd

    if os.path.exists(delta_usd_path):
        os.remove(delta_usd_path)

    stage = Usd.Stage.CreateNew(delta_usd_path)
    stage.GetRootLayer().subLayerPaths.append(raw_usd)

    # Set defaultPrim to match /Root from source scene
    root_prim = stage.GetPrimAtPath("/Root")
    if root_prim.IsValid():
        stage.SetDefaultPrim(root_prim)
    else:
        src_stage = Usd.Stage.Open(raw_usd)
        src_def = src_stage.GetDefaultPrim()
        if src_def.IsValid() and stage.GetPrimAtPath(src_def.GetPath()).IsValid():
            stage.SetDefaultPrim(stage.GetPrimAtPath(src_def.GetPath()))

    preprocessor(stage)
    stage.GetRootLayer().Save()
    print(f"[INFO] Created cached USD delta layer: '{delta_usd_path}'")
    return delta_usd_path


def get_open_door_usd(
    scene_id_or_path: str = DEFAULT_INTERIOR_AGENT_SCENE_ID,
    base_dir: str = DEFAULT_INTERIOR_AGENT_DIR,
    force_regenerate: bool = False,
) -> str:
    """Resolve an InteriorAgent scene USD with all interior doors deactivated via a delta layer.

    Args:
        scene_id_or_path: Scene identifier or path.
        base_dir: Base directory containing InteriorAgent scenes.
        force_regenerate: If True, re-creates the delta layer.

    Returns:
        Path to the open-doors delta USD file.
    """
    return get_preprocessed_usd(
        scene_id_or_path=scene_id_or_path,
        preprocessor=prepare_interior_agent_stage,
        variant_name="open_doors",
        base_dir=base_dir,
        force_regenerate=force_regenerate,
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

    # Optional sensors, populated by create_interior_agent_scene_cfg (None entries are skipped by the scene)
    camera: CameraCfg | None = None
    goal_camera: CameraCfg | None = None


def create_interior_agent_scene_cfg(
    scene_id_or_path: str = DEFAULT_INTERIOR_AGENT_SCENE_ID,
    base_dir: str = DEFAULT_INTERIOR_AGENT_DIR,
    robot_spawn_pos: tuple[float, float, float] = (0.0, -2.0, 0.25),
    robot_spawn_rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    num_envs: int = 1,
    env_spacing: float = 30.0,
    open_doors: bool = True,
    robot_name: str = "nova_carter",
    enable_camera: bool = False,
    enable_goal_camera: bool = False,
) -> InteriorAgentSceneCfg:
    """Create a configured InteriorAgentSceneCfg instance.

    Args:
        scene_id_or_path: Scene identifier (e.g. 'kujiale_0003') or path to USD.
        base_dir: Base directory for dataset scenes.
        robot_spawn_pos: (x, y, z) initial position of the robot in meters.
        robot_spawn_rot: (x, y, z, w) initial quaternion orientation of the robot.
        num_envs: Number of parallel environments.
        env_spacing: Distance between environment origins in meters.
        open_doors: If True, uses the open-door delta layer to allow free passage for PhysX and LiDAR.
        robot_name: Registered embodiment to spawn (see :func:`nav_arena.embodiments.list_embodiments`).
        enable_camera: Mount the embodiment's RGB-D camera (requires cameras enabled at app launch).
        enable_goal_camera: Add a free-standing RGB camera at ``/World/GoalCamera`` for goal-image rendering.
            Only supported with a single environment.

    Returns:
        Configured InteriorAgentSceneCfg.

    Raises:
        ValueError: If a goal camera is requested with more than one environment.
    """
    if enable_goal_camera and num_envs != 1:
        raise ValueError("The goal camera is a single global prim and requires num_envs == 1")
    embodiment = get_embodiment(robot_name)
    if open_doors:
        usd_path = get_open_door_usd(scene_id_or_path, base_dir)
    else:
        usd_path = resolve_interior_agent_usd(scene_id_or_path, base_dir)

    cfg = InteriorAgentSceneCfg(num_envs=num_envs, env_spacing=env_spacing)
    cfg.scene_asset = cfg.scene_asset.replace(
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        )
    )
    cfg.robot = embodiment.articulation_cfg.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=embodiment.articulation_cfg.init_state.replace(
            pos=robot_spawn_pos,
            rot=robot_spawn_rot,
        ),
    )
    # Sensors attach to the embodiment's USD body link (chassis_link for the Carter, base_link for the Dingo).
    if "lidar" in embodiment.sensors:
        cfg.lidar = cfg.lidar.replace(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{embodiment.body_link}",
            offset=cfg.lidar.offset.replace(pos=tuple(embodiment.lidar_offset)),
            ray_alignment=embodiment.lidar_ray_alignment,
        )
    else:
        cfg.lidar = None
    if enable_camera:
        cfg.camera = create_embodiment_camera_cfg(embodiment)
    if enable_goal_camera:
        cfg.goal_camera = create_goal_camera_cfg(embodiment)
    return cfg

