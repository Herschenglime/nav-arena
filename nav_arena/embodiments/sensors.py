# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Soft-swappable sensor configurations defined purely in Python."""

from __future__ import annotations

from dataclasses import dataclass, field

from isaaclab.sensors import MultiMeshRayCasterCfg, RayCasterCfg, patterns
from isaaclab.utils.configclass import configclass


def create_2d_lidar_cfg(
    prim_path: str = "{ENV_REGEX_NS}/Robot/chassis_link",
    mesh_prim_paths: list[str | MultiMeshRayCasterCfg.RaycastTargetCfg] | None = None,
    mount_pos: tuple[float, float, float] = (0.0, 0.0, 0.35),
    horizontal_fov_deg: float = 360.0,
    horizontal_res_deg: float = 1.0,
    max_range: float = 25.0,
    use_multi_mesh: bool = False,
) -> RayCasterCfg:
    """Create a 2D planar LiDAR sensor configuration.

    Args:
        prim_path: USD path where sensor frame is spawned.
        mesh_prim_paths: Mesh prim paths to cast rays against.
        mount_pos: (x, y, z) offset relative to parent link frame in meters.
        horizontal_fov_deg: Total horizontal field of view in degrees.
        horizontal_res_deg: Angular resolution per ray beam in degrees.
        max_range: Maximum ray distance.
        use_multi_mesh: Whether to use MultiMeshRayCaster for scenes with multiple meshes.

    Returns:
        Configured RayCasterCfg or MultiMeshRayCasterCfg instance.
    """
    half_fov = horizontal_fov_deg / 2.0
    pattern_cfg = patterns.LidarPatternCfg(
        channels=1,
        vertical_fov_range=(0.0, 0.0),
        horizontal_fov_range=(-half_fov, half_fov),
        horizontal_res=horizontal_res_deg,
    )

    if use_multi_mesh:
        if mesh_prim_paths is None:
            mesh_prim_paths = [
                MultiMeshRayCasterCfg.RaycastTargetCfg(
                    prim_expr="/World/Scene",
                    merge_prim_meshes=True,
                    track_mesh_transforms=False,
                )
            ]
        return MultiMeshRayCasterCfg(
            prim_path=prim_path,
            mesh_prim_paths=mesh_prim_paths,
            offset=MultiMeshRayCasterCfg.OffsetCfg(pos=mount_pos),
            ray_alignment="base",
            pattern_cfg=pattern_cfg,
            max_distance=max_range,
            debug_vis=False,
        )

    if mesh_prim_paths is None:
        mesh_prim_paths = ["/World/defaultGroundPlane"]

    return RayCasterCfg(
        prim_path=prim_path,
        mesh_prim_paths=mesh_prim_paths,
        offset=RayCasterCfg.OffsetCfg(pos=mount_pos),
        ray_alignment="base",
        pattern_cfg=pattern_cfg,
        max_distance=max_range,
        debug_vis=False,
    )


@dataclass
class SensorSuiteCfg:
    """Configurable sensor loadout for an embodiment."""

    lidar_2d: RayCasterCfg | None = field(default_factory=create_2d_lidar_cfg)
    enable_camera: bool = False
