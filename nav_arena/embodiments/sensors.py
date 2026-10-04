# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Soft-swappable sensor configurations defined purely in Python."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg, MultiMeshRayCasterCfg, RayCasterCfg, patterns
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from .base import RobotEmbodimentCfg


def create_2d_lidar_cfg(
    prim_path: str = "{ENV_REGEX_NS}/Robot/chassis_link",
    mesh_prim_paths: list[str | MultiMeshRayCasterCfg.RaycastTargetCfg] | None = None,
    mount_pos: tuple[float, float, float] = (0.0, 0.0, 0.35),
    horizontal_fov_deg: float = 360.0,
    horizontal_res_deg: float = 1.0,
    max_range: float = 25.0,
    use_multi_mesh: bool = False,
    update_period: float = 0.05,
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
        update_period: Update period in seconds. Defaults to 0.05s (20 Hz) to match physical LiDARs and reduce Warp compute.

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
            update_period=update_period,
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
        update_period=update_period,
        debug_vis=False,
    )


RGBD_CAMERA_DATA_TYPES = ["rgb", "distance_to_image_plane"]
"""Camera outputs consumed by the learned visual-navigation baselines: RGB and metric z-depth."""


def pinhole_intrinsics(
    width: int, height: int, focal_length: float, horizontal_aperture: float
) -> tuple[float, float, float, float]:
    """Compute pixel intrinsics ``(fx, fy, cx, cy)`` for an Isaac Sim pinhole camera.

    Omniverse cameras have square pixels, so the vertical aperture follows from the aspect ratio
    and ``fx == fy``.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        focal_length: Focal length in the same units as ``horizontal_aperture`` (USD tenths of a scene unit).
        horizontal_aperture: Horizontal film aperture, same units as ``focal_length``.

    Returns:
        Tuple ``(fx, fy, cx, cy)`` in pixels.
    """
    fx = focal_length * width / horizontal_aperture
    return fx, fx, width / 2.0, height / 2.0


def create_rgbd_camera_cfg(
    prim_path: str = "{ENV_REGEX_NS}/Robot/base_link/front_camera",
    mount_pos: tuple[float, float, float] = (0.0, 0.0, 0.30),
    mount_rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    width: int = 640,
    height: int = 360,
    focal_length: float = 1.4,
    focus_distance: float = 0.205,
    horizontal_aperture: float = 1.88,
    clipping_range: tuple[float, float] = (0.05, 20.0),
    update_period: float = 0.0,
) -> CameraCfg:
    """Create a forward-facing pinhole RGB-D camera configuration.

    Defaults reproduce the camera the NavDP baselines were trained and evaluated with on the
    Clearpath Dingo (640x360, 1.4 / 1.88 optics, ~68 deg horizontal FOV, 0.30 m above the base).

    The pose uses the ``"world"`` convention (+X forward, +Z up) so ``mount_rot`` is expressed in the
    robot frame; Isaac Lab quaternions are ``(x, y, z, w)``.

    Args:
        prim_path: Camera prim path; its parent must be a robot link.
        mount_pos: (x, y, z) offset from the parent link in meters.
        mount_rot: (x, y, z, w) rotation offset in the world convention.
        width: Image width in pixels.
        height: Image height in pixels.
        focal_length: Pinhole focal length.
        focus_distance: Pinhole focus distance.
        horizontal_aperture: Horizontal film aperture.
        clipping_range: (near, far) clipping distances in meters; farther depth returns ``inf``.
        update_period: Sensor update period in seconds. 0 refreshes whenever the environment renders,
            so the effective rate is set by ``sim.render_interval``.

    Returns:
        Configured CameraCfg providing ``rgb`` and ``distance_to_image_plane``.
    """
    # Imported lazily: only needed when a camera is actually configured.
    from isaaclab_physx.renderers import IsaacRtxRendererCfg

    return CameraCfg(
        prim_path=prim_path,
        update_period=update_period,
        height=height,
        width=width,
        data_types=list(RGBD_CAMERA_DATA_TYPES),
        renderer_cfg=IsaacRtxRendererCfg(),
        offset=CameraCfg.OffsetCfg(pos=mount_pos, rot=mount_rot, convention="world"),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=focal_length,
            focus_distance=focus_distance,
            horizontal_aperture=horizontal_aperture,
            clipping_range=clipping_range,
        ),
    )


def create_embodiment_camera_cfg(embodiment: RobotEmbodimentCfg, **kwargs) -> CameraCfg:
    """Create the RGB-D camera for an embodiment, mounted per its ``body_link`` and camera offsets.

    Args:
        embodiment: Embodiment providing ``body_link``, ``camera_offset`` and ``camera_rot``.
        **kwargs: Overrides forwarded to :func:`create_rgbd_camera_cfg`.

    Returns:
        Configured CameraCfg.
    """
    kwargs.setdefault("prim_path", f"{{ENV_REGEX_NS}}/Robot/{embodiment.body_link}/front_camera")
    kwargs.setdefault("mount_pos", tuple(embodiment.camera_offset))
    kwargs.setdefault("mount_rot", tuple(embodiment.camera_rot))
    return create_rgbd_camera_cfg(**kwargs)


@dataclass
class SensorSuiteCfg:
    """Configurable sensor loadout for an embodiment."""

    lidar_2d: RayCasterCfg | None = field(default_factory=create_2d_lidar_cfg)
    camera: CameraCfg | None = None
    enable_camera: bool = False
