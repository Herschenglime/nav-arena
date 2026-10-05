# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for sensor configuration factories (CPU-only; no Kit boot)."""

from __future__ import annotations

import math

import pytest

from nav_arena.embodiments import (
    RGBD_CAMERA_DATA_TYPES,
    SensorSuiteCfg,
    create_embodiment_camera_cfg,
    create_rgbd_camera_cfg,
    get_embodiment,
    pinhole_intrinsics,
)


def test_rgbd_camera_defaults_match_navdp_dingo_camera():
    """Verify defaults reproduce the camera the NavDP baselines were trained and evaluated with."""
    cfg = create_rgbd_camera_cfg()
    assert (cfg.width, cfg.height) == (640, 360)
    assert cfg.data_types == ["rgb", "distance_to_image_plane"]
    assert cfg.spawn.focal_length == 1.4
    assert cfg.spawn.focus_distance == 0.205
    assert cfg.spawn.horizontal_aperture == 1.88
    assert tuple(cfg.spawn.clipping_range) == (0.05, 20.0)
    assert type(cfg.renderer_cfg).__name__ == "IsaacRtxRendererCfg"


def test_rgbd_camera_pose_uses_world_convention_xyzw():
    """Verify the camera faces +X / +Z-up with an identity (x, y, z, w) rotation, 0.30 m above the link."""
    cfg = create_rgbd_camera_cfg()
    assert cfg.offset.convention == "world"
    assert tuple(cfg.offset.pos) == (0.0, 0.0, 0.30)
    assert tuple(cfg.offset.rot) == (0.0, 0.0, 0.0, 1.0)


def test_rgbd_camera_update_period_defers_to_render_interval():
    """Verify update_period=0 so the env render_interval alone sets the camera rate."""
    assert create_rgbd_camera_cfg().update_period == 0.0
    assert create_rgbd_camera_cfg(update_period=0.2).update_period == 0.2


def test_rgbd_camera_overrides():
    """Verify factory arguments override optics, resolution, and mounting."""
    cfg = create_rgbd_camera_cfg(
        prim_path="{ENV_REGEX_NS}/Robot/x/cam", mount_pos=(0.1, 0.0, 0.5), width=320, height=180, focal_length=2.0
    )
    assert cfg.prim_path == "{ENV_REGEX_NS}/Robot/x/cam"
    assert tuple(cfg.offset.pos) == (0.1, 0.0, 0.5)
    assert (cfg.width, cfg.height) == (320, 180)
    assert cfg.spawn.focal_length == 2.0


def test_camera_data_types_constant_is_not_shared_state():
    """Verify mutating a returned cfg's data types cannot corrupt the module constant."""
    cfg = create_rgbd_camera_cfg()
    cfg.data_types.append("normals")
    assert RGBD_CAMERA_DATA_TYPES == ["rgb", "distance_to_image_plane"]
    assert create_rgbd_camera_cfg().data_types == ["rgb", "distance_to_image_plane"]


def test_pinhole_intrinsics_for_baseline_camera():
    """Verify intrinsics (square pixels) and the resulting ~68 deg horizontal / ~41 deg vertical FOV."""
    fx, fy, cx, cy = pinhole_intrinsics(640, 360, 1.4, 1.88)
    assert fx == pytest.approx(476.595, abs=1e-3)
    assert fy == fx
    assert (cx, cy) == (320.0, 180.0)
    hfov = 2 * math.degrees(math.atan(320.0 / fx))
    vfov = 2 * math.degrees(math.atan(180.0 / fy))
    assert hfov == pytest.approx(68.0, abs=0.5)
    assert vfov == pytest.approx(41.2, abs=0.5)


@pytest.mark.parametrize(
    "robot,parent", [("dingo", "base_link"), ("nova_carter", "chassis_link")]
)
def test_embodiment_camera_mounts_on_body_link(robot, parent):
    """Verify the embodiment camera attaches under the robot's USD body link using its offsets."""
    emb = get_embodiment(robot)
    cfg = create_embodiment_camera_cfg(emb)
    assert cfg.prim_path == f"{{ENV_REGEX_NS}}/Robot/{parent}/front_camera"
    assert tuple(cfg.offset.pos) == tuple(emb.camera_offset)
    assert tuple(cfg.offset.rot) == tuple(emb.camera_rot)


def test_embodiment_camera_accepts_overrides():
    """Verify explicit kwargs win over embodiment-derived defaults."""
    cfg = create_embodiment_camera_cfg(get_embodiment("dingo"), mount_pos=(0.0, 0.0, 0.9), width=64, height=48)
    assert tuple(cfg.offset.pos) == (0.0, 0.0, 0.9)
    assert (cfg.width, cfg.height) == (64, 48)


def test_sensor_suite_camera_disabled_by_default():
    """Verify the default sensor suite has no camera and an unaffected LiDAR."""
    suite = SensorSuiteCfg()
    assert suite.camera is None
    assert suite.enable_camera is False
    assert suite.lidar_2d is not None
