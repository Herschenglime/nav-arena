# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for LaserScanPublisherNode range calculations and invalid mask handling."""

import math
import pytest
import rclpy
import torch

from nav_arena.ros2.sensors import LaserScanPublisherNode


@pytest.fixture(scope="module")
def rclpy_module_context():
    """Manage lifecycle of ROS 2 context for unit test module."""
    if not rclpy.ok():
        rclpy.init()
    yield
    if rclpy.ok():
        rclpy.shutdown()


@pytest.fixture
def default_node(rclpy_module_context):
    """Create default LaserScanPublisherNode."""
    node = LaserScanPublisherNode(
        node_name="test_laserscan_default",
        topic_name="/test_scan_default",
        frame_id="lidar_link",
        horizontal_fov_deg=360.0,
        horizontal_res_deg=1.0,
        range_min=0.05,
        range_max=25.0,
        scan_time=0.02,
    )
    yield node
    node.destroy_node()


def test_accurate_range_calculation(default_node):
    """Verify that ray hit Euclidean distance is calculated accurately."""
    sensor_pos = torch.tensor([1.0, 2.0, 0.5])
    # 3 hits:
    # 1. Delta = (3, 4, 0) -> dist = 5.0
    # 2. Delta = (0, 0, 3) -> dist = 3.0
    # 3. Delta = (1, 1, 1) -> dist = sqrt(3) ~ 1.73205
    ray_hits = torch.tensor([
        [4.0, 6.0, 0.5],
        [1.0, 2.0, 3.5],
        [2.0, 3.0, 1.5],
    ])

    msg = default_node.publish_from_raycaster(ray_hits, sensor_pos)
    assert len(msg.ranges) == 3
    assert math.isclose(msg.ranges[0], 5.0, rel_tol=1e-5)
    assert math.isclose(msg.ranges[1], 3.0, rel_tol=1e-5)
    assert math.isclose(msg.ranges[2], math.sqrt(3.0), rel_tol=1e-5)


def test_invalid_mask_handling(default_node):
    """Verify NaN, Inf, under-range (< min), and over-range (> max) are clamped to inf."""
    sensor_pos = torch.tensor([0.0, 0.0, 0.0])
    ray_hits = torch.tensor([
        [0.02, 0.0, 0.0],          # dist = 0.02 < range_min (0.05) -> inf
        [30.0, 0.0, 0.0],          # dist = 30.0 > range_max (25.0) -> inf
        [float("nan"), 0.0, 0.0],  # NaN -> inf
        [float("inf"), 0.0, 0.0],  # Inf -> inf
        [10.0, 0.0, 0.0],          # Valid range = 10.0
    ])

    msg = default_node.publish_from_raycaster(ray_hits, sensor_pos)
    assert len(msg.ranges) == 5
    assert math.isinf(msg.ranges[0])
    assert math.isinf(msg.ranges[1])
    assert math.isinf(msg.ranges[2])
    assert math.isinf(msg.ranges[3])
    assert math.isclose(msg.ranges[4], 10.0, rel_tol=1e-5)


def test_batched_input_dimensions(default_node):
    """Verify handling of batched (N, B, 3) and (N, 3) tensors."""
    # N=2 environments, B=4 beams
    ray_hits_batched = torch.zeros(2, 4, 3)
    sensor_pos_batched = torch.zeros(2, 3)

    # First env has hits at distance 2.0
    ray_hits_batched[0, :, 0] = 2.0
    # Second env has hits at distance 8.0 (should be ignored since node takes index 0)
    ray_hits_batched[1, :, 0] = 8.0

    msg = default_node.publish_from_raycaster(ray_hits_batched, sensor_pos_batched)
    assert len(msg.ranges) == 4
    for r in msg.ranges:
        assert math.isclose(r, 2.0, rel_tol=1e-5)


def test_timestamp_handling(default_node):
    """Verify custom stamp_time converts correctly to seconds and nanoseconds."""
    sensor_pos = torch.zeros(3)
    ray_hits = torch.tensor([[1.0, 0.0, 0.0]])

    stamp = 1718000000.123456
    msg = default_node.publish_from_raycaster(ray_hits, sensor_pos, stamp_time=stamp)
    assert msg.header.stamp.sec == 1718000000
    # Floating point precision for nanoseconds
    assert abs(msg.header.stamp.nanosec - 123456000) < 1000

    # Test stamp_time=None uses ROS clock
    msg_now = default_node.publish_from_raycaster(ray_hits, sensor_pos, stamp_time=None)
    assert msg_now.header.stamp.sec >= 0


def test_fov_angle_boundaries(rclpy_module_context):
    """Verify angle_min, angle_max, and angle_increment for 360 deg vs sub-360 deg FOV."""
    # 360 deg FOV
    node_360 = LaserScanPublisherNode(
        node_name="test_node_360",
        topic_name="/scan_360",
        horizontal_fov_deg=360.0,
        horizontal_res_deg=1.0,
    )
    try:
        assert math.isclose(node_360.angle_min, -math.pi, rel_tol=1e-5)
        # angle_max = pi - 1 deg
        expected_max = math.pi - math.radians(1.0)
        assert math.isclose(node_360.angle_max, expected_max, rel_tol=1e-5)
        assert math.isclose(node_360.angle_increment, math.radians(1.0), rel_tol=1e-5)
    finally:
        node_360.destroy_node()

    # 180 deg FOV
    node_180 = LaserScanPublisherNode(
        node_name="test_node_180",
        topic_name="/scan_180",
        horizontal_fov_deg=180.0,
        horizontal_res_deg=0.5,
    )
    try:
        assert math.isclose(node_180.angle_min, -math.pi / 2.0, rel_tol=1e-5)
        assert math.isclose(node_180.angle_max, math.pi / 2.0, rel_tol=1e-5)
        assert math.isclose(node_180.angle_increment, math.radians(0.5), rel_tol=1e-5)
    finally:
        node_180.destroy_node()
