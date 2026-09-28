# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ROS 2 sensor bridge nodes converting Isaac Lab simulation tensors to standard ROS 2 messages."""

from __future__ import annotations

import math
from typing import Optional

import torch
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import LaserScan


class LaserScanPublisherNode(Node):
    """ROS 2 node that bridges Isaac Lab 2D RayCaster hit tensors to sensor_msgs/LaserScan."""

    def __init__(
        self,
        node_name: str = "laserscan_publisher",
        topic_name: str = "/scan",
        frame_id: str = "lidar_link",
        horizontal_fov_deg: float = 360.0,
        horizontal_res_deg: float = 1.0,
        range_min: float = 0.05,
        range_max: float = 25.0,
        scan_time: float = 0.02,
    ):
        """Initialize the LaserScan publisher node.

        Args:
            node_name: ROS 2 node name.
            topic_name: Target LaserScan topic name (defaults to '/scan').
            frame_id: Stamped TF frame name for the scan.
            horizontal_fov_deg: Total FOV in degrees (defaults to 360.0).
            horizontal_res_deg: Angular increment between beams in degrees (defaults to 1.0).
            range_min: Minimum sensor range in meters.
            range_max: Maximum sensor range in meters.
            scan_time: Scan duration / period in seconds.
        """
        super().__init__(node_name)
        self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])

        self.frame_id = frame_id
        self.range_min = float(range_min)
        self.range_max = float(range_max)
        self.scan_time = float(scan_time)

        # Angular configuration matching patterns.LidarPatternCfg
        self.angle_increment = math.radians(horizontal_res_deg)
        half_fov = math.radians(horizontal_fov_deg / 2.0)
        self.angle_min = -half_fov
        # For 360 fov with 1-deg resolution, 360 beams span [-180, +179]
        if abs(horizontal_fov_deg - 360.0) < 1e-4:
            self.angle_max = half_fov - self.angle_increment
        else:
            self.angle_max = half_fov

        self.scan_pub = self.create_publisher(LaserScan, topic_name, 10)

    def publish_from_raycaster(
        self,
        ray_hits_w: torch.Tensor,
        sensor_pos_w: torch.Tensor,
        stamp_time: Optional[float] = None,
    ) -> LaserScan:
        """Calculate ranges from RayCaster world hit coordinates and publish LaserScan.

        Args:
            ray_hits_w: Tensor of shape (B, 3) or (N, B, 3) with world ray hit coordinates.
            sensor_pos_w: Tensor of shape (3,) or (N, 3) with sensor world origin.
            stamp_time: Optional simulation timestamp in seconds (uses ROS node clock if None).

        Returns:
            The published LaserScan message.
        """
        if ray_hits_w.dim() == 3:
            ray_hits = ray_hits_w[0]
        else:
            ray_hits = ray_hits_w

        if sensor_pos_w.dim() == 2:
            sensor_pos = sensor_pos_w[0]
        else:
            sensor_pos = sensor_pos_w

        # Compute Euclidean distance from sensor position
        deltas = ray_hits - sensor_pos.unsqueeze(0)
        dists = torch.linalg.norm(deltas, dim=-1)

        # Replace non-hits, infs, nans, or out-of-range readings with inf
        invalid_mask = torch.isnan(dists) | torch.isinf(dists) | (dists > self.range_max) | (dists < self.range_min)
        dists_clamped = dists.clone()
        dists_clamped[invalid_mask] = float("inf")

        ranges_list = dists_clamped.cpu().tolist()

        msg = LaserScan()
        if stamp_time is not None:
            sec = int(stamp_time)
            nanosec = int((stamp_time - sec) * 1e9)
            msg.header.stamp.sec = sec
            msg.header.stamp.nanosec = nanosec
        else:
            msg.header.stamp = self.get_clock().now().to_msg()

        msg.header.frame_id = self.frame_id
        msg.angle_min = self.angle_min
        msg.angle_max = self.angle_max
        msg.angle_increment = self.angle_increment
        msg.time_increment = 0.0
        msg.scan_time = self.scan_time
        msg.range_min = self.range_min
        msg.range_max = self.range_max
        msg.ranges = ranges_list

        self.scan_pub.publish(msg)
        return msg
