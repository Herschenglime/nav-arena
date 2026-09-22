# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ROS 2 OmniGraph and rclpy integration utilities for mobile embodiments."""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

import torch

logger = logging.getLogger(__name__)


def setup_ros2_clock(graph_path: str = "/ActionGraph/ROS_Clock") -> str | None:
    """Create ROS 2 Clock publisher OmniGraph to sync ROS 2 time with sim time.

    Args:
        graph_path: Target prim path for the ActionGraph.

    Returns:
        Graph prim path on success, None on failure.
    """
    try:
        from isaacsim.core.experimental.utils.app import enable_extension

        enable_extension("isaacsim.ros2.nodes")
        from isaacsim.ros2.nodes import Ros2ClockGraphConfig, create_ros2_clock_graph

        config = Ros2ClockGraphConfig(graph_path=graph_path)
        return create_ros2_clock_graph(config)
    except Exception as e:
        logger.warning(f"Failed to create ROS 2 clock graph: {e}")
        return None


def setup_ros2_odometry(
    articulation_root: str,
    chassis_prim: str,
    graph_path: str = "/ActionGraph/ROS_Odometry",
    odometry_topic: str = "/odom",
    tf_topic: str = "/tf",
    node_namespace: str = "",
) -> str | None:
    """Create ROS 2 Odometry and TF publisher OmniGraph for the mobile base.

    Args:
        articulation_root: Prim path to the Articulation root (e.g. ``/World/envs/env_0/Robot``).
        chassis_prim: Prim path to the chassis body (e.g. ``/World/envs/env_0/Robot/chassis_link``).
        graph_path: Target prim path for the ActionGraph.
        odometry_topic: Topic name for nav_msgs/Odometry.
        tf_topic: Topic name for tf2_msgs/TFMessage.
        node_namespace: ROS 2 node namespace.

    Returns:
        Graph prim path on success, None on failure.
    """
    try:
        from isaacsim.core.experimental.utils.app import enable_extension

        enable_extension("isaacsim.ros2.nodes")
        from isaacsim.ros2.nodes import Ros2OdometryGraphConfig, create_ros2_odometry_graph

        config = Ros2OdometryGraphConfig(
            graph_path=graph_path,
            articulation_root=articulation_root,
            chassis_prim=chassis_prim,
            odometry_topic=odometry_topic,
            tf_topic=tf_topic,
            node_namespace=node_namespace,
            publish_robot_tf=True,
        )

        return create_ros2_odometry_graph(config)

    except Exception as e:
        logger.warning(f"Failed to create ROS 2 odometry graph: {e}")
        return None


class Ros2TwistReceiver:
    """Asynchronous ROS 2 subscriber for /cmd_vel that feeds into the ActionManager.

    This class decouples ROS 2 subscriber callbacks from the simulation stepping loop,
    ensuring thread-safe consumption of velocity commands.
    """

    def __init__(
        self,
        topic_name: str = "/cmd_vel",
        node_name: str = "nav_arena_twist_receiver",
        device: str = "cuda:0",
    ):
        self.topic_name = topic_name
        self.device = device
        self._linear_x = 0.0
        self._angular_z = 0.0
        self._lock = threading.Lock()
        self._node = None
        self._executor = None
        self._thread = None
        self._is_running = False

    def start(self):
        """Initialize rclpy node and spin in a background thread."""
        try:
            import rclpy
            from geometry_msgs.msg import Twist
            from rclpy.executors import SingleThreadedExecutor

            if not rclpy.ok():
                rclpy.init()

            self._node = rclpy.create_node(self._node_name if hasattr(self, "_node_name") else "twist_receiver")
            self._sub = self._node.create_subscription(
                Twist,
                self.topic_name,
                self._twist_callback,
                10,
            )

            self._executor = SingleThreadedExecutor()
            self._executor.add_node(self._node)

            self._is_running = True
            self._thread = threading.Thread(target=self._executor.spin, daemon=True)
            self._thread.start()
            logger.info(f"Ros2TwistReceiver listening on '{self.topic_name}'")
        except Exception as e:
            logger.warning(f"Could not start ROS 2 twist receiver: {e}")

    def _twist_callback(self, msg):
        with self._lock:
            self._linear_x = float(msg.linear.x)
            self._angular_z = float(msg.angular.z)

    def get_action(self, num_envs: int = 1) -> torch.Tensor:
        """Get the latest velocity command formatted as an ActionManager action tensor.

        Args:
            num_envs: Number of environment instances.

        Returns:
            torch.Tensor with shape (num_envs, 2) containing [v, omega].
        """
        with self._lock:
            v = self._linear_x
            omega = self._angular_z

        action = torch.tensor([[v, omega]], device=self.device, dtype=torch.float32)
        if num_envs > 1:
            action = action.repeat(num_envs, 1)
        return action

    def stop(self):
        """Shutdown subscriber and background thread."""
        self._is_running = False
        if self._executor:
            self._executor.shutdown()
        if self._node:
            self._node.destroy_node()
