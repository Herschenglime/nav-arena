# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task state publisher node for publishing goal poses to ROS 2."""

import math
from typing import Optional
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from std_msgs.msg import Bool
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster


class TaskStatePublisherNode(Node):
    """ROS 2 node that broadcasts task-specific states such as navigation goal poses, static transforms, and completion status."""

    def __init__(self, node_name: str = "task_state_publisher", goal_topic: str = "/goal_pose", qos_depth: int = 1):
        """Initialize the task state publisher node.

        Args:
            node_name: ROS 2 node name.
            goal_topic: Topic name for goal pose publishing.
            qos_depth: Subscription queue depth.
        """
        super().__init__(node_name)
        # Enforce simulation time
        self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])

        # TRANSIENT_LOCAL ensures the goal pose and completion status persist for late subscribers
        latched_qos = QoSProfile(
            depth=qos_depth,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        self.goal_pub = self.create_publisher(PoseStamped, goal_topic, latched_qos)
        self.goal_reached_pub = self.create_publisher(Bool, "/goal_reached", latched_qos)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)

    def broadcast_static_tf(
        self,
        parent_frame: str,
        child_frame: str,
        translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    ):
        """Broadcast a static transform between two coordinate frames."""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = parent_frame
        t.child_frame_id = child_frame
        t.transform.translation.x = float(translation[0])
        t.transform.translation.y = float(translation[1])
        t.transform.translation.z = float(translation[2])
        t.transform.rotation.x = float(rotation[0])
        t.transform.rotation.y = float(rotation[1])
        t.transform.rotation.z = float(rotation[2])
        t.transform.rotation.w = float(rotation[3])
        self.static_tf_broadcaster.sendTransform(t)

    def publish_goal_reached(self, reached: bool = True) -> Bool:
        """Publish goal reached status.

        Args:
            reached: True if goal reached successfully, False otherwise.

        Returns:
            The published Bool message.
        """
        msg = Bool()
        msg.data = bool(reached)
        self.goal_reached_pub.publish(msg)
        return msg

    def publish_goal_pose(
        self,
        x: float,
        y: float,
        z: float = 0.0,
        heading: float = 0.0,
        quat: Optional[tuple[float, float, float, float]] = None,
        frame_id: str = "map",
    ) -> PoseStamped:
        """Publish a goal pose as a PoseStamped message.

        Args:
            x: X position in meters.
            y: Y position in meters.
            z: Z position in meters. Defaults to 0.0.
            heading: Heading/yaw angle in radians (used if quat is None).
            quat: Optional (x, y, z, w) quaternion orientation.
            frame_id: Target coordinate frame. Defaults to "odom".

        Returns:
            The published PoseStamped message.
        """
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id

        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = float(z)

        if quat is not None:
            msg.pose.orientation.x = float(quat[0])
            msg.pose.orientation.y = float(quat[1])
            msg.pose.orientation.z = float(quat[2])
            msg.pose.orientation.w = float(quat[3])
        else:
            half_yaw = heading * 0.5
            msg.pose.orientation.x = 0.0
            msg.pose.orientation.y = 0.0
            msg.pose.orientation.z = math.sin(half_yaw)
            msg.pose.orientation.w = math.cos(half_yaw)

        self.goal_pub.publish(msg)
        return msg
