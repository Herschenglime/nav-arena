# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test for TaskStatePublisherNode."""

import math
import time
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_arena.ros2.state_publisher import TaskStatePublisherNode


def test_task_state_publisher():
    rclpy.init()
    try:
        publisher_node = TaskStatePublisherNode()

        # Create a test subscriber to verify received messages
        received_msgs: list[PoseStamped] = []
        subscriber_node = rclpy.create_node("mock_goal_subscriber")
        from rclpy.qos import QoSProfile, QoSDurabilityPolicy
        goal_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        subscriber_node.create_subscription(
            PoseStamped,
            "/goal_pose",
            lambda msg: received_msgs.append(msg),
            goal_qos,
        )

        # Publish a sample goal with heading 90 degrees (pi / 2)
        test_x = 3.5
        test_y = -1.2
        test_yaw = math.pi / 2.0
        published_msg = publisher_node.publish_goal_pose(
            x=test_x,
            y=test_y,
            heading=test_yaw,
            frame_id="odom",
        )

        assert published_msg.header.frame_id == "odom"
        assert published_msg.pose.position.x == test_x
        assert published_msg.pose.position.y == test_y

        # Spin to receive message
        start_time = time.time()
        while time.time() - start_time < 2.0 and len(received_msgs) == 0:
            rclpy.spin_once(publisher_node, timeout_sec=0.1)
            rclpy.spin_once(subscriber_node, timeout_sec=0.1)

        assert len(received_msgs) > 0, "No PoseStamped message received by subscriber."
        rx = received_msgs[0]
        assert rx.header.frame_id == "odom"
        assert abs(rx.pose.position.x - test_x) < 1e-4
        assert abs(rx.pose.position.y - test_y) < 1e-4
        expected_qz = math.sin(test_yaw / 2.0)
        expected_qw = math.cos(test_yaw / 2.0)
        assert abs(rx.pose.orientation.z - expected_qz) < 1e-4
        assert abs(rx.pose.orientation.w - expected_qw) < 1e-4

    finally:
        publisher_node.destroy_node()
        subscriber_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    test_task_state_publisher()
    print("test_task_state_publisher PASSED")
