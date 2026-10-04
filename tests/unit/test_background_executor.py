# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for BackgroundRos2Executor."""

import time
import pytest
import rclpy
from std_msgs.msg import String

from nav_arena.ros2 import BackgroundRos2Executor


@pytest.fixture(scope="module")
def rclpy_context():
    """Manage lifecycle of ROS 2 context for unit test module."""
    if not rclpy.ok():
        rclpy.init()
    yield
    if rclpy.ok():
        rclpy.shutdown()


def test_executor_lifecycle_and_async_dispatch(rclpy_context):
    """Test background spinning and asynchronous message delivery."""
    rx_node = rclpy.create_node("test_rx_node")
    tx_node = rclpy.create_node("test_tx_node")

    received_msgs = []
    rx_node.create_subscription(String, "/test_executor_topic", lambda m: received_msgs.append(m.data), 10)
    pub = tx_node.create_publisher(String, "/test_executor_topic", 10)

    executor = BackgroundRos2Executor(nodes=[rx_node])
    assert not executor.is_spinning

    try:
        executor.start()
        assert executor.is_spinning

        msg = String()
        msg.data = "hello_background_executor"
        pub.publish(msg)

        # Wait for async callback to be processed by background thread
        start = time.time()
        while time.time() - start < 2.0 and not received_msgs:
            time.sleep(0.01)

        assert received_msgs == ["hello_background_executor"]
    finally:
        executor.shutdown(timeout_sec=2.0)
        assert not executor.is_spinning
        rx_node.destroy_node()
        tx_node.destroy_node()


def test_executor_add_remove_node(rclpy_context):
    """Test dynamic node addition and removal."""
    n1 = rclpy.create_node("test_node_add_1")
    n2 = rclpy.create_node("test_node_add_2")

    executor = BackgroundRos2Executor(nodes=[n1])
    assert executor.get_nodes() == [n1]

    executor.add_node(n2)
    assert executor.get_nodes() == [n1, n2]

    # Re-adding existing node should be a no-op
    executor.add_node(n1)
    assert executor.get_nodes() == [n1, n2]

    executor.remove_node(n1)
    assert executor.get_nodes() == [n2]

    executor.remove_node(n2)
    assert executor.get_nodes() == []

    executor.shutdown()
    n1.destroy_node()
    n2.destroy_node()


def test_executor_context_manager(rclpy_context):
    """Test context manager enters and exits cleanly."""
    node = rclpy.create_node("test_cm_node")
    with BackgroundRos2Executor(nodes=[node]) as executor:
        assert executor.is_spinning
    assert not executor.is_spinning
    node.destroy_node()


def test_executor_idempotent_start_shutdown(rclpy_context):
    """Test that multiple start and shutdown invocations are safe."""
    node = rclpy.create_node("test_idempotent_node")
    executor = BackgroundRos2Executor(nodes=[node])

    # Multiple starts
    executor.start()
    executor.start()
    assert executor.is_spinning

    # Multiple shutdowns
    executor.shutdown(timeout_sec=1.0)
    executor.shutdown(timeout_sec=1.0)
    assert not executor.is_spinning

    node.destroy_node()


def test_executor_restart_cycle(rclpy_context):
    """Test that BackgroundRos2Executor can be restarted after shutdown and process messages."""
    rx = rclpy.create_node("test_restart_rx")
    tx = rclpy.create_node("test_restart_tx")

    received = []
    rx.create_subscription(String, "/test_restart_topic", lambda m: received.append(m.data), 10)
    pub = tx.create_publisher(String, "/test_restart_topic", 10)

    executor = BackgroundRos2Executor(nodes=[rx])

    # First run
    executor.start()
    assert executor.is_spinning
    pub.publish(String(data="msg_1"))
    t0 = time.time()
    while time.time() - t0 < 1.0 and len(received) < 1:
        time.sleep(0.01)
    assert received == ["msg_1"]
    executor.shutdown(timeout_sec=1.0)
    assert not executor.is_spinning

    # Second run (restart)
    executor.start()
    assert executor.is_spinning
    pub.publish(String(data="msg_2"))
    t0 = time.time()
    while time.time() - t0 < 1.0 and len(received) < 2:
        time.sleep(0.01)
    assert received == ["msg_1", "msg_2"]
    executor.shutdown(timeout_sec=1.0)
    assert not executor.is_spinning

    rx.destroy_node()
    tx.destroy_node()

