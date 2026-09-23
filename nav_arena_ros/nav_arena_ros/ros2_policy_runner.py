# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ROS 2 Lifecycle Policy Runner for PointNavTask."""

from __future__ import annotations

import argparse
import sys
from abc import ABC, abstractmethod

import torch

# 1. Parse known CLI arguments before modifying sys.argv
parser = argparse.ArgumentParser(description="ROS 2 Lifecycle Policy Runner for PointNavTask.")
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args_cli, unknown_args = parser.parse_known_args()

# Save original sys.argv for clean ROS 2 / rclpy parsing
orig_sys_argv = list(sys.argv)

# 2. Insert Kit flags BEFORE any ROS 2 arguments (--ros-args) so rclpy does not reject them
kit_flags = [
    "--enable", "omni.graph",
    "--enable", "omni.graph.action",
    "--enable", "isaacsim.ros2.bridge",
    "--enable", "isaacsim.ros2.nodes",
    "--/exts/omni.kit.livestream.app/primaryStream/allowDynamicResize=true",
    "--/app/livestream/allowResize=true",
]
if "--ros-args" in sys.argv:
    ros_idx = sys.argv.index("--ros-args")
    sys.argv[ros_idx:ros_idx] = kit_flags
else:
    sys.argv.extend(kit_flags)

# 3. Launch Omniverse application
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 4. Restore original sys.argv so rclpy sees only valid ROS 2 arguments
sys.argv = orig_sys_argv

"""Simulation app is now active - Omniverse / Isaac Lab imports are safe."""

import carb.settings
_carb_settings = carb.settings.get_settings()
_carb_settings.set("/app/livestream/allowResize", True)
_carb_settings.set("/exts/omni.kit.livestream.app/primaryStream/allowDynamicResize", True)

import math
import rclpy
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
import lifecycle_msgs.msg
from geometry_msgs.msg import Twist, PoseStamped

from nav_arena.tasks import PointNavTask, PointNavEnvCfg
from nav_arena.embodiments.ros2_bridge import setup_ros2_clock, setup_ros2_odometry


class ActionAdapter(ABC):
    """Abstract base class for mapping ROS 2 commands to RL action tensors."""

    @abstractmethod
    def get_action(self) -> torch.Tensor:
        """Return the current action tensor with shape (num_envs, action_dim)."""
        pass

    @abstractmethod
    def destroy(self):
        """Cleanup subscribers and buffers."""
        pass


class TwistActionAdapter(ActionAdapter):
    """Action adapter for differential-drive mobile bases commanded via geometry_msgs/Twist."""

    def __init__(self, node: LifecycleNode, num_envs: int, device: str):
        self.node = node
        self.num_envs = num_envs
        self.device = device
        self.action = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        self.subscriber = self.node.create_subscription(
            Twist,
            '/cmd_vel',
            self._twist_callback,
            10
        )

    def _twist_callback(self, msg: Twist):
        # Maps [linear_x, angular_z] to differential drive action tensor
        self.action[0, 0] = msg.linear.x
        self.action[0, 1] = msg.angular.z

    def get_action(self) -> torch.Tensor:
        return self.action

    def destroy(self):
        if self.subscriber is not None:
            self.node.destroy_subscription(self.subscriber)
            self.subscriber = None


class Ros2PolicyRunnerNode(LifecycleNode):
    """ROS 2 Lifecycle Node encapsulating the Isaac Lab PointNavTask simulation loop."""

    def __init__(self):
        super().__init__('ros2_policy_runner')
        self.env: ManagerBasedRLEnv | None = None
        self.adapter: ActionAdapter | None = None
        self.goal_pub = None
        self.is_active: bool = False

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info("Configuring PointNavTask and ROS 2 OmniGraph bridges...")
        try:
            # 1. Setup OmniGraph clock and odometry bridges before environment creation
            setup_ros2_clock()
            setup_ros2_odometry(
                articulation_root="/World/envs/env_0/Robot",
                chassis_prim="/World/envs/env_0/Robot/chassis_link",
            )

            # 2. Instantiate PointNav RL Environment
            env_cfg = PointNavEnvCfg()
            env_cfg.scene.num_envs = 1
            self.env = PointNavTask(cfg=env_cfg)

            # 3. Setup Action Adapter (configurable for future legged robots)
            self.adapter = TwistActionAdapter(self, self.env.num_envs, self.env.device)

            # 4. Managed Lifecycle publisher for active goal pose
            self.goal_pub = self.create_lifecycle_publisher(PoseStamped, '/goal_pose', 10)

            self.get_logger().info("PointNavTask successfully configured.")
            return TransitionCallbackReturn.SUCCESS
        except Exception as e:
            self.get_logger().error(f"Failed to configure PointNavTask: {e}")
            return TransitionCallbackReturn.FAILURE

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info("Activating PointNavTask simulation loop...")
        try:
            super().on_activate(state)
            if self.env is not None:
                obs, info = self.env.reset()
                self.publish_goal()
            self.is_active = True
            self.get_logger().info("Simulation loop is now ACTIVE.")
            return TransitionCallbackReturn.SUCCESS
        except Exception as e:
            self.get_logger().error(f"Failed to activate PointNavTask: {e}")
            return TransitionCallbackReturn.FAILURE

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info("Deactivating simulation loop...")
        self.is_active = False
        return super().on_deactivate(state)

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info("Cleaning up resources...")
        self.is_active = False
        if self.adapter is not None:
            self.adapter.destroy()
            self.adapter = None
        if self.env is not None:
            self.env.close()
            self.env = None
        if self.goal_pub is not None:
            self.destroy_lifecycle_publisher(self.goal_pub)
            self.goal_pub = None
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"Shutting down from {state.label}...")
        self.on_cleanup(state)
        return TransitionCallbackReturn.SUCCESS

    def publish_goal(self):
        if self.goal_pub is None or self.env is None:
            return
        goal_x, goal_y, goal_h = self.env.get_goal_pose_w()

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "World"
        msg.pose.position.x = goal_x
        msg.pose.position.y = goal_y
        msg.pose.position.z = 0.0
        msg.pose.orientation.z = math.sin(goal_h / 2.0)
        msg.pose.orientation.w = math.cos(goal_h / 2.0)
        self.goal_pub.publish(msg)
        self.get_logger().info(f"Published active goal: ({goal_x:.2f}, {goal_y:.2f})")


def main(args=None):
    rclpy.init(args=args)
    node = Ros2PolicyRunnerNode()

    try:
        while simulation_app.is_running():
            # TODO(scaling): Refactor ROS 2 spin to background thread for multi-agent/high-throughput scaling
            rclpy.spin_once(node, timeout_sec=0.0)

            if node.is_active and node.env is not None and node.adapter is not None:
                action = node.adapter.get_action()
                obs, reward, terminated, truncated, info = node.env.step(action)
                dones = terminated | truncated
                if dones[0]:
                    node.publish_goal()
            else:
                # Keep kit rendering and compiling shaders while unconfigured or inactive
                simulation_app.update()

    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt received, stopping.")
    except Exception as e:
        node.get_logger().error(f"Error in simulation loop: {e}", exc_info=True)
    finally:
        if node.is_active:
            node.on_deactivate(LifecycleState(lifecycle_msgs.msg.State.PRIMARY_STATE_ACTIVE, 'active'))
        node.on_cleanup(LifecycleState(lifecycle_msgs.msg.State.PRIMARY_STATE_INACTIVE, 'inactive'))
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        simulation_app.close()


if __name__ == "__main__":
    main()
