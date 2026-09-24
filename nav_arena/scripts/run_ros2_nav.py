# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Main execution orchestrator for PointNav simulation with ROS 2 bridge."""

from __future__ import annotations

import argparse
import math
import sys
import rclpy

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Run nav_arena simulation with ROS 2 bridge.")
parser.add_argument("--scene", type=str, default="kujiale_0003", help="InteriorAgent scene ID or USD path.")
parser.add_argument("--num-steps", type=int, default=-1, help="Max simulation steps (-1 for infinite).")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

# 1. Mandatory flag injection before AppLauncher boots
sys.argv.extend([
    "--enable", "omni.graph",
    "--enable", "omni.graph.action",
    "--enable", "isaacsim.ros2.bridge",
    "--enable", "isaacsim.ros2.nodes",
])

# 2. Boot Isaac Lab application
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""App is now running - safe to import Omniverse / Isaac / ROS modules."""

import carb.settings

_carb_settings = carb.settings.get_settings()
_carb_settings.set("/app/livestream/allowResize", True)
_carb_settings.set("/exts/omni.kit.livestream.app/primaryStream/allowDynamicResize", True)

from nav_arena.tasks import PointNavTask, create_point_nav_env_cfg
from nav_arena.ros2.adapters.action_adapter import TwistActionAdapter
from nav_arena.ros2.state_publisher import TaskStatePublisherNode
from nav_arena.ros2.graph_builder import build_ros2_omnigraph


def main():
    print("=" * 70)
    print("nav_arena ROS 2 Simulation Bridge")
    print(f"Scene: {args_cli.scene}")
    print("=" * 70)

    # 3. Environment configuration
    spawn_pos = (-2.5, 0.0, 0.25)
    spawn_rot = (0.0, 0.0, 0.0, 1.0)
    goal_pos = (-1.0, 0.0)
    goal_heading = 0.0
    goal_threshold = 0.40

    env_cfg = create_point_nav_env_cfg(
        scene_id_or_path=args_cli.scene,
        robot_spawn_pos=spawn_pos,
        robot_spawn_rot=spawn_rot,
        goal_pos=goal_pos,
        goal_heading=goal_heading,
        goal_threshold=goal_threshold,
        episode_length_s=60.0,
        num_envs=1,
    )

    print("[INFO] Instantiating PointNavTask environment...")
    env = PointNavTask(cfg=env_cfg)

    # Frame viewport camera on robot and goal path
    env.sim.set_camera_view(eye=[-1.75, -2.8, 2.2], target=[-1.75, 0.0, 0.3])

    # 4. Build OmniGraph ROS 2 bridge for Clock, TF, and Odometry
    print("[INFO] Constructing OmniGraph ROS 2 Bridge...")
    robot_prim_path = "/World/envs/env_0/Robot/chassis_link"
    build_ros2_omnigraph(
        robot_prim_path=robot_prim_path,
        graph_path="/ActionGraph/ROS_Bridge",
        odom_topic="odom",
        odom_frame="odom",
        base_frame="chassis_link",
    )

    # 5. Environment Warmup Step (compiles shaders, initializes physics)
    print("[INFO] Performing environment warmup reset...")
    env.reset()

    # 6. Initialize ROS 2 interfaces
    print("[INFO] Initializing ROS 2 nodes...")
    rclpy.init()
    action_adapter = TwistActionAdapter(num_envs=env.num_envs, device=env.device)
    state_publisher = TaskStatePublisherNode()

    # Query initial robot pose directly from simulation environment (single source of truth)
    init_x, init_y, init_heading = env.get_robot_pose_w(0)
    half_yaw = init_heading * 0.5
    init_rot = (0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw))

    # Broadcast static transform linking canonical REP-105 'map' frame to 'odom' at robot spawn origin
    state_publisher.broadcast_static_tf(
        parent_frame="map",
        child_frame="odom",
        translation=(init_x, init_y, 0.0),
        rotation=init_rot,
    )
    print(f"[INFO] Static TF published: map -> odom at ({init_x:.2f}, {init_y:.2f}, 0.00, yaw={init_heading:.2f})")

    # Get goal coordinates and publish initial goal in 'map' frame
    goal_x, goal_y, goal_heading = env.get_goal_pose_w(0)
    state_publisher.publish_goal_pose(x=goal_x, y=goal_y, heading=goal_heading, frame_id="map")
    print(f"[INFO] Goal published to /goal_pose (frame=map): ({goal_x:.2f}, {goal_y:.2f}, heading={goal_heading:.2f})")
    print("[INFO] Bridge ready. Listening on /cmd_vel and publishing /clock, /tf, /odom, /goal_pose, /goal_reached.")

    # 7. Main execution loop
    step_count = 0
    try:
        while simulation_app.is_running():
            # Process incoming ROS 2 messages
            rclpy.spin_once(action_adapter, timeout_sec=0.0)
            rclpy.spin_once(state_publisher, timeout_sec=0.0)

            # Ingest action and advance simulation
            actions = action_adapter.get_action()
            obs, rewards, dones, timeouts, infos = env.step(actions)

            # Advance Omniverse Kit / OmniGraph execution
            simulation_app.update()

            if dones.any():
                is_goal = env.is_goal_reached(0)
                is_collision = env.is_collision(0)
                is_timeout = env.is_timed_out(0)
                print(
                    f"[INFO] Episode terminated at step {step_count}: "
                    f"goal_reached={is_goal}, collision={is_collision}, timeout={is_timeout}"
                )
                if is_goal:
                    state_publisher.publish_goal_reached(True)
                    print("[INFO] Published True to /goal_reached (TRANSIENT_LOCAL).")

            step_count += 1
            if args_cli.num_steps > 0 and step_count >= args_cli.num_steps:
                print(f"[INFO] Completed requested {args_cli.num_steps} simulation steps.")
                break

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt received. Shutting down gracefully...")

    finally:
        print("[INFO] Cleaning up ROS 2 and simulation resources...")
        action_adapter.destroy_node()
        state_publisher.destroy_node()
        rclpy.shutdown()
        env.close()
        simulation_app.close()
        print("[INFO] Simulation bridge terminated.")


if __name__ == "__main__":
    main()
