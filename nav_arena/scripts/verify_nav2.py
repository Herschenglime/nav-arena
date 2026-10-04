# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end verification script for Nav2 bringup and autonomous navigation in kujiale_0003."""

from __future__ import annotations

import argparse
import math
import os
import signal
import subprocess
import sys
import time

from isaaclab.app import AppLauncher

# 1. Parse arguments
parser = argparse.ArgumentParser(description="Verify Nav2 autonomous navigation in nav_arena.")
parser.add_argument("--scene", type=str, default="kujiale_0003", help="Scene ID or USD path.")
parser.add_argument("--max-steps", type=int, default=2500, help="Max simulation steps for navigation.")
parser.add_argument("--rviz", action="store_true", help="Launch RViz with Nav2.")
parser.add_argument("--spawn-x", type=float, default=-6.42, help="Robot spawn X coordinate (m). Default: -6.42 (West bedroom)")
parser.add_argument("--spawn-y", type=float, default=0.64, help="Robot spawn Y coordinate (m). Default: 0.64 (West bedroom)")
parser.add_argument("--spawn-yaw", type=float, default=0.0, help="Robot spawn heading (rad). Default: 0.0")
parser.add_argument("--goal-x", type=float, default=5.70, help="Navigation goal X coordinate (m). Default: 5.70 (East bedroom)")
parser.add_argument("--goal-y", type=float, default=-1.52, help="Navigation goal Y coordinate (m). Default: -1.52 (East bedroom)")
parser.add_argument("--goal-yaw", type=float, default=0.0, help="Navigation goal heading (rad). Default: 0.0")

from nav_arena.core import launch_simulation_app
from nav_arena.utils import add_logger_args, configure_logging, get_logger, managed_process

logger = get_logger("verify_nav2")

add_logger_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

configure_logging(args_cli.log_level)

from rclpy.executors import SingleThreadedExecutor
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from lifecycle_msgs.msg import State
from lifecycle_msgs.srv import GetState
from nav_msgs.msg import OccupancyGrid, Path
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rosgraph_msgs.msg import Clock

def run_nav2_verification(simulation_app):
    import omni.usd
    from nav_arena.ros2 import (
        BackgroundRos2Executor,
        LaserScanPublisherNode,
        TaskStatePublisherNode,
        build_ros2_omnigraph,
    )
    from nav_arena.ros2.adapters.action_adapter import TwistActionAdapter
    from nav_arena.tasks import PointNavTask, create_point_nav_env_cfg

    logger.section(f"VERIFYING NAV2 AUTONOMOUS NAVIGATION IN SIMULATION: Scene={args_cli.scene}")

    logger.info(f"Spawn: ({args_cli.spawn_x:.2f}, {args_cli.spawn_y:.2f}) [yaw={args_cli.spawn_yaw:.2f} rad]")
    logger.info(f"Goal:  ({args_cli.goal_x:.2f}, {args_cli.goal_y:.2f}) [yaw={args_cli.goal_yaw:.2f} rad]")

    # 4. Configure PointNavTask environment in kujiale_0003
    spawn_pos = (args_cli.spawn_x, args_cli.spawn_y, 0.25)
    half_spawn_yaw = args_cli.spawn_yaw * 0.5
    spawn_rot = (0.0, 0.0, math.sin(half_spawn_yaw), math.cos(half_spawn_yaw))
    goal_pos = (args_cli.goal_x, args_cli.goal_y)
    goal_heading = args_cli.goal_yaw
    goal_threshold = 0.40

    env_cfg = create_point_nav_env_cfg(
        scene_id_or_path=args_cli.scene,
        robot_spawn_pos=spawn_pos,
        robot_spawn_rot=spawn_rot,
        goal_pos=goal_pos,
        goal_heading=goal_heading,
        goal_threshold=goal_threshold,
        episode_length_s=120.0,
        num_envs=1,
    )

    logger.info("Instantiating PointNavTask environment...")
    env = PointNavTask(cfg=env_cfg)

    # Position viewport camera to frame the robot and path
    env.sim.set_camera_view(eye=[-1.75, -2.8, 2.5], target=[-1.75, 0.0, 0.3])

    # 5. Build OmniGraph ROS 2 bridge for Clock, TF, and Odometry
    logger.info("Constructing OmniGraph ROS 2 Bridge...")
    robot_prim_path = "/World/envs/env_0/Robot/chassis_link"
    build_ros2_omnigraph(
        robot_prim_path=robot_prim_path,
        graph_path="/ActionGraph/ROS_Bridge",
        odom_topic="odom",
        odom_frame="odom",
        base_frame="base_link",
    )

    # 6. Environment Warmup Step
    logger.info("Performing environment warmup reset...")
    env.reset()

    # 7. Initialize ROS 2 nodes
    logger.info("Initializing ROS 2 interfaces...")
    rclpy.init()
    action_adapter = TwistActionAdapter(num_envs=env.num_envs, device=env.device)
    state_publisher = TaskStatePublisherNode()
    scan_publisher = LaserScanPublisherNode(topic_name="/scan", frame_id="lidar_link")

    # Track received Nav2 diagnostic topics
    received_plans: list[Path] = []
    received_cmd_vels: list[Twist] = []
    received_costmaps: list[OccupancyGrid] = []

    test_node = rclpy.create_node("nav2_monitor")
    test_node.create_subscription(Path, "/plan", lambda m: received_plans.append(m), 10)
    test_node.create_subscription(Twist, "/cmd_vel", lambda m: received_cmd_vels.append(m), 10)
    test_node.create_subscription(OccupancyGrid, "/global_costmap/costmap", lambda m: received_costmaps.append(m), 10)
    initial_pose_pub = test_node.create_publisher(PoseWithCovarianceStamped, "/initialpose", 10)
    # NVIDIA standard lifecycle supervision: query bt_navigator/get_state
    lifecycle_client = test_node.create_client(GetState, "/bt_navigator/get_state")

    executor = BackgroundRos2Executor(nodes=[action_adapter, state_publisher, scan_publisher, test_node])
    executor.start()

    # Note: map -> odom transform is dynamically broadcast by AMCL in Nav2

    # 8. Launch Nav2 stack via ros2 launch
    script_dir = os.path.dirname(os.path.abspath(__file__))
    launch_script = os.path.abspath(os.path.join(script_dir, "..", "methods", "nav2", "launch", "nav2.launch.py"))
    nav2_cmd = [
        sys.executable,
        "/opt/ros/jazzy/bin/ros2",
        "launch",
        launch_script,
        "use_sim_time:=true",
        f"rviz:={'true' if args_cli.rviz else 'false'}",
        f"initial_pose_x:={args_cli.spawn_x}",
        f"initial_pose_y:={args_cli.spawn_y}",
        f"initial_pose_yaw:={args_cli.spawn_yaw}",
    ]
    logger.info(f"Starting Nav2 process: {' '.join(nav2_cmd)}")
    log_file_path = "/home/robopi/simulation/nav2_bringup.log"
    log_file = open(log_file_path, "w")

    try:
        with managed_process(
            nav2_cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        ) as nav2_proc:
            # 9. Warmup phase: step simulation to pump /clock and /scan until Nav2 lifecycle manager transitions all nodes to ACTIVE
            logger.info("Stepping simulation to pump /clock and /scan for Nav2 lifecycle startup...")

            warmup_start = time.time()
            nav2_ready = False
            active_req_future = None

            lidar = env.scene["lidar"]

            step = 0
            max_warmup_steps = 1500  # Up to 30s sim time
            while step < max_warmup_steps and time.time() - warmup_start < 120.0 and nav2_proc.poll() is None:
                # Advance simulation physics
                actions = action_adapter.get_action()
                env.step(actions)

                # Publish 2D LiDAR LaserScan from RayCaster
                scan_publisher.publish_from_raycaster(lidar.data.ray_hits_w, lidar.data.pos_w)

                # Advance Omniverse Kit / OmniGraph execution
                simulation_app.update()

                step += 1

                # Check Nav2 lifecycle state via bt_navigator/get_state (NVIDIA standard pattern)
                if active_req_future is None:
                    if lifecycle_client.service_is_ready():
                        active_req_future = lifecycle_client.call_async(GetState.Request())
                elif active_req_future.done():
                    try:
                        res = active_req_future.result()
                        if res is not None and res.current_state.id == State.PRIMARY_STATE_ACTIVE:
                            logger.info(f"Nav2 bt_navigator reports ACTIVE (after {time.time() - warmup_start:.1f}s, step {step})!")
                            nav2_ready = True
                            break
                    except Exception:
                        pass
                    active_req_future = None

                if step % 50 == 0:
                    logger.info(f"Warmup step {step} ({time.time() - warmup_start:.1f}s elapsed), waiting for Nav2 active...")


            if not nav2_ready:
                logger.warning("Nav2 lifecycle manager did not report active status during warmup window. Checking process status...")
                if nav2_proc.poll() is not None:
                    log_file.flush()
                    with open(log_file_path, "r") as f:
                        logger.error(f"Nav2 process died prematurely:\n{f.read()[-2000:]}")
                    assert False, "Nav2 process crashed during startup."
                else:
                    assert False, f"Nav2 failed to report active status within {max_warmup_steps} simulation steps."

            # Align AMCL localization with simulation spawn pose
            init_msg = PoseWithCovarianceStamped()
            init_msg.header.frame_id = "map"
            init_msg.header.stamp = test_node.get_clock().now().to_msg()
            init_msg.pose.pose.position.x = float(args_cli.spawn_x)
            init_msg.pose.pose.position.y = float(args_cli.spawn_y)
            init_msg.pose.pose.position.z = 0.0
            init_msg.pose.pose.orientation.z = math.sin(half_spawn_yaw)
            init_msg.pose.pose.orientation.w = math.cos(half_spawn_yaw)
            init_msg.pose.covariance[0] = 0.05
            init_msg.pose.covariance[7] = 0.05
            init_msg.pose.covariance[35] = 0.05
            initial_pose_pub.publish(init_msg)
            logger.info(f"Published initial pose to AMCL: ({args_cli.spawn_x:.2f}, {args_cli.spawn_y:.2f}) [yaw={args_cli.spawn_yaw:.2f} rad]")

            # Step simulation to allow AMCL and costmaps to integrate scans before dispatching goal
            for _ in range(30):
                actions = action_adapter.get_action()
                env.step(actions)
                scan_publisher.publish_from_raycaster(lidar.data.ray_hits_w, lidar.data.pos_w)
                simulation_app.update()

            logger.info("Nav2 is active! Dispatching goal to NavigateToPose action server...")
            goal_x, goal_y, goal_heading = env.get_goal_pose_w(0)

            nav_action_client = ActionClient(test_node, NavigateToPose, "navigate_to_pose")
            logger.info("Waiting for NavigateToPose action server...")
            server_ready = False
            wait_start = time.time()
            while time.time() - wait_start < 20.0 and nav2_proc.poll() is None:
                if nav_action_client.wait_for_server(timeout_sec=0.1):
                    server_ready = True
                    break
                simulation_app.update()

            assert server_ready, "NavigateToPose action server timed out!"

            goal_msg = NavigateToPose.Goal()
            goal_msg.pose.header.frame_id = "map"
            goal_msg.pose.header.stamp = test_node.get_clock().now().to_msg()
            goal_msg.pose.pose.position.x = float(goal_x)
            goal_msg.pose.pose.position.y = float(goal_y)
            goal_msg.pose.pose.position.z = 0.0
            half_yaw = goal_heading * 0.5
            goal_msg.pose.pose.orientation.z = math.sin(half_yaw)
            goal_msg.pose.pose.orientation.w = math.cos(half_yaw)

            goal_future = nav_action_client.send_goal_async(goal_msg)
            while not goal_future.done() and nav2_proc.poll() is None:
                simulation_app.update()
                time.sleep(0.01)

            goal_handle = goal_future.result()
            assert goal_handle.accepted, "Nav2 bt_navigator rejected the navigation goal!"
            logger.info(f"Goal successfully accepted by Nav2 bt_navigator: ({goal_x:.2f}, {goal_y:.2f}) [frame=map]")
            state_publisher.publish_goal_pose(x=goal_x, y=goal_y, heading=goal_heading, frame_id="map")

            # 10. Autonomous Navigation Loop
            logger.info("Entering closed-loop navigation loop...")
            nav_start_time = time.time()
            start_x, start_y, _ = env.get_robot_pose_w(0)
            max_traveled = 0.0
            goal_reached = False
            last_log = time.time()

            for step in range(args_cli.max_steps):
                if nav2_proc.poll() is not None:
                    log_file.flush()
                    with open(log_file_path, "r") as f:
                        logger.error(f"Nav2 process died:\n{f.read()[-2000:]}")
                    assert False, "Nav2 process crashed during navigation."

                # Apply commanded actions to robot articulation (updated concurrently by background ROS 2 thread)
                actions = action_adapter.get_action()
                obs, rewards, dones, timeouts, infos = env.step(actions)

                # Publish updated 2D LiDAR raycaster hits
                scan_publisher.publish_from_raycaster(lidar.data.ray_hits_w, lidar.data.pos_w)

                # Update Omniverse App
                simulation_app.update()

                # Track navigation progress
                curr_x, curr_y, _ = env.get_robot_pose_w(0)
                traveled = math.hypot(curr_x - start_x, curr_y - start_y)
                max_traveled = max(max_traveled, traveled)

                dist_to_goal = math.hypot(curr_x - goal_x, curr_y - goal_y)

                if time.time() - last_log > 1.0:
                    last_log = time.time()
                    latest_cmd = received_cmd_vels[-1] if len(received_cmd_vels) > 0 else Twist()
                    act = actions[0].cpu().tolist()
                    robot_joint_vel = env.scene["robot"].data.joint_vel[0].cpu().tolist()
                    logger.info(
                        f"Step {step}: Traveled={traveled:.3f}m | DistToGoal={dist_to_goal:.3f}m | "
                        f"AdapterAct=[v={act[0]:.2f}, w={act[1]:.2f}] | "
                        f"CmdVel=[v={latest_cmd.linear.x:.2f}, w={latest_cmd.angular.z:.2f}] | "
                        f"WheelVel={[round(v, 2) for v in robot_joint_vel[:2]]} | "
                        f"Plans={len(received_plans)}"
                    )

                # Check goal reached
                if dist_to_goal < goal_threshold or env.is_goal_reached(0):
                    logger.success(f"NAV2 AUTONOMOUS GOAL REACHED! (Final dist to goal: {dist_to_goal:.3f}m)")
                    goal_reached = True
                    state_publisher.publish_goal_reached(True)
                    break

                if env.is_collision(0):
                    logger.warning(f"Collision detected during navigation at step {step}!")

            # 11. Assertions
            logger.section("RUNNING GOAL 3 VERIFICATION ASSERTIONS")

            assert len(received_costmaps) > 0, "No global costmap received from Nav2 map_server!"
            logger.check("Global costmap received", len(received_costmaps) > 0, f"updates={len(received_costmaps)}")

            assert len(received_plans) > 0, "Nav2 planner failed to generate any global path plans!"
            logger.check("Nav2 path plans received", len(received_plans) > 0, f"plans={len(received_plans)}")

            assert len(received_cmd_vels) > 0, "Nav2 controller failed to issue any /cmd_vel commands!"
            logger.check("Velocity commands received", len(received_cmd_vels) > 0, f"cmd_vels={len(received_cmd_vels)}")

            assert max_traveled > 0.8, f"Robot traveled insufficient distance: {max_traveled:.3f}m"
            logger.check("Maximum robot displacement (> 0.8m)", max_traveled > 0.8, f"max_traveled={max_traveled:.3f} m")

            assert goal_reached, f"Robot failed to reach goal within {args_cli.max_steps} steps!"
            logger.check("Autonomous navigation goal reached", goal_reached, f"final_dist={dist_to_goal:.3f} m")

            logger.section("GOAL 3 NAV2 BRINGUP & AUTONOMOUS NAVIGATION FULLY VERIFIED!")

    finally:
        logger.info("Cleaning up ROS 2 nodes...")
        if executor is not None:
            executor.shutdown()
        test_node.destroy_node()
        state_publisher.destroy_node()
        scan_publisher.destroy_node()
        action_adapter.destroy_node()
        rclpy.shutdown()
        log_file.close()



def main():
    try:
        with launch_simulation_app(args_cli, enable_ros2=True, livestream=True) as simulation_app:
            run_nav2_verification(simulation_app)
    except Exception as e:
        logger.error(f"Exception occurred during Nav2 verification: {e}", exc_info=True)
        sys.exit(1)



if __name__ == "__main__":
    main()
