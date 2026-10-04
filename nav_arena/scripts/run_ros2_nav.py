# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Main execution orchestrator for PointNav simulation with ROS 2 bridge."""

from __future__ import annotations

import argparse
import math
import sys
import time
import rclpy

from isaaclab.app import AppLauncher

from nav_arena.core import launch_simulation_app
from nav_arena.utils import add_logger_args, configure_logging, get_logger

logger = get_logger("run_ros2_nav")

# Parse arguments
parser = argparse.ArgumentParser(description="Run nav_arena simulation with ROS 2 bridge.")
parser.add_argument("--scene", type=str, default="kujiale_0003", help="InteriorAgent scene ID or USD path.")
parser.add_argument("--num-steps", type=int, default=-1, help="Max simulation steps (-1 for infinite).")
add_logger_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

configure_logging(args_cli.log_level)


def main():
    with launch_simulation_app(args_cli, enable_ros2=True, livestream=True) as simulation_app:
        from nav_arena.tasks import PointNavTask, create_point_nav_env_cfg
        from nav_arena.ros2 import BackgroundRos2Executor
        from nav_arena.ros2.adapters.action_adapter import TwistActionAdapter
        from nav_arena.ros2.state_publisher import TaskStatePublisherNode
        from nav_arena.ros2.graph_builder import build_ros2_omnigraph

        logger.section(f"nav_arena ROS 2 Simulation Bridge: Scene={args_cli.scene}")


        # 3. Environment configuration
        env = None
        action_adapter = None
        state_publisher = None
        executor = None

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

        logger.info("Instantiating PointNavTask environment...")
        env = PointNavTask(cfg=env_cfg)

        # Frame viewport camera on robot and goal path
        env.sim.set_camera_view(eye=[-1.75, -2.8, 2.2], target=[-1.75, 0.0, 0.3])

        # 4. Build OmniGraph ROS 2 bridge for Clock, TF, and Odometry
        logger.info("Constructing OmniGraph ROS 2 Bridge...")
        robot_prim_path = "/World/envs/env_0/Robot/chassis_link"
        build_ros2_omnigraph(
            robot_prim_path=robot_prim_path,
            graph_path="/ActionGraph/ROS_Bridge",
            odom_topic="odom",
            odom_frame="odom",
            base_frame="chassis_link",
        )

        # 5. Environment Warmup Step (compiles shaders, initializes physics)
        logger.info("Performing environment warmup reset...")
        env.reset()

        # 6. Initialize ROS 2 interfaces
        logger.info("Initializing ROS 2 nodes...")
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
        logger.info(f"Static TF published: map -> odom at ({init_x:.2f}, {init_y:.2f}, 0.00, yaw={init_heading:.2f})")

        # Get goal coordinates and publish initial goal in 'map' frame
        goal_x, goal_y, goal_heading = env.get_goal_pose_w(0)
        state_publisher.publish_goal_pose(x=goal_x, y=goal_y, heading=goal_heading, frame_id="map")
        logger.info(f"Goal published to /goal_pose (frame=map): ({goal_x:.2f}, {goal_y:.2f}, heading={goal_heading:.2f})")

        # Spin ROS 2 nodes on a dedicated background thread to prevent callback starvation under render load
        executor = BackgroundRos2Executor(nodes=[action_adapter, state_publisher])
        executor.start()
        logger.info("BackgroundRos2Executor running on dedicated thread.")
        logger.info("Bridge ready. Listening on /cmd_vel and publishing /clock, /tf, /odom, /goal_pose, /goal_reached.")

        # 7. Main execution loop
        step_count = 0
        start_wall_time = time.time()
        try:
            while simulation_app.is_running():
                # Ingest action and advance simulation
                actions = action_adapter.get_action()
                obs, rewards, dones, timeouts, infos = env.step(actions)

                # In headless mode, step OmniGraph only when not rendered by env.step()
                if args_cli.headless and (step_count % env_cfg.sim.render_interval == 0):
                    simulation_app.update()

                if dones.any():
                    is_goal = env.is_goal_reached(0)
                    is_collision = env.is_collision(0)
                    is_timeout = env.is_timed_out(0)
                    logger.info(
                        f"Episode terminated at step {step_count}: "
                        f"goal_reached={is_goal}, collision={is_collision}, timeout={is_timeout}"
                    )
                    if is_goal:
                        state_publisher.publish_goal_reached(True)
                        logger.info("Published True to /goal_reached (TRANSIENT_LOCAL).")

                step_count += 1
                if args_cli.num_steps > 0 and step_count >= args_cli.num_steps:
                    elapsed_wall = time.time() - start_wall_time
                    sim_time = step_count * env.step_dt
                    rtf = sim_time / elapsed_wall if elapsed_wall > 0 else 0.0
                    fps = step_count / elapsed_wall if elapsed_wall > 0 else 0.0
                    logger.info(f"Completed requested {args_cli.num_steps} simulation steps.")
                    logger.info(f"Performance: {step_count} steps in {elapsed_wall:.2f}s | Rate: {fps:.1f} steps/s | RTF: {rtf:.2f}x")
                    break

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received. Shutting down gracefully...")
        except Exception as e:
            logger.error(f"Exception in simulation loop: {e}", exc_info=True)

        finally:
            logger.info("Cleaning up ROS 2 and simulation resources...")
            if executor is not None:
                executor.shutdown()
            if action_adapter is not None:
                action_adapter.destroy_node()
            if state_publisher is not None:
                state_publisher.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
            if env is not None:
                env.close()
            logger.info("Simulation bridge terminated.")



if __name__ == "__main__":
    main()
