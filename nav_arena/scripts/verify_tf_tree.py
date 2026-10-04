# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Verification script for ROS 2 transform tree (TF, static TF, and robot_state_publisher)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

from isaaclab.app import AppLauncher

from nav_arena.core import launch_simulation_app
from nav_arena.utils import add_logger_args, configure_logging, get_logger, managed_process

logger = get_logger("verify_tf_tree")

# 1. Parse arguments
parser = argparse.ArgumentParser(description="Verify ROS 2 TF Tree and robot_state_publisher.")
parser.add_argument("--num-steps", type=int, default=60, help="Number of simulation steps to run.")
add_logger_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

configure_logging(args_cli.log_level)

def run_tf_verification(simulation_app):
    import carb.settings
    import rclpy
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile
    from rosgraph_msgs.msg import Clock
    from std_msgs.msg import String
    import tf2_ros

    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.utils.configclass import configclass

    from nav_arena.embodiments import (
        NOVA_CARTER_CFG,
        NovaCarterEmbodimentCfg,
    )
    from nav_arena.ros2 import BackgroundRos2Executor, build_ros2_omnigraph

    @configclass
    class TFVerifySceneCfg(InteractiveSceneCfg):
        """Minimal verification scene with ground plane, lighting, and Nova Carter."""

        ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
        dome_light = AssetBaseCfg(
            prim_path="/World/defaultDomeLight",
            spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8)),
        )
        robot = NOVA_CARTER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


    logger.section("VERIFYING ROS 2 TF TREE & ROBOT_STATE_PUBLISHER")

    embodiment_cfg = NovaCarterEmbodimentCfg()
    logger.info(
        f"Embodiment Config: base_frame='{embodiment_cfg.base_frame}', "
        f"chassis_frame='{embodiment_cfg.chassis_frame}', lidar_frame='{embodiment_cfg.lidar_frame}'"
    )
    logger.info(f"LiDAR Offset: {embodiment_cfg.lidar_offset} m")

    executor = None

    script_dir = os.path.dirname(os.path.abspath(__file__))
    launch_file_path = os.path.abspath(os.path.join(script_dir, "..", "methods", "nav2", "launch", "robot_description.launch.py"))
    launch_cmd = [
        sys.executable,
        "/opt/ros/jazzy/bin/ros2",
        "launch",
        launch_file_path,
        "use_sim_time:=true",
        f"base_frame:={embodiment_cfg.base_frame}",
        f"chassis_frame:={embodiment_cfg.chassis_frame}",
        f"lidar_frame:={embodiment_cfg.lidar_frame}",
        f"sensor_x:={embodiment_cfg.lidar_offset[0]}",
        f"sensor_y:={embodiment_cfg.lidar_offset[1]}",
        f"sensor_z:={embodiment_cfg.lidar_offset[2]}",
        f"robot_radius:={embodiment_cfg.robot_radius}",
        f"robot_height:={embodiment_cfg.robot_height}",
    ]
    logger.info(f"Starting robot_state_publisher process: {' '.join(launch_cmd)}")
    with managed_process(launch_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as rsp_proc:
        # 5. Initialize Isaac Lab simulation context and scene
        sim_cfg = sim_utils.SimulationCfg(dt=0.02)
        sim = sim_utils.SimulationContext(sim_cfg)
        scene_cfg = TFVerifySceneCfg(num_envs=1, env_spacing=2.0)
        scene = InteractiveScene(scene_cfg)

        # 6. Initialize ROS 2 OmniGraph Bridge (Clock, Odometry, and TF)
        logger.info("Constructing ROS 2 OmniGraph Bridge...")
        build_ros2_omnigraph(
            robot_prim_path="/World/envs/env_0/Robot/chassis_link",
            base_frame=embodiment_cfg.base_frame,
        )

        # Reset simulation to start physics
        sim.reset()


        # 7. Setup rclpy node and subscriptions
        rclpy.init()
        listener_node = rclpy.create_node("tf_tree_verifier")

        received_clocks: list[Clock] = []
        received_descriptions: list[str] = []

        listener_node.create_subscription(Clock, "/clock", lambda m: received_clocks.append(m), 10)

        latched_qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        listener_node.create_subscription(
            String,
            "/robot_description",
            lambda m: received_descriptions.append(m.data),
            latched_qos,
        )

        tf_buffer = tf2_ros.Buffer()
        tf_listener = tf2_ros.TransformListener(tf_buffer, listener_node)

        executor = BackgroundRos2Executor(nodes=[listener_node])
        executor.start()
        logger.info("BackgroundRos2Executor started for listener_node.")

        try:
            logger.info(f"Stepping simulation for {args_cli.num_steps} steps...")
            start_wall_time = time.time()
            for step in range(args_cli.num_steps):
                scene.write_data_to_sim()
                sim.step()
                scene.update(dt=sim.get_physics_dt())
                simulation_app.update()


            logger.info(f"Simulation stepped {args_cli.num_steps} iterations in {time.time() - start_wall_time:.2f}s.")

            # Allow tf buffer to settle
            time.sleep(0.5)

            # 8. Assertions
            logger.section("RUNNING TF TREE ASSERTIONS")

            # Assertion 1: Clock publishing
            assert len(received_clocks) > 0, "No /clock messages received from OmniGraph ROS2PublishClock!"
            latest_clock = received_clocks[-1].clock.sec + received_clocks[-1].clock.nanosec * 1e-9
            logger.check(
                "Clock messages received",
                len(received_clocks) > 0,
                f"count={len(received_clocks)}, latest sim time={latest_clock:.3f} s",
            )

            # Assertion 2: /robot_description published
            assert len(received_descriptions) > 0, "No /robot_description message received from robot_state_publisher!"
            urdf_text = received_descriptions[-1]
            assert "<robot name=\"nova_carter\"" in urdf_text, "robot_description does not contain expected robot name!"
            assert f"link name=\"{embodiment_cfg.lidar_frame}\"" in urdf_text, f"Missing link '{embodiment_cfg.lidar_frame}' in URDF!"
            logger.check(
                "URDF contents verified",
                True,
                f"messages={len(received_descriptions)}",
            )

            # Assertion 3: Static TF base_link -> chassis_link
            can_transform_base_chassis = tf_buffer.can_transform(
                embodiment_cfg.base_frame,
                embodiment_cfg.chassis_frame,
                rclpy.time.Time(),
            )
            assert can_transform_base_chassis, f"Cannot transform from {embodiment_cfg.base_frame} to {embodiment_cfg.chassis_frame}!"
            t_base_chassis = tf_buffer.lookup_transform(
                embodiment_cfg.base_frame,
                embodiment_cfg.chassis_frame,
                rclpy.time.Time(),
            )
            logger.check(
                f"Static TF '{embodiment_cfg.base_frame}' -> '{embodiment_cfg.chassis_frame}'",
                can_transform_base_chassis,
                f"translation=({t_base_chassis.transform.translation.x:.3f}, {t_base_chassis.transform.translation.y:.3f}, {t_base_chassis.transform.translation.z:.3f})",
            )

            # Assertion 4: Static TF chassis_link -> lidar_link
            can_transform_chassis_lidar = tf_buffer.can_transform(
                embodiment_cfg.chassis_frame,
                embodiment_cfg.lidar_frame,
                rclpy.time.Time(),
            )
            assert can_transform_chassis_lidar, f"Cannot transform from {embodiment_cfg.chassis_frame} to {embodiment_cfg.lidar_frame}!"
            t_chassis_lidar = tf_buffer.lookup_transform(
                embodiment_cfg.chassis_frame,
                embodiment_cfg.lidar_frame,
                rclpy.time.Time(),
            )
            lidar_z = t_chassis_lidar.transform.translation.z
            assert abs(lidar_z - embodiment_cfg.lidar_offset[2]) < 1e-4, f"LiDAR Z offset mismatch: {lidar_z}"
            logger.check(
                f"Static TF '{embodiment_cfg.chassis_frame}' -> '{embodiment_cfg.lidar_frame}'",
                True,
                f"lidar_z={lidar_z:.3f} m (expected {embodiment_cfg.lidar_offset[2]:.3f} m)",
            )

            # Assertion 5: Dynamic TF odom -> base_link
            can_transform_odom_base = tf_buffer.can_transform(
                "odom",
                embodiment_cfg.base_frame,
                rclpy.time.Time(),
            )
            assert can_transform_odom_base, f"Cannot transform from 'odom' to '{embodiment_cfg.base_frame}'!"
            t_odom_base = tf_buffer.lookup_transform(
                "odom",
                embodiment_cfg.base_frame,
                rclpy.time.Time(),
            )
            logger.check(
                f"Dynamic TF 'odom' -> '{embodiment_cfg.base_frame}'",
                can_transform_odom_base,
                f"translation=({t_odom_base.transform.translation.x:.3f}, {t_odom_base.transform.translation.y:.3f}, {t_odom_base.transform.translation.z:.3f})",
            )

            # Assertion 6: Full chain lookup odom -> lidar_link
            can_transform_odom_lidar = tf_buffer.can_transform(
                "odom",
                embodiment_cfg.lidar_frame,
                rclpy.time.Time(),
            )
            assert can_transform_odom_lidar, f"Cannot transform from 'odom' to '{embodiment_cfg.lidar_frame}'!"
            t_odom_lidar = tf_buffer.lookup_transform(
                "odom",
                embodiment_cfg.lidar_frame,
                rclpy.time.Time(),
            )
            total_z = t_odom_lidar.transform.translation.z
            logger.check(
                f"Full chain lookup 'odom' -> '{embodiment_cfg.lidar_frame}'",
                can_transform_odom_lidar,
                f"total height={total_z:.3f} m",
            )

            logger.section("ALL TF TREE & ROBOT_STATE_PUBLISHER ASSERTIONS PASSED SUCCESSFULLY!")


        finally:
            # Clean teardown
            logger.info("Cleaning up processes and ROS nodes...")
            if executor is not None:
                executor.shutdown()
            listener_node.destroy_node()
            rclpy.shutdown()


def main():
    try:
        with launch_simulation_app(args_cli, enable_ros2=True, livestream=False) as simulation_app:
            run_tf_verification(simulation_app)
    except Exception as e:
        logger.error(f"Exception occurred during TF verification: {e}", exc_info=True)
        sys.exit(1)



if __name__ == "__main__":
    main()
