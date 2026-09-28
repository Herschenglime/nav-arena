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

# 1. Parse arguments
parser = argparse.ArgumentParser(description="Verify ROS 2 TF Tree and robot_state_publisher.")
parser.add_argument("--num-steps", type=int, default=60, help="Number of simulation steps to run.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 2. Mandatory boot-time flag injection for OmniGraph and ROS 2 bridges
sys.argv.extend([
    "--enable", "omni.graph",
    "--enable", "omni.graph.action",
    "--enable", "isaacsim.ros2.bridge",
    "--enable", "isaacsim.ros2.nodes",
])

# 3. Launch Omniverse application
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Simulation app is now active - Omniverse / Isaac Lab / ROS 2 imports are safe."""

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
    setup_ros2_clock,
    setup_ros2_odometry,
)


@configclass
class TFVerifySceneCfg(InteractiveSceneCfg):
    """Minimal verification scene with ground plane, lighting, and Nova Carter."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/defaultDomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8)),
    )
    robot = NOVA_CARTER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_tf_verification():
    print("=" * 70)
    print("VERIFYING ROS 2 TF TREE & ROBOT_STATE_PUBLISHER")
    print("=" * 70)

    embodiment_cfg = NovaCarterEmbodimentCfg()
    print(f"[INFO] Embodiment Config: base_frame='{embodiment_cfg.base_frame}', "
          f"chassis_frame='{embodiment_cfg.chassis_frame}', lidar_frame='{embodiment_cfg.lidar_frame}'")
    print(f"[INFO] LiDAR Offset: {embodiment_cfg.lidar_offset} m")

    # 4. Launch robot_state_publisher via ROS 2 launch file
    launch_file_path = os.path.abspath("nav_arena/nav_arena/methods/nav2/launch/robot_description.launch.py")
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
    print(f"[INFO] Starting robot_state_publisher process: {' '.join(launch_cmd)}")
    rsp_proc = subprocess.Popen(launch_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # 5. Initialize Isaac Lab simulation context and scene
    sim_cfg = sim_utils.SimulationCfg(dt=0.02)
    sim = sim_utils.SimulationContext(sim_cfg)
    scene_cfg = TFVerifySceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    # 6. Initialize ROS 2 OmniGraphs (Clock and Odometry)
    print("[INFO] Constructing ROS 2 Clock and Odometry OmniGraphs...")
    setup_ros2_clock()
    setup_ros2_odometry(
        articulation_root="/World/envs/env_0/Robot/chassis_link",
        chassis_prim="/World/envs/env_0/Robot/chassis_link",
        chassis_frame_id=embodiment_cfg.base_frame,
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

    print(f"[INFO] Stepping simulation for {args_cli.num_steps} steps...")
    start_wall_time = time.time()
    for step in range(args_cli.num_steps):
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt=sim.get_physics_dt())
        simulation_app.update()

        # Spin rclpy to process incoming clock, tf, and tf_static messages
        rclpy.spin_once(listener_node, timeout_sec=0.01)

    print(f"[INFO] Simulation stepped {args_cli.num_steps} iterations in {time.time() - start_wall_time:.2f}s.")

    # Additional spin to allow tf buffer to settle
    for _ in range(20):
        rclpy.spin_once(listener_node, timeout_sec=0.05)

    try:
        # 8. Assertions
        print("\n" + "=" * 70)
        print("RUNNING TF TREE ASSERTIONS")
        print("=" * 70)

        # Assertion 1: Clock publishing
        print(f"[CHECK 1] Received {len(received_clocks)} /clock messages...")
        assert len(received_clocks) > 0, "No /clock messages received from OmniGraph ROS2PublishClock!"
        latest_clock = received_clocks[-1].clock.sec + received_clocks[-1].clock.nanosec * 1e-9
        print(f"          Latest sim time: {latest_clock:.3f} s -> [PASS]")

        # Assertion 2: /robot_description published
        print(f"[CHECK 2] Received {len(received_descriptions)} /robot_description messages...")
        assert len(received_descriptions) > 0, "No /robot_description message received from robot_state_publisher!"
        urdf_text = received_descriptions[-1]
        assert "<robot name=\"nova_carter\"" in urdf_text, "robot_description does not contain expected robot name!"
        assert f"link name=\"{embodiment_cfg.lidar_frame}\"" in urdf_text, f"Missing link '{embodiment_cfg.lidar_frame}' in URDF!"
        print("          URDF contents verified -> [PASS]")

        # Assertion 3: Static TF base_link -> chassis_link
        print(f"[CHECK 3] Querying static TF '{embodiment_cfg.base_frame}' -> '{embodiment_cfg.chassis_frame}'...")
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
        print(f"          Translation: ({t_base_chassis.transform.translation.x:.3f}, "
              f"{t_base_chassis.transform.translation.y:.3f}, "
              f"{t_base_chassis.transform.translation.z:.3f}) -> [PASS]")

        # Assertion 4: Static TF chassis_link -> lidar_link
        print(f"[CHECK 4] Querying static TF '{embodiment_cfg.chassis_frame}' -> '{embodiment_cfg.lidar_frame}'...")
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
        print(f"          LiDAR Z Offset: {lidar_z:.3f} m (expected {embodiment_cfg.lidar_offset[2]:.3f} m)")
        assert abs(lidar_z - embodiment_cfg.lidar_offset[2]) < 1e-4, f"LiDAR Z offset mismatch: {lidar_z}"
        print("          Static LiDAR TF verified -> [PASS]")

        # Assertion 5: Dynamic TF odom -> base_link
        print(f"[CHECK 5] Querying dynamic TF 'odom' -> '{embodiment_cfg.base_frame}'...")
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
        print(f"          Odom->Base Translation: ({t_odom_base.transform.translation.x:.3f}, "
              f"{t_odom_base.transform.translation.y:.3f}, "
              f"{t_odom_base.transform.translation.z:.3f}) -> [PASS]")

        # Assertion 6: Full chain lookup odom -> lidar_link
        print(f"[CHECK 6] Full chain lookup 'odom' -> '{embodiment_cfg.lidar_frame}'...")
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
        print(f"          Total LiDAR height in odom frame: {total_z:.3f} m -> [PASS]")

        print("\n" + "=" * 70)
        print("ALL TF TREE & ROBOT_STATE_PUBLISHER ASSERTIONS PASSED SUCCESSFULLY!")
        print("=" * 70)

    finally:
        # Clean teardown
        print("[INFO] Cleaning up processes and ROS nodes...")
        listener_node.destroy_node()
        rclpy.shutdown()
        if rsp_proc.poll() is None:
            rsp_proc.terminate()
            try:
                rsp_proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                rsp_proc.kill()


def main():
    try:
        run_tf_verification()
    except Exception as e:
        import traceback
        print("[ERROR] Exception occurred during TF verification:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
