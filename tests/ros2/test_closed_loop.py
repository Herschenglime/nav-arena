# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Closed-loop integration test for ROS 2 navigation bridge."""

import os
import subprocess
import sys
import time
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock


def test_closed_loop():
    print("=" * 70)
    print("Running Closed-Loop ROS 2 Bridge Integration Test")
    print("=" * 70)

    # 1. Launch simulation bridge in background with sufficient steps for test
    viz_flags = ["--headless"]
    startup_timeout = 45.0
    if "--viz" in sys.argv:
        idx = sys.argv.index("--viz")
        if idx + 1 < len(sys.argv):
            viz_flags = ["--viz", sys.argv[idx + 1]]
        else:
            viz_flags = ["--viz", "kit"]
        if "kit" in viz_flags:
            startup_timeout = 90.0

    sim_script = os.path.abspath("nav_arena/nav_arena/scripts/run_ros2_nav.py")
    cmd = [
        sys.executable,
        "-u",
        sim_script,
        *viz_flags,
        "--num-steps",
        "3000",
    ]
    print(f"[INFO] Launching simulation process: {' '.join(cmd)}")
    log_file_path = "/home/robopi/simulation/closed_loop_sim.log"
    log_file = open(log_file_path, "w")
    sim_proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True)

    # 2. Initialize test ROS 2 node
    rclpy.init()
    test_node = rclpy.create_node("closed_loop_tester")

    received_odom: list[Odometry] = []
    received_clocks: list[Clock] = []
    received_goals: list[PoseStamped] = []

    test_node.create_subscription(Odometry, "/odom", lambda m: received_odom.append(m), 10)
    test_node.create_subscription(Clock, "/clock", lambda m: received_clocks.append(m), 10)

    from rclpy.qos import QoSProfile, QoSDurabilityPolicy
    goal_qos = QoSProfile(
        depth=1,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
    )
    test_node.create_subscription(PoseStamped, "/goal_pose", lambda m: received_goals.append(m), goal_qos)

    cmd_pub = test_node.create_publisher(Twist, "/cmd_vel", 10)

    forward_twist = Twist()
    forward_twist.linear.x = 0.4  # 0.4 m/s forward
    stop_twist = Twist()          # 0.0 m/s

    start_time = time.time()
    try:
        # Wait for simulation startup and bridge activation across all topics
        print(f"[INFO] Waiting up to {startup_timeout}s for simulation startup, /clock, /odom, and /goal_pose...")
        while time.time() - start_time < startup_timeout and (
            len(received_clocks) == 0 or len(received_odom) == 0 or len(received_goals) == 0
        ):
            if sim_proc.poll() is not None:
                log_file.flush()
                with open(log_file_path, "r") as f:
                    out = f.read()
                print(f"[ERROR] Simulation process exited prematurely with code {sim_proc.returncode}:\n{out[-2000:]}")
                assert False, "Simulation process died before publishing topics."
            rclpy.spin_once(test_node, timeout_sec=0.1)

        assert len(received_clocks) > 0, f"Did not receive /clock within {startup_timeout}s."
        assert len(received_odom) > 0, f"Did not receive /odom within {startup_timeout}s."
        assert len(received_goals) > 0, f"Did not receive /goal_pose within {startup_timeout}s."

        goal = received_goals[-1]
        start_odom_x = received_odom[-1].pose.pose.position.x
        print(f"[INFO] Bridge Active!")
        print(f"       Initial Sim Time: {received_clocks[-1].clock.sec}s")
        print(f"       Initial Odom X:   {start_odom_x:.3f}m")
        print(f"       Goal Pose (frame={goal.header.frame_id}): ({goal.pose.position.x:.2f}, {goal.pose.position.y:.2f})")

        # Closed-loop navigation until goal is reached
        # Robot spawns at world x=-2.5, goal is at world x=-1.0 (delta = 1.50m)
        target_goal_dist = abs(goal.pose.position.x - (-2.5))
        goal_tolerance = 0.20    # Close proximity threshold to goal (1.30m traveled)
        max_drive_timeout = 25.0 # 25 seconds is plenty for 1.50m at 0.4 m/s

        print(f"[INFO] Goal Navigation Target Distance: {target_goal_dist:.2f}m ahead (tolerance: {goal_tolerance:.2f}m)")
        print("[INFO] Streaming closed-loop /cmd_vel towards goal...")

        drive_start = time.time()
        last_report = 0.0
        goal_reached = False
        curr_traveled = 0.0

        while time.time() - drive_start < max_drive_timeout and sim_proc.poll() is None:
            rclpy.spin_once(test_node, timeout_sec=0.02)
            curr_x = received_odom[-1].pose.pose.position.x
            curr_traveled = curr_x - start_odom_x
            dist_remaining = target_goal_dist - curr_traveled

            if time.time() - last_report > 0.5:
                last_report = time.time()
                print(f"[INFO] Driving... traveled: {curr_traveled:.3f}m / {target_goal_dist:.2f}m (dist remaining: {dist_remaining:.3f}m)")

            # Check goal reached success condition
            if dist_remaining <= goal_tolerance:
                print("=" * 70)
                print(f"[SUCCESS] GOAL REACHED! Distance to goal: {dist_remaining:.3f}m <= {goal_tolerance:.2f}m")
                print("=" * 70)
                goal_reached = True

                # Command robot to halt smoothly
                for _ in range(15):
                    cmd_pub.publish(stop_twist)
                    rclpy.spin_once(test_node, timeout_sec=0.04)

                # Keep simulation viewport open for visual confirmation if GUI is active
                if "kit" in viz_flags:
                    print("[INFO] Holding stationary at goal for 3 seconds for visual inspection...")
                    time.sleep(3.0)
                break

            cmd_pub.publish(forward_twist)

        assert goal_reached, f"Robot failed to reach goal within {max_drive_timeout}s! Traveled: {curr_traveled:.3f}m"

        print("[INFO] Test complete. Terminating simulation process cleanly...")
        if sim_proc.poll() is None:
            sim_proc.terminate()
            try:
                sim_proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                sim_proc.kill()
        print("[INFO] Simulation process terminated.")

        # Final ROS spin to drain incoming queues
        for _ in range(10):
            rclpy.spin_once(test_node, timeout_sec=0.05)

        # 3. Assertions
        print(f"[INFO] Verification results:")
        print(f"       Total /clock messages received: {len(received_clocks)}")
        print(f"       Total /goal_pose messages received: {len(received_goals)}")
        print(f"       Total /odom messages received: {len(received_odom)}")

        assert len(received_clocks) > 10, "Insufficient /clock messages received."
        assert len(received_goals) > 0, "No /goal_pose messages received."
        assert len(received_odom) > 10, "Insufficient /odom messages received."

        # Verify goal pose coordinates match expected (-1.0, 0.0) in world frame
        final_goal = received_goals[-1]
        assert final_goal.header.frame_id == "world", f"Unexpected frame_id: {final_goal.header.frame_id}"
        assert abs(final_goal.pose.position.x - (-1.0)) < 0.1, f"Unexpected goal x: {final_goal.pose.position.x}"
        assert abs(final_goal.pose.position.y - 0.0) < 0.1, f"Unexpected goal y: {final_goal.pose.position.y}"
        print(f"[INFO] Goal Pose Verified: ({final_goal.pose.position.x:.2f}, {final_goal.pose.position.y:.2f}) [frame: {final_goal.header.frame_id}]")

        final_odom_x = received_odom[-1].pose.pose.position.x
        total_delta = final_odom_x - start_odom_x
        print(f"[INFO] Final Odometry: start x={start_odom_x:.3f}, end x={final_odom_x:.3f}, total delta={total_delta:.3f}m")
        assert total_delta >= (target_goal_dist - goal_tolerance - 0.05), f"Robot traveled insufficient distance: {total_delta:.3f}m"

        print("=" * 70)
        print("CLOSED-LOOP GOAL NAVIGATION TEST PASSED SUCCESSFULLY!")
        print("=" * 70)

    finally:
        if sim_proc.poll() is None:
            sim_proc.terminate()
            try:
                sim_proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                sim_proc.kill()
        test_node.destroy_node()
        rclpy.shutdown()
        log_file.close()


if __name__ == "__main__":
    test_closed_loop()
