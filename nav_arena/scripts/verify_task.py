# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Verification script for PointNav task environment, goal management, and MDP metrics."""

from __future__ import annotations

import argparse
import sys
import torch

from isaaclab.app import AppLauncher

# Parse CLI arguments
parser = argparse.ArgumentParser(description="Verify PointNav task environment and evaluation metrics.")
parser.add_argument("--scene", type=str, default="kujiale_0003", help="InteriorAgent scene ID or USD path.")
parser.add_argument("--num-steps", type=int, default=150, help="Number of simulation steps to run.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Omniverse application
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Simulation app is now active - Omniverse / Isaac Lab imports are safe."""

import carb.settings

_carb_settings = carb.settings.get_settings()
_carb_settings.set("/app/livestream/allowResize", True)
_carb_settings.set("/exts/omni.kit.livestream.app/primaryStream/allowDynamicResize", True)

from nav_arena.tasks import PointNavTask, create_point_nav_env_cfg


def run_verification():
    print("=" * 70)
    print("PointNav Task & Metrics Verification")
    print(f"Scene: {args_cli.scene}")
    print("=" * 70)

    # Robot spawn location and target goal in the wide living room opening
    spawn_pos = (-2.5, 0.0, 0.25)
    spawn_rot = (0.0, 0.0, 0.0, 1.0)
    # Target goal 1.5m straight ahead along +X
    goal_pos = (-1.0, 0.0)
    goal_heading = 0.0
    goal_threshold = 0.4

    print(f"[INFO] Configuring PointNav environment:")
    print(f"       Robot Spawn: {spawn_pos}")
    print(f"       Goal Target: {goal_pos} (heading: {goal_heading} rad, tolerance: {goal_threshold} m)")

    env_cfg = create_point_nav_env_cfg(
        scene_id_or_path=args_cli.scene,
        robot_spawn_pos=spawn_pos,
        robot_spawn_rot=spawn_rot,
        goal_pos=goal_pos,
        goal_heading=goal_heading,
        goal_threshold=goal_threshold,
        episode_length_s=60.0,
    )

    print("[INFO] Instantiating PointNavTask...")
    task = PointNavTask(cfg=env_cfg)

    # Position viewport camera to frame the robot and goal runway in the living room
    task.sim.set_camera_view(eye=[-1.75, -2.8, 2.2], target=[-1.75, 0.0, 0.3])

    # Reset environment
    print("[INFO] Resetting environment...")
    obs, extras = task.reset()

    # Step 1: Verify Goal and Robot Pose Extraction
    goal_x, goal_y, goal_h = task.get_goal_pose_w()
    robot_x, robot_y, robot_h = task.get_robot_pose_w()
    initial_dist = task.get_goal_distance()

    print(f"[VERIFY] Initial Robot Pose: ({robot_x:.3f}, {robot_y:.3f}, {robot_h:.3f})")
    print(f"[VERIFY] Target Goal Pose:   ({goal_x:.3f}, {goal_y:.3f}, {goal_h:.3f})")
    print(f"[VERIFY] Initial Goal Distance: {initial_dist:.3f} m")

    assert abs(goal_x - goal_pos[0]) < 1e-3, f"Goal X mismatch: expected {goal_pos[0]}, got {goal_x}"
    assert abs(goal_y - goal_pos[1]) < 1e-3, f"Goal Y mismatch: expected {goal_pos[1]}, got {goal_y}"
    assert initial_dist > 1.0, f"Expected initial distance > 1.0 m, got {initial_dist}"
    print("  -> [PASS] Goal pose and initial distance verified.")

    # Step 2: Drive towards the goal and verify distance decrease and goal termination
    print("\n[INFO] Driving forward towards the goal (+X)...")
    forward_action = torch.tensor([[0.6, 0.0]], device=task.device)

    goal_reached_detected = False
    step_count = 0

    for step in range(args_cli.num_steps):
        obs, rew, terminated, truncated, extras = task.step(forward_action)
        step_count += 1

        curr_dist = task.get_goal_distance()
        rx, ry, rh = task.get_robot_pose_w()

        if step % 20 == 0 or task.is_goal_reached():
            print(f"  Step {step:03d}: Robot=({rx:.2f}, {ry:.2f}) Dist={curr_dist:.2f}m Term={terminated.item()} (goal={task.is_goal_reached()}, coll={task.is_collision()})")

        if task.is_goal_reached():
            print(f"\n[VERIFY] Goal reached triggered at step {step_count}!")
            print(f"         Final Robot Pose: ({rx:.3f}, {ry:.3f})")
            print(f"         Final Goal Distance: {curr_dist:.3f} m (threshold: {goal_threshold} m)")
            goal_reached_detected = True
            break

    assert goal_reached_detected, "Robot failed to trigger goal termination during forward drive test!"
    print("  -> [PASS] Goal reached termination verified.")

    # Step 3: Verify Episode Reset
    print("\n[INFO] Testing episode reset...")
    task.reset()
    reset_x, reset_y, _ = task.get_robot_pose_w()
    reset_dist = task.get_goal_distance()
    print(f"[VERIFY] Post-reset Robot Pose: ({reset_x:.3f}, {reset_y:.3f})")
    print(f"[VERIFY] Post-reset Goal Distance: {reset_dist:.3f} m")
    assert abs(reset_x - spawn_pos[0]) < 0.2, f"Robot failed to reset to spawn X: got {reset_x}"
    assert abs(reset_y - spawn_pos[1]) < 0.2, f"Robot failed to reset to spawn Y: got {reset_y}"
    print("  -> [PASS] Reset and state restoration verified.")

    # Step 4: Verify Contact / Collision Metric by driving backward into the wall
    print("\n[INFO] Testing collision detection by driving backward towards wall (-X)...")
    reverse_action = torch.tensor([[-0.8, 0.0]], device=task.device)
    collision_detected = False

    for c_step in range(200):
        obs, rew, terminated, truncated, extras = task.step(reverse_action)
        if c_step % 25 == 0 or task.is_collision():
            rx, ry, _ = task.get_robot_pose_w()
            c_force = torch.linalg.norm(task.scene.sensors["contact_forces"].data.net_forces_w[:, 0], dim=-1).cpu().item()
            print(f"  Collision test step {c_step:03d}: Robot=({rx:.2f}, {ry:.2f}) ContactForce={c_force:.2f}N Term={terminated.item()} (coll={task.is_collision()})")

        if task.is_collision():
            c_force = torch.linalg.norm(task.scene.sensors["contact_forces"].data.net_forces_w[:, 0], dim=-1).cpu().item()
            print(f"\n[VERIFY] Collision termination triggered at step {c_step + 1}!")
            print(f"         Impact Net Force: {c_force:.2f} N (threshold: 1.0 N)")
            collision_detected = True
            break

    assert collision_detected, "Collision termination failed to trigger when driving into obstacle!"
    print("  -> [PASS] Collision termination and contact metric verified.")

    print("\n" + "=" * 70)
    print("ALL POINTNAV VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 70)

    task.close()


def main():
    try:
        run_verification()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
