# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Verification script for the PointNav task: goal management, MDP metrics, and per-embodiment sensors."""

from __future__ import annotations

import argparse
import sys
import torch

from isaaclab.app import AppLauncher

from nav_arena.core import launch_simulation_app
from nav_arena.utils import add_logger_args, configure_logging, get_logger

logger = get_logger("verify_task")

# Parse CLI arguments
parser = argparse.ArgumentParser(description="Verify PointNav task environment and evaluation metrics.")
parser.add_argument("--scene", type=str, default="kujiale_0003", help="InteriorAgent scene ID or USD path.")
parser.add_argument("--num-steps", type=int, default=150, help="Number of simulation steps to run.")
# Not restricted with argparse `choices`: listing embodiments would import Isaac Lab before the app boots.
parser.add_argument("--robot", type=str, default="nova_carter", help="Registered embodiment (e.g. nova_carter, dingo).")
parser.add_argument(
    "--camera", action="store_true", help="Also mount the RGB-D and goal cameras and verify frames and goal images."
)
add_logger_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.camera:
    # RTX camera rendering must be enabled before the app boots.
    args_cli.enable_cameras = True

configure_logging(args_cli.log_level)



def verify_dingo_usd_fixes(task) -> None:
    """Verify the derived Dingo USD's caster friction fix and ground-plane removal reached the spawned robot."""
    stage = task.sim.stage
    caster = stage.GetPrimAtPath("/World/envs/env_0/Robot/PhysicsMaterials/caster_wheel")
    mode = caster.GetAttribute("physxMaterial:frictionCombineMode").Get() if caster.IsValid() else None
    logger.check("Caster friction combine mode is 'min'", mode == "min", f"mode={mode}")
    assert mode == "min", f"Dingo caster friction fix missing (mode={mode})"
    ground = stage.GetPrimAtPath("/World/envs/env_0/Robot/GroundPlane")
    ground_off = not ground.IsValid() or not ground.IsActive()
    logger.check("Dingo's embedded ground plane is deactivated", ground_off)
    assert ground_off, "Dingo's embedded ground plane is still active"


def verify_cameras(task) -> None:
    """Verify RGB-D frames, intrinsics, and goal-image rendering."""
    import numpy as np

    for _ in range(8):  # RTX output needs a few render frames to settle
        task.sim.render()
    rgb, depth = task.get_camera_frame()
    intrinsics = task.get_camera_intrinsics()
    valid = np.isfinite(depth) & (depth > 0)
    logger.info(f"Camera frame: rgb={rgb.shape} {rgb.dtype}, depth={depth.shape} {depth.dtype}, fx={intrinsics[0, 0]:.1f}")
    assert rgb.shape == (360, 640, 3) and rgb.dtype == np.uint8, f"Unexpected RGB {rgb.shape} {rgb.dtype}"
    assert depth.shape == (360, 640), f"Unexpected depth {depth.shape}"
    assert abs(intrinsics[0, 0] - 476.595) < 0.5, f"Unexpected fx {intrinsics[0, 0]}"
    assert rgb.max() > 0, "RGB frame is black"
    assert valid.mean() > 0.05, f"Too few valid depth pixels ({valid.mean():.2f})"
    logger.check("RGB-D frame and intrinsics valid", True, f"valid_depth={valid.mean():.2f}, max_depth={depth[valid].max():.1f} m")

    goal_x, goal_y, _ = task.get_goal_pose_w()
    goal_image = task.render_goal_image((goal_x, goal_y))
    diff = float(np.abs(goal_image.astype(np.float32) - rgb.astype(np.float32)).mean())
    assert goal_image.shape == (360, 640, 3) and goal_image.dtype == np.uint8, f"Unexpected goal image {goal_image.shape}"
    assert goal_image.max() > 0, "Goal image is black"
    assert diff > 1.0, f"Goal image is identical to the live view (mean abs diff {diff:.2f})"
    logger.check("Goal image rendered from a different viewpoint", True, f"mean_abs_diff_vs_live={diff:.1f}")


def run_verification():
    from nav_arena.embodiments import get_embodiment
    from nav_arena.tasks import PointNavTask, create_point_nav_env_cfg

    embodiment = get_embodiment(args_cli.robot)
    logger.section(f"PointNav Task & Metrics Verification: '{args_cli.scene}' with '{embodiment.name}'")


    # Robot spawn location and target goal in the wide living room opening
    spawn_pos = (-2.5, -0.25)
    spawn_rot = (0.0, 0.0, 0.0, 1.0)
    # Target goal 1.5m straight ahead along +X
    goal_pos = (-1.0, -0.25)
    goal_heading = 0.0
    goal_threshold = 0.4

    logger.info("Configuring PointNav environment:")
    logger.info(f"       Robot Spawn: {spawn_pos}")
    logger.info(f"       Goal Target: {goal_pos} (heading: {goal_heading} rad, tolerance: {goal_threshold} m)")

    env_cfg = create_point_nav_env_cfg(
        scene_id_or_path=args_cli.scene,
        robot_spawn_pos=spawn_pos,
        robot_spawn_rot=spawn_rot,
        goal_pos=goal_pos,
        goal_heading=goal_heading,
        goal_threshold=goal_threshold,
        episode_length_s=60.0,
        robot_name=args_cli.robot,
        enable_camera=args_cli.camera,
        enable_goal_camera=args_cli.camera,
    )

    logger.info("Instantiating PointNavTask...")
    task = PointNavTask(cfg=env_cfg)

    # Position viewport camera to frame the robot and goal runway in the living room
    task.sim.set_camera_view(eye=[-1.75, -3.05, 2.2], target=[-1.75, -0.25, 0.3])

    # Reset environment
    logger.info("Resetting environment...")
    obs, extras = task.reset()

    if args_cli.robot == "dingo":
        verify_dingo_usd_fixes(task)
    if args_cli.camera:
        verify_cameras(task)

    # Step 1: Verify Goal and Robot Pose Extraction
    goal_x, goal_y, goal_h = task.get_goal_pose_w()
    robot_x, robot_y, robot_h = task.get_robot_pose_w()
    initial_dist = task.get_goal_distance()

    logger.info(f"Initial Robot Pose: ({robot_x:.3f}, {robot_y:.3f}, {robot_h:.3f})")
    logger.info(f"Target Goal Pose:   ({goal_x:.3f}, {goal_y:.3f}, {goal_h:.3f})")
    logger.info(f"Initial Goal Distance: {initial_dist:.3f} m")

    assert abs(goal_x - goal_pos[0]) < 1e-3, f"Goal X mismatch: expected {goal_pos[0]}, got {goal_x}"
    assert abs(goal_y - goal_pos[1]) < 1e-3, f"Goal Y mismatch: expected {goal_pos[1]}, got {goal_y}"
    assert initial_dist > 1.0, f"Expected initial distance > 1.0 m, got {initial_dist}"
    logger.check("Goal pose and initial distance verified", True, f"initial_dist={initial_dist:.3f} m")

    # Step 2: Drive towards the goal and verify distance decrease and goal termination
    logger.info("Driving forward towards the goal (+X)...")
    forward_action = torch.tensor([[0.6, 0.0, 0.0]], device=task.device)

    goal_reached_detected = False
    step_count = 0
    max_lateral_force = 0.0

    for step in range(args_cli.num_steps):
        obs, rew, terminated, truncated, extras = task.step(forward_action)
        step_count += 1

        curr_dist = task.get_goal_distance()
        rx, ry, rh = task.get_robot_pose_w()
        free_drive_forces = task.scene.sensors["contact_forces"].data.net_forces_w.torch[:, 0]
        max_lateral_force = max(max_lateral_force, float(torch.linalg.norm(free_drive_forces[:, :2], dim=-1).max()))

        if step % 20 == 0 or task.is_goal_reached():
            logger.info(f"Step {step:03d}: Robot=({rx:.2f}, {ry:.2f}) Dist={curr_dist:.2f}m Term={terminated.item()} (goal={task.is_goal_reached()}, coll={task.is_collision()})")

        if task.is_goal_reached():
            logger.info(f"Goal reached triggered at step {step_count}! Final pose: ({rx:.3f}, {ry:.3f}), dist: {curr_dist:.3f}m")
            goal_reached_detected = True
            break

    assert goal_reached_detected, "Robot failed to trigger goal termination during forward drive test!"
    assert not task.is_collision(), "Collision falsely triggered during a free drive (check contact sensing)"
    logger.check("Goal reached termination verified", goal_reached_detected, f"steps={step_count}, final_dist={curr_dist:.3f} m")
    logger.info(f"Max lateral contact force during free drive: {max_lateral_force:.3f} N (collision threshold 1.0 N)")
    assert max_lateral_force < 1.0, f"Lateral contact force {max_lateral_force:.2f} N during free drive exceeds the collision threshold"

    # Step 3: Verify Episode Reset
    logger.info("Testing episode reset...")
    task.reset()
    reset_x, reset_y, _ = task.get_robot_pose_w()
    reset_dist = task.get_goal_distance()
    logger.info(f"Post-reset Robot Pose: ({reset_x:.3f}, {reset_y:.3f}), Goal Distance: {reset_dist:.3f} m")
    assert abs(reset_x - spawn_pos[0]) < 0.2, f"Robot failed to reset to spawn X: got {reset_x}"
    assert abs(reset_y - spawn_pos[1]) < 0.2, f"Robot failed to reset to spawn Y: got {reset_y}"
    logger.check("Reset and state restoration verified", True, f"pos=({reset_x:.3f}, {reset_y:.3f}), dist={reset_dist:.3f} m")

    # Step 4: Verify Contact / Collision Metric by driving backward into the wall
    logger.info("Testing collision detection by driving backward towards wall (-X)...")
    reverse_action = torch.tensor([[-0.8, 0.0, 0.0]], device=task.device)
    collision_detected = False

    for c_step in range(200):
        obs, rew, terminated, truncated, extras = task.step(reverse_action)
        if c_step % 25 == 0 or task.is_collision():
            rx, ry, _ = task.get_robot_pose_w()
            c_force = torch.linalg.norm(task.scene.sensors["contact_forces"].data.net_forces_w.torch[:, 0], dim=-1).cpu().item()
            logger.info(f"Collision test step {c_step:03d}: Robot=({rx:.2f}, {ry:.2f}) ContactForce={c_force:.2f}N Term={terminated.item()} (coll={task.is_collision()})")

        if task.is_collision():
            c_force = torch.linalg.norm(task.scene.sensors["contact_forces"].data.net_forces_w.torch[:, 0], dim=-1).cpu().item()
            logger.info(f"Collision termination triggered at step {c_step + 1}! Impact Net Force: {c_force:.2f} N")
            collision_detected = True
            break

    assert collision_detected, "Collision termination failed to trigger when driving into obstacle!"
    assert task.get_terminal_cause() == "collision", f"Unexpected terminal cause {task.get_terminal_cause()}"
    logger.check("Collision termination and contact metric verified", collision_detected, f"step={c_step + 1}, impact_force={c_force:.2f} N")

    logger.section("ALL POINTNAV VERIFICATION CHECKS PASSED SUCCESSFULLY!")

    task.close()


def main():
    try:
        with launch_simulation_app(args_cli, enable_ros2=False, livestream=True) as simulation_app:
            run_verification()
    except Exception as e:
        logger.error(f"Exception occurred during PointNav verification: {e}", exc_info=True)
        sys.exit(1)



if __name__ == "__main__":
    main()
