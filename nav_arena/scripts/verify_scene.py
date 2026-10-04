# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Verification script for InteriorAgent scene configuration and robot embodiment."""

from __future__ import annotations

import argparse
import sys
import torch

from isaaclab.app import AppLauncher

from nav_arena.core import launch_simulation_app
from nav_arena.utils import add_logger_args, configure_logging, create_mock_env, get_logger

logger = get_logger("verify_scene")

# Parse CLI arguments
parser = argparse.ArgumentParser(description="Verify InteriorAgent scene loading and robot embodiment.")
parser.add_argument("--scene", type=str, default="kujiale_0003", help="InteriorAgent scene ID or USD path.")
parser.add_argument("--loop", action="store_true", help="Keep simulation running continuously for visual inspection.")
add_logger_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

configure_logging(args_cli.log_level)

def run_verification(simulation_app):
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation
    from isaaclab.managers import ActionManager
    from isaaclab.scene import InteractiveScene
    from isaaclab.utils.configclass import configclass

    from nav_arena.embodiments import NOVA_CARTER_ACTION_CFG
    from nav_arena.scenes import create_interior_agent_scene_cfg
    from nav_arena.ros2 import build_ros2_omnigraph

    @configclass
    class EmbodimentActionCfg:
        """Action configuration for Nova Carter differential drive."""

        robot_action = NOVA_CARTER_ACTION_CFG.replace(asset_name="robot")


    logger.section(f"INTERIORAGENT SCENE VERIFICATION: '{args_cli.scene}'")
    logger.info(f"Setting up SimulationContext with scene '{args_cli.scene}'...")
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)

    # Robot spawn location in the living room
    robot_pos = (0.0, -2.0, 0.25)
    robot_rot = (0.0, 0.0, 0.0, 1.0)

    logger.info(f"Configuring InteriorAgent scene with robot at {robot_pos}...")
    scene_cfg = create_interior_agent_scene_cfg(
        scene_id_or_path=args_cli.scene,
        robot_spawn_pos=robot_pos,
        robot_spawn_rot=robot_rot,
        num_envs=1,
    )

    logger.info("Creating InteractiveScene...")
    scene = InteractiveScene(scene_cfg)

    mock_env = create_mock_env(scene, sim)

    # Initialize ROS 2 OmniGraph bridge before timeline playback
    logger.info("Initializing ROS 2 OmniGraph bridge...")
    build_ros2_omnigraph(
        robot_prim_path="/World/envs/env_0/Robot/chassis_link",
    )


    # Reset simulator to initialize physics views and start playback
    sim.reset()

    # Initialize ActionManager
    action_manager = ActionManager(EmbodimentActionCfg(), mock_env)

    robot: Articulation = scene["robot"]

    # Frame camera in living room towards robot
    sim.set_camera_view(
        eye=[robot_pos[0] + 2.5, robot_pos[1] - 2.5, 2.0],
        target=[robot_pos[0], robot_pos[1], 0.3],
    )

    init_pos_z = robot.data.root_pos_w[0, 2].item()
    init_pos_x = robot.data.root_pos_w[0, 0].item()
    init_pos_y = robot.data.root_pos_w[0, 1].item()
    logger.info(f"Initial robot position: x={init_pos_x:.4f}, y={init_pos_y:.4f}, z={init_pos_z:.4f} m")

    if args_cli.loop:
        logger.info("Running in continuous loop for visual inspection. Press Ctrl+C to terminate.")
        step = 0
        while simulation_app.is_running():
            phase = (step // 150) % 4
            if phase == 0:
                cmd = [0.4, 0.0]
            elif phase == 1:
                cmd = [0.0, 0.8]
            elif phase == 2:
                cmd = [-0.3, 0.0]
            else:
                cmd = [0.0, -0.8]

            action = torch.tensor([cmd], device=sim.device)
            action_manager.process_action(action)
            action_manager.apply_action()

            scene.write_data_to_sim()
            sim.step()
            scene.update(dt=sim.get_physics_dt())
            simulation_app.update()
            step += 1
        return

    # Non-loop automated verification run
    logger.info("Stepping 60 steps to allow robot to settle onto floor geometry...")
    zero_action = torch.tensor([[0.0, 0.0]], device=sim.device)
    for _ in range(60):
        action_manager.process_action(zero_action)
        action_manager.apply_action()
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt=sim.get_physics_dt())

    settled_pos_z = robot.data.root_pos_w[0, 2].item()
    logger.info(f"Settled robot Z position: {settled_pos_z:.4f} m")

    # Robot must not fall through the floor into the abyss (e.g. z < -1.0)
    assert settled_pos_z > -0.5, f"Robot fell through the floor! Settled Z: {settled_pos_z:.4f} m"
    logger.check("Robot settled securely on floor geometry", settled_pos_z > -0.5, f"settled_z={settled_pos_z:.4f} m")

    # Test driving forward inside the room
    logger.info("Stepping 60 steps commanding forward velocity [v=0.4 m/s, w=0.0 rad/s]...")
    fwd_action = torch.tensor([[0.4, 0.0]], device=sim.device)
    start_pos_x = robot.data.root_pos_w[0, 0].item()
    start_pos_y = robot.data.root_pos_w[0, 1].item()

    for _ in range(60):
        action_manager.process_action(fwd_action)
        action_manager.apply_action()
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt=sim.get_physics_dt())

    final_pos_x = robot.data.root_pos_w[0, 0].item()
    final_pos_y = robot.data.root_pos_w[0, 1].item()
    displacement = ((final_pos_x - start_pos_x) ** 2 + (final_pos_y - start_pos_y) ** 2) ** 0.5
    logger.info(f"Final robot position: ({final_pos_x:.4f}, {final_pos_y:.4f}), displacement: {displacement:.4f} m")

    assert displacement > 0.05, f"Robot failed to drive forward: displacement was {displacement:.4f} m"
    logger.check("Robot floor navigation displacement", displacement > 0.05, f"displacement={displacement:.4f} m")

    # Check LiDAR sensor
    if "lidar" in scene.keys():
        lidar = scene["lidar"]
        ray_hits = lidar.data.ray_hits_w
        logger.info(f"LiDAR ray hits tensor shape: {ray_hits.shape}")
        assert ray_hits is not None and ray_hits.numel() > 0, "LiDAR returned empty ray hits!"
        logger.check("LiDAR scene ray casting", ray_hits is not None and ray_hits.numel() > 0, f"shape={tuple(ray_hits.shape)}")

    logger.section(f"PHASE 3 INTERIORAGENT SCENE VERIFICATION ('{args_cli.scene}') PASSED SUCCESSFULLY!")


def main():
    try:
        with launch_simulation_app(args_cli, enable_ros2=True, livestream=True) as simulation_app:
            run_verification(simulation_app)
    except Exception as e:
        logger.error(f"Exception occurred during verification: {e}", exc_info=True)
        sys.exit(1)
    except BaseException as e:
        logger.critical(f"BaseException occurred: {type(e)}: {e}", exc_info=True)
        sys.exit(1)



if __name__ == "__main__":
    main()
