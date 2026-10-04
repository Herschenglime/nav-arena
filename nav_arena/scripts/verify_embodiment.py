# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Verification script for Nova Carter embodiment, ActionManager, and sensors."""

from __future__ import annotations

import argparse
import sys
import torch

from isaaclab.app import AppLauncher

from nav_arena.core import launch_simulation_app
from nav_arena.utils import add_logger_args, configure_logging, create_mock_env, get_logger

logger = get_logger("verify_embodiment")

# Parse arguments
parser = argparse.ArgumentParser(description="Verify Nova Carter embodiment and ActionManager.")
parser.add_argument("--loop", action="store_true", help="Keep simulation running in a loop for livestream verification.")
add_logger_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

configure_logging(args_cli.log_level)

def run_verification(simulation_app):
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, AssetBaseCfg
    from isaaclab.managers import ActionManager
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sensors import RayCaster
    from isaaclab.utils.configclass import configclass

    from nav_arena.embodiments import (
        NOVA_CARTER_ACTION_CFG,
        NOVA_CARTER_CFG,
        create_2d_lidar_cfg,
    )
    from nav_arena.ros2 import build_ros2_omnigraph

    @configclass
    class EmbodimentVerifySceneCfg(InteractiveSceneCfg):
        """Minimal verification scene with ground plane, lighting, and Nova Carter."""

        # Ground plane
        ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

        # Lighting
        dome_light = AssetBaseCfg(
            prim_path="/World/defaultDomeLight",
            spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8)),
        )

        # Nova Carter Articulation
        robot = NOVA_CARTER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # 2D LiDAR
        lidar = create_2d_lidar_cfg(
            prim_path="{ENV_REGEX_NS}/Robot/chassis_link",
            mesh_prim_paths=["/World/defaultGroundPlane"],
        )

    @configclass
    class EmbodimentActionCfg:
        """Action configuration for Nova Carter differential drive."""

        robot_action = NOVA_CARTER_ACTION_CFG.replace(asset_name="robot")


    logger.section("VERIFYING NOVA CARTER EMBODIMENT & SENSORS")
    logger.info("Creating SimulationContext...")
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)

    logger.info("Setting up verification scene...")
    scene_cfg = EmbodimentVerifySceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    # ActionManager expects an environment object that holds .scene and .sim
    mock_env = create_mock_env(scene, sim)

    # Initialize ROS 2 OmniGraph bridge before starting simulation timeline
    logger.info("Initializing ROS 2 OmniGraph bridge...")
    build_ros2_omnigraph(
        robot_prim_path="/World/envs/env_0/Robot/chassis_link",
    )


    # Reset simulator to initialize physics views and start playback
    sim.reset()

    # Initialize ActionManager
    action_manager = ActionManager(EmbodimentActionCfg(), mock_env)

    robot: Articulation = scene["robot"]
    lidar: RayCaster = scene["lidar"]

    # Position camera to frame the robot nicely
    sim.set_camera_view(eye=[2.5, -2.5, 1.8], target=[0.0, 0.0, 0.3])

    init_pos_x = robot.data.root_pos_w[0, 0].item()
    logger.info(f"Initial robot X position: {init_pos_x:.4f} m")

    if args_cli.loop:
        logger.info("Running in continuous loop for livestream inspection. Press Ctrl+C to terminate.")
        step = 0
        while simulation_app.is_running():
            # Alternate maneuvers: Forward, Spin Left, Reverse, Spin Right
            phase = (step // 150) % 4
            if phase == 0:
                cmd = [0.6, 0.0]
            elif phase == 1:
                cmd = [0.0, 1.0]
            elif phase == 2:
                cmd = [-0.4, 0.0]
            else:
                cmd = [0.0, -1.0]

            action = torch.tensor([cmd], device=sim.device)
            action_manager.process_action(action)
            action_manager.apply_action()

            scene.write_data_to_sim()
            sim.step()
            scene.update(dt=sim.get_physics_dt())
            # Pump Omniverse Kit event loop to render frame and stream via WebRTC
            simulation_app.update()
            step += 1
        return

    # Standard non-loop test run
    # Test driving forward with linear vel = 0.5 m/s, angular vel = 0.0 rad/s
    action = torch.tensor([[0.5, 0.0]], device=sim.device)

    logger.info("Stepping simulation with action [v=0.5 m/s, w=0.0 rad/s] for 60 steps...")
    for step in range(60):
        # Process and apply action through ActionManager
        action_manager.process_action(action)
        action_manager.apply_action()

        # Step physics
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt=sim.get_physics_dt())

    final_pos_x = robot.data.root_pos_w[0, 0].item()
    displacement_x = final_pos_x - init_pos_x
    logger.info(f"Final robot X position: {final_pos_x:.4f} m (displacement: {displacement_x:.4f} m)")

    # Check that robot actually moved forward
    assert displacement_x > 0.05, f"Robot failed to drive forward: displacement was {displacement_x:.4f} m"
    logger.check("DifferentialDriveAction commanded wheel joints", displacement_x > 0.05, f"displacement={displacement_x:.4f} m")

    # Check that LiDAR sensor updated
    ray_hits = lidar.data.ray_hits_w
    logger.info(f"RayCaster data shape: {ray_hits.shape}")
    assert ray_hits is not None and ray_hits.numel() > 0, "RayCaster returned empty data!"
    logger.check("RayCaster 2D LiDAR generated valid ray hits", ray_hits is not None and ray_hits.numel() > 0, f"shape={tuple(ray_hits.shape)}")

    logger.section("PHASE 2 EMBODIMENT VERIFICATION PASSED SUCCESSFULLY!")


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
