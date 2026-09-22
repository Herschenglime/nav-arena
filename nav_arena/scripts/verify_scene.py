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

# Parse CLI arguments
parser = argparse.ArgumentParser(description="Verify InteriorAgent scene loading and robot embodiment.")
parser.add_argument("--scene", type=str, default="kujiale_0003", help="InteriorAgent scene ID or USD path.")
parser.add_argument("--loop", action="store_true", help="Keep simulation running continuously for visual inspection.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Mandatory boot-time extension flags for OmniGraph schemas and ROS 2 bridge
sys.argv.extend([
    "--enable", "omni.graph",
    "--enable", "omni.graph.action",
    "--enable", "isaacsim.ros2.bridge",
    "--enable", "isaacsim.ros2.nodes",
    "--/exts/omni.kit.livestream.app/primaryStream/allowDynamicResize=true",
    "--/app/livestream/allowResize=true",
])

# Launch Omniverse application
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Simulation app is now active - Omniverse / Isaac Lab imports are safe."""

import carb.settings

_carb_settings = carb.settings.get_settings()
_carb_settings.set("/app/livestream/allowResize", True)
_carb_settings.set("/exts/omni.kit.livestream.app/primaryStream/allowDynamicResize", True)

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ActionManager
from isaaclab.scene import InteractiveScene
from isaaclab.utils.configclass import configclass

from nav_arena.embodiments import (
    NOVA_CARTER_ACTION_CFG,
    setup_ros2_clock,
    setup_ros2_odometry,
)
from nav_arena.scenes import create_interior_agent_scene_cfg


@configclass
class EmbodimentActionCfg:
    """Action configuration for Nova Carter differential drive."""

    robot_action = NOVA_CARTER_ACTION_CFG.replace(asset_name="robot")


def run_verification():
    print(f"[INFO] Setting up SimulationContext with scene '{args_cli.scene}'...")
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)

    # Robot spawn location in the living room
    robot_pos = (0.0, -2.0, 0.25)
    robot_rot = (0.0, 0.0, 0.0, 1.0)

    print(f"[INFO] Configuring InteriorAgent scene with robot at {robot_pos}...")
    scene_cfg = create_interior_agent_scene_cfg(
        scene_id_or_path=args_cli.scene,
        robot_spawn_pos=robot_pos,
        robot_spawn_rot=robot_rot,
        num_envs=1,
    )

    print("[INFO] Creating InteractiveScene...")
    scene = InteractiveScene(scene_cfg)

    import types

    mock_env = types.SimpleNamespace(
        scene=scene,
        sim=sim,
        device=sim.device,
        num_envs=scene.num_envs,
        physics_dt=sim.get_physics_dt(),
        step_dt=sim.get_physics_dt(),
    )

    # Initialize ROS 2 OmniGraphs before timeline playback
    print("[INFO] Initializing ROS 2 Clock and Odometry OmniGraphs...")
    setup_ros2_clock()
    setup_ros2_odometry(
        articulation_root="/World/envs/env_0/Robot/chassis_link",
        chassis_prim="/World/envs/env_0/Robot/chassis_link",
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
    print(f"[INFO] Initial robot position: x={init_pos_x:.4f}, y={init_pos_y:.4f}, z={init_pos_z:.4f} m")

    if args_cli.loop:
        print("[INFO] Running in continuous loop for visual inspection. Press Ctrl+C to terminate.")
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
    print("[INFO] Stepping 60 steps to allow robot to settle onto floor geometry...")
    zero_action = torch.tensor([[0.0, 0.0]], device=sim.device)
    for _ in range(60):
        action_manager.process_action(zero_action)
        action_manager.apply_action()
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt=sim.get_physics_dt())

    settled_pos_z = robot.data.root_pos_w[0, 2].item()
    print(f"[INFO] Settled robot Z position: {settled_pos_z:.4f} m")

    # Robot must not fall through the floor into the abyss (e.g. z < -1.0)
    assert settled_pos_z > -0.5, f"Robot fell through the floor! Settled Z: {settled_pos_z:.4f} m"
    print("[SUCCESS] Robot settled securely on the floor geometry (no fall-through)!")

    # Test driving forward inside the room
    print("[INFO] Stepping 60 steps commanding forward velocity [v=0.4 m/s, w=0.0 rad/s]...")
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
    print(f"[INFO] Final robot position: ({final_pos_x:.4f}, {final_pos_y:.4f}), displacement: {displacement:.4f} m")

    assert displacement > 0.05, f"Robot failed to drive forward: displacement was {displacement:.4f} m"
    print("[SUCCESS] Robot successfully navigated on the InteriorAgent floor!")

    # Check LiDAR sensor
    if "lidar" in scene.keys():
        lidar = scene["lidar"]
        ray_hits = lidar.data.ray_hits_w
        print(f"[INFO] LiDAR ray hits tensor shape: {ray_hits.shape}")
        assert ray_hits is not None and ray_hits.numel() > 0, "LiDAR returned empty ray hits!"
        print("[SUCCESS] LiDAR sensor successfully cast rays against the scene!")

    print("=" * 60)
    print(f"PHASE 3 INTERIORAGENT SCENE VERIFICATION ('{args_cli.scene}') PASSED SUCCESSFULLY!")
    print("=" * 60)


def main():
    try:
        run_verification()
    except Exception as e:
        import traceback

        print("[ERROR] Exception occurred during verification:", file=sys.stderr)
        traceback.print_exc()
    except BaseException as e:
        import traceback

        print(f"[ERROR] BaseException occurred: {type(e)}: {e}", file=sys.stderr)
        traceback.print_exc()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
