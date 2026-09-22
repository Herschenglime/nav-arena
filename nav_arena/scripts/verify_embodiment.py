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

# Parse arguments
parser = argparse.ArgumentParser(description="Verify Nova Carter embodiment and ActionManager.")
parser.add_argument("--loop", action="store_true", help="Keep simulation running in a loop for livestream verification.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Inject boot-time extension flags for OmniGraph USD schemas and livestreaming
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


"""Simulation app is now active - subsequent Omniverse / Isaac Lab imports are safe."""

import carb.settings

# Enable dynamic viewport resizing so WebRTC client resolutions (e.g. 1440p/2560x1440) connect seamlessly
_carb_settings = carb.settings.get_settings()
_carb_settings.set("/app/livestream/allowResize", True)
_carb_settings.set("/exts/omni.kit.livestream.app/primaryStream/allowDynamicResize", True)


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
    setup_ros2_clock,
    setup_ros2_odometry,
)


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


def run_verification():
    print("[INFO] Creating SimulationContext...")
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)

    print("[INFO] Setting up verification scene...")
    scene_cfg = EmbodimentVerifySceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    import types

    # ActionManager expects an environment object that holds .scene and .sim
    mock_env = types.SimpleNamespace(
        scene=scene,
        sim=sim,
        device=sim.device,
        num_envs=scene.num_envs,
        physics_dt=sim.get_physics_dt(),
        step_dt=sim.get_physics_dt(),
    )

    # Initialize ROS 2 OmniGraphs before starting simulation timeline
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
    lidar: RayCaster = scene["lidar"]

    # Position camera to frame the robot nicely
    sim.set_camera_view(eye=[2.5, -2.5, 1.8], target=[0.0, 0.0, 0.3])

    init_pos_x = robot.data.root_pos_w[0, 0].item()
    print(f"[INFO] Initial robot X position: {init_pos_x:.4f} m")

    if args_cli.loop:
        print("[INFO] Running in continuous loop for livestream inspection. Press Ctrl+C to terminate.")
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

    print("[INFO] Stepping simulation with action [v=0.5 m/s, w=0.0 rad/s] for 60 steps...")
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
    print(f"[INFO] Final robot X position: {final_pos_x:.4f} m (displacement: {displacement_x:.4f} m)")

    # Check that robot actually moved forward
    assert displacement_x > 0.05, f"Robot failed to drive forward: displacement was {displacement_x:.4f} m"
    print("[SUCCESS] DifferentialDriveAction correctly commanded the wheel joints!")

    # Check that LiDAR sensor updated
    ray_hits = lidar.data.ray_hits_w
    print(f"[INFO] RayCaster data shape: {ray_hits.shape}")
    assert ray_hits is not None and ray_hits.numel() > 0, "RayCaster returned empty data!"
    print("[SUCCESS] RayCaster 2D LiDAR generated valid ray hit tensors!")

    print("=" * 60)
    print("PHASE 2 EMBODIMENT VERIFICATION PASSED SUCCESSFULLY!")
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
