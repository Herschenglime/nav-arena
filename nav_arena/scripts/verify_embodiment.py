# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Verification script for registered embodiments (Nova Carter, Dingo), ActionManager, and sensors."""

from __future__ import annotations

import argparse
import sys
import torch

from isaaclab.app import AppLauncher

from nav_arena.core import launch_simulation_app
from nav_arena.utils import add_logger_args, configure_logging, create_mock_env, get_logger

logger = get_logger("verify_embodiment")

# Parse arguments
parser = argparse.ArgumentParser(description="Verify a registered embodiment, its ActionManager, and sensors.")
# Not restricted with argparse `choices`: listing embodiments would import Isaac Lab before the app boots.
# The name is validated by get_embodiment() once the app is running.
parser.add_argument(
    "--robot", default="nova_carter", help="Registered embodiment to verify (e.g. nova_carter, dingo)."
)
parser.add_argument(
    "--camera", action="store_true", help="Also mount the RGB-D camera and verify RGB / depth tensors."
)
parser.add_argument("--loop", action="store_true", help="Keep simulation running in a loop for livestream verification.")
add_logger_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.camera:
    # RTX camera rendering must be enabled before the app boots.
    args_cli.enable_cameras = True

configure_logging(args_cli.log_level)

def _to_torch(data):
    """Return a torch tensor view of Isaac Lab sensor output (ProxyArray or tensor)."""
    return data.torch if hasattr(data, "torch") else data


def verify_camera(sim, camera, embodiment) -> None:
    """Render a few frames and validate RGB / metric depth tensors from the RGB-D camera."""
    import torch

    # RTX data needs a few render frames after the sim starts before it becomes valid.
    for _ in range(8):
        sim.render()
    camera.update(sim.get_physics_dt(), force_recompute=True)

    rgb = _to_torch(camera.data.output["rgb"])
    depth = _to_torch(camera.data.output["distance_to_image_plane"])
    logger.info(f"Camera rgb: shape={tuple(rgb.shape)} dtype={rgb.dtype}; depth: shape={tuple(depth.shape)} dtype={depth.dtype}")

    shape_ok = rgb.shape[1:3] == (360, 640) and depth.shape[1:3] == (360, 640)
    logger.check("Camera resolution is 640x360", shape_ok, f"rgb={tuple(rgb.shape)}, depth={tuple(depth.shape)}")
    assert shape_ok, "Unexpected camera output resolution"

    rgb_ok = rgb.dtype == torch.uint8 and int(rgb[0, ..., :3].max()) > 0
    logger.check("RGB is uint8 and non-black", rgb_ok, f"max={int(rgb[0, ..., :3].max())}")
    assert rgb_ok, "RGB image is empty or not uint8"

    valid = torch.isfinite(depth) & (depth > 0)
    valid_fraction = float(valid.float().mean())
    depth_ok = valid_fraction > 0.05
    logger.check(
        "Depth has valid metric returns", depth_ok, f"valid_fraction={valid_fraction:.2f}, max_valid={float(depth[valid].max()):.2f} m"
    )
    assert depth_ok, f"Depth camera has too few valid pixels ({valid_fraction:.2f}); check orientation/rendering"

    # The ground-plane scene guarantees a ground return below the horizon within the clipping range.
    near = float(depth[valid].min())
    assert 0.1 < near < 5.0, f"Implausible nearest depth {near:.3f} m for a camera {embodiment.camera_offset[2]:.2f} m above ground"
    logger.check("Nearest ground return is plausible", True, f"min_depth={near:.2f} m")


def run_verification(simulation_app):
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, AssetBaseCfg
    from isaaclab.managers import ActionManager
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sensors import Camera, RayCaster
    from isaaclab.sim.utils.stage import get_current_stage
    from isaaclab.utils.configclass import configclass

    from nav_arena.embodiments import (
        create_2d_lidar_cfg,
        create_embodiment_camera_cfg,
        get_embodiment,
    )
    from nav_arena.ros2 import build_ros2_omnigraph

    embodiment = get_embodiment(args_cli.robot)
    body_prim = f"/World/envs/env_0/Robot/{embodiment.body_link}"

    @configclass
    class EmbodimentVerifySceneCfg(InteractiveSceneCfg):
        """Minimal verification scene with ground plane, lighting, and the selected embodiment."""

        # Ground plane
        ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

        # Lighting
        dome_light = AssetBaseCfg(
            prim_path="/World/defaultDomeLight",
            spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8)),
        )

        # Embodiment articulation
        robot = embodiment.articulation_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # 2D LiDAR
        lidar = create_2d_lidar_cfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{embodiment.body_link}",
            mesh_prim_paths=["/World/defaultGroundPlane"],
            mount_pos=embodiment.lidar_offset,
        )

        # Optional RGB-D camera (None entries are skipped by InteractiveScene)
        camera = create_embodiment_camera_cfg(embodiment) if args_cli.camera else None

    @configclass
    class EmbodimentActionCfg:
        """Action configuration for the embodiment's differential drive."""

        robot_action = embodiment.action_cfg.replace(asset_name="robot")


    logger.section(f"VERIFYING {embodiment.name.upper()} EMBODIMENT & SENSORS")
    logger.info("Creating SimulationContext...")
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)

    logger.info("Setting up verification scene...")
    scene_cfg = EmbodimentVerifySceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    # Embodiments whose USD needs runtime fixes (stage_patch_fn) must be patched after the robot is spawned and
    # before physics initializes in sim.reset(). The Dingo's fixes are baked into its derived USD instead.
    if embodiment.stage_patch_fn is not None:
        embodiment.stage_patch_fn(get_current_stage())
        logger.info(f"Applied '{embodiment.name}' stage patch")

    # ActionManager expects an environment object that holds .scene and .sim
    mock_env = create_mock_env(scene, sim)

    # Initialize ROS 2 OmniGraph bridge before starting simulation timeline
    logger.info("Initializing ROS 2 OmniGraph bridge...")
    build_ros2_omnigraph(
        robot_prim_path=body_prim,
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
                cmd = [0.6, 0.0, 0.0]
            elif phase == 1:
                cmd = [0.0, 0.0, 1.0]
            elif phase == 2:
                cmd = [-0.4, 0.0, 0.0]
            else:
                cmd = [0.0, 0.0, -1.0]

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
    # Test driving forward with body twist [vx=0.5 m/s, vy=0.0 m/s, wz=0.0 rad/s]
    action = torch.tensor([[0.5, 0.0, 0.0]], device=sim.device)

    logger.info("Stepping simulation with twist [vx=0.5 m/s, vy=0.0 m/s, wz=0.0 rad/s] for 60 steps...")
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

    if args_cli.camera:
        verify_camera(sim, scene["camera"], embodiment)

    logger.section(f"{embodiment.name.upper()} EMBODIMENT VERIFICATION PASSED SUCCESSFULLY!")


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
