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


def verify_holonomic_reference(robot, action_term, embodiment) -> None:
    """Check our wheel speeds against Isaac Sim's own HolonomicController, built the way its Kaya example builds it.

    The reference reads the wheel layout from the USD with Isaac Sim's HolonomicRobotUsdSetup (independent of our
    reader) and takes the command at ``base_link/control_offset``, the same point as our action term's command frame.
    """
    import numpy as np
    import warp as wp

    import isaaclab.sim as sim_utils
    import omni.kit.app

    # Pure-Python extensions (no USD schemas), so enabling them after boot is fine (unlike omni.graph / ROS 2).
    manager = omni.kit.app.get_app().get_extension_manager()
    for extension in ("isaacsim.robot_motion.controllers", "isaacsim.robot.experimental.wheeled_robots"):
        manager.set_extension_enabled_immediate(extension, True)
    import isaacsim.robot_motion.controllers as motion_controllers
    import isaacsim.robot_motion.experimental.motion_generation as mg
    from isaacsim.robot.experimental.wheeled_robots.robots import HolonomicRobotUsdSetup

    from nav_arena.embodiments.kinematics import holonomic_ik

    root = sim_utils.find_first_matching_prim(robot.cfg.prim_path).GetPath().pathString
    setup = HolonomicRobotUsdSetup(robot_prim_path=root, com_prim_path=f"{root}/{embodiment.body_link}/control_offset")
    radius, positions, orientations, angles, wheel_axis, up_axis = setup.get_holonomic_controller_params()
    wheel_names = list(setup.get_articulation_controller_params())
    controller = motion_controllers.HolonomicController(
        robot_joint_space=list(robot.joint_names),
        wheel_joint_names=wheel_names,
        wheel_radius=radius,
        wheel_positions=positions,
        wheel_orientations=orientations,
        mecanum_angles=angles,
        wheel_axis=wheel_axis,
        rotation_direction=up_axis,
        device="cpu",
    )
    ours_order = list(action_term.cfg.wheel_joint_names)
    twists = [(0.3, 0.0, 0.0), (0.0, 0.3, 0.0), (0.0, 0.0, 1.0), (0.2, -0.1, 0.5), (-0.25, 0.15, -0.8)]
    worst = 0.0
    for twist in twists:
        setpoint = mg.RobotState(
            sites=mg.SpatialState.from_name(
                spatial_space=["control_point"],
                linear_velocities=(["control_point"], wp.array([[twist[0], twist[1], 0.0]], dtype=wp.float32, device="cpu")),
                angular_velocities=(["control_point"], wp.array([[0.0, 0.0, twist[2]]], dtype=wp.float32, device="cpu")),
            )
        )
        desired = controller.forward(mg.RobotState(), setpoint, 0.0)
        reference = dict(zip(desired.joints.velocity_names, np.asarray(desired.joints.velocities.numpy()).reshape(-1)))
        ours = holonomic_ik(torch.tensor([twist], dtype=torch.float32), action_term.wheel_matrix.cpu())[0]
        for name, value in zip(ours_order, ours.tolist()):
            worst = max(worst, abs(value - float(reference[name])))
    passed = worst < 1e-3
    logger.check("Wheel speeds match Isaac Sim's HolonomicController", passed, f"max |difference| = {worst:.2e} rad/s over {len(twists)} twists")
    assert passed, f"Our holonomic kinematics disagree with Isaac Sim's HolonomicController by {worst:.3e} rad/s"


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
        robot = embodiment.articulation_cfg.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=embodiment.articulation_cfg.init_state.replace(pos=(0.0, 0.0, embodiment.spawn_height)),
        )

        # 2D LiDAR
        lidar = create_2d_lidar_cfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{embodiment.body_link}",
            mesh_prim_paths=["/World/defaultGroundPlane"],
            mount_pos=embodiment.lidar_offset,
            ray_alignment=embodiment.lidar_ray_alignment,
        )

        # Optional RGB-D camera (None entries are skipped by InteractiveScene)
        camera = create_embodiment_camera_cfg(embodiment) if args_cli.camera else None

    @configclass
    class EmbodimentActionCfg:
        """Action configuration for the embodiment's differential drive."""

        robot_action = embodiment.action_cfg.replace(asset_name="robot")


    logger.section(f"VERIFYING {embodiment.name.upper()} EMBODIMENT & SENSORS")
    logger.info("Creating SimulationContext...")
    sim_cfg = sim_utils.SimulationCfg(dt=embodiment.sim_dt)
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

    # Drive checks (per axis, driven by the embodiment's limits). The pass/fail rules live in nav_arena.utils.drive_check.
    from nav_arena.embodiments.base import is_policy_driven
    from nav_arena.utils.drive_check import (
        evaluate_hold,
        evaluate_segment,
        measure_segment,
        settle_until_still,
        yaw_from_quat_xyzw,
    )

    dt = sim.get_physics_dt()

    @torch.inference_mode()  # as Isaac Lab's play scripts do; a policy-driven action term needs it
    def run(command, seconds):
        action_t = torch.tensor([command], device=sim.device)
        for _ in range(max(1, round(seconds / dt))):
            action_manager.process_action(action_t)
            action_manager.apply_action()
            scene.write_data_to_sim()
            sim.step()
            scene.update(dt=dt)

    def pose():
        position = robot.data.root_pos_w[0]
        return (float(position[0]), float(position[1])), yaw_from_quat_xyzw([float(v) for v in robot.data.root_quat_w[0]])

    def segment(command, warmup_s=0.5, measure_s=1.0, brake_s=0.5):
        run(command, warmup_s)
        start_xy, start_yaw = pose()
        run(command, measure_s / 2.0)
        mid_xy, mid_yaw = pose()
        run(command, measure_s / 2.0)
        end_xy, end_yaw = pose()
        run([0.0, 0.0, 0.0], brake_s)
        return measure_segment(command, start_xy, start_yaw, end_xy, end_yaw, measure_s, mid_xy, mid_yaw)

    settle_s = settle_until_still(lambda: run([0.0, 0.0, 0.0], dt), lambda: (*pose()[0], pose()[1]), dt)
    logger.check("Robot settles after spawn", settle_s is not None, f"still after {settle_s:.2f} s" if settle_s else "still moving after 8 s")
    assert settle_s is not None, "Robot never came to rest after spawning"
    passed, detail = evaluate_hold(segment([0.0, 0.0, 0.0], warmup_s=0.0, brake_s=0.0))
    logger.check("Zero-command hold", passed, detail)
    assert passed, f"Robot moved without a command: {detail}"

    if is_policy_driven(embodiment.drive_type):
        # A locomotion policy can wander or tip over slowly while "standing": hold 10 s, stay put and stay up.
        stand_height = float(robot.data.root_pos_w[0][2])
        passed, detail = evaluate_hold(
            segment([0.0, 0.0, 0.0], warmup_s=0.0, measure_s=10.0, brake_s=0.0), max_translation=0.1, max_yaw=0.1
        )
        height = float(robot.data.root_pos_w[0][2])
        upright = height > 0.8 * stand_height
        passed = passed and upright
        detail += f", base height {stand_height:.3f} -> {height:.3f} m"
        logger.check("Stands for 10 s under a zero command", passed, detail)
        assert passed, f"Legged robot did not stand still: {detail}"

    drive_name = type(action_manager._terms["robot_action"]).__name__
    commands = [("Forward drive (vx)", [min(0.3, embodiment.max_linear_speed), 0.0, 0.0])]
    if embodiment.max_lateral_speed > 0.0:
        commands.append(("Strafe drive (vy)", [0.0, min(0.3, embodiment.max_lateral_speed), 0.0]))
    commands.append(("In-place rotation (wz)", [0.0, 0.0, min(1.0, embodiment.max_angular_speed)]))
    if embodiment.drive_type == "holonomic":
        verify_holonomic_reference(robot, action_manager._terms["robot_action"], embodiment)

    for label, command in commands:
        # A rotation is measured after a longer spin-up: swivel casters (Nova Carter) re-orient at the start of a turn,
        # moving the turning centre by ~8 cm over the first second; after 1.5 s it holds within millimetres.
        warmup_s = 1.5 if command[2] != 0.0 else 0.5
        passed, detail = evaluate_segment(segment(command, warmup_s=warmup_s))
        logger.check(f"{drive_name} {label}", passed, detail)
        assert passed, f"{label} failed: {detail}"

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
