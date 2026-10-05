# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PointNav task environment and configuration."""

from __future__ import annotations

import math
from typing import Any, Callable, Sequence

import numpy as np
import torch

import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.commands import UniformPose2dCommandCfg
from isaaclab.managers import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.configclass import configclass

from nav_arena.embodiments import NOVA_CARTER_ACTION_CFG, get_embodiment
from nav_arena.scenes import (
    DEFAULT_INTERIOR_AGENT_SCENE_ID,
    InteriorAgentSceneCfg,
    create_interior_agent_scene_cfg,
)


##
# Custom MDP Terminations & Metrics
##


def pose_goal_reached(
    env: ManagerBasedRLEnv,
    command_name: str = "pose_2d_command",
    threshold: float = 0.5,
) -> torch.Tensor:
    """Terminate when the robot is within threshold Euclidean distance to the 2D goal pose.

    Args:
        env: Manager-based RL environment instance.
        command_name: Name of the active command term in command_manager.
        threshold: Euclidean distance tolerance in meters.

    Returns:
        Boolean tensor of shape (num_envs,) indicating goal reached condition.
    """
    command_term = env.command_manager.get_term(command_name)
    robot = command_term.robot
    # Compute horizontal (XY) Euclidean distance in world frame
    delta_pos = command_term.pos_command_w[:, :2] - robot.data.root_pos_w[:, :2]
    dist = torch.linalg.norm(delta_pos, dim=1)
    return dist < threshold


def lateral_contact(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Terminate when the horizontal contact force on the sensor bodies exceeds a threshold.

    For robots whose body link also carries a ground-contacting part (e.g. the Dingo's caster sphere), the floor's
    vertical support force would trip a net-force check immediately. Walls and obstacles act through horizontal
    normals, so only the world-frame XY force components are considered.

    Args:
        env: Manager-based RL environment instance.
        threshold: Lateral force magnitude in Newtons.
        sensor_cfg: Contact sensor entity.

    Returns:
        Boolean tensor of shape (num_envs,).
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history.torch[:, :, sensor_cfg.body_ids]
    lateral = torch.linalg.norm(forces[..., :2], dim=-1)
    return torch.any(torch.max(lateral, dim=1)[0] > threshold, dim=1)


def apply_embodiment_stage_patch(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | None,
    patch_fn: Callable[[Any], None],
) -> None:
    """Prestartup event: apply an embodiment's USD stage patch after spawning and before physics starts."""
    patch_fn(env.sim.stage)


##
# Environment Scene Configuration
##


@configclass
class PointNavSceneCfg(InteriorAgentSceneCfg):
    """PointNav scene extending InteriorAgentSceneCfg with chassis contact sensing."""

    # Contact sensor for illegal contact / collision detection on the robot chassis
    contact_forces: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/chassis_link",
        update_period=0.0,
        history_length=3,
        debug_vis=False,
    )


##
# MDP Terms Configuration
##


@configclass
class ActionsCfg:
    """Action specifications for PointNav differential drive."""

    robot_action = NOVA_CARTER_ACTION_CFG.replace(asset_name="robot")


@configclass
class CommandsCfg:
    """Command specifications for PointNav 2D target pose."""

    pose_2d_command = UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        resampling_time_range=(100000.0, 100000.0),
        debug_vis=True,
        ranges=UniformPose2dCommandCfg.Ranges(
            pos_x=(-1.0, -1.0),
            pos_y=(0.0, 0.0),
            heading=(0.0, 0.0),
        ),
    )


@configclass
class ObservationsCfg:
    """Observation specifications for PointNav."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Observation terms for evaluation / state tracking."""

        robot_pos = ObservationTermCfg(func=mdp.root_pos_w)
        robot_quat = ObservationTermCfg(func=mdp.root_quat_w)
        robot_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel)
        robot_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel)

    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    """Termination terms for PointNav evaluation metrics."""

    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    tipped = TerminationTermCfg(
        func=mdp.bad_orientation,
        params={"limit_angle": 0.6},
        time_out=False,
    )
    collision = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces"), "threshold": 1.0},
        time_out=False,
    )
    goal = TerminationTermCfg(
        func=pose_goal_reached,
        params={"command_name": "pose_2d_command", "threshold": 0.5},
        time_out=False,
    )


@configclass
class RewardsCfg:
    """Reward terms (minimal placeholder for ManagerBasedRLEnv compatibility)."""

    is_alive = RewardTermCfg(func=mdp.is_alive, weight=1.0)


@configclass
class EventCfg:
    """Event configuration for episode resets."""

    reset_scene = EventTermCfg(func=mdp.reset_scene_to_default, mode="reset")
    # Set by create_point_nav_env_cfg for embodiments that need a USD stage patch (e.g. Dingo caster friction)
    embodiment_stage_patch: EventTermCfg | None = None


##
# Full Environment Configuration
##


@configclass
class PointNavEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for PointNav evaluation environment."""

    robot_name: str = "nova_carter"
    """Registered embodiment this environment was configured for."""
    scene: PointNavSceneCfg = PointNavSceneCfg(num_envs=1, env_spacing=10.0)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    commands: CommandsCfg = CommandsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post-initialization to configure simulation and control step frequencies."""
        self.decimation = 2
        self.episode_length_s = 60.0
        self.sim.dt = 0.01
        # Set to 3 for ~33.3 FPS smooth viewport interaction while remaining 3x lighter than baseline
        self.sim.render_interval = 3


def create_point_nav_env_cfg(
    scene_id_or_path: str = DEFAULT_INTERIOR_AGENT_SCENE_ID,
    robot_spawn_pos: tuple[float, float, float] = (-2.5, 0.0, 0.25),
    robot_spawn_rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    goal_pos: tuple[float, float] = (-1.0, 0.0),
    goal_heading: float = 0.0,
    goal_threshold: float = 0.5,
    collision_threshold: float = 1.0,
    episode_length_s: float = 60.0,
    num_envs: int = 1,
    open_doors: bool = True,
    robot_name: str = "nova_carter",
    enable_camera: bool = False,
    enable_goal_camera: bool = False,
    render_interval: int | None = None,
    max_tilt: float = 0.6,
    show_goal_marker: bool = True,
    scene_queries: bool = False,
) -> PointNavEnvCfg:
    """Factory helper to create a fully customized PointNavEnvCfg.

    Args:
        scene_id_or_path: InteriorAgent scene identifier or custom USD path.
        robot_spawn_pos: (x, y, z) initial position of robot in meters.
        robot_spawn_rot: (x, y, z, w) initial quaternion orientation.
        goal_pos: (x, y) target coordinates in meters.
        goal_heading: Target heading in radians.
        goal_threshold: Distance in meters considered success.
        collision_threshold: Net contact force in Newtons considered a collision.
        episode_length_s: Episode duration timeout in seconds.
        num_envs: Number of parallel environments.
        open_doors: If True, uses the open-door delta layer to allow free passage.
        robot_name: Registered embodiment (e.g. ``"nova_carter"``, ``"dingo"``). Its articulation, action term,
            sensors, and optional USD stage patch are applied.
        enable_camera: Mount the embodiment's RGB-D camera. The app must be launched with cameras enabled.
        enable_goal_camera: Add a free-standing camera for rendering goal images (single environment only).
        render_interval: Physics steps between renders. ``None`` keeps 3 (~33 Hz viewport) without a camera and uses
            20 (5 Hz, the learned baselines' planning rate at ``dt=0.01``) when a camera is enabled, since every render
            then also pays for RTX sensor rendering.
        max_tilt: Roll/pitch magnitude in radians beyond which the robot counts as tipped over.
        show_goal_marker: Draw Isaac Lab's goal-pose arrow in the scene. The arrow is real geometry, so **cameras see
            it**: a depth planner treats it as an obstacle sitting on its own goal (iPlanner halted 3.5 m out because
            of it). Disable it whenever a policy consumes camera data.
        scene_queries: Enable PhysX scene queries (needed by the third-person follow camera's occlusion rays).

    Returns:
        Configured PointNavEnvCfg.
    """
    embodiment = get_embodiment(robot_name)
    env_cfg = PointNavEnvCfg()
    env_cfg.robot_name = robot_name
    env_cfg.scene = PointNavSceneCfg(num_envs=num_envs, env_spacing=10.0)

    # Configure scene asset and robot spawn
    base_scene = create_interior_agent_scene_cfg(
        scene_id_or_path=scene_id_or_path,
        robot_spawn_pos=robot_spawn_pos,
        robot_spawn_rot=robot_spawn_rot,
        num_envs=num_envs,
        open_doors=open_doors,
        robot_name=robot_name,
        enable_camera=enable_camera,
        enable_goal_camera=enable_goal_camera,
    )
    env_cfg.scene.scene_asset = base_scene.scene_asset
    env_cfg.scene.robot = base_scene.robot
    env_cfg.scene.lidar = base_scene.lidar
    env_cfg.scene.camera = base_scene.camera
    env_cfg.scene.goal_camera = base_scene.goal_camera

    # Bind the embodiment's action term and contact sensing to its USD body link
    env_cfg.actions.robot_action = embodiment.action_cfg.replace(asset_name="robot")
    env_cfg.scene.contact_forces = env_cfg.scene.contact_forces.replace(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{embodiment.body_link}"
    )

    # USD-level embodiment fixes must land after spawning and before physics starts. Isaac Lab only allows
    # prestartup events when physics replication is off (irrelevant for single-environment evaluation).
    if embodiment.stage_patch_fn is not None:
        env_cfg.scene.replicate_physics = False
        env_cfg.events.embodiment_stage_patch = EventTermCfg(
            func=apply_embodiment_stage_patch,
            mode="prestartup",
            params={"patch_fn": embodiment.stage_patch_fn},
        )


    # Configure hardcoded fixed goal command
    env_cfg.commands.pose_2d_command.ranges.pos_x = (goal_pos[0], goal_pos[0])
    env_cfg.commands.pose_2d_command.ranges.pos_y = (goal_pos[1], goal_pos[1])
    env_cfg.commands.pose_2d_command.ranges.heading = (goal_heading, goal_heading)

    # Configure metrics thresholds
    env_cfg.terminations.goal.params["threshold"] = goal_threshold
    env_cfg.terminations.collision.params["threshold"] = collision_threshold
    if embodiment.ground_contact_on_body:
        env_cfg.terminations.collision = TerminationTermCfg(
            func=lateral_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces"), "threshold": collision_threshold},
            time_out=False,
        )
    env_cfg.terminations.tipped.params["limit_angle"] = max_tilt
    env_cfg.commands.pose_2d_command.debug_vis = show_goal_marker
    if scene_queries:
        env_cfg.sim.enable_scene_query_support = True
    env_cfg.episode_length_s = episode_length_s
    if render_interval is not None:
        env_cfg.sim.render_interval = render_interval
    elif enable_camera:
        env_cfg.sim.render_interval = 20

    return env_cfg


##
# Task Wrapper Class
##


class PointNavTask(ManagerBasedRLEnv):
    """PointNav task wrapper around Isaac Lab's ManagerBasedRLEnv.

    Provides high-level helpers for inspecting task state, goals, and metrics.
    """

    cfg: PointNavEnvCfg

    def __init__(self, cfg: PointNavEnvCfg | None = None, render_mode: str | None = None, **kwargs):
        """Initialize PointNav task environment.

        Args:
            cfg: Environment configuration. Defaults to default PointNavEnvCfg.
            render_mode: Optional render mode ('rgb_array' or None).
        """
        if cfg is None:
            cfg = PointNavEnvCfg()
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

    @property
    def action_dim(self) -> int:
        """Width of the command tensor accepted by :meth:`step` (the body twist ``[vx, vy, wz]``)."""
        return int(self.action_manager.total_action_dim)

    def get_goal_pose_w(self, env_idx: int = 0) -> tuple[float, float, float]:
        """Get the current target 2D pose in world frame.

        Args:
            env_idx: Index of the environment.

        Returns:
            Tuple of (x, y, heading) in meters and radians.
        """
        command_term = self.command_manager.get_term("pose_2d_command")
        pos_w = command_term.pos_command_w[env_idx].cpu().numpy()
        heading_w = float(command_term.heading_command_w[env_idx].cpu().item())
        return float(pos_w[0]), float(pos_w[1]), heading_w

    def get_robot_pose_w(self, env_idx: int = 0) -> tuple[float, float, float]:
        """Get the current robot 2D pose in world frame.

        Args:
            env_idx: Index of the environment.

        Returns:
            Tuple of (x, y, heading) in meters and radians.
        """
        robot = self.scene["robot"]
        pos_w = robot.data.root_pos_w[env_idx].cpu().numpy()
        heading_w = float(robot.data.heading_w[env_idx].cpu().item())
        return float(pos_w[0]), float(pos_w[1]), heading_w

    def get_goal_distance(self, env_idx: int = 0) -> float:
        """Get the Euclidean distance from robot to goal.

        Args:
            env_idx: Index of the environment.

        Returns:
            Distance in meters.
        """
        command_term = self.command_manager.get_term("pose_2d_command")
        robot = command_term.robot
        delta_pos = command_term.pos_command_w[env_idx, :2] - robot.data.root_pos_w[env_idx, :2]
        dist = torch.linalg.norm(delta_pos).cpu().item()
        return float(dist)

    def is_goal_reached(self, env_idx: int = 0) -> bool:
        """Check if goal reached termination triggered this step."""
        return bool(self.termination_manager.get_term("goal")[env_idx].cpu().item())

    def is_collision(self, env_idx: int = 0) -> bool:
        """Check if collision termination triggered this step."""
        return bool(self.termination_manager.get_term("collision")[env_idx].cpu().item())

    def is_timed_out(self, env_idx: int = 0) -> bool:
        """Check if time_out termination triggered this step."""
        return bool(self.termination_manager.get_term("time_out")[env_idx].cpu().item())

    def is_tipped(self, env_idx: int = 0) -> bool:
        """Check if the robot tipped over (tilt beyond ``max_tilt``) this step."""
        return bool(self.termination_manager.get_term("tipped")[env_idx].cpu().item())

    def get_terminal_cause(self, env_idx: int = 0) -> str | None:
        """Why the episode ended this step: ``goal_reached``, ``collision``, ``tipped``, ``time_out``, or None."""
        for cause, check in (
            ("goal_reached", self.is_goal_reached),
            ("collision", self.is_collision),
            ("tipped", self.is_tipped),
            ("time_out", self.is_timed_out),
        ):
            if check(env_idx):
                return cause
        return None

    def get_robot_position_w(self, env_idx: int = 0) -> np.ndarray:
        """Robot root position in the world frame, shape (3,) in meters."""
        return self.scene["robot"].data.root_pos_w[env_idx].cpu().numpy().astype(np.float64)

    def get_robot_quat_w(self, env_idx: int = 0) -> np.ndarray:
        """Robot root orientation in the world frame as ``(x, y, z, w)``, shape (4,)."""
        return self.scene["robot"].data.root_quat_w[env_idx].cpu().numpy().astype(np.float64)

    def get_camera_intrinsics(self, env_idx: int = 0) -> np.ndarray:
        """Intrinsic matrix ``[3, 3]`` of the robot's RGB-D camera in pixels."""
        camera = self._get_camera("camera")
        return _to_tensor(camera.data.intrinsic_matrices)[env_idx].cpu().numpy().astype(np.float64)

    def get_camera_frame(self, env_idx: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Latest RGB-D frame from the robot camera.

        The frame is as fresh as the last render (see ``render_interval``).

        Returns:
            ``(rgb, depth)``: RGB uint8 ``[H, W, 3]`` and metric image-plane depth float32 ``[H, W]``
            (``inf`` beyond the clipping range).
        """
        camera = self._get_camera("camera")
        output = camera.data.output
        rgb = _to_tensor(output["rgb"])[env_idx, :, :, :3].cpu().numpy().copy()
        depth = _to_tensor(output["distance_to_image_plane"])[env_idx, :, :, 0].cpu().numpy().astype(np.float32)
        return rgb, depth

    def refresh_camera_frame(self, env_idx: int = 0, render_frames: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Render now and return the freshest RGB-D frame, independent of ``render_interval`` phase.

        Planners that run at a few Hz call this at planning time so their input is never stale, while the physics
        loop keeps rendering at the (cheaper) ``render_interval`` cadence in between.

        Returns:
            ``(rgb, depth)`` as in :meth:`get_camera_frame`.
        """
        camera = self._get_camera("camera")
        for _ in range(render_frames):
            self.sim.render()
        camera.update(self.step_dt, force_recompute=True)
        return self.get_camera_frame(env_idx)

    def render_goal_image(
        self,
        goal_xy: Sequence[float],
        yaw: float | None = None,
        height: float | None = None,
        env_idx: int = 0,
        render_frames: int = 4,
    ) -> np.ndarray:
        """Render the scene as seen from a goal pose, for image-goal policies (ViNT, NavDP image-goal).

        A separate virtual camera is moved to the goal; the robot is not teleported.

        Args:
            goal_xy: Goal (x, y) in the world frame.
            yaw: Viewing heading in radians. Defaults to the bearing from the robot's current position to the goal.
            height: Camera height above the world origin plane. Defaults to the robot's current root height plus
                the embodiment's camera mount height.
            env_idx: Environment index (the goal camera exists for a single environment only).
            render_frames: Render passes before reading, as RTX output needs a few frames to settle.

        Returns:
            RGB uint8 ``[H, W, 3]``.
        """
        camera = self._get_camera("goal_camera")
        robot_pos = self.get_robot_position_w(env_idx)
        gx, gy = float(goal_xy[0]), float(goal_xy[1])
        if yaw is None:
            yaw = math.atan2(gy - robot_pos[1], gx - robot_pos[0])
        if height is None:
            height = float(robot_pos[2]) + float(get_embodiment(self.cfg.robot_name).camera_offset[2])
        eye = [gx, gy, height]
        target = [gx + math.cos(yaw), gy + math.sin(yaw), height]
        camera.set_world_poses_from_view(eyes=torch.tensor([eye]), targets=torch.tensor([target]))
        for _ in range(render_frames):
            self.sim.render()
        camera.update(self.physics_dt, force_recompute=True)
        return _to_tensor(camera.data.output["rgb"])[0, :, :, :3].cpu().numpy().copy()

    def _get_camera(self, name: str):
        try:
            return self.scene[name]
        except KeyError as exc:
            flag = "enable_goal_camera" if name == "goal_camera" else "enable_camera"
            raise RuntimeError(f"Scene has no '{name}'; create the env with create_point_nav_env_cfg({flag}=True)") from exc


def _to_tensor(data: Any) -> torch.Tensor:
    """Torch view of Isaac Lab sensor output (ProxyArray or tensor)."""
    return data.torch if hasattr(data, "torch") else data
