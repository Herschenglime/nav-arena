# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PointNav task environment and configuration."""

from __future__ import annotations

import math
from typing import Sequence
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

from nav_arena.embodiments import NOVA_CARTER_ACTION_CFG
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


##
# Full Environment Configuration
##


@configclass
class PointNavEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for PointNav evaluation environment."""

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
        self.sim.render_interval = self.decimation


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

    Returns:
        Configured PointNavEnvCfg.
    """
    env_cfg = PointNavEnvCfg()
    env_cfg.scene = PointNavSceneCfg(num_envs=num_envs, env_spacing=10.0)

    # Configure scene asset and robot spawn
    base_scene = create_interior_agent_scene_cfg(
        scene_id_or_path=scene_id_or_path,
        robot_spawn_pos=robot_spawn_pos,
        robot_spawn_rot=robot_spawn_rot,
        num_envs=num_envs,
    )
    env_cfg.scene.scene_asset = base_scene.scene_asset
    env_cfg.scene.robot = base_scene.robot
    env_cfg.scene.lidar = base_scene.lidar

    # Configure hardcoded fixed goal command
    env_cfg.commands.pose_2d_command.ranges.pos_x = (goal_pos[0], goal_pos[0])
    env_cfg.commands.pose_2d_command.ranges.pos_y = (goal_pos[1], goal_pos[1])
    env_cfg.commands.pose_2d_command.ranges.heading = (goal_heading, goal_heading)

    # Configure metrics thresholds
    env_cfg.terminations.goal.params["threshold"] = goal_threshold
    env_cfg.terminations.collision.params["threshold"] = collision_threshold
    env_cfg.episode_length_s = episode_length_s

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
