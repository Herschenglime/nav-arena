# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action terms for mobile navigation embodiments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class DifferentialDriveAction(ActionTerm):
    """Action term that converts linear/angular velocity commands into differential wheel velocities."""

    cfg: DifferentialDriveActionCfg
    _asset: Articulation

    def __init__(self, cfg: DifferentialDriveActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # Resolve joint IDs from articulation
        left_joint_ids, left_joint_names = self._asset.find_joints(self.cfg.left_wheel_joint_name)
        if len(left_joint_ids) != 1:
            raise ValueError(
                f"Expected 1 matching joint for left wheel '{self.cfg.left_wheel_joint_name}', "
                f"got {len(left_joint_ids)}: {left_joint_names}"
            )
        right_joint_ids, right_joint_names = self._asset.find_joints(self.cfg.right_wheel_joint_name)
        if len(right_joint_ids) != 1:
            raise ValueError(
                f"Expected 1 matching joint for right wheel '{self.cfg.right_wheel_joint_name}', "
                f"got {len(right_joint_ids)}: {right_joint_names}"
            )

        self._left_wheel_idx = left_joint_ids[0]
        self._right_wheel_idx = right_joint_ids[0]
        self._wheel_joint_ids = [self._left_wheel_idx, self._right_wheel_idx]

        # Preallocate action tensors
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._wheel_vel_targets = torch.zeros(self.num_envs, 2, device=self.device)

        # Scale and offset tensors
        self._scale = torch.tensor(self.cfg.scale, device=self.device).unsqueeze(0)
        self._offset = torch.tensor(self.cfg.offset, device=self.device).unsqueeze(0)

        self._wheel_radius = self.cfg.wheel_radius
        self._wheel_base = self.cfg.wheel_base

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions = self._raw_actions * self._scale + self._offset

        # Clip commands
        self._processed_actions[:, 0] = torch.clamp(
            self._processed_actions[:, 0], -self.cfg.max_linear_speed, self.cfg.max_linear_speed
        )
        self._processed_actions[:, 1] = torch.clamp(
            self._processed_actions[:, 1], -self.cfg.max_angular_speed, self.cfg.max_angular_speed
        )

    def apply_actions(self):
        v = self._processed_actions[:, 0]
        omega = self._processed_actions[:, 1]

        # Differential drive inverse kinematics
        # v_left = (v - omega * L / 2) / R
        # v_right = (v + omega * L / 2) / R
        v_left = (v - omega * (self._wheel_base / 2.0)) / self._wheel_radius
        v_right = (v + omega * (self._wheel_base / 2.0)) / self._wheel_radius

        self._wheel_vel_targets[:, 0] = v_left
        self._wheel_vel_targets[:, 1] = v_right

        self._asset.set_joint_velocity_target_index(
            target=self._wheel_vel_targets,
            joint_ids=self._wheel_joint_ids,
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self._raw_actions.zero_()
            self._processed_actions.zero_()
            self._wheel_vel_targets.zero_()
        else:
            self._raw_actions[env_ids] = 0.0
            self._processed_actions[env_ids] = 0.0
            self._wheel_vel_targets[env_ids] = 0.0


@configclass
class DifferentialDriveActionCfg(ActionTermCfg):
    """Configuration for a differential drive action term.

    Maps a 2D action ``[v, omega]`` (forward linear velocity in m/s and angular
    yaw rate in rad/s) into target angular velocities for the left and right wheels:

    .. math::
        \\dot{q}_{\\text{left}} = \\frac{v - \\omega \\cdot L / 2}{R}
        \\dot{q}_{\\text{right}} = \\frac{v + \\omega \\cdot L / 2}{R}

    where :math:`L` is the wheel base (track width) and :math:`R` is the wheel radius.
    """

    class_type: type[DifferentialDriveAction] = DifferentialDriveAction

    asset_name: str = "robot"
    """Name of the articulation asset in the scene."""

    left_wheel_joint_name: str = MISSING
    """Name or regex expression for the left drive wheel joint."""

    right_wheel_joint_name: str = MISSING
    """Name or regex expression for the right drive wheel joint."""

    wheel_radius: float = 0.14
    """Wheel radius in meters. Defaults to 0.14 m (Nova Carter)."""

    wheel_base: float = 0.413
    """Distance between left and right wheels in meters. Defaults to 0.413 m (Nova Carter)."""

    scale: tuple[float, float] = (1.0, 1.0)
    """Scaling factor applied to raw actions [v_scale, omega_scale]."""

    offset: tuple[float, float] = (0.0, 0.0)
    """Offset added to raw actions [v_offset, omega_offset]."""

    max_linear_speed: float = 2.0
    """Maximum linear velocity clip (m/s)."""

    max_angular_speed: float = 3.0
    """Maximum angular velocity clip (rad/s)."""
