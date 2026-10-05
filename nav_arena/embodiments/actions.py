# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action terms for mobile navigation embodiments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING, Any

import torch

from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils.configclass import configclass

from nav_arena.embodiments.kinematics import diff_drive_ik, holonomic_ik
from nav_arena.embodiments.wheel_geometry import read_wheel_geometry

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class DifferentialDriveAction(ActionTerm):
    """Action term that converts a body twist command into differential wheel velocities.

    The command is the framework-wide body twist ``[vx, vy, wz]``. A differential drive cannot move sideways, so
    ``vy`` is accepted (keeping the command width identical across drive types) and ignored.
    """

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
        return 3

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
        self._processed_actions[:, 1] = 0.0  # a differential drive has no lateral velocity
        self._processed_actions[:, 2] = torch.clamp(
            self._processed_actions[:, 2], -self.cfg.max_angular_speed, self.cfg.max_angular_speed
        )

    def apply_actions(self):
        v = self._processed_actions[:, 0]
        omega = self._processed_actions[:, 2]

        v_left, v_right = diff_drive_ik(
            v=v,
            omega=omega,
            wheel_base=self._wheel_base,
            wheel_radius=self._wheel_radius,
        )

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

    Maps a body twist action ``[vx, vy, omega]`` (forward velocity in m/s, lateral velocity in m/s, and yaw rate in
    rad/s) into target angular velocities for the left and right wheels. ``vy`` is ignored. With :math:`v = v_x`:

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

    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Scaling factor applied to raw actions [vx_scale, vy_scale, omega_scale]."""

    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Offset added to raw actions [vx_offset, vy_offset, omega_offset]."""

    max_linear_speed: float = 2.0
    """Maximum linear velocity clip (m/s)."""

    max_angular_speed: float = 3.0
    """Maximum angular velocity clip (rad/s)."""


class HolonomicDriveAction(ActionTerm):
    """Action term that converts a body twist command into omni or mecanum wheel velocities.

    The command is the body twist ``[vx, vy, wz]`` at the chassis frame the wheel joints attach to. The wheel layout is
    read from the robot's USD (``isaacmecanumwheel:*`` attributes and joint frames) unless given in the config, and
    turned into a twist-to-wheel matrix once at construction. Passive roller joints are left alone.
    """

    cfg: HolonomicDriveActionCfg
    _asset: Articulation

    def __init__(self, cfg: HolonomicDriveActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        names = list(self.cfg.wheel_joint_names)
        joint_ids, joint_names = self._asset.find_joints(names, preserve_order=True)
        if len(joint_ids) != len(names):
            raise ValueError(f"Expected {len(names)} wheel joints matching {names}, got {len(joint_ids)}: {joint_names}")
        self._wheel_joint_ids = list(joint_ids)

        geometry = self.cfg.wheel_geometry
        if geometry is None:
            import isaaclab.sim as sim_utils

            root = sim_utils.find_first_matching_prim(self._asset.cfg.prim_path)
            if root is None:
                raise ValueError(f"Could not find the robot prim for '{self._asset.cfg.prim_path}' to read the wheel layout")
            geometry = read_wheel_geometry(sim_utils.get_current_stage(), root.GetPath().pathString, joint_names)
        self._matrix = geometry.matrix().to(self.device)

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._wheel_vel_targets = torch.zeros(self.num_envs, len(names), device=self.device)
        self._scale = torch.tensor(self.cfg.scale, device=self.device).unsqueeze(0)
        self._offset = torch.tensor(self.cfg.offset, device=self.device).unsqueeze(0)
        self._limits = torch.tensor(
            [self.cfg.max_linear_speed, self.cfg.max_lateral_speed, self.cfg.max_angular_speed], device=self.device
        ).unsqueeze(0)

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def wheel_matrix(self) -> torch.Tensor:
        """The ``[N, 3]`` twist-to-wheel-velocity matrix in use."""
        return self._matrix

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions = torch.clamp(self._raw_actions * self._scale + self._offset, -self._limits, self._limits)

    def apply_actions(self):
        self._wheel_vel_targets[:] = holonomic_ik(self._processed_actions, self._matrix)
        self._asset.set_joint_velocity_target_index(target=self._wheel_vel_targets, joint_ids=self._wheel_joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._wheel_vel_targets[env_ids] = 0.0


@configclass
class HolonomicDriveActionCfg(ActionTermCfg):
    """Configuration for an omni or mecanum drive action term.

    Maps a body twist action ``[vx, vy, omega]`` (m/s, m/s, rad/s) into target angular velocities for the wheel joints
    using the layout of the wheels (see :func:`nav_arena.embodiments.kinematics.holonomic_matrix`).
    """

    class_type: type[HolonomicDriveAction] = HolonomicDriveAction

    asset_name: str = "robot"
    """Name of the articulation asset in the scene."""

    wheel_joint_names: tuple[str, ...] = MISSING
    """Exact names of the driven wheel joints. The roller joints of omni/mecanum wheels are not listed."""

    wheel_geometry: Any = None
    """A :class:`WheelGeometry`, or None to read the layout from the robot's USD."""

    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Scaling factor applied to raw actions [vx_scale, vy_scale, omega_scale]."""

    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Offset added to raw actions [vx_offset, vy_offset, omega_offset]."""

    max_linear_speed: float = 1.0
    """Maximum forward velocity clip (m/s)."""

    max_lateral_speed: float = 1.0
    """Maximum sideways velocity clip (m/s)."""

    max_angular_speed: float = 3.0
    """Maximum yaw rate clip (rad/s)."""
