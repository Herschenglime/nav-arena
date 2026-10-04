# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Differential drive kinematics for mobile navigation embodiments."""

from __future__ import annotations

import torch


def diff_drive_ik(
    v: torch.Tensor | float,
    omega: torch.Tensor | float,
    wheel_base: float,
    wheel_radius: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differential drive inverse kinematics.

    Computes left and right wheel angular velocities (rad/s) from forward
    linear velocity `v` (m/s) and angular yaw rate `omega` (rad/s).

    Formula:
        dot_q_left  = (v - omega * L / 2) / R
        dot_q_right = (v + omega * L / 2) / R

    Args:
        v: Linear velocity in m/s (tensor or float).
        omega: Angular velocity in rad/s (tensor or float).
        wheel_base: Distance L between the left and right wheels in meters.
        wheel_radius: Radius R of the wheels in meters.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (wheel_vel_left, wheel_vel_right) in rad/s.
    """
    if wheel_radius <= 0.0:
        raise ValueError(f"wheel_radius must be positive, got {wheel_radius}")
    if wheel_base <= 0.0:
        raise ValueError(f"wheel_base must be positive, got {wheel_base}")

    if not isinstance(v, torch.Tensor):
        v = torch.as_tensor(v, dtype=torch.float32)
    if not isinstance(omega, torch.Tensor):
        omega = torch.as_tensor(omega, dtype=torch.float32)

    half_base = wheel_base / 2.0
    wheel_vel_left = (v - omega * half_base) / wheel_radius
    wheel_vel_right = (v + omega * half_base) / wheel_radius
    return wheel_vel_left, wheel_vel_right


def diff_drive_fk(
    wheel_vel_left: torch.Tensor | float,
    wheel_vel_right: torch.Tensor | float,
    wheel_base: float,
    wheel_radius: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differential drive forward kinematics.

    Computes forward linear velocity `v` (m/s) and angular yaw rate `omega` (rad/s)
    from left and right wheel angular velocities (rad/s).

    Formula:
        v = (R / 2) * (dot_q_right + dot_q_left)
        omega = (R / L) * (dot_q_right - dot_q_left)

    Args:
        wheel_vel_left: Left wheel angular velocity in rad/s (tensor or float).
        wheel_vel_right: Right wheel angular velocity in rad/s (tensor or float).
        wheel_base: Distance L between the left and right wheels in meters.
        wheel_radius: Radius R of the wheels in meters.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (v, omega) in m/s and rad/s.
    """
    if wheel_radius <= 0.0:
        raise ValueError(f"wheel_radius must be positive, got {wheel_radius}")
    if wheel_base <= 0.0:
        raise ValueError(f"wheel_base must be positive, got {wheel_base}")

    if not isinstance(wheel_vel_left, torch.Tensor):
        wheel_vel_left = torch.as_tensor(wheel_vel_left, dtype=torch.float32)
    if not isinstance(wheel_vel_right, torch.Tensor):
        wheel_vel_right = torch.as_tensor(wheel_vel_right, dtype=torch.float32)

    v = (wheel_radius / 2.0) * (wheel_vel_right + wheel_vel_left)
    omega = (wheel_radius / wheel_base) * (wheel_vel_right - wheel_vel_left)
    return v, omega
