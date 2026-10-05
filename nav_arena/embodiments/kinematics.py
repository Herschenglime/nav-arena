# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wheeled drive kinematics for mobile navigation embodiments: differential drive and omni/mecanum holonomic drive."""

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


def holonomic_matrix(
    wheel_positions: torch.Tensor | list,
    wheel_axes: torch.Tensor | list,
    wheel_radii: torch.Tensor | list | float,
    wheel_angles_deg: torch.Tensor | list | float = 90.0,
) -> torch.Tensor:
    """Matrix mapping a body twist to wheel angular velocities for an omni or mecanum drive.

    Each wheel only constrains the contact point's velocity along one direction (its no-slip axis ``a_i``); it slides
    freely along the rollers. For a body twist ``(vx, vy, wz)`` at the command site, the contact speed along ``a_i`` is

        u_i = a_i . (v + wz x r_i) = [ax, ay, -ax * ry + ay * rx] . (vx, vy, wz)

    and the wheel joint speed is ``u_i / (R_i cos(gamma_i))`` with ``gamma_i = angle_i - 90 deg``. The rolling direction
    of a wheel with axle ``w`` is ``w x up = (wy, -wx)``, and ``a_i`` is that direction rotated by ``gamma_i``. A
    positive joint speed is a positive rotation about the axle direction ``w``. The model is the one used by Isaac Sim's
    ``HolonomicController``, which an integration test checks against.

    Args:
        wheel_positions: Wheel centres ``[N, 2]`` (x, y) in the frame of the command site, in meters.
        wheel_axes: Axle directions ``[N, 2]`` in the same frame (the horizontal part of the joint axis); normalised here.
        wheel_radii: Wheel radius in meters, one value or ``[N]``.
        wheel_angles_deg: Roller angle: 90 for an omni wheel (rollers perpendicular to the rolling direction), 45 or 135
            for mecanum. One value or ``[N]``.

    Returns:
        Matrix ``M`` of shape ``[N, 3]`` such that ``wheel_velocities = twist @ M.T``.

    Raises:
        ValueError: On inconsistent shapes, a non-positive radius, a zero axle vector, or a roller angle of 0/180
            degrees (the wheel could not drive along its no-slip axis).
    """
    positions = torch.as_tensor(wheel_positions, dtype=torch.float64)
    axes = torch.as_tensor(wheel_axes, dtype=torch.float64)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(f"wheel_positions must have shape [N, 2], got {tuple(positions.shape)}")
    num_wheels = positions.shape[0]
    if axes.shape != positions.shape:
        raise ValueError(f"wheel_axes must have shape {tuple(positions.shape)}, got {tuple(axes.shape)}")
    radii = torch.broadcast_to(torch.as_tensor(wheel_radii, dtype=torch.float64), (num_wheels,))
    angles = torch.broadcast_to(torch.as_tensor(wheel_angles_deg, dtype=torch.float64), (num_wheels,))
    if bool((radii <= 0.0).any()):
        raise ValueError(f"wheel radii must be positive, got {radii.tolist()}")
    axis_norm = torch.linalg.norm(axes, dim=1)
    if bool((axis_norm < 1e-9).any()):
        raise ValueError("every wheel axle must have a non-zero horizontal direction")
    axes = axes / axis_norm[:, None]

    gamma = torch.deg2rad(angles - 90.0)
    cos_gamma = torch.cos(gamma)
    if bool((cos_gamma.abs() < 1e-9).any()):
        raise ValueError(f"roller angles of 0 or 180 degrees are not drivable, got {angles.tolist()}")

    rolling = torch.stack((axes[:, 1], -axes[:, 0]), dim=1)  # axle x up
    sin_gamma = torch.sin(gamma)
    a = torch.stack(
        (cos_gamma * rolling[:, 0] - sin_gamma * rolling[:, 1], sin_gamma * rolling[:, 0] + cos_gamma * rolling[:, 1]),
        dim=1,
    )
    # a . (wz x r) with wz x r = wz * (-ry, rx)
    yaw_term = -a[:, 0] * positions[:, 1] + a[:, 1] * positions[:, 0]
    matrix = torch.stack((a[:, 0], a[:, 1], yaw_term), dim=1) / (radii * cos_gamma)[:, None]
    return matrix.to(torch.float32)


def holonomic_ik(twist: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Wheel angular velocities (rad/s) for a body twist ``[..., 3] = (vx, vy, wz)`` and a :func:`holonomic_matrix`."""
    if twist.shape[-1] != 3:
        raise ValueError(f"twist must end in a dimension of 3 (vx, vy, wz), got shape {tuple(twist.shape)}")
    return twist @ matrix.to(twist.dtype).T


def holonomic_fk(wheel_velocities: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Body twist ``[..., 3]`` that best explains measured wheel velocities ``[..., N]`` (least squares).

    Raises:
        ValueError: If the wheel count does not match, or the wheels cannot span a full planar twist (rank below 3).
    """
    if wheel_velocities.shape[-1] != matrix.shape[0]:
        raise ValueError(f"expected {matrix.shape[0]} wheel velocities, got shape {tuple(wheel_velocities.shape)}")
    if int(torch.linalg.matrix_rank(matrix)) < 3:
        raise ValueError("the wheel layout cannot produce all of (vx, vy, wz); forward kinematics is under-determined")
    return wheel_velocities @ torch.linalg.pinv(matrix).to(wheel_velocities.dtype).T
