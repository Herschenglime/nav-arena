# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for differential drive kinematics (diff_drive_ik and diff_drive_fk)."""

import pytest
import torch

from nav_arena.embodiments.kinematics import diff_drive_fk, diff_drive_ik


def test_pure_forward_translation():
    """v > 0, omega = 0 should result in equal wheel speeds (v / R)."""
    wheel_base = 0.5
    wheel_radius = 0.1
    v = torch.tensor([1.5])
    omega = torch.tensor([0.0])

    left, right = diff_drive_ik(v, omega, wheel_base, wheel_radius)
    expected = 1.5 / 0.1  # 15.0 rad/s
    assert torch.allclose(left, torch.tensor([expected]))
    assert torch.allclose(right, torch.tensor([expected]))

    # Test reverse direction
    v_rev = torch.tensor([-1.0])
    left_rev, right_rev = diff_drive_ik(v_rev, omega, wheel_base, wheel_radius)
    assert torch.allclose(left_rev, torch.tensor([-10.0]))
    assert torch.allclose(right_rev, torch.tensor([-10.0]))


def test_pure_in_place_rotation():
    """v = 0, omega > 0 should yield counter-rotating wheels: left = -right."""
    wheel_base = 0.4
    wheel_radius = 0.1
    v = torch.tensor([0.0])
    omega = torch.tensor([2.0])

    # Left = (0 - 2 * 0.2) / 0.1 = -4.0
    # Right = (0 + 2 * 0.2) / 0.1 = +4.0
    left, right = diff_drive_ik(v, omega, wheel_base, wheel_radius)
    assert torch.allclose(left, torch.tensor([-4.0]))
    assert torch.allclose(right, torch.tensor([4.0]))

    # Negative yaw (clockwise)
    omega_cw = torch.tensor([-2.0])
    left_cw, right_cw = diff_drive_ik(v, omega_cw, wheel_base, wheel_radius)
    assert torch.allclose(left_cw, torch.tensor([4.0]))
    assert torch.allclose(right_cw, torch.tensor([-4.0]))


def test_wheel_base_and_radius_scaling():
    """Verify that scaling wheel base L and wheel radius R changes speeds proportionally."""
    v = torch.tensor([1.0])
    omega = torch.tensor([1.0])

    # Case A: L=0.5, R=0.1 -> L/2=0.25 -> left=(1 - 0.25)/0.1 = 7.5, right=(1 + 0.25)/0.1 = 12.5
    left_a, right_a = diff_drive_ik(v, omega, wheel_base=0.5, wheel_radius=0.1)
    assert torch.allclose(left_a, torch.tensor([7.5]))
    assert torch.allclose(right_a, torch.tensor([12.5]))

    # Case B: Double radius -> wheel angular speeds should halve
    left_b, right_b = diff_drive_ik(v, omega, wheel_base=0.5, wheel_radius=0.2)
    assert torch.allclose(left_b, left_a / 2.0)
    assert torch.allclose(right_b, right_a / 2.0)

    # Case C: Double wheel base -> differential component should double
    # left = (1.0 - 1.0 * 0.5) / 0.1 = 5.0
    # right = (1.0 + 1.0 * 0.5) / 0.1 = 15.0
    left_c, right_c = diff_drive_ik(v, omega, wheel_base=1.0, wheel_radius=0.1)
    assert torch.allclose(left_c, torch.tensor([5.0]))
    assert torch.allclose(right_c, torch.tensor([15.0]))


def test_batch_broadcasting():
    """Verify multi-environment tensor broadcasting across arbitrary batch shapes."""
    batch_size = 16
    v = torch.linspace(-2.0, 2.0, batch_size)
    omega = torch.linspace(-1.0, 1.0, batch_size)
    wheel_base = 0.413
    wheel_radius = 0.14

    left, right = diff_drive_ik(v, omega, wheel_base, wheel_radius)
    assert left.shape == (batch_size,)
    assert right.shape == (batch_size,)

    # 2D batch shape (num_envs, 1)
    v_2d = v.unsqueeze(-1)
    omega_2d = omega.unsqueeze(-1)
    left_2d, right_2d = diff_drive_ik(v_2d, omega_2d, wheel_base, wheel_radius)
    assert left_2d.shape == (batch_size, 1)
    assert right_2d.shape == (batch_size, 1)

    # Check consistency between 1D and 2D
    assert torch.allclose(left_2d.squeeze(-1), left)
    assert torch.allclose(right_2d.squeeze(-1), right)


def test_fk_ik_roundtrip():
    """Verify that diff_drive_fk is the exact mathematical inverse of diff_drive_ik."""
    wheel_base = 0.413
    wheel_radius = 0.14

    v_orig = torch.tensor([0.75, -1.2, 0.0, 2.0])
    omega_orig = torch.tensor([-0.5, 0.3, 1.5, 0.0])

    w_left, w_right = diff_drive_ik(v_orig, omega_orig, wheel_base, wheel_radius)
    v_recovered, omega_recovered = diff_drive_fk(w_left, w_right, wheel_base, wheel_radius)

    assert torch.allclose(v_recovered, v_orig, atol=1e-6)
    assert torch.allclose(omega_recovered, omega_orig, atol=1e-6)


def test_scalar_float_inputs():
    """Verify diff_drive_ik and diff_drive_fk accept standard Python floats."""
    w_left, w_right = diff_drive_ik(1.0, 0.0, wheel_base=0.4, wheel_radius=0.1)
    assert isinstance(w_left, torch.Tensor)
    assert isinstance(w_right, torch.Tensor)
    assert torch.isclose(w_left, torch.tensor(10.0))
    assert torch.isclose(w_right, torch.tensor(10.0))

    v, omega = diff_drive_fk(10.0, 10.0, wheel_base=0.4, wheel_radius=0.1)
    assert torch.isclose(v, torch.tensor(1.0))
    assert torch.isclose(omega, torch.tensor(0.0))


def test_invalid_parameters_raise():
    """Verify zero or negative wheel radius or wheel base raises ValueError."""
    with pytest.raises(ValueError, match="wheel_radius must be positive"):
        diff_drive_ik(1.0, 0.0, wheel_base=0.5, wheel_radius=0.0)

    with pytest.raises(ValueError, match="wheel_radius must be positive"):
        diff_drive_ik(1.0, 0.0, wheel_base=0.5, wheel_radius=-0.1)

    with pytest.raises(ValueError, match="wheel_base must be positive"):
        diff_drive_ik(1.0, 0.0, wheel_base=0.0, wheel_radius=0.1)

    with pytest.raises(ValueError, match="wheel_base must be positive"):
        diff_drive_ik(1.0, 0.0, wheel_base=-0.5, wheel_radius=0.1)

    with pytest.raises(ValueError, match="wheel_radius must be positive"):
        diff_drive_fk(1.0, 1.0, wheel_base=0.5, wheel_radius=0.0)

    with pytest.raises(ValueError, match="wheel_base must be positive"):
        diff_drive_fk(1.0, 1.0, wheel_base=-0.5, wheel_radius=0.1)
