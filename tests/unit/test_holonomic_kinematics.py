# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for omni/mecanum kinematics (holonomic_matrix, holonomic_ik, holonomic_fk)."""

import math

import pytest
import torch

from nav_arena.embodiments.kinematics import holonomic_fk, holonomic_ik, holonomic_matrix

# Kaya's three omni wheels, read from the USD (see docs/working-memory/embodiment-expansion-kaya-notes.md).
KAYA_POSITIONS = [(-0.09804319, 0.00063677), (0.04934748, -0.08452497), (0.04952906, 0.08569367)]
KAYA_AXES = [(-1.0, 0.0), (0.5, -0.8660254), (0.5, 0.8660254)]
KAYA_RADIUS = 0.04


def _kaya():
    return holonomic_matrix(KAYA_POSITIONS, KAYA_AXES, KAYA_RADIUS, 90.0)


def test_kaya_forward_drives_only_the_two_front_wheels_in_opposite_directions():
    """Verify vx = 1 leaves the rear wheel (rolling along y) idle and spins the front pair by +-sin(60)/R."""
    wheels = holonomic_ik(torch.tensor([[1.0, 0.0, 0.0]]), _kaya())[0]
    expected = 0.8660254 / KAYA_RADIUS
    assert wheels[0].item() == pytest.approx(0.0, abs=1e-6)
    assert wheels[1].item() == pytest.approx(-expected, rel=1e-5)
    assert wheels[2].item() == pytest.approx(expected, rel=1e-5)


def test_kaya_sideways_drives_the_rear_wheel_fully_and_the_front_wheels_by_half():
    """Verify vy = 1 spins the rear wheel at 1/R and the front wheels at -0.5/R."""
    wheels = holonomic_ik(torch.tensor([[0.0, 1.0, 0.0]]), _kaya())[0]
    assert wheels[0].item() == pytest.approx(1.0 / KAYA_RADIUS, rel=1e-5)
    assert wheels[1].item() == pytest.approx(-0.5 / KAYA_RADIUS, rel=1e-5)
    assert wheels[2].item() == pytest.approx(-0.5 / KAYA_RADIUS, rel=1e-5)


def test_kaya_pure_rotation_spins_all_wheels_the_same_way():
    """Verify wz > 0 turns every wheel in the same direction, at about (distance to centre) / R each."""
    wheels = holonomic_ik(torch.tensor([[0.0, 0.0, 1.0]]), _kaya())[0]
    assert bool((wheels < 0).all()) or bool((wheels > 0).all())
    distances = [math.hypot(*p) for p in KAYA_POSITIONS]
    for wheel, dist in zip(wheels, distances):
        assert abs(wheel.item()) == pytest.approx(dist / KAYA_RADIUS, rel=0.05)


def test_round_trip_twist_to_wheels_to_twist():
    """Verify forward kinematics inverts inverse kinematics for random twists (three wheels span the plane exactly)."""
    matrix = _kaya()
    twist = torch.randn(64, 3)
    recovered = holonomic_fk(holonomic_ik(twist, matrix), matrix)
    assert torch.allclose(recovered, twist, atol=1e-4)


def _mecanum_matrix(half_length=0.2, half_width=0.15, radius=0.05):
    """Standard 4-wheel mecanum: lateral axles, rollers at 45 deg on the diagonals (front-left/rear-right the same)."""
    positions = [(half_length, half_width), (half_length, -half_width), (-half_length, half_width), (-half_length, -half_width)]
    axes = [(0.0, 1.0)] * 4
    angles = [135.0, 45.0, 45.0, 135.0]
    return holonomic_matrix(positions, axes, radius, angles), radius


def test_mecanum_forward_turns_all_wheels_equally():
    """Verify a 4-wheel mecanum drives straight when all wheels turn at v / R."""
    matrix, radius = _mecanum_matrix()
    wheels = holonomic_ik(torch.tensor([[1.0, 0.0, 0.0]]), matrix)[0]
    assert torch.allclose(wheels, torch.full((4,), 1.0 / radius), atol=1e-4)


def test_mecanum_strafe_alternates_wheel_directions_at_the_same_speed():
    """Verify strafing a mecanum drive turns the wheels at +-v / R in the diagonal pattern."""
    matrix, radius = _mecanum_matrix()
    wheels = holonomic_ik(torch.tensor([[0.0, 1.0, 0.0]]), matrix)[0]
    assert torch.allclose(wheels.abs(), torch.full((4,), 1.0 / radius), atol=1e-4)
    assert wheels[0] * wheels[1] < 0 and wheels[0] * wheels[3] > 0


def test_mecanum_round_trip_is_a_least_squares_fit_with_four_wheels():
    """Verify four wheels (one redundant) still recover the twist exactly from consistent wheel speeds."""
    matrix, _ = _mecanum_matrix()
    twist = torch.randn(32, 3)
    assert torch.allclose(holonomic_fk(holonomic_ik(twist, matrix), matrix), twist, atol=1e-4)


def test_ik_is_batched_and_linear():
    """Verify IK works on any leading batch shape and scales linearly with the command."""
    matrix = _kaya()
    twist = torch.randn(2, 5, 3)
    assert holonomic_ik(twist, matrix).shape == (2, 5, 3)
    assert torch.allclose(holonomic_ik(2.0 * twist, matrix), 2.0 * holonomic_ik(twist, matrix), atol=1e-5)


def test_fk_rejects_a_layout_that_cannot_span_the_plane():
    """Verify two wheels (or parallel wheels) make forward kinematics under-determined and raise."""
    two = holonomic_matrix([(0.1, 0.0), (-0.1, 0.0)], [(0.0, 1.0), (0.0, 1.0)], 0.05, 90.0)
    with pytest.raises(ValueError, match="under-determined"):
        holonomic_fk(torch.zeros(1, 2), two)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"wheel_radii": 0.0}, "radii must be positive"),
        ({"wheel_axes": [(0.0, 0.0)] * 3}, "non-zero"),
        ({"wheel_angles_deg": 180.0}, "not drivable"),
        ({"wheel_axes": [(1.0, 0.0)] * 2}, "wheel_axes must have shape"),
        ({"wheel_positions": [0.0, 1.0, 2.0]}, "wheel_positions must have shape"),
    ],
)
def test_invalid_geometry_is_rejected(kwargs, message):
    """Verify bad wheel geometry fails loudly instead of producing a silently wrong matrix."""
    args = {"wheel_positions": KAYA_POSITIONS, "wheel_axes": KAYA_AXES, "wheel_radii": KAYA_RADIUS, "wheel_angles_deg": 90.0}
    args.update(kwargs)
    with pytest.raises(ValueError, match=message):
        holonomic_matrix(**args)


def test_ik_rejects_a_command_that_is_not_a_twist():
    """Verify IK refuses a command whose last dimension is not 3."""
    with pytest.raises(ValueError, match="twist"):
        holonomic_ik(torch.zeros(1, 2), _kaya())
