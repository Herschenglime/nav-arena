# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the per-axis drive check rules used by verify_embodiment."""

import math

import pytest

from nav_arena.utils.drive_check import (
    SegmentResult,
    evaluate_hold,
    evaluate_segment,
    measure_segment,
    wrap_angle,
    yaw_from_quat_xyzw,
)


def test_yaw_uses_isaac_lab_xyzw_order():
    """Verify yaw is read from (x, y, z, w): a 90 deg turn about z is (0, 0, sin45, cos45)."""
    half = math.radians(45.0)
    assert yaw_from_quat_xyzw((0.0, 0.0, math.sin(half), math.cos(half))) == pytest.approx(math.pi / 2)
    assert yaw_from_quat_xyzw((0.0, 0.0, 0.0, 1.0)) == pytest.approx(0.0)


def test_yaw_change_wraps_across_pi():
    """Verify a small turn across +-pi is measured as small, not as nearly a full turn."""
    assert wrap_angle(math.radians(179) - math.radians(-179)) == pytest.approx(math.radians(-2))


def test_motion_is_measured_in_the_starting_body_frame():
    """Verify a robot facing +y that moves +y in the world has moved forward (dx), not sideways."""
    result = measure_segment((0.3, 0.0, 0.0), (1.0, 1.0), math.pi / 2, (1.0, 1.3), math.pi / 2, 1.0)
    assert result.dx == pytest.approx(0.3) and result.dy == pytest.approx(0.0, abs=1e-9)


def test_clean_strafe_passes():
    """Verify a strafe at 90 % of the command with little drift passes."""
    passed, _ = evaluate_segment(SegmentResult((0.0, 0.3, 0.0), 1.0, dx=0.01, dy=0.27, dyaw=0.02))
    assert passed


@pytest.mark.parametrize(
    "result, reason",
    [
        (SegmentResult((0.3, 0.0, 0.0), 1.0, dx=0.15, dy=0.0, dyaw=0.0), "speed ratio"),
        (SegmentResult((0.0, 0.3, 0.0), 1.0, dx=0.2, dy=0.25, dyaw=0.0), "cross-axis"),  # a diagonal 'strafe'
        (SegmentResult((0.3, 0.0, 0.0), 1.0, dx=0.28, dy=0.0, dyaw=0.3), "heading"),
        (SegmentResult((0.0, 0.0, 1.0), 1.0, dx=0.1, dy=0.0, dyaw=0.9, center_shift=0.2), "turning centre moved"),
        (SegmentResult((0.0, 0.0, 1.0), 1.0, dx=0.0, dy=0.0, dyaw=0.0015, center_shift=0.0), "speed ratio"),  # old yaw bug
    ],
)
def test_bad_segments_fail_with_the_reason(result, reason):
    """Verify slow, diagonal, turning or non-rotating segments fail, naming the problem."""
    passed, detail = evaluate_segment(result)
    assert not passed and reason in detail


def test_segment_must_command_exactly_one_axis():
    """Verify a mixed command is rejected rather than checked against an arbitrary axis."""
    with pytest.raises(ValueError, match="exactly one axis"):
        evaluate_segment(SegmentResult((0.3, 0.3, 0.0), 1.0, dx=0.3, dy=0.3, dyaw=0.0))


def test_zero_command_hold():
    """Verify the hold passes when the robot stays put and fails when it creeps or spins."""
    assert evaluate_hold(SegmentResult((0.0, 0.0, 0.0), 1.0, dx=0.002, dy=0.001, dyaw=0.001))[0]
    assert not evaluate_hold(SegmentResult((0.0, 0.0, 0.0), 1.0, dx=0.03, dy=0.0, dyaw=0.0))[0]
    assert not evaluate_hold(SegmentResult((0.0, 0.0, 0.0), 1.0, dx=0.0, dy=0.0, dyaw=0.05))[0]


def _rotate_about(center, offset, yaw):
    """Pose of a root that sits `offset` from `center` (in the body frame) when the body has heading `yaw`."""
    c, s = math.cos(yaw), math.sin(yaw)
    return (center[0] + c * offset[0] - s * offset[1], center[1] + s * offset[0] + c * offset[1]), yaw


def test_turning_in_place_with_an_offset_root_passes():
    """Verify a root 0.1 m off the turning axis (like the Dingo's base_link) passes: its circle is not drift."""
    center, offset = (1.0, 2.0), (0.0, 0.1)
    poses = [_rotate_about(center, offset, yaw) for yaw in (0.0, 0.45, 0.9)]
    (p0, y0), (p1, y1), (p2, y2) = poses
    result = measure_segment((0.0, 0.0, 1.0), p0, y0, p2, y2, 1.0, p1, y1)
    assert result.center_shift == pytest.approx(0.0, abs=1e-9)
    assert math.hypot(result.dx, result.dy) > 0.08  # the root itself moved, as the Dingo's did
    assert evaluate_segment(result)[0]


def test_orbiting_instead_of_turning_in_place_fails():
    """Verify a robot whose turning centre wanders (it drives an arc) fails the in-place rotation check."""
    p0, y0 = (0.0, 0.0), 0.0
    p1, y1 = (0.3, 0.05), 0.45
    p2, y2 = (0.55, 0.2), 0.9
    passed, detail = evaluate_segment(measure_segment((0.0, 0.0, 1.0), p0, y0, p2, y2, 1.0, p1, y1))
    assert not passed and "turning centre moved" in detail


def test_rotation_without_a_mid_pose_is_refused():
    """Verify a rotation segment cannot be judged without locating its turning centre."""
    with pytest.raises(ValueError, match="mid pose"):
        evaluate_segment(SegmentResult((0.0, 0.0, 1.0), 1.0, dx=0.0, dy=0.0, dyaw=0.9))
