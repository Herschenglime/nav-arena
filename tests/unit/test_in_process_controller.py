# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for planar frame transforms and the shared path follower.

Ported from the NavDP Isaac Sim integration's adapter tests.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from nav_arena.embodiments import diff_drive_ik
from nav_arena.methods.in_process import FollowerCfg, body_to_world, follow_path, goal_body_input, world_to_body


def test_goal_frames_facing_north():
    """Verify that facing +Y, a goal to the east is robot-right (negative y) and north is forward."""
    np.testing.assert_allclose(goal_body_input([2, 4], [1, 2], math.pi / 2), [[2, -1, 0]], atol=1e-6)


def test_world_body_round_trip():
    """Verify world->body->world is the identity for arbitrary poses."""
    points = np.array([[2, -1], [3, 2]])
    np.testing.assert_allclose(
        world_to_body(body_to_world(points, [4, 1], 1.2), [4, 1], 1.2), points, atol=1e-6
    )


def test_follow_turn_sign_and_wheel_mapping():
    """Verify a path curving left yields positive yaw rate, which slows the left wheel relative to the right."""
    path = np.array([[0, 0], [0.5, 0.1], [1, 0.2]])
    v, w = follow_path(path, 2)
    assert v > 0 and w > 0
    left, right = diff_drive_ik(v, w, wheel_base=0.45, wheel_radius=0.1225)
    assert float(left) < float(right)


@pytest.mark.parametrize("invalid", [[], [[float("nan"), 1]], [[0, 0]]])
def test_invalid_paths_stop_the_robot(invalid):
    """Verify empty, non-finite, or degenerate paths command zero velocity."""
    assert follow_path(invalid, 2) == (0, 0)


def test_stops_inside_goal_tolerance():
    """Verify zero velocity once within the goal tolerance, regardless of the path."""
    path = np.array([[0, 0], [0.5, 0.1], [1, 0.2]])
    assert follow_path(path, 0.2) == (0, 0)
    assert follow_path(path, FollowerCfg().goal_tolerance) == (0, 0)


def test_turns_in_place_when_target_is_behind():
    """Verify zero forward speed (rotation only) when the lookahead target is far off-heading."""
    v, w = follow_path([[0, 0], [-1, 1]], 2)
    assert v == 0
    assert w != 0


def test_speed_ramps_down_on_goal_approach_and_respects_limits():
    """Verify forward speed is capped by max_speed far away and shrinks proportionally near the goal."""
    path = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    cfg = FollowerCfg(max_speed=0.3, goal_tolerance=0.3)
    far, _ = follow_path(path, 5.0, cfg)
    near, _ = follow_path(path, 0.4, cfg)
    capped, _ = follow_path(path, 0.5, cfg)  # 0.7 * 0.5 = 0.35 exceeds max_speed, so the cap wins
    assert far == pytest.approx(0.3)
    assert near == pytest.approx(0.7 * 0.4)
    assert capped == pytest.approx(0.3)
    assert 0 < near < far


def test_yaw_rate_is_clamped():
    """Verify steering output never exceeds max_yaw_rate."""
    cfg = FollowerCfg(max_yaw_rate=0.4)
    _, w = follow_path([[0, 0], [0.3, 0.45], [0.6, 0.9]], 3.0, cfg)
    assert abs(w) <= 0.4 + 1e-9
    _, w = follow_path([[0, 0], [0.1, 1.0]], 3.0, cfg)
    assert abs(w) == pytest.approx(0.4)


def test_follower_trims_already_passed_path():
    """Verify points behind the robot (closest-point trimming) do not drag the lookahead target backward."""
    path = np.array([[-1.0, 0.0], [-0.5, 0.0], [0.0, 0.0], [0.6, 0.0], [1.2, 0.0]])
    v, w = follow_path(path, 3.0)
    assert v > 0
    assert w == pytest.approx(0.0, abs=1e-6)
