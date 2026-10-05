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
    v, vy, w = follow_path(path, 2)
    assert v > 0 and w > 0 and vy == 0
    left, right = diff_drive_ik(v, w, wheel_base=0.45, wheel_radius=0.1225)
    assert float(left) < float(right)


@pytest.mark.parametrize("invalid", [[], [[float("nan"), 1]], [[0, 0]]])
def test_invalid_paths_stop_the_robot(invalid):
    """Verify empty, non-finite, or degenerate paths command zero velocity."""
    assert follow_path(invalid, 2) == (0, 0, 0)


def test_stops_inside_goal_tolerance():
    """Verify zero velocity once within the goal tolerance, regardless of the path."""
    path = np.array([[0, 0], [0.5, 0.1], [1, 0.2]])
    assert follow_path(path, 0.2) == (0, 0, 0)
    assert follow_path(path, FollowerCfg().goal_tolerance) == (0, 0, 0)


def test_turns_in_place_when_target_is_behind():
    """Verify zero forward speed (rotation only) when the lookahead target is far off-heading."""
    v, vy, w = follow_path([[0, 0], [-1, 1]], 2)
    assert v == 0 and vy == 0
    assert w != 0


def test_speed_ramps_down_on_goal_approach_and_respects_limits():
    """Verify forward speed is capped by max_speed far away and shrinks proportionally near the goal."""
    path = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    cfg = FollowerCfg(max_speed=0.3, goal_tolerance=0.3)
    far, _, _ = follow_path(path, 5.0, cfg)
    near, _, _ = follow_path(path, 0.4, cfg)
    capped, _, _ = follow_path(path, 0.5, cfg)  # 0.7 * 0.5 = 0.35 exceeds max_speed, so the cap wins
    assert far == pytest.approx(0.3)
    assert near == pytest.approx(0.7 * 0.4)
    assert capped == pytest.approx(0.3)
    assert 0 < near < far


def test_yaw_rate_is_clamped():
    """Verify steering output never exceeds max_yaw_rate."""
    cfg = FollowerCfg(max_yaw_rate=0.4)
    _, _, w = follow_path([[0, 0], [0.3, 0.45], [0.6, 0.9]], 3.0, cfg)
    assert abs(w) <= 0.4 + 1e-9
    _, _, w = follow_path([[0, 0], [0.1, 1.0]], 3.0, cfg)
    assert abs(w) == pytest.approx(0.4)


def test_follower_trims_already_passed_path():
    """Verify points behind the robot (closest-point trimming) do not drag the lookahead target backward."""
    path = np.array([[-1.0, 0.0], [-0.5, 0.0], [0.0, 0.0], [0.6, 0.0], [1.2, 0.0]])
    v, vy, w = follow_path(path, 3.0)
    assert v > 0 and vy == 0
    assert w == pytest.approx(0.0, abs=1e-6)


def test_policy_reseed_makes_random_draws_repeatable():
    """Verify reseed() records the seed and restores identical numpy/torch random state."""
    import numpy as np
    import torch

    from nav_arena.methods.in_process.base import InProcessPolicy, InProcessPolicyCfg, Plan

    class Dummy(InProcessPolicy):
        name = "dummy"

        def step(self, obs):  # pragma: no cover - never planned
            return Plan(path=np.zeros((1, 3)), stop=False)

    policy = Dummy(InProcessPolicyCfg())
    policy.reseed(5)
    first = (np.random.rand(), torch.rand(1).item())
    policy.reseed(5)
    second = (np.random.rand(), torch.rand(1).item())
    assert first == second and policy.cfg.seed == 5
    policy.reseed(None)
    assert policy.cfg.seed is None


def _legacy_follow_path(path_body, goal_distance, cfg):
    """The pre-refactor unicycle follower, kept verbatim as the reference for the bit-identical equivalence test."""
    path = np.asarray(path_body, dtype=np.float32)
    if goal_distance <= cfg.goal_tolerance or path.ndim != 2 or len(path) == 0:
        return 0.0, 0.0
    if path.shape[1] < 2 or not np.isfinite(path).all():
        return 0.0, 0.0
    path = path[np.argmin(np.linalg.norm(path[:, :2], axis=1)) :]
    distances = np.linalg.norm(path[:, :2], axis=1)
    candidates = np.flatnonzero(distances >= cfg.lookahead)
    target = path[candidates[0] if len(candidates) else -1, :2]
    if np.linalg.norm(target) < 0.05:
        return 0.0, 0.0
    heading = math.atan2(float(target[1]), float(target[0]))
    if abs(heading) > cfg.turn_in_place_angle:
        return 0.0, float(np.clip(1.5 * heading, -cfg.max_yaw_rate, cfg.max_yaw_rate))
    speed = min(cfg.max_speed, cfg.goal_slowdown_gain * goal_distance) * max(0.0, math.cos(heading))
    curvature = 2.0 * float(target[1]) / max(float(target @ target), 0.01)
    return speed, float(np.clip(speed * curvature, -cfg.max_yaw_rate, cfg.max_yaw_rate))


def test_without_lateral_speed_the_follower_matches_the_old_unicycle_exactly():
    """Verify (vx, wz) is bit-identical to the pre-refactor follower and vy is 0 on random paths and goal distances."""
    rng = np.random.default_rng(0)
    cfg = FollowerCfg()
    for _ in range(500):
        path = np.cumsum(rng.normal(0.3, 0.4, size=(rng.integers(1, 12), 2)), axis=0) * rng.choice([-1, 1], size=2)
        goal_distance = float(rng.uniform(0.0, 4.0))
        vx, vy, wz = follow_path(path, goal_distance, cfg)
        assert (vx, wz) == _legacy_follow_path(path, goal_distance, cfg)
        assert vy == 0.0


def test_holonomic_follower_translates_while_turning_and_uses_sideways_speed():
    """Verify with lateral speed the robot turns toward a left target while moving left instead of stopping."""
    cfg = FollowerCfg(max_lateral_speed=0.3)
    vx, vy, wz = follow_path([[0, 0], [0.5, 0.5], [1.0, 1.0]], 3.0, cfg)  # target 45 deg to the left
    assert vx > 0 and vy > 0 and wz > 0
    assert vy == pytest.approx(vx)  # translation points straight at the target
    vx, vy, wz = follow_path([[0, 0], [0.5, -0.5], [1.0, -1.0]], 3.0, cfg)
    assert vx > 0 and vy < 0 and wz < 0


def test_holonomic_follower_never_drives_away_from_a_target_behind_it():
    """Verify a target behind the robot produces rotation only, so the camera never leads a reversing robot."""
    vx, vy, wz = follow_path([[0, 0], [-1.0, 0.2]], 2.0, FollowerCfg(max_lateral_speed=0.3))
    assert (vx, vy) == (0.0, 0.0) and wz != 0.0


def test_holonomic_follower_clips_sideways_speed():
    """Verify |vy| never exceeds max_lateral_speed even when the target is mostly to the side."""
    cfg = FollowerCfg(max_speed=0.6, max_lateral_speed=0.1)
    _, vy, _ = follow_path([[0, 0], [0.4, 0.6], [0.8, 1.2]], 5.0, cfg)
    assert abs(vy) == pytest.approx(0.1)
