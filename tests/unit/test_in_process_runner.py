# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the closed-loop episode runner using a fake kinematic task (CPU-only)."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from nav_arena.methods.in_process import EpisodeCfg, FollowerCfg, run_episode
from nav_arena.methods.in_process.base import InProcessPolicy, InProcessPolicyCfg, Plan


class FakeTask:
    """Unicycle with the PointNavTask interface the runner uses. Auto-resets on termination like the real env."""

    step_dt = 0.02
    device = "cpu"

    def __init__(self, goal=(3.0, 0.0), threshold=0.4, depth_valid=True, forced_cause=None, truncate_after=None):
        self.start = np.array([0.0, 0.0, 0.0])
        self.goal = np.array(goal)
        self.threshold = threshold
        self.depth_valid = depth_valid
        self.forced_cause = forced_cause
        self.truncate_after = truncate_after
        self.pose = self.start.copy()
        self.cause = None
        self.goal_image_calls = 0
        self.zero_steps = 0
        self.steps_taken = 0
        self.actions = []
        self.renders = 0
        self.sim = SimpleNamespace(render=self._render)

    def _render(self):
        self.renders += 1

    def reset(self):
        self.pose = self.start.copy()
        self.cause = None
        self.steps_taken = 0

    def step(self, action):
        v, w = (float(a) for a in action[0])
        self.actions.append((v, w))
        if v == 0.0 and w == 0.0:
            self.zero_steps += 1
        x, y, yaw = self.pose
        self.pose = np.array([x + v * math.cos(yaw) * self.step_dt, y + v * math.sin(yaw) * self.step_dt, yaw + w * self.step_dt])
        self.steps_taken += 1
        terminated = truncated = False
        if self.forced_cause is not None and self.steps_taken >= 5:
            terminated, self.cause = True, self.forced_cause
        elif np.linalg.norm(self.pose[:2] - self.goal) < self.threshold:
            terminated, self.cause = True, "goal_reached"
        elif self.truncate_after is not None and self.steps_taken >= self.truncate_after:
            truncated = True
        done = terminated or truncated
        if done:
            self.pose = self.start.copy()  # the real env resets immediately on termination
        return None, None, torch.tensor([terminated]), torch.tensor([truncated]), {}

    def get_robot_pose_w(self):
        return tuple(self.pose)

    def get_robot_position_w(self):
        return np.array([self.pose[0], self.pose[1], 0.1])

    def get_robot_quat_w(self):
        return np.array([0.0, 0.0, math.sin(self.pose[2] / 2), math.cos(self.pose[2] / 2)])

    def get_goal_pose_w(self):
        return float(self.goal[0]), float(self.goal[1]), 0.0

    def get_goal_distance(self):
        return float(np.linalg.norm(self.pose[:2] - self.goal))

    def get_terminal_cause(self):
        return self.cause

    def refresh_camera_frame(self):
        rgb = np.full((6, 8, 3), 128, dtype=np.uint8)
        depth = np.full((6, 8), 2.0 if self.depth_valid else np.inf, dtype=np.float32)
        return rgb, depth

    def render_goal_image(self, goal_xy):
        self.goal_image_calls += 1
        return np.full((6, 8, 3), 7, dtype=np.uint8)


class GoalSeeker(InProcessPolicy):
    """Plans a straight line to the point goal."""

    name = "goal_seeker"
    supported_tasks = ("pointgoal", "imagegoal")

    def __init__(self, cfg=None, stop=False, requires_pose=False):
        super().__init__(cfg or InProcessPolicyCfg())
        self._stop = stop
        self.requires_pose = requires_pose
        self.observations = []
        self.resets = 0

    def reset(self):
        self.resets += 1

    def step(self, obs):
        self.observations.append(obs)
        g = obs.goal_body[0, :2]
        t = np.linspace(0.1, 1.0, 10)[:, None]
        return Plan(path=t * g[None, :], stop=self._stop, diagnostics={"calls": len(self.observations)})


def _cfg(**kw):
    kw.setdefault("follower", FollowerCfg(goal_tolerance=0.4))
    return EpisodeCfg(**kw)


def test_reaches_goal_and_reports_metrics():
    """Verify a straight route succeeds with sensible time-to-goal, remaining distance, and path length."""
    task = FakeTask()
    result = run_episode(task, GoalSeeker(), _cfg())
    assert result.success and result.terminal_cause == "goal_reached"
    assert result.time_to_goal_s == pytest.approx(result.steps * task.step_dt)
    assert result.initial_goal_distance_m == pytest.approx(3.0)
    # Distance is read before the terminal step; reading it after the env auto-reset would give 3.0 again.
    assert result.final_goal_distance_m < 0.6
    assert 2.4 < result.path_length_m < 3.1
    assert result.mean_inference_ms >= 0.0 and result.wall_time_s > 0


def test_plan_cadence_follows_plan_hz():
    """Verify replanning every round(1 / (plan_hz * dt)) steps (5 Hz at dt=0.02 -> every 10 steps)."""
    task, policy = FakeTask(), GoalSeeker()
    result = run_episode(task, policy, _cfg())
    assert result.plans == math.ceil(result.steps / 10)
    slow = GoalSeeker(InProcessPolicyCfg(plan_hz=2.0))
    result = run_episode(FakeTask(), slow, _cfg())
    assert result.plans == math.ceil(result.steps / 25)


def test_warmup_steps_are_zero_velocity_and_policy_is_reset():
    """Verify warmup holds the robot still before the first plan and the policy history is cleared."""
    task, policy = FakeTask(), GoalSeeker()
    run_episode(task, policy, _cfg(warmup_steps=7))
    assert task.actions[:7] == [(0.0, 0.0)] * 7
    assert policy.resets == 1


def test_policy_stop_holds_the_robot_until_the_step_budget_ends():
    """Verify stop requests command zero velocity; the episode ends by step budget, not success."""
    task, policy = FakeTask(), GoalSeeker(stop=True)
    result = run_episode(task, policy, _cfg(max_steps=60, warmup_steps=0))
    assert not result.success and result.terminal_cause == "max_steps"
    assert result.stop_requests == result.plans > 0
    assert result.path_length_m == 0.0
    assert all(a == (0.0, 0.0) for a in task.actions)


@pytest.mark.parametrize("cause", ["collision", "tipped"])
def test_terminal_causes_are_reported(cause):
    """Verify collision and tipped terminations are surfaced and are not successes."""
    result = run_episode(FakeTask(forced_cause=cause), GoalSeeker(), _cfg())
    assert result.terminal_cause == cause and not result.success
    assert result.time_to_goal_s is None


def test_env_timeout_is_reported():
    """Verify a truncation with no other cause is reported as time_out."""
    result = run_episode(FakeTask(truncate_after=30), GoalSeeker(stop=True), _cfg(warmup_steps=0))
    assert result.terminal_cause == "time_out"


def test_depth_guard_aborts_on_blind_camera():
    """Verify too few valid depth pixels aborts instead of driving blind."""
    with pytest.raises(RuntimeError, match="too few valid pixels"):
        run_episode(FakeTask(depth_valid=False), GoalSeeker(), _cfg())


def test_imagegoal_renders_goal_image_once_and_passes_it():
    """Verify image-goal policies receive a goal image rendered at the goal pose, once per episode."""
    task = FakeTask()
    policy = GoalSeeker(InProcessPolicyCfg(task="imagegoal"))
    run_episode(task, policy, _cfg())
    assert task.goal_image_calls == 1
    assert all(o.goal_image is not None and o.goal_image.max() == 7 for o in policy.observations)


def test_provided_goal_image_skips_rendering():
    """Verify a user-supplied goal image is used as-is."""
    task = FakeTask()
    policy = GoalSeeker(InProcessPolicyCfg(task="imagegoal"))
    image = np.full((6, 8, 3), 99, dtype=np.uint8)
    run_episode(task, policy, _cfg(goal_image=image))
    assert task.goal_image_calls == 0
    assert policy.observations[0].goal_image is image


def test_pose_is_supplied_only_to_policies_that_require_it():
    """Verify world pose / XYZW quaternion arrive with shapes [1,3] / [1,4] for pose-based policies only."""
    plain, posed = GoalSeeker(), GoalSeeker(requires_pose=True)
    run_episode(FakeTask(), plain, _cfg())
    run_episode(FakeTask(), posed, _cfg())
    assert plain.observations[0].robot_position is None
    obs = posed.observations[0]
    assert obs.robot_position.shape == (1, 3) and obs.robot_quaternion.shape == (1, 4)
    assert np.linalg.norm(obs.robot_quaternion) == pytest.approx(1.0)


def test_goal_body_input_is_in_the_robot_frame():
    """Verify the policy sees the goal ahead (+x, ~0 y) at the start and nearer on later plans."""
    policy = GoalSeeker()
    run_episode(FakeTask(), policy, _cfg())
    first, last = policy.observations[0].goal_body, policy.observations[-1].goal_body
    assert first[0, 0] == pytest.approx(3.0, abs=0.01) and abs(first[0, 1]) < 0.01
    assert last[0, 0] < first[0, 0]


def test_recorder_writes_run_artifacts(tmp_path):
    """Verify settings, per-plan rows, summary, goal image and snapshots are recorded."""
    out = tmp_path / "run"
    policy = GoalSeeker(InProcessPolicyCfg(task="imagegoal"))
    result = run_episode(FakeTask(), policy, _cfg(output_dir=out, snapshot_every_s=0.5))
    settings = json.loads((out / "settings.json").read_text())
    rows = [json.loads(line) for line in (out / "steps.jsonl").read_text().splitlines()]
    summary = json.loads((out / "summary.json").read_text())
    assert settings["policy"] == "goal_seeker" and settings["plan_period_steps"] == 10
    assert len(rows) == result.plans
    assert {"step", "position", "goal_distance", "stop", "diagnostics", "inference_ms", "path_world"} <= set(rows[0])
    assert summary["terminal_cause"] == "goal_reached" and summary["success"] is True
    assert (out / "goal_rgb.png").is_file()
    assert list(out.glob("rgb_*.png")) and list(out.glob("depth_m_*.npy"))


def test_recorder_refuses_to_overwrite_a_previous_run(tmp_path):
    """Verify an existing output directory is never reused."""
    out = tmp_path / "run"
    out.mkdir()
    with pytest.raises(FileExistsError):
        run_episode(FakeTask(), GoalSeeker(), _cfg(output_dir=out))


class FakeViewer:
    def __init__(self):
        self.updates = []

    def update(self, position, yaw, dt=0.02):
        self.updates.append((tuple(position), yaw, dt))


def test_stalled_episode_ends_early_with_a_clear_cause():
    """Verify a robot held still by a stop request ends as 'stalled' after the timeout, not at the step budget."""
    result = run_episode(
        FakeTask(), GoalSeeker(stop=True), _cfg(max_steps=5000, warmup_steps=0, stall_timeout_s=1.0, progress_every_s=0)
    )
    assert result.terminal_cause == "stalled" and not result.success
    assert 50 <= result.steps <= 60  # 1 s of simulated time at 50 Hz
    assert result.stop_requests == result.plans > 0


def test_stall_check_can_be_disabled_and_does_not_fire_while_moving():
    """Verify stall detection is skippable, and a driving robot never trips it."""
    held = run_episode(FakeTask(), GoalSeeker(stop=True), _cfg(max_steps=120, warmup_steps=0, stall_timeout_s=None))
    assert held.terminal_cause == "max_steps"
    moving = run_episode(FakeTask(), GoalSeeker(), _cfg(stall_timeout_s=1.0))
    assert moving.terminal_cause == "goal_reached"


def test_viewer_follows_every_step_and_adds_display_renders():
    """Verify the viewer is updated each step with the robot pose and extra renders run at viewer_render_hz."""
    task, viewer = FakeTask(), FakeViewer()
    result = run_episode(task, GoalSeeker(), _cfg(viewer=viewer, viewer_render_hz=25.0))
    assert len(viewer.updates) == result.steps
    assert viewer.updates[0][0] == pytest.approx((0.0, 0.0, 0.1))
    assert all(dt == pytest.approx(0.02) for _, _, dt in viewer.updates)
    # 25 Hz at dt=0.02 -> every 2nd step; every plan also forces its own render only inside the real task
    assert task.renders == math.ceil(result.steps / 2)


def test_no_viewer_means_no_display_renders():
    """Verify display renders happen only when a viewer is attached."""
    task = FakeTask()
    run_episode(task, GoalSeeker(), _cfg())
    assert task.renders == 0


def test_recorder_handles_a_viewer_in_the_settings(tmp_path):
    """Verify a (non-serializable) viewer is recorded by type name and does not break settings.json."""
    out = tmp_path / "run"
    run_episode(FakeTask(), GoalSeeker(), _cfg(output_dir=out, viewer=FakeViewer()))
    settings = json.loads((out / "settings.json").read_text())
    assert settings["episode"]["viewer"] == "FakeViewer"
    assert settings["episode"]["stall_timeout_s"] == 10.0
    assert settings["episode"]["follower"]["goal_tolerance"] == 0.4


class FakeOverlay:
    def __init__(self):
        self.updates = []
        self.cleared = 0

    def update(self, path_xy, stop, z):
        self.updates.append((np.asarray(path_xy).copy(), stop, z))

    def clear(self):
        self.cleared += 1


def test_overlay_shows_the_goal_then_every_plan_and_clears_at_the_end():
    """Verify the overlay is drawn at the start (empty plan), once per plan with its stop flag, and cleared at the end."""
    overlay = FakeOverlay()
    result = run_episode(FakeTask(), GoalSeeker(), _cfg(overlay=overlay))
    assert len(overlay.updates) == 1 + result.plans
    first_path, first_stop, z = overlay.updates[0]
    assert first_path.shape == (0, 2) and first_stop is False and z == pytest.approx(0.1)
    # plans are drawn in the world frame: the straight-line plan from the origin heads toward the goal at x=3
    path, stop, _ = overlay.updates[1]
    assert path[-1][0] == pytest.approx(3.0, abs=0.05) and stop is False
    assert overlay.cleared == 1


def test_overlay_receives_the_stop_flag_and_is_recorded_in_settings(tmp_path):
    """Verify stop requests reach the overlay, and the settings record the overlay type."""
    overlay = FakeOverlay()
    out = tmp_path / "run"
    run_episode(FakeTask(), GoalSeeker(stop=True), _cfg(overlay=overlay, max_steps=30, warmup_steps=0, output_dir=out))
    assert all(stop for _, stop, _ in overlay.updates[1:])
    assert json.loads((out / "settings.json").read_text())["episode"]["overlay"] == "FakeOverlay"
