# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the NavDP-family in-process adapters (mocked upstream agents; CPU-only).

The contract tests are ported from the NavDP Isaac Sim integration (``test_isaac_policies.py``): model-specific goal
handling, metre-depth ownership, and native stop behavior.
"""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import Mock

import numpy as np
import pytest

from nav_arena.methods.in_process import PolicyObservation, get_policy, list_policies, register_policy
from nav_arena.methods.in_process.base import InProcessPolicy, Plan
from nav_arena.methods.in_process.navdp_adapter import loader
from nav_arena.methods.in_process.navdp_adapter.checkpoints import CHECKPOINTS, default_checkpoint, sha256_file
from nav_arena.methods.in_process.navdp_adapter.observation import as_path, prepare_observation
from nav_arena.methods.in_process.navdp_adapter.policies import (
    IPlannerPolicy,
    NavDPPolicy,
    ViNTPolicy,
    VIPlannerPolicy,
    XNavDPPolicy,
)

K = np.eye(3)


@pytest.fixture
def rgb():
    return np.full((3, 4, 3), [240, 100, 10], dtype=np.uint8)


@pytest.fixture
def depth():
    return np.full((3, 4), 7.0, dtype=np.float32)


@pytest.fixture
def goal():
    return np.array([[12, -8, 0]], dtype=np.float32)


def _obs(rgb, depth, goal=None, **kw):
    return PolicyObservation(rgb=rgb, depth=depth, goal_body=goal, **kw)


def test_rgb_order_depth_units_and_ownership(rgb, depth):
    """Verify RGB is passed through untouched, invalid depth is zeroed, and the simulator buffer is never mutated."""
    depth[0, :3] = [np.inf, np.nan, -1]
    before = depth.copy()
    images, depths = prepare_observation(rgb, depth)
    np.testing.assert_array_equal(images[0], rgb)
    np.testing.assert_array_equal(depths[0, 0, :, 0], [0, 0, 0, 7])
    depths[:] = 100  # A mutating upstream agent must not alter simulator data.
    np.testing.assert_array_equal(depth, before)


def test_prepare_observation_validates_inputs(rgb, depth):
    """Verify dtype and alignment errors are raised before any model runs."""
    with pytest.raises(ValueError):
        prepare_observation(rgb.astype(np.float32), depth)
    with pytest.raises(ValueError):
        prepare_observation(rgb, depth[:2])


def test_iplanner_clamp_and_fear_stop(rgb, depth, goal):
    """Verify iPlanner clamps the goal to +/-5 m, receives raw (unsanitized) depth, and stops on high fear."""
    p = IPlannerPolicy(K, agent=Mock())
    depth[0, 0] = np.inf
    p.agent.step_pointgoal.return_value = (None, np.ones((1, 5, 3)), np.array([0.9]))
    result = p.step(_obs(rgb, depth, goal))
    np.testing.assert_array_equal(p.agent.step_pointgoal.call_args.args[1], [[5, -5, 0]])
    assert np.isinf(p.agent.step_pointgoal.call_args.args[0][0, 0, 0, 0])
    assert result.stop
    assert result.diagnostics["fear"] == pytest.approx(0.9)


def test_iplanner_low_fear_does_not_stop(rgb, depth, goal):
    """Verify fear below the threshold produces a driving plan."""
    p = IPlannerPolicy(K, agent=Mock())
    p.agent.step_pointgoal.return_value = (None, np.ones((1, 5, 3)), np.array([0.1]))
    result = p.step(_obs(rgb, depth, goal))
    assert not result.stop
    assert result.path.shape == (5, 3)


def test_vint_receives_image_goal_and_preserves_native_stop(rgb, depth):
    """Verify ViNT gets only images (no point goal) and an all-zero path is reported as a native stop."""
    p = ViNTPolicy(K, agent=Mock())
    p.agent.step_imagegoal.return_value = (None, np.zeros((1, 51, 3)))
    goal_image = 255 - rgb
    result = p.step(_obs(rgb, depth, goal_image=goal_image))
    args = p.agent.step_imagegoal.call_args.args
    assert len(args) == 2  # No world/point-goal information sent to ViNT.
    np.testing.assert_array_equal(args[0], goal_image[None])
    np.testing.assert_array_equal(args[1], rgb[None])
    assert result.stop


def test_vint_requires_goal_image_and_defaults_to_3hz(rgb, depth):
    """Verify ViNT rejects observations without a goal image and uses its native 3 Hz rate."""
    p = ViNTPolicy(K, agent=Mock())
    assert p.plan_hz == 3.0
    assert p.cfg.task == "imagegoal"
    with pytest.raises(ValueError, match="goal image"):
        p.step(_obs(rgb, depth))


def test_navdp_preserves_its_goal_range_and_native_recovery(rgb, depth, goal):
    """Verify NavDP gets the unclamped goal and low critic values flag recovery without a fear-style stop."""
    p = NavDPPolicy(K, agent=Mock())
    recovery = np.array([[[0, 1, 0], [0, 1, 0]]], dtype=np.float32)
    p.agent.step_pointgoal.return_value = (recovery, None, np.array([[-4, -5]]), None)
    result = p.step(_obs(rgb, depth, goal))
    np.testing.assert_array_equal(p.agent.step_pointgoal.call_args.args[0], goal)
    np.testing.assert_array_equal(result.path, recovery[0])
    assert not result.stop  # Do not apply iPlanner's fear gate to NavDP.
    assert result.diagnostics["native_critic_recovery"]


def test_navdp_imagegoal_task(rgb, depth):
    """Verify NavDP's image-goal mode routes the goal image to step_imagegoal."""
    p = NavDPPolicy(K, task="imagegoal", agent=Mock())
    p.agent.step_imagegoal.return_value = (np.ones((1, 4, 3)), None, np.array([[1.0, 2.0]]), None)
    result = p.step(_obs(rgb, depth, goal_image=rgb))
    p.agent.step_imagegoal.assert_called_once()
    assert result.diagnostics["critic_max"] == 2.0


def test_invalid_plans_are_rejected_before_driving():
    """Verify multi-path, empty, and non-finite trajectories raise instead of reaching the controller."""
    for path in [np.zeros((2, 5, 3)), np.zeros((1, 0, 3)), np.full((1, 2, 3), np.nan)]:
        with pytest.raises(ValueError):
            as_path(path)


def test_viplanner_semantic_input_channel_order_and_depth_order(rgb, depth, goal):
    """Verify VIPlanner receives BGR images, a +/-10 m clamped goal, and raw depth."""
    p = VIPlannerPolicy(K, agent=Mock())
    depth[0, 0] = np.inf
    p.agent.step_pointgoal.return_value = (None, np.ones((1, 5, 3)), np.array([0.9]))
    result = p.step(_obs(rgb, depth, goal))
    bgr, depths, clamped = p.agent.step_pointgoal.call_args.args
    np.testing.assert_array_equal(bgr[0], rgb[..., ::-1])
    np.testing.assert_array_equal(clamped, [[10, -8, 0]])
    assert np.isinf(depths[0, 0, 0, 0])
    assert result.stop
    assert result.diagnostics["semantic_source"] == "mask2former_rgb_prediction"


def test_xnavdp_requires_and_preserves_pose_for_native_guidance(rgb, depth, goal):
    """Verify X-NavDP rejects missing/invalid poses and forwards a valid XYZW pose to native guidance."""
    p = XNavDPPolicy(K, agent=Mock())
    p.agent.embodiment, p.agent.is_real = 0, True
    p.agent.is_stuck = np.array([True])
    trajectory = np.arange(72, dtype=np.float32).reshape(1, 24, 3) / 100
    p.agent.step_pointgoal_with_guidance.return_value = (trajectory, None, np.array([[1, 2]]), None)
    position = np.array([[-10, -8, 0.1]])
    quaternion = np.array([[0, 0, 1, 0]])  # XYZW: 180 degrees about Z.
    for pose in [{}, dict(robot_position=position, robot_quaternion=np.zeros((1, 4)))]:
        with pytest.raises(ValueError):
            p.step(_obs(rgb, depth, goal, **pose))
    p.agent.step_pointgoal_with_guidance.assert_not_called()
    result = p.step(_obs(rgb, depth, goal, robot_position=position, robot_quaternion=quaternion))
    g, images, _, pos, quat = p.agent.step_pointgoal_with_guidance.call_args.args
    np.testing.assert_array_equal(g, goal)
    np.testing.assert_array_equal(images[0], rgb)
    np.testing.assert_array_equal(pos, position)
    np.testing.assert_array_equal(quat, quaternion)
    np.testing.assert_array_equal(result.path, trajectory[0])
    assert result.diagnostics["native_stuck"]
    assert result.diagnostics["embodiment"] == "dingo"
    assert result.diagnostics["is_real"]
    assert not result.stop  # Native recovery is not an added stop gate.
    assert XNavDPPolicy.requires_pose


def test_reset_preserves_native_threshold_and_clears_only_stateful_agents():
    """Verify reset() calls each upstream agent's reset with its native arguments, and stateless ones not at all."""
    for cls in (IPlannerPolicy, VIPlannerPolicy):
        p = cls(K, agent=Mock())
        p.reset()
        p.agent.reset.assert_not_called()
    p = NavDPPolicy(K, agent=Mock())
    p.reset()
    p.agent.reset.assert_called_once_with(1, -3.0)
    for cls in (XNavDPPolicy, ViNTPolicy):
        p = cls(K, agent=Mock())
        p.reset()
        p.agent.reset.assert_called_once_with(1)


def test_point_goal_policies_reject_malformed_goals(rgb, depth):
    """Verify missing, wrongly shaped, and non-finite point goals are rejected."""
    p = IPlannerPolicy(K, agent=Mock())
    for bad in [None, np.zeros((3,)), np.array([[np.nan, 0, 0]]), np.zeros((2, 3))]:
        with pytest.raises(ValueError, match="point goal"):
            p.step(_obs(rgb, depth, bad))
    p.agent.step_pointgoal.assert_not_called()


def test_unsupported_task_rejected():
    """Verify a policy refuses goal modalities it cannot consume."""
    with pytest.raises(ValueError, match="supports tasks"):
        IPlannerPolicy(K, task="imagegoal", agent=Mock())
    with pytest.raises(ValueError, match="supports tasks"):
        ViNTPolicy(K, task="pointgoal", agent=Mock())


def test_intrinsic_shape_validated():
    """Verify a malformed camera matrix is rejected at construction."""
    with pytest.raises(ValueError, match="intrinsic"):
        IPlannerPolicy(np.eye(4), agent=Mock())


def test_plan_rate_override():
    """Verify plan_hz falls back to the native rate unless overridden."""
    assert IPlannerPolicy(K, agent=Mock()).plan_hz == 5.0
    assert IPlannerPolicy(K, plan_hz=2.0, agent=Mock()).plan_hz == 2.0


def test_iplanner_depth_batch_units_and_input_ownership(monkeypatch):
    """Verify the NavDP fork's iPlanner depth preprocessing keeps metres, batches correctly, and never mutates input.

    Exercises the fix carried on the NavDP fork (``process_depth`` used to write through an expanded view).
    """
    torch = pytest.importorskip("torch")
    resize = pytest.importorskip("torchvision.transforms").Resize
    folder = default_checkpoint("iplanner").parents[1]
    if not (folder / "iplanner_agent.py").is_file():
        pytest.skip(f"NavDP iPlanner baseline not found at {folder}")
    monkeypatch.syspath_prepend(str(folder))
    from iplanner_agent import IPlannerAgent

    agent = IPlannerAgent.__new__(IPlannerAgent)
    agent.transform = resize((2, 3), antialias=None)
    agent.max_depth = 15
    depth = torch.tensor(
        [[[7.0, float("inf"), -1.0], [float("nan"), 16.0, 2.0]], [[3.0, 3.0, 3.0], [3.0, 3.0, 3.0]]]
    )
    original = depth.clone()
    prepared = agent.process_depth(depth)
    assert tuple(prepared.shape) == (2, 3, 2, 3)
    assert torch.isfinite(prepared).all()
    assert prepared[0, 0, 0, 0].item() == 7  # No uint16 / 6.55 m clipping.
    assert (prepared[1] == 3).all()
    torch.testing.assert_close(depth, original, equal_nan=True)
    torch.testing.assert_close(prepared[:, 0], prepared[:, 2])


# -- Registry ---------------------------------------------------------------------------------------------------------


def test_default_policies_registered():
    """Verify the five baselines are registered under underscore names."""
    assert list_policies() == ["iplanner", "navdp", "vint", "viplanner", "x_navdp"]


def test_unknown_policy_raises_keyerror():
    """Verify unknown names raise KeyError listing available policies."""
    with pytest.raises(KeyError, match="Available policies"):
        get_policy("nope", K)


def test_hyphenated_names_are_accepted(monkeypatch):
    """Verify upstream's hyphenated spelling resolves to the underscore registry entry."""
    created = {}

    class Dummy(InProcessPolicy):
        name = "dummy_bot"

        def step(self, obs):
            return Plan(np.zeros((1, 2)))

    register_policy("dummy-bot", lambda intrinsic, **kw: created.setdefault("p", Dummy(Mock(task="pointgoal"))))
    try:
        assert "dummy_bot" in list_policies()
        assert get_policy("dummy-bot", K) is created["p"]
    finally:
        from nav_arena.methods.in_process import registry

        registry._POLICY_REGISTRY.pop("dummy_bot", None)


# -- Loader -----------------------------------------------------------------------------------------------------------


@pytest.fixture
def fake_navdp(tmp_path):
    for folder in ("iplanner", "navdp", "x-navdp/eval/src"):
        (tmp_path / "baselines" / folder).mkdir(parents=True)
    return tmp_path


@pytest.fixture(autouse=True)
def _clean_loader_state():
    saved = list(sys.path)
    loader._reset_active_planner_for_tests()
    yield
    loader._reset_active_planner_for_tests()
    sys.path[:] = saved


def test_loader_adds_baseline_to_path_and_allows_same_planner_twice(fake_navdp):
    """Verify activating a baseline exposes its directory and re-activating the same one is allowed."""
    folder = loader.activate_planner("iplanner", fake_navdp)
    assert str(folder) in sys.path
    assert loader.active_planner() == "iplanner"
    assert loader.activate_planner("iplanner", fake_navdp) == folder


def test_loader_rejects_second_planner(fake_navdp):
    """Verify loading a different baseline in the same process raises PlannerConflictError."""
    loader.activate_planner("navdp", fake_navdp)
    with pytest.raises(loader.PlannerConflictError, match="one planner per process"):
        loader.activate_planner("x_navdp", fake_navdp)
    assert loader.active_planner() == "navdp"


def test_loader_xnavdp_adds_eval_src_first(fake_navdp):
    """Verify X-NavDP's eval/src (its policy_agent) takes precedence over the baseline root."""
    loader.activate_planner("x_navdp", fake_navdp)
    assert sys.path[0] == str(fake_navdp / "baselines" / "x-navdp" / "eval" / "src")


def test_loader_missing_checkout_is_actionable(tmp_path):
    """Verify a missing baseline directory explains how to fix it, and leaves no planner active."""
    with pytest.raises(FileNotFoundError, match="NAV_ARENA_NAVDP_ROOT"):
        loader.activate_planner("vint", tmp_path)
    assert loader.active_planner() is None
    with pytest.raises(KeyError):
        loader.activate_planner("bogus", tmp_path)


# -- Checkpoints ------------------------------------------------------------------------------------------------------


def test_default_checkpoint_paths(tmp_path):
    """Verify checkpoint paths follow NavDP/baselines/<folder>/checkpoints/<file>."""
    assert default_checkpoint("x_navdp", tmp_path) == tmp_path / "baselines/x-navdp/checkpoints/x-navdp_posttrain.ckpt"
    assert default_checkpoint("vint", tmp_path).name == "vint.pth"


def test_sha256_file(tmp_path):
    """Verify the digest helper against a known value."""
    f = tmp_path / "x.bin"
    f.write_bytes(b"abc")
    assert sha256_file(f) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


@pytest.mark.parametrize("planner", [k for k, v in CHECKPOINTS.items() if v.sha256])
def test_installed_checkpoints_match_recorded_hashes(planner):
    """Verify any installed checkpoint is the recorded file (skipped when the NavDP checkout lacks it)."""
    path = default_checkpoint(planner)
    if not path.is_file():
        pytest.skip(f"{path} not installed")
    assert sha256_file(path) == CHECKPOINTS[planner].sha256


def test_policy_modules_import_no_simulator_or_ros():
    """Verify importing the policy layer pulls in no Omniverse, ROS 2, or model-framework modules."""
    import subprocess

    code = (
        "import sys, nav_arena.methods\n"
        "bad = [m for m in sys.modules if m.split('.')[0] in ('isaacsim', 'omni', 'rclpy', 'mmdet', 'diffusers', 'carb')]\n"
        "sys.exit(1 if bad else 0)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
