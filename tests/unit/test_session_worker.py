# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for SessionKey, EpisodeSpec, RunSession, worker subprocess, and verify_baseline shim."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from nav_arena.benchmarks.session import (
    EpisodeSpec,
    RunSession,
    SessionKey,
    print_result_summary,
)
from nav_arena.benchmarks.spec import EpisodeLimits, RunSpec, VizCfg
from nav_arena.benchmarks.worker import (
    load_run_spec,
    main as worker_main,
)
from nav_arena.scripts.verify_baseline import _parse_policy_args, build_spec_from_cli


@dataclass
class DummyResult:
    """Mock EpisodeResult for testing."""

    terminal_cause: str
    success: bool
    steps: int = 100
    sim_time_s: float = 2.0
    time_to_goal_s: float | None = 2.0
    initial_goal_distance_m: float = 5.0
    final_goal_distance_m: float = 0.1
    path_length_m: float = 5.2
    plans: int = 10
    stop_requests: int = 0
    mean_inference_ms: float = 12.0
    wall_time_s: float = 3.5


def _sample_spec(method: str = "iplanner", route: str | None = "hall_straight") -> RunSpec:
    return RunSpec(
        method=method,
        robot="dingo",
        scene="kujiale_0003",
        route=route,
        seed=42,
        method_params={"plan_hz": 5.0, "fear_threshold": 0.5},
        limits=EpisodeLimits(max_steps=1200, goal_tolerance=0.35, max_speed=0.3, stall_timeout_s=8.0),
        viz=VizCfg(gui=False, follow_camera=False, goal_overlay=True),
        output_dir=Path("/tmp/runs/test_run"),
    )


# ---------------------------------------------------------------------------
# SessionKey Tests
# ---------------------------------------------------------------------------


def test_session_key_from_spec():
    spec = _sample_spec()
    key = SessionKey.from_spec(spec)

    assert key.scene == "kujiale_0003"
    assert key.robot == "dingo"
    assert key.method == "iplanner"
    assert key.method_params == {"plan_hz": 5.0, "fear_threshold": 0.5}
    assert key.viz == VizCfg(gui=False, follow_camera=False, goal_overlay=True)


def test_session_key_equality_and_hash():
    spec1 = _sample_spec()
    spec2 = _sample_spec(route="around_table")  # different route, same invariant key
    spec3 = _sample_spec(method="viplanner")

    key1 = SessionKey.from_spec(spec1)
    key2 = SessionKey.from_spec(spec2)
    key3 = SessionKey.from_spec(spec3)

    assert key1 == key2
    assert hash(key1) == hash(key2)
    assert key1 != key3
    assert hash(key1) != hash(key3)

    # Verify key can be stored in sets and dicts
    lookup = {key1: "session_1"}
    assert lookup[key2] == "session_1"
    key_set = {key1, key2, key3}
    assert len(key_set) == 2


# ---------------------------------------------------------------------------
# EpisodeSpec Tests
# ---------------------------------------------------------------------------


def test_episode_spec_from_spec_named_route():
    spec = _sample_spec(route="hall_straight")
    ep = EpisodeSpec.from_spec(spec)

    assert ep.route == "hall_straight"
    assert ep.spawn == (-6.4, 0.5)
    assert ep.goal == (-0.4, 0.5)
    assert ep.seed == 42
    assert ep.limits.max_steps == 1200
    assert ep.limits.goal_tolerance == 0.35
    assert ep.output_dir == Path("/tmp/runs/test_run")
    assert ep.spawn_yaw is not None


def test_episode_spec_from_spec_custom_coordinates():
    spec = RunSpec(
        method="iplanner",
        robot="dingo",
        scene="kujiale_0003",
        spawn=(-2.0, 1.0),
        goal=(2.0, 1.0),
        spawn_yaw=None,
        seed=10,
    )
    ep = EpisodeSpec.from_spec(spec)

    assert ep.route == "custom"
    assert ep.spawn == (-2.0, 1.0)
    assert ep.goal == (2.0, 1.0)
    assert pytest.approx(ep.spawn_yaw) == 0.0  # pointing towards +X
    assert ep.seed == 10


def test_episode_spec_unknown_route():
    spec = RunSpec(
        method="iplanner",
        robot="dingo",
        scene="kujiale_0003",
        route="non_existent_route_xyz",
    )
    with pytest.raises(KeyError, match="non_existent_route_xyz"):
        EpisodeSpec.from_spec(spec)


# ---------------------------------------------------------------------------
# RunSession Tests (nav2 reserved, mocking execution)
# ---------------------------------------------------------------------------


def test_run_session_nav2_reserved_not_implemented():
    spec = RunSpec(
        method="nav2",
        method_family="ros2",
        robot="dingo",
        scene="kujiale_0003",
        route="hall_straight",
    )
    session = RunSession(spec)
    with pytest.raises(NotImplementedError, match="Nav2 backend is reserved and not yet implemented"):
        session.start()


def test_run_session_mocked_execution(tmp_path):
    spec = _sample_spec(route="hall_straight")
    spec.output_dir = tmp_path / "run_dir"

    mock_app = MagicMock()
    mock_task = MagicMock()
    mock_task.device = "cpu"
    mock_task.get_camera_intrinsics.return_value = MagicMock()
    mock_policy = MagicMock()
    dummy_res = DummyResult(terminal_cause="goal_reached", success=True)

    with (
        patch("nav_arena.tasks.PointNavTask", return_value=mock_task) as mock_task_cls,
        patch("nav_arena.tasks.create_point_nav_env_cfg", return_value=MagicMock()) as mock_cfg_fn,
        patch("nav_arena.methods.in_process.get_policy", return_value=mock_policy) as mock_get_policy,
        patch("nav_arena.methods.in_process.run_episode", return_value=dummy_res) as mock_run_ep,
    ):
        with RunSession(spec, simulation_app=mock_app) as session:
            assert session.is_open
            mock_cfg_fn.assert_called_once()
            mock_task_cls.assert_called_once()
            assert mock_task_cls.call_args.kwargs["cfg"].seed == spec.seed
            mock_get_policy.assert_called_once()

            result = session.run_episode()
            assert result.success
            assert result.terminal_cause == "goal_reached"
            mock_run_ep.assert_called_once()

            # Verify episode_cfg passed to run_episode
            call_args = mock_run_ep.call_args[0]
            assert call_args[0] == mock_task
            assert call_args[1] == mock_policy
            episode_cfg = call_args[2]
            assert episode_cfg.max_steps == 1200
            assert episode_cfg.follower.goal_tolerance == 0.35
            assert episode_cfg.output_dir == spec.output_dir

        # After exiting context, task should be closed
        mock_task.close.assert_called_once()
        assert not session.is_open


def test_print_result_summary():
    result = DummyResult(terminal_cause="goal_reached", success=True)
    mock_logger = MagicMock()
    print_result_summary(result, output_dir=Path("/tmp/test"), custom_logger=mock_logger)
    mock_logger.section.assert_called_with("RESULT")
    mock_logger.check.assert_called_with("Goal reached", True, "cause=goal_reached")


# ---------------------------------------------------------------------------
# Worker Tests
# ---------------------------------------------------------------------------


def test_worker_load_run_spec_from_json_string():
    spec = _sample_spec()
    spec_json = spec.to_json()

    loaded = load_run_spec(spec_json)
    assert loaded.method == "iplanner"
    assert loaded.route == "hall_straight"
    assert loaded.seed == 42


def test_worker_load_run_spec_from_file(tmp_path):
    spec = _sample_spec(route="around_table")
    json_path = tmp_path / "test_spec.json"
    json_path.write_text(spec.to_json(), encoding="utf-8")

    loaded = load_run_spec(str(json_path))
    assert loaded.method == "iplanner"
    assert loaded.route == "around_table"
    assert loaded.seed == 42


def test_worker_load_run_spec_invalid_json():
    with pytest.raises(ValueError, match="Failed to deserialize RunSpec JSON"):
        load_run_spec("not a json string")


def test_worker_load_run_spec_validation_failure():
    bad_spec = {"method": "invalid_method_xyz", "robot": "dingo", "scene": "kujiale_0003"}
    with pytest.raises(ValueError, match="Invalid method"):
        load_run_spec(json.dumps(bad_spec))


def test_worker_exit_code_goal_reached(tmp_path):
    spec = _sample_spec()
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(spec.to_json())

    dummy_res = DummyResult(terminal_cause="goal_reached", success=True)
    mock_session = MagicMock()
    mock_session.__enter__.return_value = mock_session
    mock_session.run_episode.return_value = dummy_res

    with patch("nav_arena.benchmarks.worker.RunSession", return_value=mock_session):
        with pytest.raises(SystemExit) as exc_info:
            worker_main([str(spec_path)])
        assert exc_info.value.code == 0


def test_worker_exit_code_failure_collision(tmp_path):
    spec = _sample_spec()
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(spec.to_json())

    dummy_res = DummyResult(terminal_cause="collision", success=False)
    mock_session = MagicMock()
    mock_session.__enter__.return_value = mock_session
    mock_session.run_episode.return_value = dummy_res

    with patch("nav_arena.benchmarks.worker.RunSession", return_value=mock_session):
        with pytest.raises(SystemExit) as exc_info:
            worker_main([str(spec_path)])
        assert exc_info.value.code == 2


def test_worker_exit_code_infra_error():
    with pytest.raises(SystemExit) as exc_info:
        worker_main(["invalid_file_or_json_content"])
    assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# verify_baseline Shim Tests
# ---------------------------------------------------------------------------


def test_parse_policy_args():
    args = ["fear_threshold=0.7", "name='test'", "count=10", "enabled=True"]
    parsed = _parse_policy_args(args)
    assert parsed["fear_threshold"] == 0.7
    assert parsed["name"] == "test"
    assert parsed["count"] == 10
    assert parsed["enabled"] is True


def test_parse_policy_args_invalid():
    with pytest.raises(ValueError, match="KEY=VALUE"):
        _parse_policy_args(["invalid_format_without_equals"])


def test_build_spec_from_cli():
    mock_args = argparse.Namespace(
        method="iplanner",
        robot="dingo",
        scene="kujiale_0003",
        route="hall_straight",
        spawn=None,
        goal=None,
        spawn_yaw=None,
        task="pointgoal",
        max_steps=2000,
        goal_dist=0.5,
        max_speed=0.4,
        plan_hz=10.0,
        planner_device="cuda:0",
        seed=7,
        policy_arg=["test_k=123"],
        stall_timeout=15.0,
        follow_camera=True,
        follow_distance=2.0,
        follow_height=1.5,
        goal_overlay=False,
        show_goal_marker=False,
        output=Path("/tmp/custom_run"),
        visualizer=None,
    )
    spec = build_spec_from_cli(mock_args)

    assert spec.method == "iplanner"
    assert spec.robot == "dingo"
    assert spec.route == "hall_straight"
    assert spec.seed == 7
    assert spec.method_params["task"] == "pointgoal"
    assert spec.method_params["plan_hz"] == 10.0
    assert spec.method_params["device"] == "cuda:0"
    assert spec.viz.follow_distance == 2.0
    assert spec.viz.follow_height == 1.5
    assert "follow_distance" not in spec.method_params
    assert spec.method_params["test_k"] == 123
    assert spec.limits.max_steps == 2000
    assert spec.limits.goal_tolerance == 0.5
    assert spec.limits.max_speed == 0.4
    assert spec.limits.stall_timeout_s == 15.0
    assert spec.viz.follow_camera is True
    assert spec.viz.goal_overlay is False
    assert spec.output_dir == Path("/tmp/custom_run")


def test_session_key_dict_viz():
    key = SessionKey(
        scene="kujiale_0003",
        robot="dingo",
        method="iplanner",
        viz={"gui": True, "follow_camera": True, "goal_overlay": False},  # type: ignore
    )
    assert isinstance(key.viz, VizCfg)
    assert key.viz.gui is True
    assert isinstance(key.key_hash, str)
    assert len(key.key_hash) == 64


def test_episode_spec_lone_spawn_or_goal_raises():
    spec_lone_spawn = RunSpec(
        method="iplanner",
        robot="dingo",
        scene="kujiale_0003",
        spawn=(-1.0, 1.0),
        goal=None,
    )
    with pytest.raises(ValueError, match="Both spawn and goal must be specified together"):
        EpisodeSpec.from_spec(spec_lone_spawn)

    spec_lone_goal = RunSpec(
        method="iplanner",
        robot="dingo",
        scene="kujiale_0003",
        spawn=None,
        goal=(1.0, 1.0),
    )
    with pytest.raises(ValueError, match="Both spawn and goal must be specified together"):
        EpisodeSpec.from_spec(spec_lone_goal)


def test_run_session_initialized_with_key_and_run_episode_with_spec(tmp_path):
    spec = _sample_spec(route="hall_straight")
    key = SessionKey.from_spec(spec)
    ep = EpisodeSpec.from_spec(spec)

    mock_app = MagicMock()
    mock_task = MagicMock()
    mock_task.device = "cpu"
    mock_task.get_camera_intrinsics.return_value = MagicMock()
    mock_policy = MagicMock()
    dummy_res = DummyResult(terminal_cause="goal_reached", success=True)

    with (
        patch("nav_arena.tasks.PointNavTask", return_value=mock_task),
        patch("nav_arena.tasks.create_point_nav_env_cfg", return_value=MagicMock()),
        patch("nav_arena.methods.in_process.get_policy", return_value=mock_policy),
        patch("nav_arena.methods.in_process.run_episode", return_value=dummy_res) as mock_run_ep,
    ):
        session = RunSession(key, simulation_app=mock_app)
        assert not session.is_open
        result = session.run_episode(ep)
        assert session.is_open
        assert result.success
        mock_run_ep.assert_called_once()
        session.close()
        assert not session.is_open


def test_run_session_startup_failure_cleans_up():
    spec = _sample_spec()
    mock_app_cm = MagicMock()
    mock_app_cm.__enter__.return_value = MagicMock()

    with (
        patch("nav_arena.benchmarks.session.launch_simulation_app", return_value=mock_app_cm),
        patch("nav_arena.tasks.PointNavTask", side_effect=RuntimeError("GPU allocation failed")),
        patch("nav_arena.tasks.create_point_nav_env_cfg", return_value=MagicMock()),
    ):
        session = RunSession(spec)
        with pytest.raises(RuntimeError, match="GPU allocation failed"):
            session.start()

        # launch_simulation_app context manager MUST be closed with exc info
        mock_app_cm.__exit__.assert_called_once()
        assert not session.is_open


def test_worker_merge_remaining_cli_args(tmp_path):
    spec = _sample_spec()
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(spec.to_json())

    dummy_res = DummyResult(terminal_cause="goal_reached", success=True)
    mock_session = MagicMock()
    mock_session.__enter__.return_value = mock_session
    mock_session.run_episode.return_value = dummy_res

    with patch("nav_arena.benchmarks.worker.RunSession", return_value=mock_session) as mock_session_cls:
        with pytest.raises(SystemExit) as exc_info:
            worker_main([str(spec_path), "--headless", "--device", "cuda:1"])
        assert exc_info.value.code == 0
        call_kwargs = mock_session_cls.call_args[1]
        args_cli = call_kwargs["args_cli"]
        assert getattr(args_cli, "headless") is True
        assert getattr(args_cli, "device") == "cuda:1"


def test_build_spec_from_cli_headless_overrides_viz_kit():
    mock_args = argparse.Namespace(
        method="iplanner",
        robot="dingo",
        scene="kujiale_0003",
        visualizer=["kit"],
        headless=True,
    )
    spec = build_spec_from_cli(mock_args)
    assert spec.viz.gui is False



@pytest.mark.parametrize(
    ("gui", "follow", "overlay", "expect_viewer", "expect_overlay"),
    [
        (False, None, None, False, False),  # headless default: no display aids
        (True, None, None, True, True),  # --gui: follow camera and overlay on by default
        (True, False, None, False, True),  # --gui --no-follow-camera
        (False, True, True, True, True),  # explicit opt-in without a GUI
    ],
)
def test_run_session_builds_viewer_and_overlay_from_resolved_viz(tmp_path, gui, follow, overlay, expect_viewer, expect_overlay):
    """Verify the follow camera and goal overlay are created exactly when the resolved viz settings ask for them."""
    spec = _sample_spec(route="hall_straight")
    spec.output_dir = tmp_path / "run_dir"
    spec.viz = VizCfg(gui=gui, follow_camera=follow, goal_overlay=overlay)
    mock_task = MagicMock()
    mock_task.device = "cpu"
    dummy_res = DummyResult(terminal_cause="goal_reached", success=True)

    with (
        patch("nav_arena.tasks.PointNavTask", return_value=mock_task),
        patch("nav_arena.tasks.create_point_nav_env_cfg", return_value=MagicMock()),
        patch("nav_arena.methods.in_process.get_policy", return_value=MagicMock()),
        patch("nav_arena.methods.in_process.run_episode", return_value=dummy_res) as mock_run_ep,
        patch("nav_arena.utils.viewer.ThirdPersonView") as mock_view,
        patch("nav_arena.utils.viewer.DebugOverlay") as mock_overlay,
    ):
        with RunSession(spec, simulation_app=MagicMock()) as session:
            session.run_episode()
        episode_cfg = mock_run_ep.call_args[0][2]

    assert (episode_cfg.viewer is not None) == expect_viewer
    assert (episode_cfg.overlay is not None) == expect_overlay
    assert mock_view.called == expect_viewer
    assert mock_overlay.called == expect_overlay
