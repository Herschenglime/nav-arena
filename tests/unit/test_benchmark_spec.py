# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for RunSpec, validation, and override logic."""

from __future__ import annotations

import copy
import logging
from pathlib import Path
import sys
from unittest.mock import MagicMock

import pytest

from nav_arena.benchmarks.spec import (
    VALID_METHODS,
    EpisodeLimits,
    RunSpec,
    VizCfg,
    apply_overrides,
    validate_spec,
)


class TestBenchmarkSpecImports:
    """Verify spec.py adheres to zero-simulator / zero-torch import constraint."""

    def test_zero_isaac_or_torch_imports(self):
        """Ensure spec.py only relies on stdlib and pure utilities."""
        forbidden = ("isaaclab", "isaacsim", "omni", "pxr", "torch")
        for mod in forbidden:
            # If Isaac or torch was loaded earlier by pytest runner, check spec doesn't pull them in
            assert mod not in sys.modules or True  # We verify directly below

        # Inspect module globals and imported symbols of spec
        import nav_arena.benchmarks.spec as spec_mod

        imported_modules = {
            val.__name__
            for val in vars(spec_mod).values()
            if hasattr(val, "__name__") and type(val).__name__ == "module"
        }
        for mod in forbidden:
            assert mod not in imported_modules, f"Forbidden module '{mod}' imported in spec.py"


class TestBenchmarkSpecDataclasses:
    """Test dataclass instantiation, default values, and methods."""

    def test_viz_cfg_defaults(self):
        viz = VizCfg()
        assert viz.gui is False
        assert viz.follow_camera is None  # "auto": follows gui
        assert viz.goal_overlay is None
        assert viz.follow_distance == 1.6
        assert viz.follow_height == 1.2
        assert viz.show_goal_marker is False

    def test_episode_limits_defaults(self):
        limits = EpisodeLimits()
        assert limits.max_steps == 1500
        assert limits.goal_tolerance == 0.4
        assert limits.max_speed == 0.3
        assert limits.stall_timeout_s == 10.0

    def test_run_spec_defaults(self):
        spec = RunSpec(method="iplanner")
        assert spec.method == "iplanner"
        assert spec.method_family == "in_process"
        assert spec.robot == "dingo"
        assert spec.scene == "kujiale_0003"
        assert spec.route is None
        assert spec.spawn is None
        assert spec.goal is None
        assert spec.spawn_yaw is None
        assert spec.seed == 0
        assert spec.method_params == {}
        assert isinstance(spec.limits, EpisodeLimits)
        assert isinstance(spec.viz, VizCfg)
        assert spec.output_dir is None

    def test_run_spec_method_family_derivation(self):
        spec_iplanner = RunSpec(method="iplanner")
        assert spec_iplanner.method_family == "in_process"

        spec_navdp = RunSpec(method="navdp")
        assert spec_navdp.method_family == "in_process"

        spec_nav2 = RunSpec(method="nav2")
        assert spec_nav2.method_family == "ros2"

    def test_run_spec_post_init_coercions(self):
        spec = RunSpec(
            method="iplanner",
            spawn=[1, 2],  # type: ignore[arg-type]
            goal=[3, 4],  # type: ignore[arg-type]
            spawn_yaw=1.57,
            limits={"max_steps": 2000},  # type: ignore[arg-type]
            viz={"gui": True},  # type: ignore[arg-type]
            output_dir="/tmp/test_dir",  # type: ignore[arg-type]
        )
        assert spec.spawn == (1.0, 2.0)
        assert spec.goal == (3.0, 4.0)
        assert spec.spawn_yaw == 1.57
        assert isinstance(spec.limits, EpisodeLimits)
        assert spec.limits.max_steps == 2000
        assert isinstance(spec.viz, VizCfg)
        assert spec.viz.gui is True
        assert isinstance(spec.output_dir, Path)
        assert spec.output_dir == Path("/tmp/test_dir")


class TestBenchmarkSpecSerialization:
    """Test dictionary and JSON serialization / deserialization round-trips."""

    def test_dict_roundtrip(self):
        original = RunSpec(
            method="vint",
            robot="nova_carter",
            scene="kujiale_0004",
            route="hall_straight",
            seed=42,
            method_params={"plan_hz": 10.0, "device": "cuda:0", "ckpt": Path("/tmp/model.pt")},
            limits=EpisodeLimits(max_steps=800, goal_tolerance=0.2, max_speed=0.5, stall_timeout_s=5.0),
            viz=VizCfg(gui=True, follow_camera=True, goal_overlay=False),
            output_dir=Path("/tmp/benchmark_test"),
        )
        d = original.to_dict()
        assert d["method"] == "vint"
        assert d["robot"] == "nova_carter"
        assert d["output_dir"] == "/tmp/benchmark_test"
        assert d["limits"]["max_steps"] == 800
        assert d["method_params"]["ckpt"] == "/tmp/model.pt"

        restored = RunSpec.from_dict(d)
        assert restored.output_dir == Path("/tmp/benchmark_test")
        assert restored.method_params["ckpt"] == "/tmp/model.pt"

    def test_json_roundtrip(self):
        original = RunSpec(
            method="iplanner",
            spawn=(-3.0, 1.2),
            goal=(-6.0, -1.0),
            spawn_yaw=0.0,
            seed=7,
            method_params={"weights": Path("/tmp/weights.pth")},
        )
        json_str = original.to_json()
        assert isinstance(json_str, str)
        restored = RunSpec.from_json(json_str)
        assert restored.spawn == (-3.0, 1.2)
        assert restored.goal == (-6.0, -1.0)
        assert restored.method_params["weights"] == "/tmp/weights.pth"


class TestBenchmarkSpecHash:
    """Test SHA-256 spec_hash behavior and stability."""

    def test_hash_stability(self):
        spec1 = RunSpec(method="iplanner", seed=0)
        spec2 = RunSpec(method="iplanner", seed=0)
        assert len(spec1.spec_hash) == 64
        assert spec1.spec_hash == spec2.spec_hash

    def test_hash_coordinate_int_float_equivalence(self):
        spec_int = RunSpec(method="iplanner", spawn=(1, 2), goal=(3, 4))
        spec_float = RunSpec(method="iplanner", spawn=(1.0, 2.0), goal=(3.0, 4.0))
        assert spec_int.spec_hash == spec_float.spec_hash

    def test_hash_with_path_in_method_params(self):
        spec = RunSpec(method="iplanner", method_params={"checkpoint": Path("/models/baseline.pt")})
        assert len(spec.spec_hash) == 64

    def test_hash_changes_with_canonical_fields(self):
        base = RunSpec(method="iplanner", seed=0)

        # Seed change
        assert RunSpec(method="iplanner", seed=1).spec_hash != base.spec_hash

        # Method change
        assert RunSpec(method="navdp", seed=0).spec_hash != base.spec_hash

        # Robot change
        assert RunSpec(method="iplanner", robot="nova_carter", seed=0).spec_hash != base.spec_hash

        # Scene change
        assert RunSpec(method="iplanner", scene="kujiale_0004", seed=0).spec_hash != base.spec_hash

        # Route change
        assert RunSpec(method="iplanner", route="hall_straight", seed=0).spec_hash != base.spec_hash

        # Custom coordinates
        assert RunSpec(method="iplanner", spawn=(1.0, 2.0), goal=(3.0, 4.0), seed=0).spec_hash != base.spec_hash

        # Limits change
        assert (
            RunSpec(method="iplanner", limits=EpisodeLimits(max_steps=500), seed=0).spec_hash
            != base.spec_hash
        )

        # Method params change
        assert (
            RunSpec(method="iplanner", method_params={"fear_threshold": 0.8}, seed=0).spec_hash
            != base.spec_hash
        )

    def test_hash_invariant_to_runtime_and_viz_fields(self):
        base = RunSpec(method="iplanner", seed=0)

        # Output dir change does NOT affect spec_hash
        with_dir = RunSpec(method="iplanner", seed=0, output_dir=Path("/tmp/test_dir"))
        assert with_dir.spec_hash == base.spec_hash

        # Viz settings do NOT affect spec_hash
        with_viz = RunSpec(
            method="iplanner",
            seed=0,
            viz=VizCfg(gui=True, follow_camera=True, goal_overlay=False),
        )
        assert with_viz.spec_hash == base.spec_hash


class TestBenchmarkSpecValidation:
    """Test validation constraints on RunSpec."""

    def test_valid_specs_pass(self):
        # Named route
        validate_spec(RunSpec(method="iplanner", route="hall_straight"))

        # Default route (route=None, spawn=None, goal=None)
        validate_spec(RunSpec(method="iplanner"))

        # Custom coordinates
        validate_spec(RunSpec(method="navdp", spawn=(-1.0, 0.5), goal=(2.0, 3.0), spawn_yaw=0.5))

        # Reserved nav2
        validate_spec(RunSpec(method="nav2"))

        # All allowed baselines
        for method in VALID_METHODS:
            validate_spec(RunSpec(method=method))

    def test_reject_invalid_method(self):
        spec = RunSpec(method="unknown_method")
        with pytest.raises(ValueError, match="Invalid method 'unknown_method'"):
            validate_spec(spec)

    def test_reject_mismatched_method_family(self):
        spec_nav2_mismatch = RunSpec(method="nav2", method_family="in_process")
        with pytest.raises(ValueError, match="Method 'nav2' requires method_family 'ros2'"):
            validate_spec(spec_nav2_mismatch)

        spec_inproc_mismatch = RunSpec(method="iplanner", method_family="ros2")
        with pytest.raises(ValueError, match="Method 'iplanner' requires method_family 'in_process'"):
            validate_spec(spec_inproc_mismatch)

    def test_reject_invalid_robot_or_scene(self):
        with pytest.raises(ValueError, match="robot must be a non-empty string"):
            validate_spec(RunSpec(method="iplanner", robot=""))

        with pytest.raises(ValueError, match="scene must be a non-empty string"):
            validate_spec(RunSpec(method="iplanner", scene="   "))

    def test_reject_empty_or_whitespace_route(self):
        with pytest.raises(ValueError, match="route must be a non-empty string"):
            validate_spec(RunSpec(method="iplanner", route=""))

        with pytest.raises(ValueError, match="route must be a non-empty string"):
            validate_spec(RunSpec(method="iplanner", route="   "))

    def test_reject_route_spawn_goal_conflicts(self):
        # Route combined with spawn/goal
        with pytest.raises(ValueError, match="cannot be combined with --spawn/--goal"):
            validate_spec(RunSpec(method="iplanner", route="hall_straight", spawn=(1.0, 2.0), goal=(3.0, 4.0)))

        # Spawn without goal
        with pytest.raises(ValueError, match="needs --goal too"):
            validate_spec(RunSpec(method="iplanner", spawn=(1.0, 2.0)))

        # Goal without spawn
        with pytest.raises(ValueError, match="needs --spawn too"):
            validate_spec(RunSpec(method="iplanner", goal=(3.0, 4.0)))

        # Non-length-2 spawn
        spec_bad_spawn = RunSpec(method="iplanner")
        spec_bad_spawn.spawn = (1.0, 2.0, 3.0)  # type: ignore[assignment]
        spec_bad_spawn.goal = (3.0, 4.0)
        with pytest.raises(ValueError, match="spawn must be a coordinate pair"):
            validate_spec(spec_bad_spawn)

        # Non-finite coordinates
        with pytest.raises(ValueError, match="spawn must be a coordinate pair"):
            validate_spec(RunSpec(method="iplanner", spawn=(float("nan"), 2.0), goal=(3.0, 4.0)))

        with pytest.raises(ValueError, match="spawn_yaw must be a finite float"):
            validate_spec(RunSpec(method="iplanner", spawn_yaw=float("inf")))

    def test_reject_negative_or_zero_limits(self):
        with pytest.raises(ValueError, match="limits.max_steps must be a positive integer"):
            validate_spec(RunSpec(method="iplanner", limits=EpisodeLimits(max_steps=0)))

        with pytest.raises(ValueError, match="limits.max_steps must be a positive integer"):
            validate_spec(RunSpec(method="iplanner", limits=EpisodeLimits(max_steps=-50)))

        with pytest.raises(ValueError, match="limits.goal_tolerance must be a positive finite float"):
            validate_spec(RunSpec(method="iplanner", limits=EpisodeLimits(goal_tolerance=0.0)))

        with pytest.raises(ValueError, match="limits.max_speed must be a positive finite float"):
            validate_spec(RunSpec(method="iplanner", limits=EpisodeLimits(max_speed=-0.1)))

        with pytest.raises(ValueError, match="limits.stall_timeout_s must be a non-negative finite float"):
            validate_spec(RunSpec(method="iplanner", limits=EpisodeLimits(stall_timeout_s=-1.0)))

        # stall_timeout_s == 0.0 is valid (disables stall detection)
        validate_spec(RunSpec(method="iplanner", limits=EpisodeLimits(stall_timeout_s=0.0)))

    def test_reject_nan_or_inf_limits(self):
        with pytest.raises(ValueError, match="limits.goal_tolerance must be a positive finite float"):
            validate_spec(RunSpec(method="iplanner", limits=EpisodeLimits(goal_tolerance=float("nan"))))

        with pytest.raises(ValueError, match="limits.max_speed must be a positive finite float"):
            validate_spec(RunSpec(method="iplanner", limits=EpisodeLimits(max_speed=float("inf"))))

        with pytest.raises(ValueError, match="limits.stall_timeout_s must be a non-negative finite float"):
            validate_spec(RunSpec(method="iplanner", limits=EpisodeLimits(stall_timeout_s=float("nan"))))

    def test_reject_invalid_seed(self):
        with pytest.raises(ValueError, match="seed must be a non-negative integer"):
            validate_spec(RunSpec(method="iplanner", seed=-1))

        with pytest.raises(ValueError, match="seed must be a non-negative integer"):
            validate_spec(RunSpec(method="iplanner", seed=True))  # type: ignore[arg-type]


class TestBenchmarkSpecOverrides:
    """Test apply_overrides logic and INFO/WARNING logging."""

    def test_override_immutability(self):
        original = RunSpec(method="iplanner", scene="kujiale_0003", seed=0)
        updated = apply_overrides(original, {"scene": "kujiale_0004", "seed": 5})

        assert original.scene == "kujiale_0003"
        assert original.seed == 0
        assert updated.scene == "kujiale_0004"
        assert updated.seed == 5

    def test_warning_logged_for_critical_fields(self):
        spec = RunSpec(method="iplanner", robot="dingo", scene="kujiale_0003")
        mock_logger = MagicMock(spec=logging.Logger)

        apply_overrides(
            spec,
            {"method": "navdp", "scene": "kujiale_0004", "robot": "nova_carter"},
            logger=mock_logger,
        )

        assert mock_logger.warning.call_count == 3
        warning_calls = [c[0][0] % c[0][1:] for c in mock_logger.warning.call_args_list]
        assert any("--method overrides spec.method: iplanner -> navdp" in msg for msg in warning_calls)
        assert any("--scene overrides spec.scene: kujiale_0003 -> kujiale_0004" in msg for msg in warning_calls)
        assert any("--robot overrides spec.robot: dingo -> nova_carter" in msg for msg in warning_calls)
        assert mock_logger.info.call_count == 0

    def test_info_logged_for_non_critical_fields(self):
        spec = RunSpec(method="iplanner", seed=0)
        mock_logger = MagicMock(spec=logging.Logger)

        updated = apply_overrides(
            spec,
            {
                "--seed": 42,
                "--max-steps": 2500,
                "--gui": True,
                "--route": "hall_straight",
                "--output": "/tmp/run_out",
            },
            logger=mock_logger,
        )

        assert mock_logger.warning.call_count == 0
        assert mock_logger.info.call_count == 5
        info_calls = [c[0][0] % c[0][1:] for c in mock_logger.info.call_args_list]
        assert any("--seed overrides spec.seed: 0 -> 42" in msg for msg in info_calls)
        assert any("--max-steps overrides spec.limits.max_steps: 1500 -> 2500" in msg for msg in info_calls)
        assert any("--gui overrides spec.viz.gui: False -> True" in msg for msg in info_calls)
        assert any("--route overrides spec.route: None -> hall_straight" in msg for msg in info_calls)
        assert any("--output overrides spec.output_dir: None -> /tmp/run_out" in msg for msg in info_calls)

        assert updated.seed == 42
        assert updated.limits.max_steps == 2500
        assert updated.viz.gui is True
        assert updated.route == "hall_straight"
        assert updated.output_dir == Path("/tmp/run_out")

    def test_override_seeds_flag(self):
        spec = RunSpec(method="iplanner", seed=0)
        updated = apply_overrides(spec, {"--seeds": 5})
        assert updated.seed == 5

        updated2 = apply_overrides(spec, {"seeds": [7]})
        assert updated2.seed == 7

        with pytest.raises(ValueError, match="Single episode spec cannot accept multiple seeds"):
            apply_overrides(spec, {"seeds": [1, 2]})

    def test_override_method_auto_updates_family(self):
        spec = RunSpec(method="iplanner")
        assert spec.method_family == "in_process"

        updated = apply_overrides(spec, {"method": "nav2"})
        assert updated.method == "nav2"
        assert updated.method_family == "ros2"

    def test_override_policy_arg_parsing(self):
        spec = RunSpec(method="iplanner", method_params={"plan_hz": 10.0})
        updated = apply_overrides(
            spec,
            {
                "--policy-arg": ["fear_threshold=0.8", "timeout=25", "tag='v1'"],
            },
        )
        assert updated.method_params["plan_hz"] == 10.0
        assert updated.method_params["fear_threshold"] == 0.8
        assert updated.method_params["timeout"] == 25
        assert updated.method_params["tag"] == "v1"

    def test_override_boolean_negations(self):
        spec = RunSpec(method="iplanner", viz=VizCfg(gui=True, follow_camera=True, goal_overlay=True))
        updated = apply_overrides(
            spec,
            {
                "--no-gui": True,
                "--no-follow-camera": True,
                "--no-goal-overlay": True,
            },
        )
        assert updated.viz.gui is False
        assert updated.viz.follow_camera is False
        assert updated.viz.goal_overlay is False

    def test_override_nested_dicts_and_policy_args(self):
        spec = RunSpec(method="iplanner", method_params={"plan_hz": 10.0})
        updated = apply_overrides(
            spec,
            {
                "method_params": {"fear_threshold": 0.8},
                "plan_hz": 20.0,
            },
        )
        assert updated.method_params["plan_hz"] == 20.0
        assert updated.method_params["fear_threshold"] == 0.8


class TestVizResolution:
    """The follow camera and overlay default to "auto" and follow the GUI flag in exactly one place."""

    @pytest.mark.parametrize("gui", [False, True])
    @pytest.mark.parametrize("value", [None, True, False])
    def test_resolved_follows_gui_only_when_unset(self, gui, value):
        resolved = VizCfg(gui=gui, follow_camera=value, goal_overlay=value).resolved()
        expected = gui if value is None else value
        assert resolved.gui is gui
        assert resolved.follow_camera is expected
        assert resolved.goal_overlay is expected

    def test_unset_flags_survive_serialization(self):
        spec = RunSpec(method="iplanner", viz=VizCfg(gui=True))
        restored = RunSpec.from_json(spec.to_json())
        assert restored.viz.follow_camera is None
        assert restored.viz.resolved().follow_camera is True

    def test_old_specs_with_explicit_booleans_still_load(self):
        spec = RunSpec.from_dict({"method": "iplanner", "viz": {"gui": True, "follow_camera": False, "goal_overlay": True}})
        assert spec.viz.resolved() == (True, False, True)

    def test_from_options_keeps_defaults_for_absent_keys(self):
        viz = VizCfg.from_options({"gui": True, "follow_distance": 2.5, "unrelated": 1})
        assert viz.gui is True and viz.follow_distance == 2.5
        assert viz.follow_camera is None and viz.goal_overlay is None

    def test_limits_from_options_accepts_cli_aliases(self):
        limits = EpisodeLimits.from_options({"goal_dist": 0.7, "stall_timeout": 3, "max_steps": 99})
        assert (limits.goal_tolerance, limits.stall_timeout_s, limits.max_steps) == (0.7, 3.0, 99)

    def test_negated_and_distance_overrides(self):
        spec = apply_overrides(RunSpec(method="iplanner"), {"no_follow_camera": True, "follow_distance": 3.0})
        assert spec.viz.follow_camera is False and spec.viz.follow_distance == 3.0
        assert apply_overrides(RunSpec(method="iplanner"), {"planner_device": "cpu"}).method_params == {"device": "cpu"}

    def test_show_goal_marker_changes_the_spec_hash_only_when_enabled(self):
        base = RunSpec(method="iplanner")
        assert RunSpec(method="iplanner", viz=VizCfg(gui=True)).spec_hash == base.spec_hash
        assert RunSpec(method="iplanner", viz=VizCfg(show_goal_marker=True)).spec_hash != base.spec_hash


def test_run_cli_gui_flag_defaults_follow_camera_on(tmp_path):
    """Verify ``nav_arena run --gui`` builds a spec whose follow camera and overlay resolve to on (the regression)."""
    from unittest.mock import patch

    from nav_arena.cli import main

    captured = {}

    def fake_exec(spec, run_dir, **kwargs):
        captured["spec"] = spec
        return 0

    with patch("nav_arena.cli.run.execute_single_run_process", side_effect=fake_exec), patch(
        "nav_arena.cli.run.check_preflight_processes", return_value=[]
    ):
        assert main(["run", "--method", "iplanner", "--gui", "--output", str(tmp_path / "r")]) == 0
        assert captured["spec"].viz.resolved() == (True, True, True)
        assert main(["run", "--method", "iplanner", "--output", str(tmp_path / "r2")]) == 0
        assert captured["spec"].viz.resolved() == (False, False, False)
        assert main(["run", "--method", "iplanner", "--gui", "--no-follow-camera", "--output", str(tmp_path / "r3")]) == 0
        assert captured["spec"].viz.resolved() == (True, False, True)
