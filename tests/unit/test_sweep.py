# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for sweep matrix planner, batch tracking, CLI wiring, and execution."""

from __future__ import annotations

from contextlib import contextmanager
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Generator
from unittest.mock import MagicMock, patch

import pytest
import yaml

from nav_arena.benchmarks.manifest import BatchManifest, RunRecord, RunStatus
from nav_arena.benchmarks.spec import VALID_METHODS, EpisodeLimits, RunSpec, VizCfg
from nav_arena.benchmarks.sweep import (
    RESULTS_CSV_COLUMNS,
    SweepSpec,
    append_results_row,
    apply_sweep_overrides,
    check_preflight_processes,
    compute_default_timeout,
    execute_sweep,
    expand_sweep_matrix,
    format_results_row,
    get_git_info,
    get_host_info,
    resolve_batch_for_resume,
    setup_batch_directory,
    validate_sweep_spec,
)
from nav_arena.cli import main


class TestSweepImports:
    """Verify sweep.py adheres to zero-simulator / zero-torch import constraint."""

    def test_zero_isaac_or_torch_imports(self):
        """Ensure sweep.py relies exclusively on lightweight stdlib and utilities."""
        forbidden = ("isaaclab", "isaacsim", "omni", "pxr", "torch", "rclpy")
        import nav_arena.benchmarks.sweep as sweep_mod

        for name, val in vars(sweep_mod).items():
            mod = getattr(val, "__module__", "")
            for f in forbidden:
                assert not mod.startswith(f), f"Forbidden import '{f}' detected in sweep symbol '{name}'"

    def test_zero_heavy_imports_in_fresh_process(self):
        """Verify importing sweep in clean interpreter loads zero simulation or deep learning modules."""
        code = (
            "import sys\n"
            "import nav_arena.benchmarks.sweep\n"
            "heavy = {'torch', 'isaaclab', 'isaacsim', 'omni', 'pxr', 'rclpy'}\n"
            "loaded = heavy.intersection(sys.modules.keys())\n"
            "assert not loaded, f'Heavy modules loaded: {loaded}'\n"
        )
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert res.returncode == 0, f"Subprocess import failed:\n{res.stderr}"


class TestSweepSpec:
    """Tests for SweepSpec data model, validation, and overrides."""

    def test_default_sweep_spec(self):
        spec = SweepSpec(
            name="test_sweep",
            methods=["iplanner"],
            routes=["hall_straight"],
        )
        assert spec.name == "test_sweep"
        assert spec.scene == "kujiale_0003"
        assert spec.robots == ["dingo"]
        assert spec.methods == ["iplanner"]
        assert spec.routes == ["hall_straight"]
        assert spec.seeds == [0]
        assert spec.options == {}
        assert spec.method_params == {}
        assert spec.timeout_s is None

    def test_serialization_roundtrip(self):
        spec = SweepSpec(
            name="kujiale_study",
            scene="kujiale_0004",
            robots=["dingo", "nova_carter"],
            methods=["iplanner", "navdp"],
            routes=["hall_straight", {"name": "custom1", "spawn": [1.0, 2.0], "goal": [3.0, 4.0]}],
            seeds=[0, 1],
            options={"max_steps": 1200, "goal_dist": 0.35},
            method_params={"iplanner": {"fear_threshold": 0.75}},
            timeout_s=150.0,
        )
        d = spec.to_dict()
        loaded = SweepSpec.from_dict(d)
        assert loaded.name == spec.name
        assert loaded.scene == spec.scene
        assert loaded.robots == spec.robots
        assert loaded.methods == spec.methods
        assert loaded.routes == spec.routes
        assert loaded.seeds == spec.seeds
        assert loaded.options == spec.options
        assert loaded.method_params == spec.method_params
        assert loaded.timeout_s == spec.timeout_s

        yaml_str = spec.to_yaml()
        loaded_yaml = SweepSpec.from_yaml(yaml_str)
        assert loaded_yaml == loaded

    def test_load_from_yaml_file(self, tmp_path: Path):
        yaml_content = """
name: sample_yaml_sweep
scene: kujiale_0003
robots: [dingo]
methods: [iplanner, viplanner]
routes: [hall_straight, around_table]
seeds: [10, 20]
options:
  max_steps: 1000
timeout_s: 120.0
"""
        spec_file = tmp_path / "sweep.yaml"
        spec_file.write_text(yaml_content, encoding="utf-8")
        spec = SweepSpec.load(spec_file)
        assert spec.name == "sample_yaml_sweep"
        assert spec.methods == ["iplanner", "viplanner"]
        assert spec.seeds == [10, 20]
        assert spec.timeout_s == 120.0

    def test_baselines_kujiale_0003_yaml(self):
        sweep_yaml_path = Path(__file__).resolve().parents[2] / "sweeps" / "baselines_kujiale_0003.yaml"
        assert sweep_yaml_path.is_file()
        spec = SweepSpec.load(sweep_yaml_path)
        validate_sweep_spec(spec)
        assert spec.name == "baselines_kujiale_0003"
        assert spec.scene == "kujiale_0003"
        assert spec.robots == ["dingo"]
        assert spec.methods == ["iplanner", "navdp", "x_navdp", "viplanner"]
        assert len(spec.routes) == 4
        assert len(spec.seeds) == 3
        assert spec.timeout_s == 180.0
        runs = expand_sweep_matrix(spec)
        assert len(runs) == 1 * 4 * 4 * 3  # 48 runs
        assert runs[0][0] == "001_iplanner_dingo_hall_straight_s0"
        assert runs[-1][0] == "048_viplanner_dingo_to_far_room_s2"

    def test_validation_passes_valid_spec(self):
        spec = SweepSpec(
            name="valid_spec",
            scene="kujiale_0003",
            robots=["dingo"],
            methods=["iplanner", "nav2"],
            routes=["hall_straight", {"name": "c1", "spawn": [0.0, 0.0], "goal": [1.0, 1.0]}],
            seeds=[0, 1],
            options={"goal_tolerance": 0.4, "max_speed": 0.3, "max_steps": 1500, "stall_timeout_s": 10.0},
            timeout_s=100.0,
        )
        validate_sweep_spec(spec)

    def test_validation_errors(self):
        with pytest.raises(ValueError, match="Sweep name must be a non-empty string"):
            validate_sweep_spec(SweepSpec(name="", methods=["iplanner"], routes=["r1"]))

        with pytest.raises(ValueError, match="Sweep scene must be a non-empty string"):
            validate_sweep_spec(SweepSpec(name="test", scene="", methods=["iplanner"], routes=["r1"]))

        with pytest.raises(ValueError, match="Sweep robots must be a non-empty list"):
            validate_sweep_spec(SweepSpec(name="test", robots=[], methods=["iplanner"], routes=["r1"]))

        with pytest.raises(ValueError, match="Invalid method 'invalid_method'"):
            validate_sweep_spec(SweepSpec(name="test", methods=["invalid_method"], routes=["r1"]))

        with pytest.raises(ValueError, match="Sweep routes must be a non-empty list"):
            validate_sweep_spec(SweepSpec(name="test", methods=["iplanner"], routes=[]))

        with pytest.raises(ValueError, match="Inline route.*must specify.*spawn"):
            validate_sweep_spec(SweepSpec(name="test", methods=["iplanner"], routes=[{"name": "bad", "goal": [1, 2]}]))

        with pytest.raises(ValueError, match="Seed must be a non-negative integer"):
            validate_sweep_spec(SweepSpec(name="test", methods=["iplanner"], routes=["r1"], seeds=[-1]))

        with pytest.raises(ValueError, match="timeout_s must be a positive number"):
            validate_sweep_spec(SweepSpec(name="test", methods=["iplanner"], routes=["r1"], timeout_s=-10.0))

    def test_apply_sweep_overrides(self):
        base = SweepSpec(
            name="base_sweep",
            scene="kujiale_0003",
            robots=["dingo"],
            methods=["iplanner"],
            routes=["hall_straight"],
            seeds=[0],
            options={"max_steps": 1500},
            timeout_s=100.0,
        )
        overrides = {
            "name": "overridden_sweep",
            "scene": "kujiale_0004",
            "robots": "dingo,nova_carter",
            "methods": "iplanner,navdp",
            "routes": "hall_straight,around_table",
            "seeds": "1,2,3",
            "timeout_s": 200.0,
            "max_steps": 2000,
            "goal_dist": 0.5,
        }
        updated = apply_sweep_overrides(base, overrides)
        assert updated.name == "overridden_sweep"
        assert updated.scene == "kujiale_0004"
        assert updated.robots == ["dingo", "nova_carter"]
        assert updated.methods == ["iplanner", "navdp"]
        assert updated.routes == ["hall_straight", "around_table"]
        assert updated.seeds == [1, 2, 3]
        assert updated.timeout_s == 200.0
        assert updated.options["max_steps"] == 2000
        assert updated.options["goal_tolerance"] == 0.5


class TestMatrixExpansion:
    """Test Cartesian matrix expansion and RunSpec generation."""

    def test_cartesian_product_and_naming(self):
        spec = SweepSpec(
            name="baselines",
            scene="kujiale_0003",
            robots=["dingo"],
            methods=["iplanner", "navdp"],
            routes=["hall_straight", "around_table"],
            seeds=[0, 1],
            options={"max_steps": 1200, "goal_dist": 0.35, "max_speed": 0.25, "stall_timeout_s": 8.0},
            method_params={"iplanner": {"fear_threshold": 0.75}},
        )
        runs = expand_sweep_matrix(spec)
        assert len(runs) == 8  # 1 robot * 2 methods * 2 routes * 2 seeds = 8

        expected_ids = [
            "001_iplanner_dingo_hall_straight_s0",
            "002_iplanner_dingo_hall_straight_s1",
            "003_iplanner_dingo_around_table_s0",
            "004_iplanner_dingo_around_table_s1",
            "005_navdp_dingo_hall_straight_s0",
            "006_navdp_dingo_hall_straight_s1",
            "007_navdp_dingo_around_table_s0",
            "008_navdp_dingo_around_table_s1",
        ]
        actual_ids = [r[0] for r in runs]
        assert actual_ids == expected_ids

        run_id, r_spec, _label = runs[0]
        assert run_id == "001_iplanner_dingo_hall_straight_s0"
        assert r_spec.method == "iplanner"
        assert r_spec.robot == "dingo"
        assert r_spec.scene == "kujiale_0003"
        assert r_spec.route == "hall_straight"
        assert r_spec.seed == 0
        assert r_spec.method_params == {"fear_threshold": 0.75}
        assert r_spec.limits.max_steps == 1200
        assert r_spec.limits.goal_tolerance == 0.35
        assert r_spec.limits.max_speed == 0.25
        assert r_spec.limits.stall_timeout_s == 8.0

        run_id_navdp, r_spec_navdp, _ = runs[4]
        assert r_spec_navdp.method == "navdp"
        assert r_spec_navdp.method_params == {}

    def test_inline_route_expansion(self):
        spec = SweepSpec(
            name="inline_route_sweep",
            scene="kujiale_0003",
            robots=["dingo"],
            methods=["iplanner"],
            routes=[
                {"name": "custom_hall", "spawn": [0.5, -1.0], "goal": [4.0, 2.0], "spawn_yaw": 1.57},
            ],
            seeds=[0],
        )
        runs = expand_sweep_matrix(spec)
        assert len(runs) == 1
        run_id, r_spec, _label = runs[0]
        assert run_id == "001_iplanner_dingo_custom_hall_s0"
        assert r_spec.route is None
        assert r_spec.spawn == (0.5, -1.0)
        assert r_spec.goal == (4.0, 2.0)
        assert r_spec.spawn_yaw == 1.57


class TestBatchSetup:
    """Test batch directory structure and artifact creation."""

    def test_setup_batch_directory(self, tmp_path: Path):
        spec = SweepSpec(
            name="test_batch_setup",
            scene="kujiale_0003",
            robots=["dingo"],
            methods=["iplanner"],
            routes=["hall_straight"],
            seeds=[0],
        )
        batch_dir, manifest, expanded_runs = setup_batch_directory(
            spec, runs_dir=tmp_path, batch_id="20261004-1200_test_batch_setup"
        )
        assert batch_dir == tmp_path / "20261004-1200_test_batch_setup"
        assert (batch_dir / "batch.yaml").is_file()
        assert (batch_dir / "manifest.json").is_file()
        assert (batch_dir / "results.csv").is_file()

        # Check batch.yaml
        batch_data = yaml.safe_load((batch_dir / "batch.yaml").read_text(encoding="utf-8"))
        assert batch_data["batch_id"] == "20261004-1200_test_batch_setup"
        assert batch_data["name"] == "test_batch_setup"
        assert batch_data["status"] == "running"
        assert "git" in batch_data
        assert "host" in batch_data
        assert batch_data["spec"]["name"] == "test_batch_setup"

        # Check manifest.json
        loaded_manifest = BatchManifest.load(batch_dir / "manifest.json")
        assert loaded_manifest.batch_id == "20261004-1200_test_batch_setup"
        assert len(loaded_manifest.runs) == 1
        rec = loaded_manifest.runs[0]
        assert rec.id == "001_iplanner_dingo_hall_straight_s0"
        assert rec.status == RunStatus.QUEUED.value

        # Check results.csv header
        with open(batch_dir / "results.csv", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            assert header == RESULTS_CSV_COLUMNS


@contextmanager
def mock_worker_subprocess(
    exit_code: int = 0,
    summary_payload: dict[str, Any] | None = None,
    hang: bool = False,
    simulate_sigint: bool = False,
) -> Generator[None, None, None]:
    """Mock managed_process to simulate worker runs with canned results."""
    from subprocess import TimeoutExpired

    @contextmanager
    def _fake_managed_process(cmd, **kwargs):
        class FakeProc:
            pid = 99999

            def __init__(self):
                self.stdout = None

            def wait(self, timeout=None):
                if simulate_sigint:
                    raise KeyboardInterrupt()
                if hang:
                    raise TimeoutExpired(cmd=cmd, timeout=timeout)
                # cmd[-1] is run_spec_path
                run_spec_path = Path(cmd[-1])
                run_dir = run_spec_path.parent
                if summary_payload is not None:
                    summary_path = run_dir / "summary.json"
                    summary_path.write_text(json.dumps(summary_payload), encoding="utf-8")
                return exit_code

        yield FakeProc()

    with patch("nav_arena.benchmarks.launcher.managed_process", side_effect=_fake_managed_process):
        yield


class TestSweepExecution:
    """Test sweep execution loop with mock workers."""

    def test_successful_sweep_execution(self, tmp_path: Path):
        spec = SweepSpec(
            name="sweep_success",
            scene="kujiale_0003",
            robots=["dingo"],
            methods=["iplanner"],
            routes=["hall_straight"],
            seeds=[0, 1],
            timeout_s=60.0,
        )
        canned_summary = {
            "terminal_cause": "goal_reached",
            "success": True,
            "time_to_goal_s": 18.5,
            "sim_time_s": 18.5,
            "path_length_m": 5.4,
            "initial_goal_distance_m": 5.0,
            "final_goal_distance_m": 0.2,
            "plans": 90,
            "stop_requests": 0,
            "mean_inference_ms": 11.2,
            "wall_time_s": 22.0,
        }

        with mock_worker_subprocess(exit_code=0, summary_payload=canned_summary):
            code = execute_sweep(
                spec,
                runs_dir=tmp_path,
                batch_id="batch_success",
                force=True,
                quiet=True,
            )

        assert code == 0
        batch_dir = tmp_path / "batch_success"
        manifest = BatchManifest.load(batch_dir / "manifest.json")
        assert manifest.is_finished()
        assert manifest.counts_by_status()[RunStatus.DONE.value] == 2

        with open(batch_dir / "results.csv", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
            assert len(rows) == 2
            assert rows[0]["terminal_cause"] == "goal_reached"
            assert rows[0]["success"] == "True"
            assert rows[0]["time_to_goal_s"] == "18.5"

        batch_yaml = yaml.safe_load((batch_dir / "batch.yaml").read_text(encoding="utf-8"))
        assert batch_yaml["status"] == "completed"
        assert batch_yaml["ended_at"] is not None

    def test_episode_not_reached_still_done(self, tmp_path: Path):
        """Worker exits 2 (collision); episode finished, so run is done and sweep exits 0."""
        spec = SweepSpec(
            name="sweep_collision",
            scene="kujiale_0003",
            robots=["dingo"],
            methods=["iplanner"],
            routes=["hall_straight"],
            seeds=[0],
        )
        canned_summary = {
            "terminal_cause": "collision",
            "success": False,
            "time_to_goal_s": 0.0,
        }

        with mock_worker_subprocess(exit_code=2, summary_payload=canned_summary):
            code = execute_sweep(
                spec,
                runs_dir=tmp_path,
                batch_id="batch_collision",
                force=True,
                quiet=True,
            )

        assert code == 0
        batch_dir = tmp_path / "batch_collision"
        manifest = BatchManifest.load(batch_dir / "manifest.json")
        assert manifest.runs[0].status == RunStatus.DONE.value
        assert manifest.runs[0].terminal_cause == "collision"

    def test_infra_failure_exits_one(self, tmp_path: Path):
        """Worker exits 1 (infra failure); run fails and sweep exits 1."""
        spec = SweepSpec(
            name="sweep_fail",
            scene="kujiale_0003",
            robots=["dingo"],
            methods=["iplanner"],
            routes=["hall_straight"],
            seeds=[0],
        )
        with mock_worker_subprocess(exit_code=1, summary_payload=None):
            code = execute_sweep(
                spec,
                runs_dir=tmp_path,
                batch_id="batch_fail",
                force=True,
                quiet=True,
            )

        assert code == 1
        batch_dir = tmp_path / "batch_fail"
        manifest = BatchManifest.load(batch_dir / "manifest.json")
        assert manifest.runs[0].status == RunStatus.FAILED.value
        assert "Worker failed with exit code 1" in manifest.runs[0].error

    def test_timeout_exits_one(self, tmp_path: Path):
        """Worker hangs; timeout triggers, run is marked timeout and sweep exits 1."""
        spec = SweepSpec(
            name="sweep_hang",
            scene="kujiale_0003",
            robots=["dingo"],
            methods=["iplanner"],
            routes=["hall_straight"],
            seeds=[0],
            timeout_s=5.0,
        )
        with mock_worker_subprocess(hang=True):
            code = execute_sweep(
                spec,
                runs_dir=tmp_path,
                batch_id="batch_hang",
                force=True,
                quiet=True,
            )

        assert code == 1
        batch_dir = tmp_path / "batch_hang"
        manifest = BatchManifest.load(batch_dir / "manifest.json")
        assert manifest.runs[0].status == RunStatus.TIMEOUT.value
        assert manifest.runs[0].terminal_cause == "timeout"

    def test_keyboard_interrupt_handles_cleanly(self, tmp_path: Path):
        """Ctrl-C marks active run failed and leaves manifest consistent."""
        spec = SweepSpec(
            name="sweep_sigint",
            scene="kujiale_0003",
            robots=["dingo"],
            methods=["iplanner"],
            routes=["hall_straight"],
            seeds=[0],
        )
        with mock_worker_subprocess(simulate_sigint=True):
            code = execute_sweep(
                spec,
                runs_dir=tmp_path,
                batch_id="batch_sigint",
                force=True,
                quiet=True,
            )

        assert code == 1
        batch_dir = tmp_path / "batch_sigint"
        manifest = BatchManifest.load(batch_dir / "manifest.json")
        assert manifest.runs[0].status == RunStatus.FAILED.value
        assert "SIGINT" in manifest.runs[0].error
        batch_yaml = yaml.safe_load((batch_dir / "batch.yaml").read_text(encoding="utf-8"))
        assert batch_yaml["status"] == "interrupted"

    def test_preflight_refuses_when_busy(self, tmp_path: Path):
        spec = SweepSpec(name="sweep_conflict", methods=["iplanner"], routes=["hall_straight"])
        with patch(
            "nav_arena.benchmarks.sweep.check_preflight_processes",
            return_value=[(1234, "python verify_baseline.py")],
        ):
            code = execute_sweep(spec, runs_dir=tmp_path, force=False)
            assert code == 1


class TestResumeLogic:
    """Test resuming an interrupted or partially failed batch sweep."""

    def test_resume_skips_done_and_retries_incomplete(self, tmp_path: Path):
        spec = SweepSpec(
            name="sweep_resumable",
            scene="kujiale_0003",
            robots=["dingo"],
            methods=["iplanner"],
            routes=["hall_straight"],
            seeds=[0, 1, 2],
        )
        batch_dir, manifest, _ = setup_batch_directory(
            spec, runs_dir=tmp_path, batch_id="batch_resume_test"
        )
        # Manually mark run 0 as done, run 1 as failed, run 2 queued
        manifest.mark_running("001_iplanner_dingo_hall_straight_s0", pid=101)
        manifest.mark_done("001_iplanner_dingo_hall_straight_s0", exit_code=0, terminal_cause="goal_reached")
        row0 = {col: "done_val" for col in RESULTS_CSV_COLUMNS}
        row0["run_id"] = "001_iplanner_dingo_hall_straight_s0"
        append_results_row(batch_dir / "results.csv", row0)

        manifest.mark_running("002_iplanner_dingo_hall_straight_s1", pid=102)
        manifest.mark_failed("002_iplanner_dingo_hall_straight_s1", exit_code=1, error="network drop")
        manifest.save(batch_dir / "manifest.json")

        executed_runs: list[str] = []

        @contextmanager
        def _tracking_managed_process(cmd, **kwargs):
            class FakeProc:
                pid = 55555

                def __init__(self):
                    self.stdout = None

                def wait(self, timeout=None):
                    run_spec_path = Path(cmd[-1])
                    run_id = run_spec_path.parent.name
                    executed_runs.append(run_id)
                    summary = {"terminal_cause": "goal_reached", "success": True}
                    (run_spec_path.parent / "summary.json").write_text(json.dumps(summary))
                    return 0

            yield FakeProc()

        with patch("nav_arena.benchmarks.launcher.managed_process", side_effect=_tracking_managed_process):
            code = execute_sweep(
                spec,
                batch_dir=batch_dir,
                resume=True,
                force=True,
                quiet=True,
            )

        assert code == 0
        # Run 001 was done so it must NOT be re-executed!
        assert "001_iplanner_dingo_hall_straight_s0" not in executed_runs
        assert "002_iplanner_dingo_hall_straight_s1" in executed_runs
        assert "003_iplanner_dingo_hall_straight_s2" in executed_runs

        updated_manifest = BatchManifest.load(batch_dir / "manifest.json")
        assert updated_manifest.counts_by_status()[RunStatus.DONE.value] == 3


class TestCliSweepCommand:
    """Test nav_arena sweep CLI subcommand arguments, dispatch, and help."""

    def test_sweep_help(self, capsys: pytest.CaptureFixture):
        with pytest.raises(SystemExit) as exc:
            main(["sweep", "--help"])
        assert exc.value.code == 0
        captured = capsys.readouterr()
        assert "sweep" in captured.out
        assert "--methods" in captured.out
        assert "--routes" in captured.out
        assert "--seeds" in captured.out
        assert "--resume" in captured.out

    def test_sweep_cli_execution_with_flags(self, tmp_path: Path):
        summary_payload = {"terminal_cause": "goal_reached", "success": True}
        with mock_worker_subprocess(exit_code=0, summary_payload=summary_payload):
            with patch("nav_arena.benchmarks.sweep.RUNS_DIR", tmp_path):
                exit_code = main([
                    "sweep",
                    "--name", "cli_test_sweep",
                    "--scene", "kujiale_0003",
                    "--methods", "iplanner",
                    "--routes", "hall_straight",
                    "--seeds", "0",
                    "--timeout", "60",
                    "--force",
                    "--quiet",
                ])
                assert exit_code == 0

    def test_sweep_cli_resume_without_spec(self, tmp_path: Path):
        """Verify nav_arena sweep --resume without spec finds the most recent batch and resumes."""
        spec = SweepSpec(
            name="batch_to_resume",
            scene="kujiale_0003",
            robots=["dingo"],
            methods=["iplanner"],
            routes=["hall_straight"],
            seeds=[0],
        )
        with patch("nav_arena.benchmarks.sweep.RUNS_DIR", tmp_path):
            batch_dir, manifest, _ = setup_batch_directory(spec, runs_dir=tmp_path)
            # Manifest has run 0 queued.
            summary_payload = {"terminal_cause": "goal_reached", "success": True}
            with mock_worker_subprocess(exit_code=0, summary_payload=summary_payload):
                exit_code = main(["sweep", "--resume", "--force", "--quiet"])
                assert exit_code == 0
            updated_manifest = BatchManifest.load(batch_dir / "manifest.json")
            assert updated_manifest.runs[0].status == RunStatus.DONE.value

    def test_sweep_cli_resume_from_batch_dir(self, tmp_path: Path):
        """Verify nav_arena sweep <batch_dir> --resume resumes using the batch directory."""
        spec = SweepSpec(
            name="batch_dir_resume",
            scene="kujiale_0003",
            robots=["dingo"],
            methods=["iplanner"],
            routes=["hall_straight"],
            seeds=[0],
        )
        batch_dir, manifest, _ = setup_batch_directory(spec, runs_dir=tmp_path)
        summary_payload = {"terminal_cause": "goal_reached", "success": True}
        with mock_worker_subprocess(exit_code=0, summary_payload=summary_payload):
            with patch("nav_arena.benchmarks.sweep.RUNS_DIR", tmp_path):
                exit_code = main(["sweep", str(batch_dir), "--resume", "--force", "--quiet"])
                assert exit_code == 0
        updated_manifest = BatchManifest.load(batch_dir / "manifest.json")
        assert updated_manifest.runs[0].status == RunStatus.DONE.value


class TestSweepEdgeCasesAndPreflight:
    """Test pre-flight detection, CSV deduplication, and complex identifier handling."""

    def test_inline_route_with_underscores_in_method_and_robot(self, tmp_path: Path):
        spec = SweepSpec(
            name="underscore_test",
            scene="kujiale_0003",
            robots=["nova_carter"],
            methods=["x_navdp"],
            routes=[
                {"name": "complex_custom_route", "spawn": [0.0, 0.0], "goal": [2.0, 2.0]},
            ],
            seeds=[0],
        )
        batch_dir, manifest, expanded_runs = setup_batch_directory(spec, runs_dir=tmp_path)
        assert len(manifest.runs) == 1
        rec = manifest.runs[0]
        assert rec.id == "001_x_navdp_nova_carter_complex_custom_route_s0"
        # Route should be 'complex_custom_route', NOT 'carter_complex_custom_route' or 'custom'
        assert rec.route == "complex_custom_route"

        # Now run with mock worker and verify results.csv gets the route name
        summary_payload = {"terminal_cause": "goal_reached", "success": True}
        with mock_worker_subprocess(exit_code=0, summary_payload=summary_payload):
            code = execute_sweep(spec, batch_dir=batch_dir, force=True, quiet=True)
            assert code == 0

        with open(batch_dir / "results.csv", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
            assert len(rows) == 1
            assert rows[0]["route"] == "complex_custom_route"

    def test_results_csv_deduplication_on_resume(self, tmp_path: Path):
        """Verify that retrying an incomplete run replaces the row in results.csv instead of duplicating."""
        csv_file = tmp_path / "results.csv"
        row1 = {col: "" for col in RESULTS_CSV_COLUMNS}
        row1["run_id"] = "001_iplanner_dingo_hall_straight_s0"
        row1["terminal_cause"] = "failed"
        row1["success"] = "False"
        append_results_row(csv_file, row1)

        with open(csv_file, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
            assert len(rows) == 1
            assert rows[0]["terminal_cause"] == "failed"

        # Now update row on retry
        row1_updated = dict(row1)
        row1_updated["terminal_cause"] = "goal_reached"
        row1_updated["success"] = "True"
        append_results_row(csv_file, row1_updated)

        with open(csv_file, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
            # Must have exactly 1 row, not 2!
            assert len(rows) == 1
            assert rows[0]["terminal_cause"] == "goal_reached"
            assert rows[0]["success"] == "True"

    def test_apply_sweep_overrides_logging_levels(self):
        base = SweepSpec(
            name="base_sweep",
            scene="kujiale_0003",
            robots=["dingo"],
            methods=["iplanner"],
            routes=["hall_straight"],
            seeds=[0],
        )
        mock_logger = MagicMock()
        apply_sweep_overrides(
            base,
            {
                "scene": "kujiale_0004",
                "robots": "nova_carter",
                "methods": "navdp",
                "seeds": "1,2",
            },
            logger=mock_logger,
        )
        # scene, robots, methods -> warning
        assert mock_logger.warning.call_count == 3
        # seeds -> info
        assert mock_logger.info.call_count == 1


def test_sweep_rejects_parallel_jobs(tmp_path):
    """Verify --jobs other than 1 is rejected, because parallel runs are not supported yet."""
    spec_path = tmp_path / "s.yaml"
    spec_path.write_text("name: j\nscene: kujiale_0003\nmethods: [iplanner]\nroutes: [hall_straight]\nseeds: [0]\n")
    assert main(["sweep", str(spec_path), "--jobs", "2", "--force"]) == 1
