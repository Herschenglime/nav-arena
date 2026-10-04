# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for nav_arena CLI run subcommand, argument parsing, results.csv, and worker management."""

from __future__ import annotations

from contextlib import contextmanager
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Generator
from unittest.mock import MagicMock, patch

import pytest
import yaml

from nav_arena.benchmarks.spec import EpisodeLimits, RunSpec, VizCfg
from nav_arena.cli import (
    RESULTS_CSV_COLUMNS,
    append_results_row,
    check_preflight_processes,
    compute_default_timeout,
    format_results_row,
    generate_run_dir,
    load_spec_file,
    main,
)


class TestCliRunArgParsing:
    """Test argument parsing and spec resolution for nav_arena run."""

    def test_run_explicit_flags(self, tmp_path: Path):
        run_output = tmp_path / "custom_run"
        argv = [
            "run",
            "--method", "vint",
            "--robot", "nova_carter",
            "--scene", "kujiale_0004",
            "--route", "hall_straight",
            "--seed", "42",
            "--max-steps", "1200",
            "--goal-dist", "0.35",
            "--max-speed", "0.25",
            "--stall-timeout", "8.0",
            "--gui",
            "--follow-camera",
            "--no-goal-overlay",
            "--output", str(run_output),
            "--quiet",
            "--timeout", "120.0",
            "--force",
        ]

        with patch("nav_arena.benchmarks.launcher.managed_process") as mock_mp:
            mock_proc = MagicMock()
            mock_proc.stdout = io.StringIO("step 1\nstep 2\n")
            mock_proc.wait.return_value = 0
            mock_mp.return_value.__enter__.return_value = mock_proc

            exit_code = main(argv)
            assert exit_code == 0

        # Verify saved run_spec.json
        saved_spec_path = run_output / "run_spec.json"
        assert saved_spec_path.is_file()
        spec_dict = json.loads(saved_spec_path.read_text(encoding="utf-8"))
        assert spec_dict["method"] == "vint"
        assert spec_dict["robot"] == "nova_carter"
        assert spec_dict["scene"] == "kujiale_0004"
        assert spec_dict["route"] == "hall_straight"
        assert spec_dict["seed"] == 42
        assert spec_dict["limits"]["max_steps"] == 1200
        assert spec_dict["limits"]["goal_tolerance"] == 0.35
        assert spec_dict["limits"]["max_speed"] == 0.25
        assert spec_dict["limits"]["stall_timeout_s"] == 8.0
        assert spec_dict["viz"]["gui"] is True
        assert spec_dict["viz"]["follow_camera"] is True
        assert spec_dict["viz"]["goal_overlay"] is False

    def test_run_custom_coordinates(self, tmp_path: Path):
        run_output = tmp_path / "coords_run"
        argv = [
            "run",
            "--method", "iplanner",
            "--spawn", "1.0", "2.0",
            "--goal", "4.0", "5.0",
            "--spawn-yaw", "1.57",
            "--output", str(run_output),
        ]

        with patch("nav_arena.benchmarks.launcher.managed_process") as mock_mp:
            mock_proc = MagicMock()
            mock_proc.stdout = io.StringIO("")
            mock_proc.wait.return_value = 0
            mock_mp.return_value.__enter__.return_value = mock_proc

            exit_code = main(argv)
            assert exit_code == 0

        saved_spec = json.loads((run_output / "run_spec.json").read_text(encoding="utf-8"))
        assert saved_spec["spawn"] == [1.0, 2.0]
        assert saved_spec["goal"] == [4.0, 5.0]
        assert pytest.approx(saved_spec["spawn_yaw"]) == 1.57

    def test_run_policy_args_parsing(self, tmp_path: Path):
        run_output = tmp_path / "policy_run"
        argv = [
            "run",
            "--method", "iplanner",
            "--route", "hall_straight",
            "--policy-arg", "fear_threshold=0.6",
            "--policy-arg", "plan_hz=5.0",
            "--output", str(run_output),
        ]

        with patch("nav_arena.benchmarks.launcher.managed_process") as mock_mp:
            mock_proc = MagicMock()
            mock_proc.stdout = io.StringIO("")
            mock_proc.wait.return_value = 0
            mock_mp.return_value.__enter__.return_value = mock_proc

            exit_code = main(argv)
            assert exit_code == 0

        saved_spec = json.loads((run_output / "run_spec.json").read_text(encoding="utf-8"))
        assert saved_spec["method_params"] == {"fear_threshold": 0.6, "plan_hz": 5.0}

    def test_run_spec_yaml_loading_and_cli_overrides(self, tmp_path: Path):
        spec_yaml = tmp_path / "base_spec.yaml"
        spec_yaml.write_text(
            yaml.dump({
                "method": "iplanner",
                "robot": "dingo",
                "scene": "kujiale_0003",
                "route": "hall_straight",
                "seed": 10,
                "limits": {
                    "max_steps": 1000,
                    "goal_tolerance": 0.5,
                    "max_speed": 0.4,
                    "stall_timeout_s": 12.0,
                },
                "viz": {
                    "gui": False,
                    "follow_camera": False,
                    "goal_overlay": True,
                },
            }),
            encoding="utf-8",
        )

        run_output = tmp_path / "yaml_run"
        argv = [
            "run",
            "--spec", str(spec_yaml),
            "--scene", "kujiale_0004",
            "--seed", "99",
            "--gui",
            "--output", str(run_output),
        ]

        with patch("nav_arena.benchmarks.launcher.managed_process") as mock_mp:
            mock_proc = MagicMock()
            mock_proc.stdout = io.StringIO("")
            mock_proc.wait.return_value = 0
            mock_mp.return_value.__enter__.return_value = mock_proc

            exit_code = main(argv)
            assert exit_code == 0

        saved_spec = json.loads((run_output / "run_spec.json").read_text(encoding="utf-8"))
        assert saved_spec["method"] == "iplanner"
        assert saved_spec["scene"] == "kujiale_0004"  # overridden
        assert saved_spec["seed"] == 99  # overridden
        assert saved_spec["viz"]["gui"] is True  # overridden
        assert saved_spec["limits"]["max_steps"] == 1000  # preserved from YAML

    def test_run_seeds_flag_alias(self, tmp_path: Path):
        run_output = tmp_path / "seeds_run"
        argv = [
            "run",
            "--method", "iplanner",
            "--route", "hall_straight",
            "--seeds", "77",
            "--output", str(run_output),
        ]
        with patch("nav_arena.benchmarks.launcher.managed_process") as mock_mp:
            mock_proc = MagicMock()
            mock_proc.stdout = io.StringIO("")
            mock_proc.wait.return_value = 0
            mock_mp.return_value.__enter__.return_value = mock_proc

            exit_code = main(argv)
            assert exit_code == 0

        saved_spec = json.loads((run_output / "run_spec.json").read_text(encoding="utf-8"))
        assert saved_spec["seed"] == 77

    def test_run_route_overrides_spec_coordinates(self, tmp_path: Path):
        spec_yaml = tmp_path / "coords_spec.yaml"
        spec_yaml.write_text(
            yaml.dump({
                "method": "iplanner",
                "spawn": [1.0, 2.0],
                "goal": [3.0, 4.0],
            }),
            encoding="utf-8",
        )
        run_output = tmp_path / "route_override_run"
        argv = [
            "run",
            "--spec", str(spec_yaml),
            "--route", "hall_straight",
            "--output", str(run_output),
        ]
        with patch("nav_arena.benchmarks.launcher.managed_process") as mock_mp:
            mock_proc = MagicMock()
            mock_proc.stdout = io.StringIO("")
            mock_proc.wait.return_value = 0
            mock_mp.return_value.__enter__.return_value = mock_proc

            exit_code = main(argv)
            assert exit_code == 0

        saved_spec = json.loads((run_output / "run_spec.json").read_text(encoding="utf-8"))
        assert saved_spec["route"] == "hall_straight"
        assert saved_spec["spawn"] is None
        assert saved_spec["goal"] is None

    def test_run_invalid_timeout(self):
        exit_code_zero = main(["run", "--method", "iplanner", "--route", "hall_straight", "--timeout", "0"])
        assert exit_code_zero == 1
        exit_code_neg = main(["run", "--method", "iplanner", "--route", "hall_straight", "--timeout", "-5"])
        assert exit_code_neg == 1

    def test_run_invalid_spec_missing_method(self):
        # When no spec is given and method is omitted, should exit with 1
        exit_code = main(["run", "--scene", "kujiale_0003"])
        assert exit_code == 1

    def test_run_invalid_route_coordinates(self):
        # Passing --spawn without --goal
        exit_code = main(["run", "--method", "iplanner", "--spawn", "1.0", "2.0"])
        assert exit_code == 1


class TestFastStartupAndZeroHeavyImports:
    """Verify nav_arena.cli starts up quickly and does not import heavy dependencies."""

    def test_zero_heavy_imports(self):
        check_code = (
            "import sys\n"
            "import nav_arena.cli\n"
            "banned = ['isaaclab', 'isaacsim', 'omni', 'pxr', 'rclpy', 'torch']\n"
            "imported = [m for m in banned if m in sys.modules]\n"
            "if imported:\n"
            "    print(f'BANNED_IMPORTED:{imported}')\n"
            "    sys.exit(1)\n"
            "sys.exit(0)\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", check_code],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, f"Heavy imports detected: {proc.stdout} {proc.stderr}"

    def test_cli_fast_startup_time(self):
        start = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, "-m", "nav_arena.cli", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        elapsed = time.perf_counter() - start
        assert proc.returncode == 0
        assert elapsed < 1.0, f"CLI startup took {elapsed:.2f}s, expected < 1.0s"


class TestResultsCsvFormatting:
    """Test results.csv row generation and persistence."""

    def test_results_csv_columns(self):
        assert len(RESULTS_CSV_COLUMNS) == 22
        assert RESULTS_CSV_COLUMNS[0] == "batch_id"
        assert RESULTS_CSV_COLUMNS[-1] == "spec_hash"

    def test_format_results_row_success(self):
        spec = RunSpec(
            method="iplanner",
            robot="dingo",
            scene="kujiale_0003",
            route="hall_straight",
            seed=42,
            limits=EpisodeLimits(goal_tolerance=0.4, max_speed=0.3),
        )
        summary = {
            "terminal_cause": "goal_reached",
            "success": True,
            "time_to_goal_s": 14.5,
            "sim_time_s": 14.5,
            "path_length_m": 4.8,
            "initial_goal_distance_m": 5.0,
            "final_goal_distance_m": 0.2,
            "plans": 30,
            "stop_requests": 0,
            "mean_inference_ms": 12.4,
            "wall_time_s": 2.5,
        }

        row = format_results_row(spec, summary, run_id="run_123", batch_id="batch_abc")

        assert row["batch_id"] == "batch_abc"
        assert row["run_id"] == "run_123"
        assert row["scene"] == "kujiale_0003"
        assert row["robot"] == "dingo"
        assert row["method"] == "iplanner"
        assert row["method_family"] == "in_process"
        assert row["route"] == "hall_straight"
        assert row["seed"] == 42
        assert row["terminal_cause"] == "goal_reached"
        assert row["success"] is True
        assert row["time_to_goal_s"] == 14.5
        assert row["goal_tolerance"] == 0.4
        assert row["max_speed"] == 0.3
        assert row["spec_hash"] == spec.spec_hash

    def test_format_results_row_timeout(self):
        spec = RunSpec(method="iplanner", route="hall_straight")
        row = format_results_row(spec, summary_data=None, run_id="run_timeout", timed_out=True)

        assert row["terminal_cause"] == "timeout"
        assert row["success"] is False
        assert row["time_to_goal_s"] == ""

    def test_format_results_row_failure(self):
        spec = RunSpec(method="iplanner", route="hall_straight")
        row = format_results_row(spec, summary_data=None, run_id="run_fail", timed_out=False)

        assert row["terminal_cause"] == "failed"
        assert row["success"] is False

    def test_append_results_row_creates_header_and_appends(self, tmp_path: Path):
        csv_file = tmp_path / "results.csv"
        spec1 = RunSpec(method="iplanner", route="hall_straight", seed=1)
        spec2 = RunSpec(method="vint", route="hall_straight", seed=2)

        row1 = format_results_row(spec1, {"terminal_cause": "goal_reached", "success": True}, run_id="run_1")
        row2 = format_results_row(spec2, {"terminal_cause": "collision", "success": False}, run_id="run_2")

        append_results_row(csv_file, row1)
        append_results_row(csv_file, row2)

        with csv_file.open("r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))

        assert len(reader) == 2
        assert reader[0]["run_id"] == "run_1"
        assert reader[0]["method"] == "iplanner"
        assert reader[0]["terminal_cause"] == "goal_reached"
        assert reader[1]["run_id"] == "run_2"
        assert reader[1]["method"] == "vint"
        assert reader[1]["terminal_cause"] == "collision"

    def test_append_results_row_to_existing_file(self, tmp_path: Path):
        csv_file = tmp_path / "existing_results.csv"
        csv_file.write_text("batch_id,run_id,scene,robot,method,method_family,route,seed,terminal_cause,success,time_to_goal_s,sim_time_s,path_length_m,initial_goal_distance_m,final_goal_distance_m,plans,stop_requests,mean_inference_ms,wall_time_s,goal_tolerance,max_speed,spec_hash\nold_batch,run_old,kujiale_0003,dingo,iplanner,in_process,hall_straight,0,goal_reached,True,10.0,10.0,3.0,3.5,0.1,20,0,10.0,2.0,0.4,0.3,hash123\n", encoding="utf-8")

        spec = RunSpec(method="vint", route="hall_straight", seed=5)
        row = format_results_row(spec, {"terminal_cause": "collision", "success": False}, run_id="run_new")
        append_results_row(csv_file, row)

        lines = csv_file.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 3  # 1 header + 2 data rows
        assert lines[0].startswith("batch_id,run_id")
        assert not lines[2].startswith("batch_id")
        assert "run_new" in lines[2]


class TestWorkerProcessExecution:
    """Test managed_process orchestration, timeout, exit codes, and streaming."""

    def test_compute_default_timeout(self):
        spec = RunSpec(
            method="iplanner",
            limits=EpisodeLimits(max_steps=1500, stall_timeout_s=10.0),
        )
        assert compute_default_timeout(spec) == (1500 / 50.0) + 10.0 + 60.0  # 100.0s

        spec2 = RunSpec(
            method="iplanner",
            limits=EpisodeLimits(max_steps=2000, stall_timeout_s=5.0),
        )
        assert compute_default_timeout(spec2) == (2000 / 50.0) + 5.0 + 60.0  # 105.0s

    def test_generate_run_dir(self, tmp_path: Path):
        spec = RunSpec(method="iplanner", robot="dingo", route="hall_straight")
        run_dir = generate_run_dir(spec, base_dir=tmp_path)
        assert str(run_dir).startswith(str(tmp_path))
        assert "iplanner_dingo_hall_straight" in run_dir.name

    def test_worker_success_exit_code_and_artifacts(self, tmp_path: Path):
        run_output = tmp_path / "success_run"

        @contextmanager
        def mock_managed_process(cmd, timeout=5.0, **popen_kwargs):
            mock_p = MagicMock()
            mock_p.stdout = io.StringIO("Sim step 1\nGoal reached!\n")
            # Create summary.json in output dir as real worker would
            summary_path = run_output / "summary.json"
            summary_path.write_text(
                json.dumps({"terminal_cause": "goal_reached", "success": True}),
                encoding="utf-8",
            )
            mock_p.wait.return_value = 0
            yield mock_p

        with patch("nav_arena.benchmarks.launcher.managed_process", side_effect=mock_managed_process):
            exit_code = main(["run", "--method", "iplanner", "--route", "hall_straight", "--output", str(run_output)])

        assert exit_code == 0
        assert (run_output / "worker.log").is_file()
        assert "Goal reached!" in (run_output / "worker.log").read_text()
        assert (run_output / "results.csv").is_file()
        with (run_output / "results.csv").open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["terminal_cause"] == "goal_reached"
        assert rows[0]["success"] == "True"

    def test_worker_not_reached_exit_code_2(self, tmp_path: Path):
        run_output = tmp_path / "not_reached_run"

        @contextmanager
        def mock_managed_process(cmd, timeout=5.0, **popen_kwargs):
            mock_p = MagicMock()
            mock_p.stdout = io.StringIO("Collision detected!\n")
            summary_path = run_output / "summary.json"
            summary_path.write_text(
                json.dumps({"terminal_cause": "collision", "success": False}),
                encoding="utf-8",
            )
            mock_p.wait.return_value = 2
            yield mock_p

        with patch("nav_arena.benchmarks.launcher.managed_process", side_effect=mock_managed_process):
            exit_code = main(["run", "--method", "iplanner", "--route", "hall_straight", "--output", str(run_output)])

        assert exit_code == 2
        with (run_output / "results.csv").open() as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["terminal_cause"] == "collision"
        assert rows[0]["success"] == "False"

    def test_worker_timeout_handling(self, tmp_path: Path):
        run_output = tmp_path / "timeout_run"

        @contextmanager
        def mock_managed_process(cmd, timeout=5.0, **popen_kwargs):
            mock_p = MagicMock()
            mock_p.stdout = io.StringIO("Running...\n")
            mock_p.wait.side_effect = subprocess.TimeoutExpired(cmd, timeout)
            yield mock_p

        with patch("nav_arena.benchmarks.launcher.managed_process", side_effect=mock_managed_process):
            exit_code = main([
                "run",
                "--method", "iplanner",
                "--route", "hall_straight",
                "--output", str(run_output),
                "--timeout", "10.0",
            ])

        assert exit_code == 1
        assert (run_output / "results.csv").is_file()
        with (run_output / "results.csv").open() as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["terminal_cause"] == "timeout"
        assert rows[0]["success"] == "False"

    def test_stale_summary_deleted_before_worker_run(self, tmp_path: Path):
        run_output = tmp_path / "stale_run"
        run_output.mkdir(parents=True, exist_ok=True)
        stale_summary = run_output / "summary.json"
        stale_summary.write_text(json.dumps({"terminal_cause": "goal_reached", "success": True}), encoding="utf-8")

        @contextmanager
        def mock_managed_process_failing(cmd, timeout=5.0, **popen_kwargs):
            mock_p = MagicMock()
            mock_p.stdout = io.StringIO("Fatal error during boot\n")
            mock_p.wait.return_value = 1  # Crash without writing summary.json
            yield mock_p

        with patch("nav_arena.benchmarks.launcher.managed_process", side_effect=mock_managed_process_failing):
            exit_code = main([
                "run",
                "--method", "iplanner",
                "--route", "hall_straight",
                "--output", str(run_output),
            ])

        assert exit_code == 1
        with (run_output / "results.csv").open() as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["terminal_cause"] == "failed"
        assert rows[0]["success"] == "False"

    def test_worker_keyboard_interrupt_handling(self, tmp_path: Path):
        run_output = tmp_path / "sigint_run"

        @contextmanager
        def mock_managed_process_interrupted(cmd, timeout=5.0, **popen_kwargs):
            mock_p = MagicMock()
            mock_p.stdout = io.StringIO("Aborted...\n")
            mock_p.wait.side_effect = KeyboardInterrupt()
            yield mock_p

        with patch("nav_arena.benchmarks.launcher.managed_process", side_effect=mock_managed_process_interrupted):
            exit_code = main([
                "run",
                "--method", "iplanner",
                "--route", "hall_straight",
                "--output", str(run_output),
            ])

        assert exit_code == 1
        with (run_output / "results.csv").open() as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["terminal_cause"] == "failed"
        assert rows[0]["success"] == "False"

    def test_preflight_conflict_detection(self, tmp_path: Path):
        with patch("nav_arena.cli.check_preflight_processes", return_value=[(9999, "python verify_baseline.py")]):
            with patch("nav_arena.benchmarks.launcher.managed_process") as mock_mp:
                # Without --force -> blocked
                code = main(["run", "--method", "iplanner", "--route", "hall_straight"])
                assert code == 1
                mock_mp.assert_not_called()

                # With --force -> proceeds
                mock_proc = MagicMock()
                mock_proc.stdout = io.StringIO("")
                mock_proc.wait.return_value = 0
                mock_mp.return_value.__enter__.return_value = mock_proc

                code_forced = main(["run", "--method", "iplanner", "--route", "hall_straight", "--force", "--output", str(tmp_path)])
                assert code_forced == 0
                mock_mp.assert_called_once()


class TestSubcommandsAndHelp:
    """Test CLI top-level help and stubs."""

    def test_top_level_help(self, capsys):
        exit_code = main([])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Unified navigation benchmarking and evaluation framework" in captured.out

    def test_subcommands_without_action_exit_1(self):
        for sub in ("runs", "routes", "map"):
            exit_code = main([sub])
            assert exit_code == 1
