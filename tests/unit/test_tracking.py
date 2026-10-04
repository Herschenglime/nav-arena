# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for nav_arena benchmark tracking, inspection, comparison, and rerun queries."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from nav_arena.benchmarks.manifest import (
    BatchManifest,
    RunRecord,
    RunStatus,
)
from nav_arena.benchmarks.spec import (
    EpisodeLimits,
    RunSpec,
    VizCfg,
)
from nav_arena.benchmarks.sweep import RESULTS_CSV_COLUMNS
from nav_arena.benchmarks.tracking import (
    compare_batch,
    find_run_spec_for_rerun,
    list_runs,
    show_run,
)
from nav_arena.cli import main


@pytest.fixture
def synthetic_runs_dir(tmp_path: Path) -> Path:
    """Create a synthetic runs directory with two batches and standalone runs."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Batch 1: 20261004-1500_sweep_test1 (completed batch)
    b1_dir = runs_dir / "20261004-1500_sweep_test1"
    b1_dir.mkdir()

    b1_yaml = {
        "batch_id": "20261004-1500_sweep_test1",
        "name": "sweep_test1",
        "created_at": "2026-10-04T15:00:00Z",
        "started_at": "2026-10-04T15:00:01Z",
        "ended_at": "2026-10-04T15:05:01Z",
        "status": "done",
        "spec": {
            "name": "sweep_test1",
            "scene": "kujiale_0003",
            "robots": ["dingo"],
            "methods": ["iplanner", "navdp"],
            "routes": ["hall_straight"],
            "seeds": [0],
            "options": {"goal_dist": 0.4, "max_speed": 0.3},
        },
    }
    (b1_dir / "batch.yaml").write_text(yaml.dump(b1_yaml), encoding="utf-8")

    r1_rec = RunRecord(
        id="000_iplanner_dingo_hall_straight_s0",
        status=RunStatus.DONE.value,
        method="iplanner",
        method_family="in_process",
        robot="dingo",
        scene="kujiale_0003",
        route="hall_straight",
        seed=0,
        exit_code=0,
        terminal_cause="goal_reached",
        run_dir="000_iplanner_dingo_hall_straight_s0",
    )
    r2_rec = RunRecord(
        id="001_navdp_dingo_hall_straight_s0",
        status=RunStatus.DONE.value,
        method="navdp",
        method_family="in_process",
        robot="dingo",
        scene="kujiale_0003",
        route="hall_straight",
        seed=0,
        exit_code=2,
        terminal_cause="collision",
        run_dir="001_navdp_dingo_hall_straight_s0",
    )
    b1_manifest = BatchManifest(
        batch_id="20261004-1500_sweep_test1",
        runs=[r1_rec, r2_rec],
        created_at="2026-10-04T15:00:00Z",
        metadata={"name": "sweep_test1"},
    )
    b1_manifest.save(b1_dir / "manifest.json")

    # Results CSV for Batch 1
    csv_rows_b1 = [
        {
            "batch_id": "20261004-1500_sweep_test1",
            "run_id": "000_iplanner_dingo_hall_straight_s0",
            "scene": "kujiale_0003",
            "robot": "dingo",
            "method": "iplanner",
            "method_family": "in_process",
            "route": "hall_straight",
            "seed": 0,
            "terminal_cause": "goal_reached",
            "success": True,
            "time_to_goal_s": 18.50,
            "sim_time_s": 18.50,
            "path_length_m": 3.40,
            "initial_goal_distance_m": 5.0,
            "final_goal_distance_m": 0.2,
            "plans": 90,
            "stop_requests": 0,
            "mean_inference_ms": 9.2,
            "wall_time_s": 35.0,
            "goal_tolerance": 0.4,
            "max_speed": 0.3,
            "spec_hash": "hash1",
        },
        {
            "batch_id": "20261004-1500_sweep_test1",
            "run_id": "001_navdp_dingo_hall_straight_s0",
            "scene": "kujiale_0003",
            "robot": "dingo",
            "method": "navdp",
            "method_family": "in_process",
            "route": "hall_straight",
            "seed": 0,
            "terminal_cause": "collision",
            "success": False,
            "time_to_goal_s": "",
            "sim_time_s": 5.20,
            "path_length_m": 1.25,
            "initial_goal_distance_m": 5.0,
            "final_goal_distance_m": 3.8,
            "plans": 26,
            "stop_requests": 0,
            "mean_inference_ms": 165.0,
            "wall_time_s": 12.0,
            "goal_tolerance": 0.4,
            "max_speed": 0.3,
            "spec_hash": "hash2",
        },
    ]
    with (b1_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_CSV_COLUMNS)
        writer.writeheader()
        for row in csv_rows_b1:
            writer.writerow(row)

    # Subdirectories for runs in Batch 1
    r1_dir = b1_dir / "000_iplanner_dingo_hall_straight_s0"
    r1_dir.mkdir()
    r1_spec = RunSpec(
        method="iplanner",
        robot="dingo",
        scene="kujiale_0003",
        route="hall_straight",
        seed=0,
        limits=EpisodeLimits(max_steps=1500, goal_tolerance=0.4, max_speed=0.3),
        viz=VizCfg(gui=False),
        output_dir=r1_dir,
    )
    (r1_dir / "run_spec.json").write_text(r1_spec.to_json(), encoding="utf-8")
    (r1_dir / "summary.json").write_text(
        json.dumps({
            "success": True,
            "terminal_cause": "goal_reached",
            "time_to_goal_s": 18.50,
            "path_length_m": 3.40,
            "mean_inference_ms": 9.2,
            "steps": 925,
            "final_goal_distance_m": 0.2,
        }),
        encoding="utf-8",
    )
    (r1_dir / "worker.log").write_text("worker logged ok\n", encoding="utf-8")

    r2_dir = b1_dir / "001_navdp_dingo_hall_straight_s0"
    r2_dir.mkdir()
    r2_spec = RunSpec(
        method="navdp",
        robot="dingo",
        scene="kujiale_0003",
        route="hall_straight",
        seed=0,
        limits=EpisodeLimits(max_steps=1500, goal_tolerance=0.4, max_speed=0.3),
        viz=VizCfg(gui=False),
        output_dir=r2_dir,
    )
    (r2_dir / "run_spec.json").write_text(r2_spec.to_json(), encoding="utf-8")
    (r2_dir / "summary.json").write_text(
        json.dumps({
            "success": False,
            "terminal_cause": "collision",
            "time_to_goal_s": None,
            "path_length_m": 1.25,
            "mean_inference_ms": 165.0,
            "steps": 260,
            "final_goal_distance_m": 3.8,
        }),
        encoding="utf-8",
    )
    (r2_dir / "worker.log").write_text("worker collision log\n", encoding="utf-8")

    # Batch 2: 20261004-1600_sweep_test2 (in-progress batch)
    b2_dir = runs_dir / "20261004-1600_sweep_test2"
    b2_dir.mkdir()
    b2_yaml = {
        "batch_id": "20261004-1600_sweep_test2",
        "name": "sweep_test2",
        "status": "running",
        "spec": {"name": "sweep_test2", "scene": "kujiale_0003"},
    }
    (b2_dir / "batch.yaml").write_text(yaml.dump(b2_yaml), encoding="utf-8")
    r3_rec = RunRecord(
        id="000_vint_dingo_hall_straight_s0",
        status=RunStatus.DONE.value,
        method="vint",
        method_family="in_process",
        robot="dingo",
        scene="kujiale_0003",
        route="hall_straight",
        seed=0,
        exit_code=0,
        terminal_cause="goal_reached",
    )
    r4_rec = RunRecord(
        id="001_vint_dingo_hall_straight_s1",
        status=RunStatus.RUNNING.value,
        method="vint",
        method_family="in_process",
        robot="dingo",
        scene="kujiale_0003",
        route="hall_straight",
        seed=1,
    )
    b2_manifest = BatchManifest(
        batch_id="20261004-1600_sweep_test2",
        runs=[r3_rec, r4_rec],
        created_at="2026-10-04T16:00:00Z",
    )
    b2_manifest.save(b2_dir / "manifest.json")

    # Standalone single run under runs_dir
    standalone_dir = runs_dir / "20261004_170000_iplanner_dingo_hall_straight_abcd"
    standalone_dir.mkdir()
    st_spec = RunSpec(
        method="iplanner",
        robot="dingo",
        scene="kujiale_0003",
        route="hall_straight",
        seed=99,
        limits=EpisodeLimits(),
        viz=VizCfg(),
        output_dir=standalone_dir,
    )
    (standalone_dir / "run_spec.json").write_text(st_spec.to_json(), encoding="utf-8")
    (standalone_dir / "summary.json").write_text(
        json.dumps({
            "success": True,
            "terminal_cause": "goal_reached",
            "time_to_goal_s": 15.2,
            "path_length_m": 3.1,
            "mean_inference_ms": 8.5,
            "steps": 760,
            "final_goal_distance_m": 0.15,
        }),
        encoding="utf-8",
    )
    (standalone_dir / "worker.log").write_text("standalone log\n", encoding="utf-8")

    return runs_dir


class TestListRuns:
    """Tests for list_runs function."""

    def test_list_runs_all_batches(self, synthetic_runs_dir: Path):
        batches = list_runs(synthetic_runs_dir)
        assert len(batches) == 2
        batch_ids = [b["batch_id"] for b in batches]
        assert "20261004-1500_sweep_test1" in batch_ids
        assert "20261004-1600_sweep_test2" in batch_ids

        # Find batch 1
        b1 = next(b for b in batches if b["batch_id"] == "20261004-1500_sweep_test1")
        assert b1["total_runs"] == 2
        assert b1["completed"] == 2
        assert b1["success_rate"] == 50.0
        assert b1["status"] == "done"

        # Find batch 2
        b2 = next(b for b in batches if b["batch_id"] == "20261004-1600_sweep_test2")
        assert b2["total_runs"] == 2
        assert b2["completed"] == 1
        assert b2["success_rate"] == 100.0  # 1 success out of 1 completed
        assert b2["status"] == "running"

    def test_list_runs_empty_dir(self, tmp_path: Path):
        empty_dir = tmp_path / "empty_runs"
        assert list_runs(empty_dir) == []

    def test_list_runs_specific_batch(self, synthetic_runs_dir: Path):
        runs = list_runs(synthetic_runs_dir, batch_id="20261004-1500_sweep_test1")
        assert len(runs) == 2

        r1 = runs[0]
        assert r1["run_id"] == "000_iplanner_dingo_hall_straight_s0"
        assert r1["method"] == "iplanner"
        assert r1["robot"] == "dingo"
        assert r1["route"] == "hall_straight"
        assert r1["seed"] == 0
        assert r1["status"] == "done"
        assert r1["terminal_cause"] == "goal_reached"
        assert r1["success"] is True
        assert r1["time_to_goal"] == pytest.approx(18.50)

        r2 = runs[1]
        assert r2["run_id"] == "001_navdp_dingo_hall_straight_s0"
        assert r2["method"] == "navdp"
        assert r2["success"] is False
        assert r2["terminal_cause"] == "collision"
        assert r2["time_to_goal"] is None

    def test_list_runs_nonexistent_batch_raises(self, synthetic_runs_dir: Path):
        with pytest.raises(FileNotFoundError):
            list_runs(synthetic_runs_dir, batch_id="does_not_exist")


class TestShowRun:
    """Tests for show_run function."""

    def test_show_batch(self, synthetic_runs_dir: Path):
        info = show_run("20261004-1500_sweep_test1", synthetic_runs_dir)
        assert info["type"] == "batch"
        assert info["batch_id"] == "20261004-1500_sweep_test1"
        assert info["status"] == "done"
        assert info["progress"]["total"] == 2
        assert info["progress"]["done"] == 2
        assert info["progress"]["failed"] == 0
        assert info["success_rate"] == 50.0
        assert info["duration"]["duration_s"] == 300.0
        assert "spec" in info

    def test_show_run_nested_in_batch(self, synthetic_runs_dir: Path):
        info = show_run("000_iplanner_dingo_hall_straight_s0", synthetic_runs_dir)
        assert info["type"] == "run"
        assert info["run_id"] == "000_iplanner_dingo_hall_straight_s0"
        assert info["status"] == "done"
        assert info["outcome"]["success"] is True
        assert info["outcome"]["terminal_cause"] == "goal_reached"
        assert info["outcome"]["exit_code"] == 0
        assert info["metrics"]["time_to_goal_s"] == pytest.approx(18.50)
        assert info["metrics"]["path_length_m"] == pytest.approx(3.40)
        assert info["metrics"]["mean_inference_ms"] == pytest.approx(9.2)
        assert info["metrics"]["steps"] == 925
        assert info["log_path"] is not None

    def test_show_standalone_run(self, synthetic_runs_dir: Path):
        run_name = "20261004_170000_iplanner_dingo_hall_straight_abcd"
        info = show_run(run_name, synthetic_runs_dir)
        assert info["type"] == "run"
        assert info["run_id"] == run_name
        assert info["outcome"]["success"] is True
        assert info["metrics"]["time_to_goal_s"] == pytest.approx(15.2)

    def test_show_nonexistent_raises(self, synthetic_runs_dir: Path):
        with pytest.raises(FileNotFoundError):
            show_run("nonexistent_run_or_batch", synthetic_runs_dir)


class TestCompareBatch:
    """Tests for compare_batch function."""

    def test_compare_by_method(self, synthetic_runs_dir: Path):
        table = compare_batch("20261004-1500_sweep_test1", synthetic_runs_dir, by="method")
        assert "| Method" in table
        assert "| Total" in table
        assert "| Success" in table
        assert "| Rate (%)" in table
        assert "| Time to Goal (s)" in table
        assert "| Path Length (m)" in table
        assert "| Collisions" in table
        assert "| Mean Infer (ms)" in table

        # iplanner row check
        assert "iplanner" in table
        assert "18.50 ± 0.00" in table
        assert "9.2" in table

        # navdp row check
        assert "navdp" in table
        assert "165.0" in table

    def test_compare_multi_seed_statistics(self, tmp_path: Path):
        """Test statistical calculations (mean, std) across multiple seeds."""
        batch_dir = tmp_path / "stat_batch"
        batch_dir.mkdir()
        (batch_dir / "batch.yaml").write_text("batch_id: stat_batch\n", encoding="utf-8")

        csv_rows = [
            # iplanner: 2 successful runs
            {
                "batch_id": "stat_batch",
                "run_id": "r1",
                "robot": "dingo",
                "method": "iplanner",
                "route": "hall_straight",
                "seed": 0,
                "terminal_cause": "goal_reached",
                "success": True,
                "time_to_goal_s": 10.0,
                "path_length_m": 2.0,
                "mean_inference_ms": 10.0,
            },
            {
                "batch_id": "stat_batch",
                "run_id": "r2",
                "robot": "dingo",
                "method": "iplanner",
                "route": "hall_straight",
                "seed": 1,
                "terminal_cause": "goal_reached",
                "success": True,
                "time_to_goal_s": 20.0,
                "path_length_m": 4.0,
                "mean_inference_ms": 20.0,
            },
            # navdp: 2 runs, 1 collision, 1 timeout
            {
                "batch_id": "stat_batch",
                "run_id": "r3",
                "robot": "dingo",
                "method": "navdp",
                "route": "hall_straight",
                "seed": 0,
                "terminal_cause": "collision",
                "success": False,
                "time_to_goal_s": "",
                "path_length_m": 1.0,
                "mean_inference_ms": 50.0,
            },
            {
                "batch_id": "stat_batch",
                "run_id": "r4",
                "robot": "dingo",
                "method": "navdp",
                "route": "hall_straight",
                "seed": 1,
                "terminal_cause": "timeout",
                "success": False,
                "time_to_goal_s": "",
                "path_length_m": 3.0,
                "mean_inference_ms": 50.0,
            },
        ]
        with (batch_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            for r in csv_rows:
                writer.writerow(r)

        table = compare_batch(str(batch_dir), tmp_path, by="method")

        # iplanner mean ttg = 15.00, sample std = sqrt((5^2 + 5^2)/1) = 7.07
        assert "15.00 ± 7.07" in table
        # iplanner mean path = 3.00, std = 1.41
        assert "3.00 ± 1.41" in table
        # iplanner mean inference = 15.0
        assert "15.0" in table

        # navdp ttg is '-' because 0 successful runs
        lines = table.splitlines()
        navdp_line = next(line for line in lines if "navdp" in line)
        assert "-" in navdp_line
        assert "1" in navdp_line  # 1 collision

    def test_compare_by_route_and_robot(self, synthetic_runs_dir: Path):
        route_table = compare_batch("20261004-1500_sweep_test1", synthetic_runs_dir, by="route")
        assert "| Route" in route_table
        assert "hall_straight" in route_table

        robot_table = compare_batch("20261004-1500_sweep_test1", synthetic_runs_dir, by="robot")
        assert "| Robot" in robot_table
        assert "dingo" in robot_table

    def test_compare_invalid_by_raises(self, synthetic_runs_dir: Path):
        with pytest.raises(ValueError, match="Invalid grouping key"):
            compare_batch("20261004-1500_sweep_test1", synthetic_runs_dir, by="invalid_dimension")

    def test_compare_missing_results_raises(self, tmp_path: Path):
        empty_b = tmp_path / "empty_batch"
        empty_b.mkdir()
        (empty_b / "batch.yaml").write_text("batch_id: empty\n", encoding="utf-8")
        with pytest.raises(FileNotFoundError):
            compare_batch(str(empty_b), tmp_path)


class TestFindRunSpecForRerun:
    """Tests for find_run_spec_for_rerun."""

    def test_find_standalone_run(self, synthetic_runs_dir: Path):
        run_name = "20261004_170000_iplanner_dingo_hall_straight_abcd"
        run_dir, spec = find_run_spec_for_rerun(run_name, synthetic_runs_dir)
        assert run_dir == (synthetic_runs_dir / run_name).resolve()
        assert spec.method == "iplanner"
        assert spec.seed == 99

    def test_find_batch_run_with_spec_file(self, synthetic_runs_dir: Path):
        run_name = "000_iplanner_dingo_hall_straight_s0"
        run_dir, spec = find_run_spec_for_rerun(run_name, synthetic_runs_dir)
        assert run_dir.name == run_name
        assert spec.method == "iplanner"
        assert spec.robot == "dingo"

    def test_find_batch_run_manifest_reconstruction(self, synthetic_runs_dir: Path):
        # r4 in batch 2 was running and has no run_spec.json yet on disk
        run_name = "001_vint_dingo_hall_straight_s1"
        run_dir, spec = find_run_spec_for_rerun(run_name, synthetic_runs_dir)
        assert spec.method == "vint"
        assert spec.seed == 1
        assert spec.robot == "dingo"

    def test_find_nonexistent_raises(self, synthetic_runs_dir: Path):
        with pytest.raises(FileNotFoundError):
            find_run_spec_for_rerun("phantom_run_123", synthetic_runs_dir)


class TestCliRunsCommands:
    """Tests for CLI invocation of runs subcommands."""

    def test_cli_runs_no_action_prints_help(self, capsys: pytest.CaptureFixture[str]):
        exit_code = main(["runs"])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "usage: nav_arena runs" in captured.out

    def test_cli_runs_list(self, synthetic_runs_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
        monkeypatch.setattr("nav_arena.cli.RUNS_DIR", synthetic_runs_dir)
        exit_code = main(["runs", "list"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "20261004-1500_sweep_test1" in captured.out
        assert "20261004-1600_sweep_test2" in captured.out

    def test_cli_runs_list_batch(self, synthetic_runs_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
        monkeypatch.setattr("nav_arena.cli.RUNS_DIR", synthetic_runs_dir)
        exit_code = main(["runs", "list", "--batch", "20261004-1500_sweep_test1"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "000_iplanner_dingo_hall_straight_s0" in captured.out
        assert "001_navdp_dingo_hall_straight_s0" in captured.out

    def test_cli_runs_show_batch(self, synthetic_runs_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
        monkeypatch.setattr("nav_arena.cli.RUNS_DIR", synthetic_runs_dir)
        exit_code = main(["runs", "show", "20261004-1500_sweep_test1"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Batch: 20261004-1500_sweep_test1" in captured.out
        assert "Progress:" in captured.out

    def test_cli_runs_show_run(self, synthetic_runs_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
        monkeypatch.setattr("nav_arena.cli.RUNS_DIR", synthetic_runs_dir)
        exit_code = main(["runs", "show", "000_iplanner_dingo_hall_straight_s0"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Run: 000_iplanner_dingo_hall_straight_s0" in captured.out
        assert "Time to Goal:" in captured.out

    def test_cli_runs_compare(self, synthetic_runs_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
        monkeypatch.setattr("nav_arena.cli.RUNS_DIR", synthetic_runs_dir)
        exit_code = main(["runs", "compare", "20261004-1500_sweep_test1", "--by", "method"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Method" in captured.out
        assert "iplanner" in captured.out

    def test_cli_runs_rerun(self, synthetic_runs_dir: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr("nav_arena.cli.RUNS_DIR", synthetic_runs_dir)
        with patch("nav_arena.cli.execute_single_run_process", return_value=0) as mock_exec:
            exit_code = main(["runs", "rerun", "000_iplanner_dingo_hall_straight_s0", "--gui", "--quiet", "--force"])
            assert exit_code == 0
            mock_exec.assert_called_once()
            call_spec = mock_exec.call_args[1]["spec"]
            assert call_spec.viz.gui is True
            assert mock_exec.call_args[1]["quiet"] is True

    def test_cli_runs_nonexistent_fails(self, synthetic_runs_dir: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr("nav_arena.cli.RUNS_DIR", synthetic_runs_dir)
        assert main(["runs", "show", "nonexistent"]) == 1
        assert main(["runs", "compare", "nonexistent"]) == 1
        assert main(["runs", "rerun", "nonexistent", "--force"]) == 1


class TestEdgeCasesAndRobustness:
    """Targeted edge case tests verifying bug fixes."""

    def test_show_run_running_or_queued_without_disk_dir(self, synthetic_runs_dir: Path):
        """Verify show_run inspects run recorded in manifest even if episode directory does not exist on disk."""
        info = show_run("001_vint_dingo_hall_straight_s1", synthetic_runs_dir)
        assert info["type"] == "run"
        assert info["run_id"] == "001_vint_dingo_hall_straight_s1"
        assert info["status"] == "running"
        assert info["outcome"]["success"] is False
        assert info["outcome"]["terminal_cause"] is None
        assert info["settings"]["method"] == "vint"
        assert info["settings"]["seed"] == 1
        assert info["settings"]["robot"] == "dingo"

    def test_compare_batch_deduplication(self, tmp_path: Path):
        """Verify compare_batch deduplicates multiple rows for the same run_id (taking latest)."""
        batch_dir = tmp_path / "dedup_batch"
        batch_dir.mkdir()
        (batch_dir / "batch.yaml").write_text("batch_id: dedup_batch\n", encoding="utf-8")

        csv_rows = [
            # First attempt: collision
            {
                "batch_id": "dedup_batch",
                "run_id": "run_001",
                "method": "iplanner",
                "robot": "dingo",
                "route": "hall_straight",
                "seed": 0,
                "terminal_cause": "collision",
                "success": False,
                "time_to_goal_s": "",
                "path_length_m": 1.0,
                "mean_inference_ms": 10.0,
            },
            # Second attempt (rerun): success
            {
                "batch_id": "dedup_batch",
                "run_id": "run_001",
                "method": "iplanner",
                "robot": "dingo",
                "route": "hall_straight",
                "seed": 0,
                "terminal_cause": "goal_reached",
                "success": True,
                "time_to_goal_s": 12.5,
                "path_length_m": 3.0,
                "mean_inference_ms": 10.0,
            },
        ]
        with (batch_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            for r in csv_rows:
                writer.writerow(r)

        table = compare_batch(str(batch_dir), tmp_path, by="method")
        lines = [line for line in table.splitlines() if "iplanner" in line]
        assert len(lines) == 1
        # Total must be 1, success must be 1, rate 100.0%, 0 collisions
        parts = [p.strip() for p in lines[0].split("|") if p.strip()]
        assert parts[0] == "iplanner"
        assert parts[1] == "1"  # Total
        assert parts[2] == "1"  # Success
        assert parts[3] == "100.0%"  # Rate
        assert "12.50" in parts[4]  # TTG
        assert parts[6] == "0"  # Collisions

    def test_compare_batch_none_fields_handling(self, tmp_path: Path):
        """Verify compare_batch does not crash when results.csv has None/empty values."""
        batch_dir = tmp_path / "none_batch"
        batch_dir.mkdir()
        (batch_dir / "batch.yaml").write_text("batch_id: none_batch\n", encoding="utf-8")

        csv_rows = [
            {
                "batch_id": "none_batch",
                "run_id": "run_none",
                "method": "iplanner",
                "robot": "dingo",
                "route": "hall_straight",
                "seed": 0,
                "terminal_cause": None,
                "success": None,
                "time_to_goal_s": None,
                "path_length_m": None,
                "mean_inference_ms": None,
            },
        ]
        with (batch_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            for r in csv_rows:
                writer.writerow(r)

        table = compare_batch(str(batch_dir), tmp_path, by="method")
        assert "iplanner" in table

    def test_find_run_spec_custom_inline_route(self, tmp_path: Path):
        """Verify find_run_spec_for_rerun reconstructs coordinates for inline custom routes."""
        b_dir = tmp_path / "custom_route_batch"
        b_dir.mkdir()

        b_yaml = {
            "batch_id": "custom_route_batch",
            "spec": {
                "name": "custom_test",
                "scene": "kujiale_0003",
                "routes": [
                    {
                        "name": "custom_hall",
                        "spawn": [-3.0, 0.9],
                        "goal": [-6.3, -1.2],
                        "spawn_yaw": 1.57,
                    }
                ],
                "options": {
                    "max_steps": 800,
                    "goal_tolerance": 0.3,
                },
            },
        }
        (b_dir / "batch.yaml").write_text(yaml.dump(b_yaml), encoding="utf-8")

        rec = RunRecord(
            id="000_iplanner_dingo_custom_hall_s0",
            status=RunStatus.DONE.value,
            method="iplanner",
            method_family="in_process",
            robot="dingo",
            scene="kujiale_0003",
            route="custom_hall",
            seed=0,
        )
        manifest = BatchManifest(batch_id="custom_route_batch", runs=[rec])
        manifest.save(b_dir / "manifest.json")

        run_dir, spec = find_run_spec_for_rerun("000_iplanner_dingo_custom_hall_s0", tmp_path)
        assert spec.spawn == (-3.0, 0.9)
        assert spec.goal == (-6.3, -1.2)
        assert spec.spawn_yaw == 1.57
        assert spec.limits.max_steps == 800
        assert spec.limits.goal_tolerance == 0.3

    def test_cli_runs_rerun_syncs_batch(self, synthetic_runs_dir: Path, monkeypatch: pytest.MonkeyPatch):
        """Verify runs rerun updates the parent batch manifest and results.csv."""
        monkeypatch.setattr("nav_arena.cli.RUNS_DIR", synthetic_runs_dir)
        target_run_id = "001_navdp_dingo_hall_straight_s0"
        b1_dir = synthetic_runs_dir / "20261004-1500_sweep_test1"
        r2_dir = b1_dir / target_run_id

        # Mock execute_single_run_process to simulate a successful rerun and update summary.json
        def fake_exec(spec, run_dir, timeout=None, quiet=False, batch_id=""):
            (run_dir / "summary.json").write_text(
                json.dumps({
                    "success": True,
                    "terminal_cause": "goal_reached",
                    "time_to_goal_s": 22.1,
                }),
                encoding="utf-8",
            )
            # Write a row in run_dir/results.csv
            with (run_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=RESULTS_CSV_COLUMNS)
                writer.writeheader()
                writer.writerow({
                    "batch_id": batch_id,
                    "run_id": target_run_id,
                    "scene": "kujiale_0003",
                    "robot": "dingo",
                    "method": "navdp",
                    "method_family": "in_process",
                    "route": "hall_straight",
                    "seed": 0,
                    "terminal_cause": "goal_reached",
                    "success": True,
                    "time_to_goal_s": 22.1,
                })
            return 0

        with patch("nav_arena.cli.execute_single_run_process", side_effect=fake_exec):
            exit_code = main(["runs", "rerun", target_run_id, "--force"])
            assert exit_code == 0

        # Check manifest was updated
        m = BatchManifest.load(b1_dir / "manifest.json")
        rec = m.get_run(target_run_id)
        assert rec.status == RunStatus.DONE.value
        assert rec.terminal_cause == "goal_reached"
        assert rec.exit_code == 0

        # Check compare_batch reflects the rerun
        table = compare_batch("20261004-1500_sweep_test1", synthetic_runs_dir, by="method")
        lines = [line for line in table.splitlines() if "navdp" in line]
        assert len(lines) == 1
        assert "22.10 ± 0.00" in lines[0]
