# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for BatchManifest and RunRecord state machine, serialization, and queries."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

from nav_arena.benchmarks.manifest import (
    VALID_RUN_STATUSES,
    BatchManifest,
    RunRecord,
    RunStatus,
)


class TestManifestImports:
    """Verify manifest.py adheres to zero-simulator / zero-torch import constraint."""

    def test_zero_isaac_or_torch_imports(self):
        """Ensure manifest.py relies exclusively on stdlib."""
        forbidden = ("isaaclab", "isaacsim", "omni", "pxr", "torch", "rclpy")
        import nav_arena.benchmarks.manifest as manifest_mod

        for name, val in vars(manifest_mod).items():
            mod = getattr(val, "__module__", "")
            for f in forbidden:
                assert not mod.startswith(f), f"Forbidden import '{f}' detected in manifest symbol '{name}'"

    def test_zero_heavy_imports_in_fresh_process(self):
        """Verify importing manifest in clean interpreter loads zero simulation or deep learning modules."""
        import subprocess

        code = (
            "import sys\n"
            "import nav_arena.benchmarks.manifest\n"
            "heavy = {'torch', 'isaaclab', 'isaacsim', 'omni', 'pxr', 'rclpy'}\n"
            "loaded = heavy.intersection(sys.modules.keys())\n"
            "assert not loaded, f'Heavy modules loaded: {loaded}'\n"
        )
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert res.returncode == 0, f"Subprocess import failed:\n{res.stderr}"


class TestRunStatusEnum:
    """Test RunStatus enumeration and helpers."""

    def test_run_status_values(self):
        assert RunStatus.QUEUED.value == "queued"
        assert RunStatus.RUNNING.value == "running"
        assert RunStatus.DONE.value == "done"
        assert RunStatus.FAILED.value == "failed"
        assert RunStatus.TIMEOUT.value == "timeout"
        assert RunStatus.SKIPPED.value == "skipped"

    def test_run_status_equality_with_strings(self):
        assert RunStatus.QUEUED == "queued"
        assert RunStatus.DONE == "done"
        assert "running" in VALID_RUN_STATUSES


class TestRunRecord:
    """Test RunRecord dataclass instantiation, validation, and serialization."""

    def test_run_record_defaults(self):
        rec = RunRecord(
            id="001_iplanner_dingo_hall_straight_s0",
            status=RunStatus.QUEUED,
            method="iplanner",
            method_family="in_process",
        )
        assert rec.id == "001_iplanner_dingo_hall_straight_s0"
        assert rec.status == "queued"
        assert rec.method == "iplanner"
        assert rec.method_family == "in_process"
        assert rec.method_params == {}
        assert rec.robot == "dingo"
        assert rec.scene == "kujiale_0003"
        assert rec.route is None
        assert rec.seed == 0
        assert rec.pid is None
        assert rec.started is None
        assert rec.ended is None
        assert rec.exit_code is None
        assert rec.terminal_cause is None
        assert rec.run_dir is None
        assert rec.error is None

    def test_invalid_status_raises(self):
        with pytest.raises(ValueError, match="Invalid status 'invalid'"):
            RunRecord(
                id="run_1",
                status="invalid",
                method="iplanner",
                method_family="in_process",
            )

    def test_run_record_dict_roundtrip(self):
        rec = RunRecord(
            id="run_1",
            status=RunStatus.DONE,
            method="navdp",
            method_family="in_process",
            method_params={"fear_threshold": 0.7},
            robot="dingo",
            scene="kujiale_0004",
            route="around_table",
            seed=42,
            pid=12345,
            started="2026-10-04T12:00:00+00:00",
            ended="2026-10-04T12:00:30+00:00",
            exit_code=0,
            terminal_cause="goal_reached",
            run_dir="run_1_dir",
            error=None,
        )
        data = rec.to_dict()
        assert data["id"] == "run_1"
        assert data["status"] == "done"
        assert data["method_params"] == {"fear_threshold": 0.7}
        assert data["exit_code"] == 0

        restored = RunRecord.from_dict(data)
        assert restored == rec

    def test_run_record_null_and_missing_fields_handling(self):
        """Verify robust deserialization when fields are None or omitted in dict."""
        raw = {
            "id": "run_null_test",
            "method": "iplanner",
            "method_params": None,
            "seed": None,
            "robot": None,
            "scene": None,
            "method_family": None,
        }
        rec = RunRecord.from_dict(raw)
        assert rec.id == "run_null_test"
        assert rec.method == "iplanner"
        assert rec.method_params == {}
        assert rec.seed == 0
        assert rec.robot == "dingo"
        assert rec.scene == "kujiale_0003"
        assert rec.method_family == "in_process"
        assert rec.status == "queued"


class TestBatchManifestInitAndSerialization:
    """Test BatchManifest initialization, round-trips, and file operations."""

    def test_batch_manifest_init_defaults(self):
        manifest = BatchManifest(batch_id="batch_001")
        assert manifest.batch_id == "batch_001"
        assert manifest.runs == []
        assert manifest.created_at is not None
        assert isinstance(manifest.metadata, dict)

    def test_batch_manifest_with_runs_dicts(self):
        raw_runs = [
            {
                "id": "run_0",
                "status": "queued",
                "method": "iplanner",
                "method_family": "in_process",
            },
            {
                "id": "run_1",
                "status": "done",
                "method": "navdp",
                "method_family": "in_process",
                "exit_code": 0,
            },
        ]
        manifest = BatchManifest(batch_id="batch_002", runs=raw_runs)  # type: ignore[arg-type]
        assert len(manifest.runs) == 2
        assert isinstance(manifest.runs[0], RunRecord)
        assert manifest.runs[0].id == "run_0"
        assert manifest.runs[0].status == "queued"
        assert isinstance(manifest.runs[1], RunRecord)
        assert manifest.runs[1].id == "run_1"
        assert manifest.runs[1].status == "done"

    def test_json_roundtrip(self):
        run = RunRecord(
            id="run_0",
            status="queued",
            method="iplanner",
            method_family="in_process",
        )
        manifest = BatchManifest(
            batch_id="batch_test",
            runs=[run],
            metadata={"git_sha": "abc1234"},
        )
        json_str = manifest.to_json()
        restored = BatchManifest.from_json(json_str)

        assert restored.batch_id == "batch_test"
        assert restored.metadata == {"git_sha": "abc1234"}
        assert len(restored.runs) == 1
        assert restored.runs[0].id == "run_0"

    def test_batch_manifest_duplicate_run_ids_rejected(self):
        """Verify BatchManifest rejects duplicate run IDs during initialization."""
        r1 = RunRecord(id="dup_id", status="queued", method="iplanner", method_family="in_process")
        r2 = RunRecord(id="dup_id", status="queued", method="iplanner", method_family="in_process")
        with pytest.raises(ValueError, match="Duplicate run record ID 'dup_id'"):
            BatchManifest(batch_id="batch_dup", runs=[r1, r2])

    def test_batch_manifest_null_fields_handling(self):
        """Verify BatchManifest deserializes gracefully when runs or metadata are null."""
        raw = {
            "batch_id": "batch_null_test",
            "runs": None,
            "created_at": None,
            "metadata": None,
        }
        manifest = BatchManifest.from_dict(raw)
        assert manifest.batch_id == "batch_null_test"
        assert manifest.runs == []
        assert manifest.metadata == {}
        assert manifest.created_at != "None"
        assert len(manifest.created_at) > 0


class TestManifestAtomicSaveAndLoad:
    """Test atomic file writing and loading behavior."""

    def test_atomic_save_and_load(self, tmp_path: Path):
        batch_dir = tmp_path / "runs" / "batch_123"
        manifest_path = batch_dir / "manifest.json"

        run = RunRecord(
            id="run_a",
            status="queued",
            method="iplanner",
            method_family="in_process",
        )
        manifest = BatchManifest(batch_id="batch_123", runs=[run])

        # Verify atomic save creates directories and file
        manifest.save(manifest_path)
        assert manifest_path.is_file()
        assert not manifest_path.with_suffix(".tmp").exists()

        # Load back
        loaded = BatchManifest.load(manifest_path)
        assert loaded.batch_id == "batch_123"
        assert len(loaded.runs) == 1
        assert loaded.runs[0].id == "run_a"

    def test_atomic_save_uses_temporary_replace_and_fsync(self, tmp_path: Path):
        manifest_path = tmp_path / "manifest.json"
        manifest = BatchManifest(batch_id="test_replace")

        with patch("os.replace", wraps=__import__("os").replace) as mock_replace, \
             patch("os.fsync", wraps=__import__("os").fsync) as mock_fsync:
            manifest.save(manifest_path)
            mock_fsync.assert_called_once()
            mock_replace.assert_called_once()
            tmp_arg, dst_arg = mock_replace.call_args[0]
            assert str(tmp_arg).endswith(".tmp")
            assert Path(dst_arg) == manifest_path.resolve()

    def test_atomic_save_leaves_target_intact_if_write_fails(self, tmp_path: Path):
        """Verify original manifest remains uncorrupted if a save operation crashes midway."""
        manifest_path = tmp_path / "manifest.json"
        manifest = BatchManifest(batch_id="batch_v1")
        manifest.save(manifest_path)

        original_content = manifest_path.read_text(encoding="utf-8")

        # Attempt to save a corrupted or failing manifest
        with patch("os.fsync", side_effect=OSError("Disk write error")):
            with pytest.raises(OSError, match="Disk write error"):
                manifest.save(manifest_path)

        # Confirm target file is untouched and valid
        assert manifest_path.read_text(encoding="utf-8") == original_content
        loaded = BatchManifest.load(manifest_path)
        assert loaded.batch_id == "batch_v1"

    def test_load_nonexistent_raises_file_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="Manifest file not found"):
            BatchManifest.load(tmp_path / "nonexistent.json")

    def test_load_corrupt_json_raises_value_error(self, tmp_path: Path):
        corrupt_file = tmp_path / "corrupt.json"
        corrupt_file.write_text("{ not valid json ...", encoding="utf-8")
        with pytest.raises(ValueError, match="Corrupt manifest JSON file"):
            BatchManifest.load(corrupt_file)


class TestStateTransitions:
    """Test state machine transitions, invariants, and illegal calls."""

    @pytest.fixture
    def sample_manifest(self) -> BatchManifest:
        run = RunRecord(
            id="run_1",
            status=RunStatus.QUEUED,
            method="iplanner",
            method_family="in_process",
            route="hall_straight",
        )
        return BatchManifest(batch_id="test_batch", runs=[run])

    def test_happy_path_queued_to_running_to_done(self, sample_manifest: BatchManifest):
        # 1. Start running
        sample_manifest.mark_running("run_1", pid=9999, started="2026-10-04T10:00:00Z")
        rec = sample_manifest.get_run("run_1")
        assert rec.status == "running"
        assert rec.pid == 9999
        assert rec.started == "2026-10-04T10:00:00Z"
        assert rec.ended is None

        # 2. Complete done
        sample_manifest.mark_done(
            "run_1",
            exit_code=0,
            terminal_cause="goal_reached",
            ended="2026-10-04T10:00:20Z",
        )
        assert rec.status == "done"
        assert rec.exit_code == 0
        assert rec.terminal_cause == "goal_reached"
        assert rec.ended == "2026-10-04T10:00:20Z"

    def test_failure_path_running_to_failed(self, sample_manifest: BatchManifest):
        sample_manifest.mark_running("run_1", pid=8888)
        sample_manifest.mark_failed("run_1", exit_code=1, error="CUDA out of memory")
        rec = sample_manifest.get_run("run_1")
        assert rec.status == "failed"
        assert rec.exit_code == 1
        assert rec.error == "CUDA out of memory"
        assert rec.ended is not None

    def test_pre_launch_failure_queued_to_failed(self, sample_manifest: BatchManifest):
        sample_manifest.mark_failed("run_1", exit_code=1, error="Subprocess spawn failed")
        rec = sample_manifest.get_run("run_1")
        assert rec.status == "failed"
        assert rec.error == "Subprocess spawn failed"

    def test_timeout_path_running_to_timeout(self, sample_manifest: BatchManifest):
        sample_manifest.mark_running("run_1", pid=7777)
        sample_manifest.mark_timeout("run_1", ended="2026-10-04T10:05:00Z")
        rec = sample_manifest.get_run("run_1")
        assert rec.status == "timeout"
        assert rec.terminal_cause == "timeout"
        assert "timed out" in rec.error.lower()
        assert rec.ended == "2026-10-04T10:05:00Z"

    def test_skip_path_queued_to_skipped(self, sample_manifest: BatchManifest):
        sample_manifest.mark_skipped("run_1", reason="Unsupported scene")
        rec = sample_manifest.get_run("run_1")
        assert rec.status == "skipped"
        assert rec.error == "Unsupported scene"
        assert rec.ended is not None

    def test_resume_failed_or_timeout_to_running(self, sample_manifest: BatchManifest):
        # Fail first
        sample_manifest.mark_running("run_1", pid=100)
        sample_manifest.mark_failed("run_1", exit_code=1, error="Crash")

        # Resume / retry
        sample_manifest.mark_running("run_1", pid=200)
        rec = sample_manifest.get_run("run_1")
        assert rec.status == "running"
        assert rec.pid == 200
        assert rec.error is None
        assert rec.ended is None

        # Now succeed
        sample_manifest.mark_done("run_1", exit_code=0, terminal_cause="goal_reached")
        assert rec.status == "done"

    def test_reset_run_back_to_queued(self, sample_manifest: BatchManifest):
        sample_manifest.mark_running("run_1", pid=555)
        sample_manifest.mark_timeout("run_1")
        rec = sample_manifest.get_run("run_1")
        assert rec.status == "timeout"

        sample_manifest.reset_run("run_1")
        assert rec.status == "queued"
        assert rec.pid is None
        assert rec.error is None
        assert rec.started is None
        assert rec.ended is None

    def test_illegal_transitions_raise(self, sample_manifest: BatchManifest):
        # Cannot mark_done while queued
        with pytest.raises(ValueError, match="Expected 'running'"):
            sample_manifest.mark_done("run_1", exit_code=0, terminal_cause="goal_reached")

        # Cannot mark_timeout while queued
        with pytest.raises(ValueError, match="Expected 'running'"):
            sample_manifest.mark_timeout("run_1")

        # Transition to done
        sample_manifest.mark_running("run_1", pid=123)
        sample_manifest.mark_done("run_1", exit_code=0, terminal_cause="goal_reached")

        # Cannot re-run done run
        with pytest.raises(ValueError, match="terminal state 'done'"):
            sample_manifest.mark_running("run_1", pid=456)

        # Cannot mark_failed on done run
        with pytest.raises(ValueError, match="Cannot transition"):
            sample_manifest.mark_failed("run_1", exit_code=1)

        # Cannot reset done run
        with pytest.raises(ValueError, match="terminal state 'done'"):
            sample_manifest.reset_run("run_1")

        # Cannot skip done run
        with pytest.raises(ValueError, match="Expected 'queued'"):
            sample_manifest.mark_skipped("run_1")

    def test_illegal_transitions_from_skipped(self, sample_manifest: BatchManifest):
        sample_manifest.mark_skipped("run_1", reason="skip reason")
        rec = sample_manifest.get_run("run_1")
        assert rec.status == "skipped"
        assert rec.terminal_cause == "skipped"

        # Cannot mark running from skipped
        with pytest.raises(ValueError, match="terminal state 'skipped'"):
            sample_manifest.mark_running("run_1", pid=123)

        # Cannot mark done from skipped
        with pytest.raises(ValueError, match="Expected 'running'"):
            sample_manifest.mark_done("run_1", exit_code=0, terminal_cause="goal_reached")

        # Cannot mark failed from skipped
        with pytest.raises(ValueError, match="Cannot transition"):
            sample_manifest.mark_failed("run_1", exit_code=1)

        # Cannot mark timeout from skipped
        with pytest.raises(ValueError, match="Expected 'running'"):
            sample_manifest.mark_timeout("run_1")

        # Cannot reset from skipped
        with pytest.raises(ValueError, match="terminal state 'skipped'"):
            sample_manifest.reset_run("run_1")

    def test_transitions_with_run_dir(self, sample_manifest: BatchManifest):
        sample_manifest.mark_running("run_1", pid=1234, run_dir="dir_running")
        rec = sample_manifest.get_run("run_1")
        assert rec.run_dir == "dir_running"

        sample_manifest.mark_done("run_1", exit_code=0, terminal_cause="goal_reached", run_dir="dir_done")
        assert rec.run_dir == "dir_done"

    def test_idempotent_updates_on_failed_and_timeout(self, sample_manifest: BatchManifest):
        sample_manifest.mark_running("run_1", pid=101)
        sample_manifest.mark_failed("run_1", exit_code=1, error="first error")
        rec = sample_manifest.get_run("run_1")
        assert rec.status == "failed"
        assert rec.error == "first error"

        # Updating failed run with refined error or exit code should succeed
        sample_manifest.mark_failed("run_1", exit_code=137, error="SIGKILL received")
        assert rec.status == "failed"
        assert rec.exit_code == 137
        assert rec.error == "SIGKILL received"

    def test_missing_run_id_raises_key_error(self, sample_manifest: BatchManifest):
        with pytest.raises(KeyError, match="Run record 'nonexistent' not found"):
            sample_manifest.get_run("nonexistent")

        with pytest.raises(KeyError):
            sample_manifest.mark_running("nonexistent", pid=1)

        with pytest.raises(KeyError):
            sample_manifest.mark_done("nonexistent", exit_code=0, terminal_cause="goal_reached")

        with pytest.raises(KeyError):
            sample_manifest.mark_failed("nonexistent", exit_code=1)

        with pytest.raises(KeyError):
            sample_manifest.mark_timeout("nonexistent")

        with pytest.raises(KeyError):
            sample_manifest.mark_skipped("nonexistent")


class TestManifestQueries:
    """Test counts, completion check, and summaries."""

    def test_counts_and_finished(self):
        r1 = RunRecord(id="r1", status=RunStatus.QUEUED, method="m1", method_family="f1")
        r2 = RunRecord(id="r2", status=RunStatus.RUNNING, method="m1", method_family="f1")
        r3 = RunRecord(id="r3", status=RunStatus.DONE, method="m1", method_family="f1")
        manifest = BatchManifest(batch_id="b", runs=[r1, r2, r3])

        counts = manifest.counts_by_status()
        assert counts["queued"] == 1
        assert counts["running"] == 1
        assert counts["done"] == 1
        assert counts["failed"] == 0
        assert counts["timeout"] == 0
        assert counts["skipped"] == 0
        assert not manifest.is_finished()

        # Finish r1 and r2
        manifest.mark_skipped("r1", reason="test")
        manifest.mark_done("r2", exit_code=0, terminal_cause="goal_reached")
        assert manifest.is_finished()

        counts_after = manifest.counts_by_status()
        assert counts_after["queued"] == 0
        assert counts_after["running"] == 0
        assert counts_after["done"] == 2
        assert counts_after["skipped"] == 1

    def test_empty_manifest_is_finished(self):
        empty = BatchManifest(batch_id="empty")
        assert empty.is_finished()
        assert empty.counts_by_status() == {
            "queued": 0,
            "running": 0,
            "done": 0,
            "failed": 0,
            "timeout": 0,
            "skipped": 0,
        }

    def test_summary_dict(self):
        r1 = RunRecord(id="r1", status=RunStatus.DONE, method="m1", method_family="f1")
        manifest = BatchManifest(batch_id="b_sum", runs=[r1])
        summary = manifest.summary_dict()

        assert summary["batch_id"] == "b_sum"
        assert summary["total"] == 1
        assert summary["counts"]["done"] == 1
        assert summary["is_finished"] is True
        assert "created_at" in summary

    def test_add_run_and_duplicate_prevention(self):
        manifest = BatchManifest(batch_id="b_add")
        r1 = RunRecord(id="run_alpha", status="queued", method="iplanner", method_family="in_process")
        manifest.add_run(r1)
        assert len(manifest.runs) == 1

        with pytest.raises(ValueError, match="already exists"):
            manifest.add_run(r1)
