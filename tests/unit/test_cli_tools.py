# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for tool subcommands: doctor, routes, and map."""

from __future__ import annotations

import io
from pathlib import Path
import subprocess
import sys
from unittest.mock import MagicMock, patch

from PIL import Image
import pytest
import yaml

from nav_arena.benchmarks.doctor import (
    CheckResult,
    check_gpu_device,
    check_map_cache,
    check_model_checkpoints,
    check_navdp_repository,
    check_python_environment,
    check_running_processes,
    format_doctor_results,
    run_doctor,
    run_doctor_checks,
)
from nav_arena.cli import main
from nav_arena.methods.in_process.navdp_adapter.checkpoints import CHECKPOINTS, default_checkpoint


# ==============================================================================
# 1. Doctor Unit Checks
# ==============================================================================


class TestDoctorChecks:
    """Test individual health checks in benchmarks/doctor.py."""

    def test_python_environment_in_venv(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("VIRTUAL_ENV", "/mock/venv")
        res = check_python_environment()
        assert res.status == "PASS"
        assert "virtualenv" in res.message

    def test_python_environment_no_venv(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)
        with patch.object(sys, "prefix", "/usr"), patch.object(sys, "base_prefix", "/usr"):
            res = check_python_environment()
            assert res.status == "WARN"
            assert "without an active virtualenv" in res.message

    def test_python_environment_old_version(self):
        with patch.object(sys, "version_info", (3, 8, 2)):
            res = check_python_environment()
            assert res.status == "FAIL"
            assert "unsupported" in res.message

    def test_gpu_device_success(self):
        mock_output = "NVIDIA GeForce RTX 4090, 24564, 20120, 550.54.14\n"
        with patch("shutil.which", return_value="/usr/bin/nvidia-smi"), patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(args=[], returncode=0, stdout=mock_output),
        ):
            res = check_gpu_device()
            assert res.status == "PASS"
            assert "Found 1 GPU(s)" in res.message
            assert "RTX 4090" in res.message

    def test_gpu_device_missing_smi(self):
        with patch("shutil.which", return_value=None):
            res = check_gpu_device()
            assert res.status == "FAIL"
            assert "nvidia-smi not found" in res.message

    def test_gpu_device_smi_failed(self):
        with patch("shutil.which", return_value="/usr/bin/nvidia-smi"), patch(
            "subprocess.run",
            side_effect=subprocess.CalledProcessError(1, ["nvidia-smi"]),
        ):
            res = check_gpu_device()
            assert res.status == "FAIL"
            assert "nvidia-smi query failed" in res.message

    def test_gpu_device_na_memory(self):
        mock_output = "NVIDIA GB10, [N/A], [N/A], 580.178.04\n"
        with patch("shutil.which", return_value="/usr/bin/nvidia-smi"), patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(args=[], returncode=0, stdout=mock_output),
        ):
            res = check_gpu_device()
            assert res.status == "PASS"
            assert "unified memory" in res.message

    def test_running_processes_clean(self):
        with patch("nav_arena.benchmarks.doctor.check_preflight_processes", return_value=[]):
            res = check_running_processes()
            assert res.status == "PASS"
            assert "No conflicting" in res.message

    def test_running_processes_conflicts(self):
        with patch(
            "nav_arena.benchmarks.doctor.check_preflight_processes",
            return_value=[(9999, "python verify_baseline.py")],
        ):
            res = check_running_processes()
            assert res.status == "WARN"
            assert "1 active simulator process" in res.message
            assert any("9999" in d for d in res.details)

    def test_navdp_repository_missing(self, tmp_path: Path):
        missing_root = tmp_path / "NavDP_does_not_exist"
        res = check_navdp_repository(navdp_root=missing_root)
        assert res.status == "FAIL"
        assert "NavDP checkout missing" in res.message

    def test_navdp_repository_present_with_git(self, tmp_path: Path):
        navdp_dir = tmp_path / "NavDP"
        navdp_dir.mkdir()

        def mock_run(cmd, **kwargs):
            if "rev-parse" in cmd:
                return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="8b9ee13\n")
            if "status" in cmd:
                return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="")
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="")

        with patch("shutil.which", return_value="/usr/bin/git"), patch("subprocess.run", side_effect=mock_run):
            res = check_navdp_repository(navdp_root=navdp_dir)
            assert res.status == "PASS"
            assert "8b9ee13" in res.message
            assert "clean" in res.message

    def test_navdp_repository_os_error(self, tmp_path: Path):
        navdp_dir = tmp_path / "NavDP"
        navdp_dir.mkdir()
        with patch("shutil.which", return_value="/usr/bin/git"), patch(
            "subprocess.run",
            side_effect=OSError("permission denied"),
        ):
            res = check_navdp_repository(navdp_root=navdp_dir)
            assert res.status == "PASS"
            assert "not a git repo or detached" in res.message

    def test_model_checkpoints_all_present(self, tmp_path: Path):
        # Create all checkpoints under tmp_path
        for name, info in CHECKPOINTS.items():
            ckpt_path = default_checkpoint(name, navdp_root=tmp_path)
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            ckpt_path.write_bytes(b"dummy_weights")

        res = check_model_checkpoints(navdp_root=tmp_path)
        assert res.status == "PASS"
        assert f"All {len(CHECKPOINTS)}" in res.message

    def test_model_checkpoints_missing_all(self, tmp_path: Path):
        empty_navdp = tmp_path / "empty_navdp"
        empty_navdp.mkdir()
        res = check_model_checkpoints(navdp_root=empty_navdp)
        assert res.status == "FAIL"
        assert "No baseline checkpoints found" in res.message

    def test_model_checkpoints_partial(self, tmp_path: Path):
        # Only create iplanner checkpoint
        ckpt_path = default_checkpoint("iplanner", navdp_root=tmp_path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        ckpt_path.write_bytes(b"dummy_weights")

        res = check_model_checkpoints(navdp_root=tmp_path)
        assert res.status == "WARN"
        assert "checkpoints missing" in res.message
        assert any("[FOUND] iplanner" in d for d in res.details)
        assert any("[MISSING] vint" in d for d in res.details)

    def test_map_cache_present(self, tmp_path: Path):
        map_dir = tmp_path / "maps" / "kujiale_0003" / "cs0.05"
        map_dir.mkdir(parents=True)
        (map_dir / "map.yaml").write_text("image: map.png\nresolution: 0.05\n", encoding="utf-8")

        res = check_map_cache(cache_dir=tmp_path)
        assert res.status == "PASS"
        assert "kujiale_0003" in res.message

    def test_map_cache_missing_dir(self, tmp_path: Path):
        empty_cache = tmp_path / "empty_cache"
        res = check_map_cache(cache_dir=empty_cache)
        assert res.status == "WARN"
        assert "does not exist" in res.message

    def test_map_cache_empty_dir(self, tmp_path: Path):
        maps_dir = tmp_path / "maps"
        maps_dir.mkdir(parents=True)
        res = check_map_cache(cache_dir=tmp_path)
        assert res.status == "WARN"
        assert "No cached maps found" in res.message

    def test_run_doctor_overall(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("VIRTUAL_ENV", "/mock/venv")
        # All pass
        with patch("shutil.which", return_value="/usr/bin/nvidia-smi"), patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(args=[], returncode=0, stdout="GPU 0: Test GPU, 8000, 4000, 550.0\n"),
        ), patch("nav_arena.benchmarks.doctor.check_preflight_processes", return_value=[]):
            # Create mock checkpoints
            for name in CHECKPOINTS:
                p = default_checkpoint(name, navdp_root=tmp_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"x")

            # Create mock map
            m_dir = tmp_path / "maps" / "test_scene"
            m_dir.mkdir(parents=True)
            (m_dir / "map.yaml").write_text("resolution: 0.05\n", encoding="utf-8")

            exit_code = run_doctor(navdp_root=tmp_path, cache_dir=tmp_path)
            assert exit_code == 0

        # When a failure check exists
        with patch("shutil.which", return_value=None):
            exit_code_fail = run_doctor(navdp_root=tmp_path, cache_dir=tmp_path)
            assert exit_code_fail == 1


# ==============================================================================
# 2. CLI Routes Subcommands
# ==============================================================================


class TestCliRoutesSubcommands:
    """Test 'routes list' and 'routes show' subcommands."""

    def test_routes_list_default(self, capsys: pytest.CaptureFixture[str]):
        code = main(["routes", "list"])
        assert code == 0
        captured = capsys.readouterr().out
        assert "Registered routes for scene 'kujiale_0003':" in captured
        assert "hall_straight" in captured
        assert "around_table" in captured
        assert "through_doorway" in captured
        assert "to_far_room" in captured

    def test_routes_list_custom_scene(self, capsys: pytest.CaptureFixture[str]):
        code = main(["routes", "list", "--scene", "kujiale_0003"])
        assert code == 0
        captured = capsys.readouterr().out
        assert "hall_straight" in captured

    def test_routes_list_unknown_scene(self, capsys: pytest.CaptureFixture[str]):
        code = main(["routes", "list", "--scene", "nonexistent_scene_123"])
        assert code == 1

    def test_routes_show_valid(self, capsys: pytest.CaptureFixture[str]):
        code = main(["routes", "show", "hall_straight"])
        assert code == 0
        captured = capsys.readouterr().out
        assert "Route:          hall_straight" in captured
        assert "Scene:          kujiale_0003" in captured
        assert "Spawn:          (-6.40, 0.50)" in captured
        assert "Goal:           (-0.40, 0.50)" in captured
        assert "Distance:       6.00 m" in captured
        assert "Reference Path: 6.00 m" in captured
        assert "Description:    Straight run" in captured

    def test_routes_show_unknown_route(self, capsys: pytest.CaptureFixture[str]):
        code = main(["routes", "show", "nonexistent_route"])
        assert code == 1

    def test_routes_no_action(self, capsys: pytest.CaptureFixture[str]):
        code = main(["routes"])
        assert code == 1

    def test_routes_zero_heavy_imports(self):
        """Verify routes list in clean interpreter loads zero simulation or heavy modules."""
        code = (
            "import sys\n"
            "from nav_arena.cli import main\n"
            "exit_code = main(['routes', 'list'])\n"
            "assert exit_code == 0\n"
            "forbidden = {'isaacsim', 'isaaclab', 'omni', 'pxr', 'rclpy', 'torch'}\n"
            "loaded = forbidden.intersection(m.split('.')[0] for m in sys.modules)\n"
            "assert not loaded, f'Heavy modules loaded in routes list: {loaded}'\n"
        )
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert res.returncode == 0, f"Routes list leaked heavy modules:\n{res.stdout}\n{res.stderr}"


# ==============================================================================
# 3. CLI Map Subcommands
# ==============================================================================


class TestCliMapSubcommands:
    """Test 'map generate' and 'map show' subcommands."""

    def test_map_generate_wiring(self):
        with patch("subprocess.run", return_value=subprocess.CompletedProcess(args=[], returncode=0)) as mock_run:
            code = main([
                "map",
                "generate",
                "--scene",
                "kujiale_0003",
                "--resolution",
                "0.1",
                "--force",
            ])
            assert code == 0
            assert mock_run.called
            call_cmd = mock_run.call_args[0][0]
            assert "nav_arena.tools.map_generator" in call_cmd
            assert "--scene" in call_cmd
            assert "kujiale_0003" in call_cmd
            assert "--cell-size" in call_cmd
            assert "0.1" in call_cmd
            assert "--force" in call_cmd

    def test_map_show_cached(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
        # Setup mock cached map
        map_dir = tmp_path / "mock_scene_open_doors" / "sub"
        map_dir.mkdir(parents=True)
        img_path = map_dir / "map.png"

        # Create 100x100 white PNG
        img = Image.new("L", (100, 100), color=255)
        img.save(img_path)

        meta = {
            "image": "map.png",
            "resolution": 0.05,
            "origin": [-5.0, -5.0, 0.0],
            "occupied_thresh": 0.65,
            "free_thresh": 0.196,
        }
        (map_dir / "map.yaml").write_text(yaml.dump(meta), encoding="utf-8")

        with patch("nav_arena.tools.route_map.find_cached_map", return_value=map_dir):
            code = main(["map", "show", "--scene", "mock_scene"])
            assert code == 0
            captured = capsys.readouterr().out
            assert "Scene:            mock_scene" in captured
            assert "Resolution:       0.05 m/cell" in captured
            assert "Dimensions:       100 x 100 pixels" in captured
            assert "5.00 m x 5.00 m" in captured

    def test_map_show_missing_yaml_file(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
        map_dir = tmp_path / "mock_scene_no_yaml"
        map_dir.mkdir(parents=True)
        with patch("nav_arena.tools.route_map.find_cached_map", return_value=map_dir):
            code = main(["map", "show", "--scene", "mock_scene"])
            assert code == 1

    def test_map_show_not_found(self, capsys: pytest.CaptureFixture[str]):
        with patch("nav_arena.tools.route_map.find_cached_map", side_effect=FileNotFoundError("Map not found")):
            code = main(["map", "show", "--scene", "missing_scene"])
            assert code == 1

    def test_map_no_action(self, capsys: pytest.CaptureFixture[str]):
        code = main(["map"])
        assert code == 1


# ==============================================================================
# 4. CLI Doctor Wiring
# ==============================================================================


class TestCliDoctorWiring:
    """Test 'doctor' subcommand invocation from CLI."""

    def test_cli_doctor_dispatch(self):
        with patch("nav_arena.benchmarks.doctor.run_doctor", return_value=0) as mock_doc:
            code = main(["doctor"])
            assert code == 0
            assert mock_doc.called
            assert mock_doc.call_args[1].get("verbose") is False

    def test_cli_doctor_verbose(self):
        with patch("nav_arena.benchmarks.doctor.run_doctor", return_value=0) as mock_doc:
            code = main(["doctor", "-v"])
            assert code == 0
            assert mock_doc.called
            assert mock_doc.call_args[1].get("verbose") is True
