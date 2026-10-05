# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Integration tests for the NVIDIA Kaya holonomic embodiment and baselines (headless Isaac Sim)."""

import os
import subprocess
import sys

import pytest

from nav_arena.methods.in_process.navdp_adapter.checkpoints import default_checkpoint

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.abspath(os.path.join(_TEST_DIR, "..", "..", "nav_arena", "scripts"))
_SIM_ROOT = os.path.abspath(os.path.join(_TEST_DIR, "..", "..", ".."))


def _run(script: str, *args: str, timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-u", os.path.join(_SCRIPTS, script), "--headless", *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
        cwd=_SIM_ROOT,
    )


def _assert_ok(result: subprocess.CompletedProcess, name: str) -> None:
    assert result.returncode == 0, f"{name} exited with {result.returncode}.\nOutput:\n{result.stdout[-6000:]}"


@pytest.mark.integration
def test_kaya_embodiment_with_mast_camera():
    """Verify Kaya spawns, executes 3-axis holonomic motion, and generates valid RGB-D frames (mast variant)."""
    result = _run("verify_embodiment.py", "--robot", "kaya", "--camera", timeout=300)
    _assert_ok(result, "verify_embodiment.py --robot kaya")
    assert "[PASS] Zero-command hold" in result.stdout
    for check in ("Forward drive (vx)", "Strafe drive (vy)", "In-place rotation (wz)"):
        assert f"[PASS] HolonomicDriveAction {check}" in result.stdout, check
    assert "Camera resolution is 640x360" in result.stdout


@pytest.mark.integration
def test_kaya_embodiment_with_native_camera():
    """Verify Kaya native RealSense D435 variant (pitched down 20 deg) produces valid metric depth."""
    result = _run("verify_embodiment.py", "--robot", "kaya.native", "--camera", timeout=300)
    _assert_ok(result, "verify_embodiment.py --robot kaya.native")
    assert "Depth has valid metric returns" in result.stdout
    assert "Nearest ground return is plausible" in result.stdout


@pytest.mark.integration
def test_kaya_pointnav_task():
    """Verify PointNav task on Kaya: goal reaching, 0 false collisions during free drive, and wall collision."""
    result = _run("verify_task.py", "--robot", "kaya", "--camera", "--num-steps", "150", timeout=300)
    _assert_ok(result, "verify_task.py --robot kaya")
    assert "Goal reached termination verified" in result.stdout
    assert "Max lateral contact force during free drive: 0.000 N" in result.stdout
    assert "Collision termination and contact metric verified" in result.stdout


@pytest.mark.integration
@pytest.mark.skipif(
    not default_checkpoint("iplanner").is_file(), reason="NavDP checkout / iPlanner checkpoint not installed"
)
def test_iplanner_reaches_goal_on_kaya():
    """Verify closed-loop navigation: iPlanner drives Kaya through hall_straight to the goal."""
    result = _run(
        "verify_baseline.py",
        "--method",
        "iplanner",
        "--robot",
        "kaya",
        "--route",
        "hall_straight",
        timeout=600,
    )
    _assert_ok(result, "verify_baseline.py --method iplanner --robot kaya")
    assert "Terminal cause:       goal_reached" in result.stdout.replace("\x1b[0m", "")
