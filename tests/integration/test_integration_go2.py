# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Integration tests for the Unitree Go2 quadruped walked by NVIDIA's locomotion policy (headless Isaac Sim)."""

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
def test_go2_stands_and_walks_on_every_axis():
    """Verify Go2 stands 10 s under a zero command and tracks forward, sideways and turning commands; camera works."""
    result = _run("verify_embodiment.py", "--robot", "go2", "--camera", timeout=600)
    _assert_ok(result, "verify_embodiment.py --robot go2")
    for check in ("Robot settles after spawn", "Zero-command hold", "Stands for 10 s under a zero command"):
        assert f"[PASS] {check}" in result.stdout, check
    for check in ("Forward drive (vx)", "Strafe drive (vy)", "In-place rotation (wz)"):
        assert f"[PASS] PreTrainedPolicyAction {check}" in result.stdout, check
    assert "[PASS] Nearest ground return is plausible" in result.stdout


@pytest.mark.integration
def test_go2_pointnav_task():
    """Verify PointNav on Go2: goal reaching, no false collisions while walking (feet touch the floor), wall collision."""
    result = _run("verify_task.py", "--robot", "go2", "--camera", "--num-steps", "250", timeout=600)
    _assert_ok(result, "verify_task.py --robot go2")
    assert "Goal reached termination verified" in result.stdout
    assert "Collision termination and contact metric verified" in result.stdout


@pytest.mark.integration
@pytest.mark.skipif(
    not default_checkpoint("iplanner").is_file(), reason="NavDP checkout / iPlanner checkpoint not installed"
)
def test_iplanner_drives_go2_through_hall_straight():
    """Verify closed loop: iPlanner's path, followed through the locomotion policy, gets Go2 to the goal."""
    result = _run("verify_baseline.py", "--method", "iplanner", "--robot", "go2", "--route", "hall_straight", timeout=900)
    _assert_ok(result, "verify_baseline.py --method iplanner --robot go2")
    assert "Terminal cause:       goal_reached" in result.stdout.replace("\x1b[0m", "")
