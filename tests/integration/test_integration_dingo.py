# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Integration tests for the Clearpath Dingo embodiment and the in-process baselines (headless Isaac Sim)."""

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
def test_dingo_embodiment_with_camera():
    """Verify the Dingo spawns, drives, applies its stage patch, and produces valid RGB / depth frames."""
    _assert_ok(_run("verify_embodiment.py", "--robot", "dingo", "--camera", timeout=300), "verify_embodiment.py")


@pytest.mark.integration
def test_dingo_pointnav_task_without_false_collisions():
    """Verify the Dingo task: caster friction patch, goal reaching, no false collisions, wall collision, goal image."""
    result = _run("verify_task.py", "--robot", "dingo", "--camera", "--num-steps", "150", timeout=300)
    _assert_ok(result, "verify_task.py --robot dingo")
    # The caster-friction and ground-plane patches must be confirmed by the script itself, not assumed.
    assert "Caster friction combine mode is 'min'" in result.stdout
    assert "Max lateral contact force during free drive: 0.000 N" in result.stdout


@pytest.mark.integration
@pytest.mark.skipif(
    not default_checkpoint("iplanner").is_file(), reason="NavDP checkout / iPlanner checkpoint not installed"
)
def test_iplanner_reaches_goal_on_dingo():
    """Verify a full closed-loop episode: iPlanner drives the Dingo across the hall to the goal."""
    # The route passes within ~1 m of a dining chair, where iPlanner's predicted fear peaks at ~0.70. Its stop gate has
    # no recovery, so a 0.7 threshold makes this test hinge on GPU float noise; 0.8 keeps it about integration.
    result = _run(
        "verify_baseline.py",
        "--method",
        "iplanner",
        "--robot",
        "dingo",
        "--route",
        "hall_straight",
        "--policy-arg",
        "fear_threshold=0.8",
        timeout=600,
    )
    _assert_ok(result, "verify_baseline.py --method iplanner")
    assert "Terminal cause:       goal_reached" in result.stdout.replace("\x1b[0m", "")
