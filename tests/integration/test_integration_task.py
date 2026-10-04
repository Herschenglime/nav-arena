# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Integration test for PointNav task environment and metrics."""

import os
import subprocess
import sys
import pytest


@pytest.mark.integration
def test_integration_task():
    """Verify PointNav task environment in headless simulation."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.abspath(os.path.join(test_dir, "..", "..", "nav_arena", "scripts", "verify_task.py"))
    sim_root = os.path.abspath(os.path.join(test_dir, "..", "..", ".."))
    cmd = [
        sys.executable,
        "-u",
        script_path,
        "--headless",
        "--num-steps",
        # Reaching the 1.5 m goal at 0.6 m/s takes ~110 steps; 30 steps can never satisfy the script's own check.
        "150",
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=300,
        cwd=sim_root,
    )

    assert result.returncode == 0, (
        f"verify_task.py exited with return code {result.returncode}.\n"
        f"Output:\n{result.stdout}"
    )
