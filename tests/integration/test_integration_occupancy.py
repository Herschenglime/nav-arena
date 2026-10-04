# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Integration test for 2D occupancy map generation."""

import os
import subprocess
import sys
import pytest


@pytest.mark.integration
def test_integration_occupancy():
    """Verify 2D occupancy map generation in headless simulation."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.abspath(os.path.join(test_dir, "..", "..", "nav_arena", "scripts", "verify_occupancy_map.py"))
    sim_root = os.path.abspath(os.path.join(test_dir, "..", "..", ".."))
    cmd = [
        sys.executable,
        "-u",
        script_path,
        "--headless",
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=120,
        cwd=sim_root,
    )

    assert result.returncode == 0, (
        f"verify_occupancy_map.py exited with return code {result.returncode}.\n"
        f"Output:\n{result.stdout}"
    )
