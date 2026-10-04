# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Integration test for ROS 2 transform tree (TF, static TF, and robot_state_publisher)."""

import os
import subprocess
import sys
import pytest


@pytest.mark.integration
@pytest.mark.ros2
def test_integration_tf_tree():
    """Verify TF tree and robot_state_publisher in headless simulation."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.abspath(os.path.join(test_dir, "..", "..", "nav_arena", "scripts", "verify_tf_tree.py"))
    sim_root = os.path.abspath(os.path.join(test_dir, "..", "..", ".."))
    cmd = [
        sys.executable,
        "-u",
        script_path,
        "--headless",
        "--num-steps",
        "30",
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
        f"verify_tf_tree.py exited with return code {result.returncode}.\n"
        f"Output:\n{result.stdout}"
    )
