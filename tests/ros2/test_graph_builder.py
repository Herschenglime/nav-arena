# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test for build_ros2_omnigraph."""

import sys

# 1. Mandatory flag injection before AppLauncher boots
sys.argv.extend([
    "--enable", "omni.graph",
    "--enable", "omni.graph.action",
    "--enable", "isaacsim.ros2.bridge",
    "--enable", "isaacsim.ros2.nodes",
])

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import omni.graph.core as og
import omni.usd
from pxr import UsdGeom
from nav_arena.ros2.graph_builder import build_ros2_omnigraph


def test_build_ros2_omnigraph():
    stage = omni.usd.get_context().get_stage()

    # Create dummy robot prim
    robot_prim_path = "/World/TestRobot"
    UsdGeom.Xform.Define(stage, robot_prim_path)

    # Build the ROS 2 OmniGraph
    graph_path = "/ActionGraph/ROS_Bridge"
    graph = build_ros2_omnigraph(robot_prim_path=robot_prim_path, graph_path=graph_path)

    assert graph is not None
    assert graph.is_valid()

    # Verify that the expected nodes exist in the graph
    node_names = [
        "OnPlaybackTick",
        "ReadSimTime",
        "PublishClock",
        "ComputeOdom",
        "PublishOdom",
        "PublishTF",
    ]
    for name in node_names:
        node = og.Controller.node(f"{graph_path}/{name}")
        assert node.is_valid(), f"Node {name} was not found or is invalid in graph {graph_path}."

    # Run a few simulation steps to verify graph execution without errors
    for _ in range(10):
        simulation_app.update()

    print("test_build_ros2_omnigraph PASSED")


if __name__ == "__main__":
    test_build_ros2_omnigraph()
    simulation_app.close()
