# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test for build_ros2_omnigraph."""

import os
import subprocess
import sys
import pytest


def _run_omnigraph_verification():
    from nav_arena.core import launch_simulation_app

    with launch_simulation_app(headless=True, enable_ros2=True) as simulation_app:
        import omni.graph.core as og
        import omni.usd
        from pxr import UsdGeom
        from nav_arena.ros2.graph_builder import build_ros2_omnigraph

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
            "PublishRawTF",
        ]
        for name in node_names:
            node = og.Controller.node(f"{graph_path}/{name}")
            assert node.is_valid(), f"Node {name} was not found or is invalid in graph {graph_path}."

        # Verify chassis prim relationship
        odom_prim = stage.GetPrimAtPath(f"{graph_path}/ComputeOdom")
        assert odom_prim.IsValid()
        rel = odom_prim.GetRelationship("inputs:chassisPrim")
        assert rel.IsValid()
        assert rel.GetTargets() == [robot_prim_path]

        # Verify modular graph: clock only
        clock_graph_path = "/ActionGraph/ClockOnly"
        clock_graph = build_ros2_omnigraph(
            graph_path=clock_graph_path,
            enable_clock=True,
            enable_odom=False,
            enable_tf=False,
        )
        assert clock_graph is not None and clock_graph.is_valid()
        assert og.Controller.node(f"{clock_graph_path}/PublishClock").is_valid()
        try:
            assert not og.Controller.node(f"{clock_graph_path}/ComputeOdom").is_valid()
        except og.OmniGraphValueError:
            pass  # Expected: node does not exist

        # Verify modular graph: odom only
        odom_graph_path = "/ActionGraph/OdomOnly"
        odom_graph = build_ros2_omnigraph(
            robot_prim_path=robot_prim_path,
            graph_path=odom_graph_path,
            enable_clock=False,
            enable_odom=True,
            enable_tf=False,
        )
        assert odom_graph is not None and odom_graph.is_valid()
        assert og.Controller.node(f"{odom_graph_path}/PublishOdom").is_valid()
        try:
            assert not og.Controller.node(f"{odom_graph_path}/PublishClock").is_valid()
        except og.OmniGraphValueError:
            pass  # Expected: node does not exist
        try:
            assert not og.Controller.node(f"{odom_graph_path}/PublishRawTF").is_valid()
        except og.OmniGraphValueError:
            pass  # Expected: node does not exist

        # Verify modular graph: tf only
        tf_graph_path = "/ActionGraph/TFOnly"
        tf_graph = build_ros2_omnigraph(
            robot_prim_path=robot_prim_path,
            graph_path=tf_graph_path,
            enable_clock=False,
            enable_odom=False,
            enable_tf=True,
        )
        assert tf_graph is not None and tf_graph.is_valid()
        assert og.Controller.node(f"{tf_graph_path}/PublishRawTF").is_valid()
        assert og.Controller.node(f"{tf_graph_path}/ComputeOdom").is_valid()
        try:
            assert not og.Controller.node(f"{tf_graph_path}/PublishClock").is_valid()
        except og.OmniGraphValueError:
            pass
        try:
            assert not og.Controller.node(f"{tf_graph_path}/PublishOdom").is_valid()
        except og.OmniGraphValueError:
            pass

        # Verify input validation errors
        try:
            build_ros2_omnigraph(
                robot_prim_path=None,
                graph_path="/ActionGraph/InvalidOdom",
                enable_odom=True,
            )
            assert False, "Expected ValueError when enable_odom=True without robot_prim_path"
        except ValueError:
            pass

        try:
            build_ros2_omnigraph(
                graph_path="/ActionGraph/InvalidEmpty",
                enable_clock=False,
                enable_odom=False,
                enable_tf=False,
            )
            assert False, "Expected ValueError when all flags are False"
        except ValueError:
            pass

        # Run a few simulation steps to verify graph execution without errors
        for _ in range(10):
            simulation_app.update()

        print("test_build_ros2_omnigraph PASSED")


@pytest.mark.ros2
def test_build_ros2_omnigraph():
    cmd = [sys.executable, "-u", os.path.abspath(__file__), "--run-sim"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, f"Graph builder test failed with code {result.returncode}:\n{result.stderr}\n{result.stdout}"


if __name__ == "__main__":
    _run_omnigraph_verification()
