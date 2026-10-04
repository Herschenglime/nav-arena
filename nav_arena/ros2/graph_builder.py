# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OmniGraph ROS 2 bridge builder for Isaac Sim."""

from __future__ import annotations

from typing import Optional
import omni.graph.core as og
import omni.usd
from pxr import Sdf


def build_ros2_omnigraph(
    robot_prim_path: Optional[str] = None,
    graph_path: str = "/ActionGraph/ROS_Bridge",
    enable_clock: bool = True,
    enable_odom: bool = True,
    enable_tf: bool = True,
    odom_topic: str = "odom",
    odom_frame: str = "odom",
    base_frame: str = "base_link",
) -> og.Graph:
    """Build programmatic OmniGraph for ROS 2 Clock, TF, and Odometry publishing.

    Args:
        robot_prim_path: Absolute USD path to the robot root or articulation prim.
            Required if enable_odom or enable_tf is True.
        graph_path: Path in USD stage for the ActionGraph. Defaults to '/ActionGraph/ROS_Bridge'.
        enable_clock: Whether to publish ROS 2 /clock. Defaults to True.
        enable_odom: Whether to compute and publish nav_msgs/Odometry. Defaults to True.
        enable_tf: Whether to broadcast dynamic odom -> base_frame TF. Defaults to True.
        odom_topic: ROS 2 topic for odometry. Defaults to 'odom'.
        odom_frame: Parent frame ID for odometry and TF. Defaults to 'odom'.
        base_frame: Robot base frame ID. Defaults to 'base_link'.

    Returns:
        The created OmniGraph instance.
    """
    if not (enable_clock or enable_odom or enable_tf):
        raise ValueError("At least one of enable_clock, enable_odom, or enable_tf must be True")

    if (enable_odom or enable_tf) and not robot_prim_path:
        raise ValueError("robot_prim_path must be specified when enable_odom or enable_tf is True")

    keys = og.Controller.Keys

    create_nodes = []
    connect = []
    set_values = []

    # Common timing nodes required if any bridge is enabled
    needs_tick = enable_clock or enable_odom or enable_tf
    if needs_tick:
        create_nodes.append(("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"))
        create_nodes.append(("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"))

    # ROS 2 Clock Publisher
    if enable_clock:
        create_nodes.append(("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"))
        connect.extend([
            ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
            ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
        ])
        set_values.append(("PublishClock.inputs:topicName", "clock"))

    # Odometry computation (shared by PublishOdom and PublishRawTF)
    needs_compute_odom = enable_odom or enable_tf
    if needs_compute_odom:
        create_nodes.append(("ComputeOdom", "isaacsim.core.nodes.IsaacComputeOdometry"))
        connect.append(("OnPlaybackTick.outputs:tick", "ComputeOdom.inputs:execIn"))

    # ROS 2 Odometry Publisher
    if enable_odom:
        create_nodes.append(("PublishOdom", "isaacsim.ros2.bridge.ROS2PublishOdometry"))
        connect.extend([
            ("ComputeOdom.outputs:execOut", "PublishOdom.inputs:execIn"),
            ("ReadSimTime.outputs:simulationTime", "PublishOdom.inputs:timeStamp"),
            ("ComputeOdom.outputs:position", "PublishOdom.inputs:position"),
            ("ComputeOdom.outputs:orientation", "PublishOdom.inputs:orientation"),
            ("ComputeOdom.outputs:linearVelocity", "PublishOdom.inputs:linearVelocity"),
            ("ComputeOdom.outputs:angularVelocity", "PublishOdom.inputs:angularVelocity"),
        ])
        set_values.extend([
            ("PublishOdom.inputs:topicName", odom_topic),
            ("PublishOdom.inputs:odomFrameId", odom_frame),
            ("PublishOdom.inputs:chassisFrameId", base_frame),
        ])

    # ROS 2 Dynamic Transform Tree Publisher (odom -> base_frame)
    if enable_tf:
        create_nodes.append(("PublishRawTF", "isaacsim.ros2.bridge.ROS2PublishRawTransformTree"))
        connect.extend([
            ("OnPlaybackTick.outputs:tick", "PublishRawTF.inputs:execIn"),
            ("ReadSimTime.outputs:simulationTime", "PublishRawTF.inputs:timeStamp"),
            ("ComputeOdom.outputs:position", "PublishRawTF.inputs:translation"),
            ("ComputeOdom.outputs:orientation", "PublishRawTF.inputs:rotation"),
        ])
        set_values.extend([
            ("PublishRawTF.inputs:topicName", "tf"),
            ("PublishRawTF.inputs:parentFrameId", odom_frame),
            ("PublishRawTF.inputs:childFrameId", base_frame),
        ])

    # 1. Create nodes, connections, and values
    if create_nodes:
        og.Controller.edit(
            {"graph_path": graph_path, "evaluator_name": "execution"},
            {
                keys.CREATE_NODES: create_nodes,
                keys.CONNECT: connect,
                keys.SET_VALUES: set_values,
            },
        )

    # 2. Configure target prim relationships on USD stage
    if needs_compute_odom and robot_prim_path:
        stage = omni.usd.get_context().get_stage()
        target_path = Sdf.Path(robot_prim_path)

        odom_node_prim = stage.GetPrimAtPath(f"{graph_path}/ComputeOdom")
        if odom_node_prim.IsValid():
            rel = odom_node_prim.GetRelationship("inputs:chassisPrim")
            if not rel.IsValid():
                rel = odom_node_prim.CreateRelationship("inputs:chassisPrim")
            rel.SetTargets([target_path])

    graph = og.get_graph_by_path(graph_path)
    return graph
