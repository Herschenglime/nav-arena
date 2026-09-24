# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OmniGraph ROS 2 bridge builder for Isaac Sim."""

from typing import Optional
import omni.graph.core as og
import omni.usd
from pxr import Sdf


def build_ros2_omnigraph(
    robot_prim_path: str,
    graph_path: str = "/ActionGraph/ROS_Bridge",
    odom_topic: str = "odom",
    odom_frame: str = "odom",
    base_frame: str = "base_link",
) -> og.Graph:
    """Build programmatic OmniGraph for ROS 2 Clock, TF, and Odometry publishing.

    Args:
        robot_prim_path: Absolute USD path to the robot root or articulation prim.
        graph_path: Path in USD stage for the ActionGraph. Defaults to '/ActionGraph/ROS_Bridge'.
        odom_topic: ROS 2 topic for odometry. Defaults to 'odom'.
        odom_frame: Parent frame ID for odometry. Defaults to 'odom'.
        base_frame: Robot base frame ID. Defaults to 'base_link'.

    Returns:
        The created OmniGraph instance.
    """
    keys = og.Controller.Keys

    # 1. Create nodes and connections
    og.Controller.edit(
        {"graph_path": graph_path, "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
                ("ComputeOdom", "isaacsim.core.nodes.IsaacComputeOdometry"),
                ("PublishOdom", "isaacsim.ros2.bridge.ROS2PublishOdometry"),
                ("PublishTF", "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
            ],
            keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
                ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
                ("OnPlaybackTick.outputs:tick", "ComputeOdom.inputs:execIn"),
                ("ComputeOdom.outputs:execOut", "PublishOdom.inputs:execIn"),
                ("ReadSimTime.outputs:simulationTime", "PublishOdom.inputs:timeStamp"),
                ("ComputeOdom.outputs:position", "PublishOdom.inputs:position"),
                ("ComputeOdom.outputs:orientation", "PublishOdom.inputs:orientation"),
                ("ComputeOdom.outputs:linearVelocity", "PublishOdom.inputs:linearVelocity"),
                ("ComputeOdom.outputs:angularVelocity", "PublishOdom.inputs:angularVelocity"),
                ("OnPlaybackTick.outputs:tick", "PublishTF.inputs:execIn"),
                ("ReadSimTime.outputs:simulationTime", "PublishTF.inputs:timeStamp"),
            ],
            keys.SET_VALUES: [
                ("PublishClock.inputs:topicName", "clock"),
                ("PublishOdom.inputs:topicName", odom_topic),
                ("PublishOdom.inputs:odomFrameId", odom_frame),
                ("PublishOdom.inputs:chassisFrameId", base_frame),
                ("PublishTF.inputs:topicName", "tf"),
            ],
        },
    )

    # 2. Configure target prim relationships on USD stage
    stage = omni.usd.get_context().get_stage()
    target_path = Sdf.Path(robot_prim_path)

    # Chassis prim for odometry computation
    odom_node_prim = stage.GetPrimAtPath(f"{graph_path}/ComputeOdom")
    if odom_node_prim.IsValid():
        rel = odom_node_prim.GetRelationship("inputs:chassisPrim")
        if not rel.IsValid():
            rel = odom_node_prim.CreateRelationship("inputs:chassisPrim")
        rel.SetTargets([target_path])

    # Target prims for transform tree publishing
    tf_node_prim = stage.GetPrimAtPath(f"{graph_path}/PublishTF")
    if tf_node_prim.IsValid():
        rel = tf_node_prim.GetRelationship("inputs:targetPrims")
        if not rel.IsValid():
            rel = tf_node_prim.CreateRelationship("inputs:targetPrims")
        rel.SetTargets([target_path])

    graph = og.get_graph_by_path(graph_path)
    return graph
