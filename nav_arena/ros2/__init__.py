# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ROS 2 bridge components for simulation interaction."""

from .sensors import LaserScanPublisherNode
from .state_publisher import TaskStatePublisherNode

__all__ = [
    "LaserScanPublisherNode",
    "TaskStatePublisherNode",
    "build_ros2_omnigraph",
]


def __getattr__(name: str):
    if name == "build_ros2_omnigraph":
        from .graph_builder import build_ros2_omnigraph
        return build_ros2_omnigraph
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
