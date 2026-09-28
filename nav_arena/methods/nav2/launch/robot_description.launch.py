# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ROS 2 launch file for robot_state_publisher with parameterized in-memory URDF."""

from __future__ import annotations

from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from nav_arena.embodiments.urdf import generate_minimal_urdf


def launch_setup(context: LaunchContext, *args, **kwargs):
    use_sim_time = LaunchConfiguration("use_sim_time").perform(context).lower() in ("true", "1")
    robot_name = LaunchConfiguration("robot_name").perform(context)
    base_frame = LaunchConfiguration("base_frame").perform(context)
    chassis_frame = LaunchConfiguration("chassis_frame").perform(context)
    lidar_frame = LaunchConfiguration("lidar_frame").perform(context)

    sensor_x = float(LaunchConfiguration("sensor_x").perform(context))
    sensor_y = float(LaunchConfiguration("sensor_y").perform(context))
    sensor_z = float(LaunchConfiguration("sensor_z").perform(context))
    robot_radius = float(LaunchConfiguration("robot_radius").perform(context))
    robot_height = float(LaunchConfiguration("robot_height").perform(context))
    chassis_size_x = float(LaunchConfiguration("chassis_size_x").perform(context))
    chassis_size_y = float(LaunchConfiguration("chassis_size_y").perform(context))
    chassis_size_z = float(LaunchConfiguration("chassis_size_z").perform(context))
    chassis_offset_x = float(LaunchConfiguration("chassis_offset_x").perform(context))
    chassis_offset_y = float(LaunchConfiguration("chassis_offset_y").perform(context))
    chassis_offset_z = float(LaunchConfiguration("chassis_offset_z").perform(context))

    urdf_xml = generate_minimal_urdf(
        name=robot_name,
        base_frame=base_frame,
        chassis_frame=chassis_frame,
        lidar_frame=lidar_frame,
        lidar_offset=(sensor_x, sensor_y, sensor_z),
        robot_radius=robot_radius,
        robot_height=robot_height,
        chassis_size=(chassis_size_x, chassis_size_y, chassis_size_z),
        chassis_offset=(chassis_offset_x, chassis_offset_y, chassis_offset_z),
    )

    rsp_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[
            {
                "robot_description": urdf_xml,
                "use_sim_time": use_sim_time,
            }
        ],
    )

    return [rsp_node]


def generate_launch_description() -> LaunchDescription:
    declared_arguments = [
        DeclareLaunchArgument("use_sim_time", default_value="true", description="Use simulation clock (/clock)"),
        DeclareLaunchArgument("robot_name", default_value="nova_carter", description="Robot name"),
        DeclareLaunchArgument("base_frame", default_value="base_link", description="Root base link"),
        DeclareLaunchArgument("chassis_frame", default_value="chassis_link", description="Chassis link"),
        DeclareLaunchArgument("lidar_frame", default_value="lidar_link", description="LiDAR link"),
        DeclareLaunchArgument("sensor_x", default_value="0.0", description="LiDAR X offset (m)"),
        DeclareLaunchArgument("sensor_y", default_value="0.0", description="LiDAR Y offset (m)"),
        DeclareLaunchArgument("sensor_z", default_value="0.35", description="LiDAR Z offset (m)"),
        DeclareLaunchArgument("robot_radius", default_value="0.28", description="Robot footprint radius (m)"),
        DeclareLaunchArgument("robot_height", default_value="0.40", description="Robot body height (m)"),
        DeclareLaunchArgument("chassis_size_x", default_value="0.747", description="Chassis box length (m)"),
        DeclareLaunchArgument("chassis_size_y", default_value="0.50", description="Chassis box width (m)"),
        DeclareLaunchArgument("chassis_size_z", default_value="0.40", description="Chassis box height (m)"),
        DeclareLaunchArgument("chassis_offset_x", default_value="-0.2335", description="Chassis box X offset (m)"),
        DeclareLaunchArgument("chassis_offset_y", default_value="0.0", description="Chassis box Y offset (m)"),
        DeclareLaunchArgument("chassis_offset_z", default_value="0.20", description="Chassis box Z offset (m)"),
        OpaqueFunction(function=launch_setup),
    ]

    return LaunchDescription(declared_arguments)
