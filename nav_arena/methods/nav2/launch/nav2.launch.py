# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unified ROS 2 launch file for Nav2 stack, robot_state_publisher, and map server."""

from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from nav2_common.launch import RewrittenYaml


def generate_launch_description() -> LaunchDescription:
    # Default paths
    base_nav_arena_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    default_map_path = os.path.join(
        base_nav_arena_dir,
        "cache",
        "maps",
        "kujiale_0003",
        "cs0.05_z0.01-0.60",
        "map.yaml",
    )
    default_params_file = os.path.join(
        os.path.dirname(__file__),
        "..",
        "config",
        "nav2_params.yaml",
    )
    robot_desc_launch_file = os.path.join(
        os.path.dirname(__file__),
        "robot_description.launch.py",
    )

    # Launch configuration variables
    use_sim_time = LaunchConfiguration("use_sim_time")
    map_yaml_file = LaunchConfiguration("map")
    params_file = LaunchConfiguration("params_file")
    autostart = LaunchConfiguration("autostart")
    use_rviz = LaunchConfiguration("rviz")

    # Declare launch arguments
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation (OmniGraph) clock if true",
    )

    declare_map_yaml_cmd = DeclareLaunchArgument(
        "map",
        default_value=default_map_path,
        description="Full path to map.yaml to load",
    )

    declare_params_file_cmd = DeclareLaunchArgument(
        "params_file",
        default_value=default_params_file,
        description="Full path to the ROS2 parameters file to use for all launched nodes",
    )

    declare_autostart_cmd = DeclareLaunchArgument(
        "autostart",
        default_value="true",
        description="Automatically startup the nav2 stack",
    )

    declare_use_rviz_cmd = DeclareLaunchArgument(
        "rviz",
        default_value="false",
        description="Whether to start RViz",
    )

    # Initial pose configuration for AMCL
    initial_pose_x = LaunchConfiguration("initial_pose_x")
    initial_pose_y = LaunchConfiguration("initial_pose_y")
    initial_pose_z = LaunchConfiguration("initial_pose_z")
    initial_pose_yaw = LaunchConfiguration("initial_pose_yaw")

    declare_initial_pose_x_cmd = DeclareLaunchArgument(
        "initial_pose_x",
        default_value="0.0",
        description="Initial pose X coordinate for AMCL",
    )
    declare_initial_pose_y_cmd = DeclareLaunchArgument(
        "initial_pose_y",
        default_value="0.0",
        description="Initial pose Y coordinate for AMCL",
    )
    declare_initial_pose_z_cmd = DeclareLaunchArgument(
        "initial_pose_z",
        default_value="0.0",
        description="Initial pose Z coordinate for AMCL",
    )
    declare_initial_pose_yaw_cmd = DeclareLaunchArgument(
        "initial_pose_yaw",
        default_value="0.0",
        description="Initial pose Yaw orientation for AMCL",
    )

    param_substitutions = {
        "amcl.ros__parameters.initial_pose.x": initial_pose_x,
        "amcl.ros__parameters.initial_pose.y": initial_pose_y,
        "amcl.ros__parameters.initial_pose.z": initial_pose_z,
        "amcl.ros__parameters.initial_pose.yaw": initial_pose_yaw,
        "amcl.ros__parameters.set_initial_pose": "true",
    }

    configured_params = RewrittenYaml(
        source_file=params_file,
        param_rewrites=param_substitutions,
        convert_types=True,
    )

    # 1. Robot description and static TF (Goal 2)
    robot_description_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(robot_desc_launch_file),
        launch_arguments={"use_sim_time": use_sim_time}.items(),
    )

    # 2. Upstream Nav2 bringup launch
    nav2_bringup_dir = get_package_share_directory("nav2_bringup")
    nav2_bringup_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_bringup_dir, "launch", "bringup_launch.py")
        ),
        launch_arguments={
            "map": map_yaml_file,
            "use_sim_time": use_sim_time,
            "params_file": configured_params,
            "autostart": autostart,
        }.items(),
    )

    # 3. Optional RViz visualization
    default_rviz_config = os.path.join(
        os.path.dirname(__file__),
        "..",
        "config",
        "nav_arena.rviz",
    )
    declare_rviz_config_cmd = DeclareLaunchArgument(
        "rviz_config",
        default_value=default_rviz_config,
        description="Full path to the RViz configuration file to use",
    )
    rviz_config_file = LaunchConfiguration("rviz_config")

    rviz_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_bringup_dir, "launch", "rviz_launch.py")
        ),
        condition=IfCondition(use_rviz),
        launch_arguments={
            "namespace": "",
            "use_namespace": "False",
            "use_sim_time": use_sim_time,
            "rviz_config": rviz_config_file,
        }.items(),
    )

    ld = LaunchDescription()
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_map_yaml_cmd)
    ld.add_action(declare_params_file_cmd)
    ld.add_action(declare_autostart_cmd)
    ld.add_action(declare_use_rviz_cmd)
    ld.add_action(declare_rviz_config_cmd)
    ld.add_action(declare_initial_pose_x_cmd)
    ld.add_action(declare_initial_pose_y_cmd)
    ld.add_action(declare_initial_pose_z_cmd)
    ld.add_action(declare_initial_pose_yaw_cmd)

    ld.add_action(robot_description_cmd)
    ld.add_action(nav2_bringup_cmd)
    ld.add_action(rviz_cmd)

    return ld
