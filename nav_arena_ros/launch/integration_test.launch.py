# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch file for orchestrating PointNavTask lifecycle runner and ROS 2 test policy."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import LifecycleNode, Node
from launch_ros.events.lifecycle import ChangeState
from launch_ros.event_handlers import OnStateTransition
import lifecycle_msgs.msg


def generate_launch_description():
    viz_arg = DeclareLaunchArgument(
        'viz',
        default_value='kit',
        description='Visualization mode for Isaac Lab (kit or none)'
    )

    # Isaac Lab simulation runner managed as a LifecycleNode
    runner_node = LifecycleNode(
        package='nav_arena_ros',
        executable='ros2_policy_runner',
        name='ros2_policy_runner',
        namespace='',
        output='screen',
        arguments=['--viz', LaunchConfiguration('viz')],
    )

    # Dummy ROS 2 Policy Node (launches only when simulation is ACTIVE)
    dummy_policy_node = Node(
        package='nav_arena_ros',
        executable='dummy_ros2_policy',
        name='dummy_policy_node',
        output='screen',
    )

    # RViz2 visualization instance (only in kit GUI mode when simulation is ACTIVE)
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        condition=IfCondition(
            PythonExpression(["'", LaunchConfiguration('viz'), "' == 'kit'"])
        ),
    )

    # 1. Trigger CONFIGURE as soon as the runner process starts
    configure_event = RegisterEventHandler(
        OnProcessStart(
            target_action=runner_node,
            on_start=[
                EmitEvent(
                    event=ChangeState(
                        lifecycle_node_matcher=lambda node: True,
                        transition_id=lifecycle_msgs.msg.Transition.TRANSITION_CONFIGURE,
                    )
                ),
            ],
        )
    )

    # 2. Trigger ACTIVATE once runner reaches INACTIVE (configuration finished)
    activate_event = RegisterEventHandler(
        OnStateTransition(
            target_lifecycle_node=runner_node,
            goal_state='inactive',
            entities=[
                EmitEvent(
                    event=ChangeState(
                        lifecycle_node_matcher=lambda node: True,
                        transition_id=lifecycle_msgs.msg.Transition.TRANSITION_ACTIVATE,
                    )
                ),
            ],
        )
    )

    # 3. Launch dependent nodes (policy and RViz2) only after runner reaches ACTIVE
    on_active_event = RegisterEventHandler(
        OnStateTransition(
            target_lifecycle_node=runner_node,
            goal_state='active',
            entities=[
                dummy_policy_node,
                rviz_node,
            ],
        )
    )

    return LaunchDescription([
        viz_arg,
        runner_node,
        configure_event,
        activate_event,
        on_active_event,
    ])
