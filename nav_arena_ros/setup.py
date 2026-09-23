import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'nav_arena_ros'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robopi',
    maintainer_email='robopi@todo.todo',
    description='ROS 2 integration and lifecycle runner for nav_arena',
    license='BSD-3-Clause',
    entry_points={
        'console_scripts': [
            'ros2_policy_runner = nav_arena_ros.ros2_policy_runner:main',
            'dummy_ros2_policy = nav_arena_ros.dummy_ros2_policy:main',
        ],
    },
)
