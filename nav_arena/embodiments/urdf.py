# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""In-memory parameterized URDF generator for mobile embodiments.

Generates self-contained, valid URDF descriptions with geometric primitives
(without external CAD mesh dependencies) for robot_state_publisher and Nav2.
"""

from __future__ import annotations

from typing import Any
import xml.etree.ElementTree as ET


def generate_minimal_urdf(
    cfg: Any | None = None,
    name: str = "mobile_robot",
    base_frame: str = "base_link",
    chassis_frame: str = "chassis_link",
    lidar_frame: str = "lidar_link",
    lidar_offset: tuple[float, float, float] = (0.0, 0.0, 0.35),
    robot_radius: float = 0.28,
    robot_height: float = 0.40,
) -> str:
    """Generate a clean, minimal URDF XML string from an embodiment configuration.

    Args:
        cfg: Optional embodiment configuration dataclass (e.g. NovaCarterEmbodimentCfg).
             If provided, attributes are extracted from it; otherwise kwargs are used.
        name: Name of the robot.
        base_frame: Root link frame ID (typically 'base_link' or 'base_footprint').
        chassis_frame: Main body chassis frame ID.
        lidar_frame: 2D LiDAR scanner frame ID.
        lidar_offset: (x, y, z) translation offset of LiDAR relative to chassis_frame.
        robot_radius: Radius in meters for the cylindrical visual/collision body.
        robot_height: Height in meters for the cylindrical visual/collision body.

    Returns:
        A formatted, valid URDF XML string ready for robot_state_publisher.
    """
    if cfg is not None:
        name = getattr(cfg, "name", name)
        base_frame = getattr(cfg, "base_frame", base_frame)
        chassis_frame = getattr(cfg, "chassis_frame", chassis_frame)
        lidar_frame = getattr(cfg, "lidar_frame", lidar_frame)
        if hasattr(cfg, "lidar_offset"):
            lidar_offset = cfg.lidar_offset
        elif hasattr(cfg, "sensor_height"):
            lidar_offset = (0.0, 0.0, float(cfg.sensor_height))
        robot_radius = getattr(cfg, "robot_radius", robot_radius)
        robot_height = getattr(cfg, "robot_height", robot_height)

    robot = ET.Element("robot", name=str(name))

    # 1. Base footprint / root link
    base_link = ET.SubElement(robot, "link", name=str(base_frame))

    # 2. Chassis link with geometric cylinder representation
    chassis_link = ET.SubElement(robot, "link", name=str(chassis_frame))

    # Visual
    visual = ET.SubElement(chassis_link, "visual")
    ET.SubElement(visual, "origin", xyz=f"0 0 {robot_height / 2.0:.4f}", rpy="0 0 0")
    vis_geom = ET.SubElement(visual, "geometry")
    ET.SubElement(vis_geom, "cylinder", radius=f"{robot_radius:.4f}", length=f"{robot_height:.4f}")
    mat = ET.SubElement(visual, "material", name="chassis_mat")
    ET.SubElement(mat, "color", rgba="0.2 0.5 0.8 0.8")

    # Collision
    collision = ET.SubElement(chassis_link, "collision")
    ET.SubElement(collision, "origin", xyz=f"0 0 {robot_height / 2.0:.4f}", rpy="0 0 0")
    coll_geom = ET.SubElement(collision, "geometry")
    ET.SubElement(coll_geom, "cylinder", radius=f"{robot_radius:.4f}", length=f"{robot_height:.4f}")

    # Joint: base_frame -> chassis_frame
    base_to_chassis = ET.SubElement(robot, "joint", name=f"{base_frame}_to_{chassis_frame}", type="fixed")
    ET.SubElement(base_to_chassis, "parent", link=str(base_frame))
    ET.SubElement(base_to_chassis, "child", link=str(chassis_frame))
    ET.SubElement(base_to_chassis, "origin", xyz="0 0 0", rpy="0 0 0")

    # 3. LiDAR link
    lidar_link = ET.SubElement(robot, "link", name=str(lidar_frame))
    lidar_vis = ET.SubElement(lidar_link, "visual")
    ET.SubElement(lidar_vis, "origin", xyz="0 0 0", rpy="0 0 0")
    lidar_geom = ET.SubElement(lidar_vis, "geometry")
    ET.SubElement(lidar_geom, "cylinder", radius="0.0500", length="0.0600")
    lidar_mat = ET.SubElement(lidar_vis, "material", name="lidar_mat")
    ET.SubElement(lidar_mat, "color", rgba="0.1 0.1 0.1 0.9")

    # Joint: chassis_frame -> lidar_frame
    chassis_to_lidar = ET.SubElement(robot, "joint", name=f"{chassis_frame}_to_{lidar_frame}", type="fixed")
    ET.SubElement(chassis_to_lidar, "parent", link=str(chassis_frame))
    ET.SubElement(chassis_to_lidar, "child", link=str(lidar_frame))
    ET.SubElement(
        chassis_to_lidar,
        "origin",
        xyz=f"{lidar_offset[0]:.4f} {lidar_offset[1]:.4f} {lidar_offset[2]:.4f}",
        rpy="0 0 0",
    )

    ET.indent(robot, space="  ")
    return ET.tostring(robot, encoding="unicode", xml_declaration=True)
