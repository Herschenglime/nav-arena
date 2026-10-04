# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Characterization unit tests for minimal URDF generator."""

from dataclasses import dataclass
import xml.etree.ElementTree as ET

import pytest

from nav_arena.embodiments.urdf import generate_minimal_urdf


def test_default_urdf_generation():
    """Verify default generation produces a valid XML document with cylinder geometry."""
    xml_str = generate_minimal_urdf()
    root = ET.fromstring(xml_str)

    assert root.tag == "robot"
    assert root.attrib.get("name") == "mobile_robot"

    # Link names
    links = [child.attrib.get("name") for child in root.findall("link")]
    assert "base_link" in links
    assert "chassis_link" in links
    assert "lidar_link" in links

    # Joint names
    joints = {j.attrib.get("name"): j for j in root.findall("joint")}
    assert "base_link_to_chassis_link" in joints
    assert "chassis_link_to_lidar_link" in joints

    # Verify joints connect correct parent/child
    b2c = joints["base_link_to_chassis_link"]
    assert b2c.attrib.get("type") == "fixed"
    assert b2c.find("parent").attrib.get("link") == "base_link"
    assert b2c.find("child").attrib.get("link") == "chassis_link"

    c2l = joints["chassis_link_to_lidar_link"]
    assert c2l.attrib.get("type") == "fixed"
    assert c2l.find("parent").attrib.get("link") == "chassis_link"
    assert c2l.find("child").attrib.get("link") == "lidar_link"
    assert c2l.find("origin").attrib.get("xyz") == "0.0000 0.0000 0.3500"

    # Verify default chassis geometry is cylinder
    chassis = root.find("./link[@name='chassis_link']")
    visual_geom = chassis.find("visual/geometry")
    assert visual_geom.find("cylinder") is not None
    cylinder = visual_geom.find("cylinder")
    assert cylinder.attrib.get("radius") == "0.2800"
    assert cylinder.attrib.get("length") == "0.4000"

    collision_geom = chassis.find("collision/geometry")
    assert collision_geom.find("cylinder") is not None


def test_box_chassis_geometry():
    """Verify box geometry visual and collision tags when chassis_size is provided."""
    xml_str = generate_minimal_urdf(
        chassis_size=(0.7, 0.5, 0.3),
        chassis_offset=(0.1, 0.0, 0.15),
    )
    root = ET.fromstring(xml_str)
    chassis = root.find("./link[@name='chassis_link']")

    # Visual box
    visual = chassis.find("visual")
    box_vis = visual.find("geometry/box")
    assert box_vis is not None
    assert box_vis.attrib.get("size") == "0.7000 0.5000 0.3000"
    assert visual.find("origin").attrib.get("xyz") == "0.1000 0.0000 0.1500"

    # Collision box
    collision = chassis.find("collision")
    box_coll = collision.find("geometry/box")
    assert box_coll is not None
    assert box_coll.attrib.get("size") == "0.7000 0.5000 0.3000"
    assert collision.find("origin").attrib.get("xyz") == "0.1000 0.0000 0.1500"


def test_box_chassis_default_offset():
    """Verify default offset calculation when chassis_offset is None."""
    xml_str = generate_minimal_urdf(chassis_size=(0.6, 0.4, 0.2))
    root = ET.fromstring(xml_str)
    chassis = root.find("./link[@name='chassis_link']")

    visual = chassis.find("visual")
    # Expected default ox, oy, oz = (-0.2335, 0.0, 0.2 / 2.0 = 0.1)
    assert visual.find("origin").attrib.get("xyz") == "-0.2335 0.0000 0.1000"


def test_custom_frame_names_and_sensor_offset():
    """Verify custom link and frame names, and LiDAR offset."""
    xml_str = generate_minimal_urdf(
        name="scout_v2",
        base_frame="base_footprint",
        chassis_frame="scout_body",
        lidar_frame="velodyne",
        lidar_offset=(0.25, 0.12, 0.55),
    )
    root = ET.fromstring(xml_str)
    assert root.attrib.get("name") == "scout_v2"

    links = [child.attrib.get("name") for child in root.findall("link")]
    assert "base_footprint" in links
    assert "scout_body" in links
    assert "velodyne" in links

    joints = {j.attrib.get("name"): j for j in root.findall("joint")}
    assert "base_footprint_to_scout_body" in joints
    assert "scout_body_to_velodyne" in joints

    velodyne_joint = joints["scout_body_to_velodyne"]
    assert velodyne_joint.find("origin").attrib.get("xyz") == "0.2500 0.1200 0.5500"


def test_config_object_passing():
    """Verify attributes extracted correctly from configuration objects."""

    @dataclass
    class DummyCfg:
        name: str = "nova_carter"
        base_frame: str = "base_link"
        chassis_frame: str = "chassis_link"
        lidar_frame: str = "lidar_link"
        lidar_offset: tuple[float, float, float] = (0.05, 0.0, 0.38)
        chassis_size: tuple[float, float, float] = (0.65, 0.45, 0.25)
        chassis_offset: tuple[float, float, float] = (0.0, 0.0, 0.125)

    xml_str = generate_minimal_urdf(cfg=DummyCfg())
    root = ET.fromstring(xml_str)
    assert root.attrib.get("name") == "nova_carter"

    lidar_joint = root.find("./joint[@name='chassis_link_to_lidar_link']")
    assert lidar_joint.find("origin").attrib.get("xyz") == "0.0500 0.0000 0.3800"

    chassis = root.find("./link[@name='chassis_link']")
    box_geom = chassis.find("visual/geometry/box")
    assert box_geom.attrib.get("size") == "0.6500 0.4500 0.2500"


def test_config_sensor_height_fallback():
    """Verify fallback to sensor_height attribute when lidar_offset is absent."""

    @dataclass
    class HeightOnlyCfg:
        name: str = "simple_bot"
        sensor_height: float = 0.42

    xml_str = generate_minimal_urdf(cfg=HeightOnlyCfg())
    root = ET.fromstring(xml_str)
    lidar_joint = root.find("./joint[@name='chassis_link_to_lidar_link']")
    assert lidar_joint.find("origin").attrib.get("xyz") == "0.0000 0.0000 0.4200"
