# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wheel layout of an omni or mecanum robot, read from the attributes authored in its USD.

Isaac Sim robot assets author ``isaacmecanumwheel:radius`` and ``isaacmecanumwheel:angle`` on each wheel joint, and the
joint frame gives the wheel position and axle direction. Reading them from the stage means the numbers are never copied
by hand, so they cannot drift from the asset. Needs ``pxr`` only (no simulator).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

import torch

from nav_arena.embodiments.kinematics import holonomic_matrix

if TYPE_CHECKING:
    from pxr import Usd

RADIUS_ATTR = "isaacmecanumwheel:radius"
ANGLE_ATTR = "isaacmecanumwheel:angle"


@dataclass(frozen=True)
class WheelGeometry:
    """Wheel layout in the frame of the body the wheel joints attach to (the command frame)."""

    positions: tuple[tuple[float, float], ...]
    """Wheel centres (x, y) in meters."""
    axes: tuple[tuple[float, float], ...]
    """Horizontal part of each wheel's axle direction."""
    radii: tuple[float, ...]
    """Wheel radii in meters."""
    angles_deg: tuple[float, ...]
    """Roller angles: 90 for an omni wheel, 45 or 135 for mecanum."""

    def matrix(self) -> torch.Tensor:
        """The ``[N, 3]`` twist-to-wheel-velocity matrix for this layout (see :func:`holonomic_matrix`)."""
        return holonomic_matrix(self.positions, self.axes, self.radii, self.angles_deg)


def read_wheel_geometry(stage: Usd.Stage, robot_root_path: str, joint_names: Sequence[str]) -> WheelGeometry:
    """Read the wheel layout of the named revolute joints under a robot prim.

    Each joint is described by its frame on ``body0`` (position ``localPos0``; axle direction ``localRot0`` applied to
    the joint axis), so the layout is expressed in the frame of ``body0``. All wheels must attach to the same body, which
    is the frame the body twist command is given in (the robot's chassis).

    Args:
        stage: The USD stage holding the robot.
        robot_root_path: Prim path of the robot (one environment's copy).
        joint_names: Prim names of the wheel joints, in the order the wheel velocities will be produced.

    Returns:
        The wheel geometry, ordered like ``joint_names``.

    Raises:
        ValueError: If a joint is missing or ambiguous, the wheels attach to different bodies, a wheel axle is vertical,
            or a joint lacks the ``isaacmecanumwheel`` attributes.
    """
    from pxr import Gf, Usd, UsdPhysics

    root = stage.GetPrimAtPath(robot_root_path)
    if not root.IsValid():
        raise ValueError(f"Robot prim '{robot_root_path}' does not exist on the stage")
    found: dict[str, list] = {}
    for prim in Usd.PrimRange(root):
        if prim.GetName() in joint_names and prim.IsA(UsdPhysics.RevoluteJoint):
            found.setdefault(prim.GetName(), []).append(prim)

    positions, axes, radii, angles = [], [], [], []
    body0_paths = set()
    axis_vectors = {"X": Gf.Vec3d(1, 0, 0), "Y": Gf.Vec3d(0, 1, 0), "Z": Gf.Vec3d(0, 0, 1)}
    for name in joint_names:
        prims = found.get(name, [])
        if len(prims) != 1:
            raise ValueError(f"Expected exactly one revolute joint named '{name}' under '{robot_root_path}', found {len(prims)}")
        prim = prims[0]
        joint = UsdPhysics.RevoluteJoint(prim)
        body0_paths.add(tuple(str(t) for t in joint.GetBody0Rel().GetTargets()))
        local_pos = joint.GetLocalPos0Attr().Get()
        axle = Gf.Rotation(Gf.Quatd(joint.GetLocalRot0Attr().Get())).TransformDir(axis_vectors[joint.GetAxisAttr().Get()])
        if math.hypot(axle[0], axle[1]) < 1e-6:
            raise ValueError(f"Wheel joint '{prim.GetPath()}' has a vertical axle; omni/mecanum wheels need a horizontal one")
        radius_attr, angle_attr = prim.GetAttribute(RADIUS_ATTR), prim.GetAttribute(ANGLE_ATTR)
        if not (radius_attr.HasAuthoredValue() and angle_attr.HasAuthoredValue()):
            raise ValueError(
                f"Wheel joint '{prim.GetPath()}' has no '{RADIUS_ATTR}'/'{ANGLE_ATTR}' attributes; "
                "give the wheel geometry explicitly in the action config"
            )
        positions.append((float(local_pos[0]), float(local_pos[1])))
        axes.append((float(axle[0]), float(axle[1])))
        radii.append(float(radius_attr.Get()))
        angles.append(float(angle_attr.Get()))
    if len(body0_paths) != 1:
        raise ValueError(f"Wheel joints attach to different bodies ({sorted(body0_paths)}); expected one chassis body")
    return WheelGeometry(tuple(positions), tuple(axes), tuple(radii), tuple(angles))
