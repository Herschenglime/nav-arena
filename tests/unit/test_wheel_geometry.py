# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for reading omni/mecanum wheel layouts from USD, and for HolonomicDriveAction's command handling."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

pxr = pytest.importorskip("pxr")
from pxr import Gf, Sdf, Usd, UsdPhysics  # noqa: E402

from nav_arena.embodiments.actions import HolonomicDriveAction  # noqa: E402
from nav_arena.embodiments.kinematics import holonomic_ik  # noqa: E402
from nav_arena.embodiments.wheel_geometry import WheelGeometry, read_wheel_geometry  # noqa: E402

# Kaya's wheels: (name, localPos0 on base_link, axle yaw about +z in degrees)
KAYA_WHEELS = [
    ("axle_0_joint", (-0.09804319, 0.00063677, -0.0505), 180.0),
    ("axle_1_joint", (0.04934748, -0.08452497, -0.0505), -60.0),
    ("axle_2_joint", (0.04952906, 0.08569367, -0.0505), 60.0),
]


def _kaya_like_stage(angle=90.0, radius=0.04, with_attrs=True):
    """An in-memory stage laid out like the Kaya USD: base_link with three revolute joints whose axis is X."""
    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/kaya", "Xform")
    base = stage.DefinePrim("/kaya/base_link", "Xform")
    UsdPhysics.RigidBodyAPI.Apply(base)
    for index, (name, pos, yaw) in enumerate(KAYA_WHEELS):
        axle = stage.DefinePrim(f"/kaya/axle_{index}", "Xform")
        UsdPhysics.RigidBodyAPI.Apply(axle)
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"/kaya/base_link/{name}")
        joint.GetBody0Rel().SetTargets(["/kaya/base_link"])
        joint.GetBody1Rel().SetTargets([f"/kaya/axle_{index}"])
        joint.GetAxisAttr().Set("X")
        joint.GetLocalPos0Attr().Set(Gf.Vec3f(*pos))
        half = math.radians(yaw) / 2.0
        joint.GetLocalRot0Attr().Set(Gf.Quatf(math.cos(half), Gf.Vec3f(0.0, 0.0, math.sin(half))))
        if with_attrs:
            prim = joint.GetPrim()
            prim.CreateAttribute("isaacmecanumwheel:radius", Sdf.ValueTypeNames.Float).Set(radius)
            prim.CreateAttribute("isaacmecanumwheel:angle", Sdf.ValueTypeNames.Float).Set(angle)
    return stage


NAMES = [name for name, _, _ in KAYA_WHEELS]


def test_kaya_like_layout_is_read_from_joint_frames_and_attributes():
    """Verify positions, axle directions, radii and roller angles come out of the stage in joint order."""
    geometry = read_wheel_geometry(_kaya_like_stage(), "/kaya", NAMES)
    assert geometry.radii == pytest.approx((0.04,) * 3) and geometry.angles_deg == pytest.approx((90.0,) * 3)
    assert geometry.positions[1] == pytest.approx((0.04934748, -0.08452497), abs=1e-6)
    assert geometry.axes[0] == pytest.approx((-1.0, 0.0), abs=1e-6)
    assert geometry.axes[1] == pytest.approx((0.5, -0.8660254), abs=1e-6)
    assert geometry.axes[2] == pytest.approx((0.5, 0.8660254), abs=1e-6)


def test_order_follows_the_requested_joint_names():
    """Verify reversing the joint names reverses the wheels, so wheel velocities line up with the action's joint ids."""
    forward = read_wheel_geometry(_kaya_like_stage(), "/kaya", NAMES)
    backward = read_wheel_geometry(_kaya_like_stage(), "/kaya", NAMES[::-1])
    assert backward.positions == tuple(reversed(forward.positions))


def test_geometry_from_the_stage_gives_the_expected_kaya_wheel_speeds():
    """Verify the full chain (USD -> geometry -> matrix) reproduces the hand-derived forward-drive wheel speeds."""
    matrix = read_wheel_geometry(_kaya_like_stage(), "/kaya", NAMES).matrix()
    wheels = holonomic_ik(torch.tensor([[1.0, 0.0, 0.0]]), matrix)[0]
    assert wheels.tolist() == pytest.approx([0.0, -0.8660254 / 0.04, 0.8660254 / 0.04], abs=1e-3)


@pytest.mark.parametrize(
    "build, message",
    [
        (lambda: (_kaya_like_stage(), "/missing", NAMES), "does not exist"),
        (lambda: (_kaya_like_stage(), "/kaya", NAMES + ["axle_9_joint"]), "found 0"),
        (lambda: (_kaya_like_stage(with_attrs=False), "/kaya", NAMES), "isaacmecanumwheel"),
    ],
)
def test_bad_stage_is_rejected_with_a_clear_message(build, message):
    """Verify a missing robot, missing joint or missing wheel attributes fail loudly."""
    stage, root, names = build()
    with pytest.raises(ValueError, match=message):
        read_wheel_geometry(stage, root, names)


def test_wheels_on_different_bodies_are_rejected():
    """Verify joints attached to different chassis bodies cannot be mixed into one command frame."""
    stage = _kaya_like_stage()
    stage.DefinePrim("/kaya/other_link", "Xform")
    UsdPhysics.RevoluteJoint(stage.GetPrimAtPath("/kaya/base_link/axle_2_joint")).GetBody0Rel().SetTargets(["/kaya/other_link"])
    with pytest.raises(ValueError, match="different bodies"):
        read_wheel_geometry(stage, "/kaya", NAMES)


def test_vertical_axle_is_rejected():
    """Verify a wheel whose axle points up is refused (it cannot drive a planar robot)."""
    stage = _kaya_like_stage()
    UsdPhysics.RevoluteJoint(stage.GetPrimAtPath("/kaya/base_link/axle_0_joint")).GetAxisAttr().Set("Z")
    with pytest.raises(ValueError, match="vertical axle"):
        read_wheel_geometry(stage, "/kaya", NAMES)


# --- HolonomicDriveAction command handling (constructed without a simulator) ---------------------------------------


class _RecordingAsset:
    def __init__(self):
        self.calls = []

    def set_joint_velocity_target_index(self, *, target, joint_ids):
        self.calls.append((target.clone(), list(joint_ids)))


def _action_term(num_envs=2, **cfg_overrides):
    """A HolonomicDriveAction with its simulator-dependent construction skipped and a recording asset."""
    geometry = WheelGeometry(
        positions=((-0.098, 0.0), (0.049, -0.085), (0.049, 0.086)),
        axes=((-1.0, 0.0), (0.5, -0.866), (0.5, 0.866)),
        radii=(0.04,) * 3,
        angles_deg=(90.0,) * 3,
    )
    cfg = SimpleNamespace(max_linear_speed=0.5, max_lateral_speed=0.25, max_angular_speed=2.0, scale=(1.0, 1.0, 1.0), offset=(0.0, 0.0, 0.0))
    for key, value in cfg_overrides.items():
        setattr(cfg, key, value)
    term = object.__new__(HolonomicDriveAction)
    term.cfg = cfg
    term._asset = _RecordingAsset()
    term._wheel_joint_ids = [4, 5, 6]
    term._matrix = geometry.matrix()
    term._raw_actions = torch.zeros(num_envs, 3)
    term._processed_actions = torch.zeros(num_envs, 3)
    term._wheel_vel_targets = torch.zeros(num_envs, 3)
    term._scale = torch.tensor(cfg.scale).unsqueeze(0)
    term._offset = torch.tensor(cfg.offset).unsqueeze(0)
    term._limits = torch.tensor([cfg.max_linear_speed, cfg.max_lateral_speed, cfg.max_angular_speed]).unsqueeze(0)
    return term


def test_command_is_clipped_per_axis_before_it_becomes_wheel_speeds():
    """Verify vx, vy and wz are each clamped to their own limit."""
    term = _action_term()
    term.process_actions(torch.tensor([[2.0, -2.0, 9.0], [0.1, 0.1, -9.0]]))
    assert torch.allclose(term.processed_actions, torch.tensor([[0.5, -0.25, 2.0], [0.1, 0.1, -2.0]]))


def test_apply_writes_wheel_velocity_targets_for_the_driven_joints_only():
    """Verify the targets are the IK of the processed command and go to the configured wheel joint ids."""
    term = _action_term()
    term.process_actions(torch.tensor([[0.4, 0.0, 0.0], [0.0, 0.2, 0.0]]))
    term.apply_actions()
    (targets, joint_ids), = term._asset.calls
    assert joint_ids == [4, 5, 6]
    assert targets[0].tolist() == pytest.approx([0.0, -0.4 * 0.866 / 0.04, 0.4 * 0.866 / 0.04], abs=1e-3)
    assert targets[1, 0].item() == pytest.approx(0.2 / 0.04, abs=1e-3)


def test_scale_and_offset_apply_before_clipping_and_reset_zeroes_the_selected_envs():
    """Verify scale/offset are applied first, and reset clears only the requested environments."""
    term = _action_term(scale=(2.0, 2.0, 2.0), offset=(0.1, 0.0, 0.0))
    term.process_actions(torch.tensor([[0.1, 0.05, 0.5], [0.1, 0.05, 0.5]]))
    assert term.processed_actions[0].tolist() == pytest.approx([0.3, 0.1, 1.0])
    term.reset([1])
    assert term.processed_actions[1].abs().sum().item() == 0.0 and term.processed_actions[0].abs().sum().item() > 0.0
    term.reset()
    assert term.processed_actions.abs().sum().item() == 0.0
