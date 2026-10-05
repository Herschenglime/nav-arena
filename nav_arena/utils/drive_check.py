# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-axis drive checks: does a body twist command move the robot along that axis, and only that axis?

Pure math (no simulator), so the pass/fail rules are unit-tested; ``scripts/verify_embodiment.py`` runs the robot and
feeds the measured poses in. Each segment holds one command, lets the robot reach speed, then measures over a window
in the body frame the robot had when the window started.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Sequence

AXES = ("vx", "vy", "wz")


def yaw_from_quat_xyzw(quat: Sequence[float]) -> float:
    """Yaw (rad) of a quaternion in Isaac Lab 3.0 order ``(x, y, z, w)``."""
    x, y, z, w = (float(v) for v in quat)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap_angle(angle: float) -> float:
    """Wrap an angle to ``(-pi, pi]``."""
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass(frozen=True)
class DriveLimits:
    """Pass thresholds for one drive segment."""

    min_speed_ratio: float = 0.7
    """Measured speed along the commanded axis must reach this fraction of the command."""
    max_cross_ratio: float = 0.15
    """Translation off the commanded axis, as a fraction of the distance travelled along it."""
    max_yaw_drift: float = 0.1
    """Heading change (rad) allowed while translating."""
    max_center_shift: float = 0.05
    """While rotating in place, how far (m) the turning centre may move between the window's two halves. The root
    itself may trace a circle (its origin need not sit on the turning axis), so its raw translation is not checked."""


@dataclass(frozen=True)
class SegmentResult:
    """Measured motion over a segment's measurement window, in the body frame at the window's start."""

    command: tuple[float, float, float]
    duration_s: float
    dx: float
    dy: float
    dyaw: float
    center_shift: float | None = None
    """For rotation segments: distance (m) between the turning centres of the window's two halves."""

    @property
    def axis(self) -> str:
        """The commanded axis (the single non-zero command component)."""
        nonzero = [name for name, value in zip(AXES, self.command) if value != 0.0]
        if len(nonzero) != 1:
            raise ValueError(f"A drive segment commands exactly one axis, got {self.command}")
        return nonzero[0]

    def measured_rate(self) -> float:
        """Average rate along the commanded axis (m/s or rad/s)."""
        travel = {"vx": self.dx, "vy": self.dy, "wz": self.dyaw}[self.axis]
        return travel / self.duration_s


def turning_center(start_xy: Sequence[float], start_yaw: float, end_xy: Sequence[float], end_yaw: float) -> tuple[float, float] | None:
    """The fixed point of the planar rigid motion between two poses, or None if it barely rotated.

    Solves ``(I - R) c = p1 - R p0`` with ``R`` the rotation by the heading change.
    """
    dtheta = wrap_angle(end_yaw - start_yaw)
    if abs(dtheta) < 1e-3:
        return None
    c, s = math.cos(dtheta), math.sin(dtheta)
    bx = float(end_xy[0]) - (c * float(start_xy[0]) - s * float(start_xy[1]))
    by = float(end_xy[1]) - (s * float(start_xy[0]) + c * float(start_xy[1]))
    # (I - R) = [[1 - c, s], [-s, 1 - c]]; det = (1 - c)^2 + s^2 = 2 (1 - c)
    det = 2.0 * (1.0 - c)
    return ((1.0 - c) * bx - s * by) / det, (s * bx + (1.0 - c) * by) / det


def measure_segment(
    command: Sequence[float],
    start_xy: Sequence[float],
    start_yaw: float,
    end_xy: Sequence[float],
    end_yaw: float,
    duration_s: float,
    mid_xy: Sequence[float] | None = None,
    mid_yaw: float | None = None,
) -> SegmentResult:
    """Express a segment's world-frame motion in the body frame at its start.

    For a rotation command with a mid pose, also measures how far the turning centre moved between the window's halves.
    """
    wx, wy = float(end_xy[0]) - float(start_xy[0]), float(end_xy[1]) - float(start_xy[1])
    c, s = math.cos(start_yaw), math.sin(start_yaw)
    center_shift = None
    if float(command[2]) != 0.0 and mid_xy is not None and mid_yaw is not None:
        first = turning_center(start_xy, start_yaw, mid_xy, mid_yaw)
        second = turning_center(mid_xy, mid_yaw, end_xy, end_yaw)
        center_shift = math.inf if first is None or second is None else math.dist(first, second)
    return SegmentResult(
        command=tuple(float(v) for v in command),
        duration_s=duration_s,
        dx=c * wx + s * wy,
        dy=-s * wx + c * wy,
        dyaw=wrap_angle(end_yaw - start_yaw),
        center_shift=center_shift,
    )


def evaluate_segment(result: SegmentResult, limits: DriveLimits | None = None) -> tuple[bool, str]:
    """Check one segment: enough speed along the commanded axis, little motion on the others.

    Returns:
        ``(passed, detail)`` where ``detail`` lists the measured values for the log.
    """
    limits = limits or DriveLimits()
    axis = result.axis
    commanded = {"vx": result.command[0], "vy": result.command[1], "wz": result.command[2]}[axis]
    ratio = result.measured_rate() / commanded
    problems = []
    if ratio < limits.min_speed_ratio:
        problems.append(f"speed ratio {ratio:.2f} < {limits.min_speed_ratio}")
    if axis == "wz":
        if result.center_shift is None:
            raise ValueError("A rotation segment needs a mid pose to locate the turning centre")
        if result.center_shift > limits.max_center_shift:
            problems.append(f"turning centre moved {result.center_shift:.3f} m (not turning in place)")
    else:
        along, across = (result.dx, result.dy) if axis == "vx" else (result.dy, result.dx)
        if abs(across) > limits.max_cross_ratio * abs(along):
            problems.append(f"cross-axis drift {abs(across):.3f} m vs {abs(along):.3f} m along")
        if abs(result.dyaw) > limits.max_yaw_drift:
            problems.append(f"heading drifted {result.dyaw:.3f} rad")
    detail = (
        f"cmd {axis}={commanded:+.2f}, rate={result.measured_rate():+.3f} (ratio {ratio:.2f}), "
        f"dx={result.dx:+.3f} m, dy={result.dy:+.3f} m, dyaw={result.dyaw:+.3f} rad"
    )
    if result.center_shift is not None:
        detail += f", centre shift={result.center_shift:.3f} m"
    if problems:
        detail += " | " + "; ".join(problems)
    return not problems, detail


def evaluate_hold(result: SegmentResult, max_translation: float = 0.01, max_yaw: float = 0.02) -> tuple[bool, str]:
    """Check a zero-command hold: the robot must stay put (catches roller jitter and creeping wheels)."""
    translation = math.hypot(result.dx, result.dy)
    passed = translation <= max_translation and abs(result.dyaw) <= max_yaw
    detail = f"drift={translation:.4f} m (max {max_translation}), dyaw={result.dyaw:+.4f} rad (max {max_yaw}) over {result.duration_s:.1f} s"
    return passed, detail


@dataclass(frozen=True)
class SettleCfg:
    """When a robot counts as at rest: its pose changes by less than the limits over consecutive windows."""

    window_s: float = 0.5
    still_windows: int = 2
    max_shift: float = 0.002
    """Metres the root may move within one window."""
    max_turn: float = 0.005
    """Radians the heading may change within one window."""
    max_s: float = 8.0
    """Give up after this long."""


def settle_until_still(step: Callable[[], None], pose: Callable[[], tuple[float, float, float]], step_dt: float, cfg: SettleCfg | None = None) -> float | None:
    """Hold the robot (``step`` applies a zero command) until its pose stops changing.

    Stillness is judged on the pose over whole windows rather than the instantaneous velocity, which chatters for robots
    with many contacts (Kaya's rollers) even when they are not going anywhere.

    Args:
        step: Advances the simulation by one ``step_dt`` with a zero command.
        pose: Returns the robot's ``(x, y, yaw)``.
        step_dt: Seconds per ``step`` call.
        cfg: Stillness limits.

    Returns:
        Seconds until the robot was still, or None if it was still moving after ``cfg.max_s``.
    """
    cfg = cfg or SettleCfg()
    steps_per_window = max(1, round(cfg.window_s / step_dt))
    quiet, elapsed = 0, 0.0
    last = pose()
    while elapsed < cfg.max_s:
        for _ in range(steps_per_window):
            step()
        elapsed += steps_per_window * step_dt
        current = pose()
        moved = math.dist(current[:2], last[:2]) > cfg.max_shift or abs(wrap_angle(current[2] - last[2])) > cfg.max_turn
        quiet = 0 if moved else quiet + 1
        last = current
        if quiet >= cfg.still_windows:
            return elapsed
    return None
