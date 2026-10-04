# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the viewport goal/plan overlay (fake backend; CPU-only)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from nav_arena.utils.viewer import DebugOverlay


class FakeBackend:
    """Records what the overlay asks the viewport layer to draw."""

    def __init__(self, fail_on_draw=False):
        self.lines = None
        self.points = None
        self.draws = 0
        self.clears = 0
        self.fail_on_draw = fail_on_draw

    def draw(self, lines, points):
        if self.fail_on_draw:
            raise RuntimeError("scene view destroyed")
        self.draws += 1
        self.lines, self.points = list(lines), list(points)

    def clear(self):
        self.clears += 1
        self.lines = self.points = None


def _overlay(**kw):
    backend = kw.pop("backend", None) or FakeBackend()
    return DebugOverlay((2.0, 1.0), tolerance=0.4, backend=backend, **kw), backend


def test_goal_pin_and_tolerance_ring_are_drawn_with_an_empty_plan():
    """Verify the goal pin, a closed tolerance ring, and a goal point are drawn when there is no plan yet."""
    overlay, backend = _overlay(ring_segments=16)
    overlay.update(np.empty((0, 2)), False, z=0.12)
    assert len(backend.lines) == 1 + 16  # pin + ring segments
    (pin_start, pin_end, pin_color, _), ring = backend.lines[0], backend.lines[1:]
    assert pin_start == (2.0, 1.0, 0.12) and pin_end == (2.0, 1.0, 1.12) and pin_color == DebugOverlay.GOAL_COLOR
    assert all(math.dist(start[:2], (2.0, 1.0)) == pytest.approx(0.4) for start, _, _, _ in ring)
    assert ring[-1][1][:2] == pytest.approx(ring[0][0][:2])  # the ring closes
    assert all(color == DebugOverlay.GOAL_COLOR for _, _, color, _ in backend.lines)
    assert backend.points == [((2.0, 1.0, 1.12), DebugOverlay.GOAL_COLOR, 14.0)]


def test_plan_is_drawn_as_a_polyline_in_cyan_or_red_when_stopped():
    """Verify path segments connect consecutive waypoints, and the colour reflects a stop request."""
    path = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.5]])
    overlay, backend = _overlay(ring_segments=8)
    overlay.update(path, False, z=0.1)
    plan = backend.lines[9:]
    assert [(s[:2], e[:2]) for s, e, _, _ in plan] == [((0.0, 0.0), (0.5, 0.0)), ((0.5, 0.0), (1.0, 0.5))]
    assert all(color == DebugOverlay.PATH_COLOR for _, _, color, _ in plan)
    assert backend.points[-1][0] == (1.0, 0.5, 0.1)  # end-of-plan point

    overlay.update(path, True, z=0.1)
    assert all(color == DebugOverlay.STOP_COLOR for _, _, color, _ in backend.lines[9:])
    assert backend.points[-1][1] == DebugOverlay.STOP_COLOR


def test_each_update_replaces_the_previous_drawing():
    """Verify an update hands the backend the full new scene (the backend replaces, never accumulates)."""
    overlay, backend = _overlay()
    overlay.update(np.array([[0.0, 0.0], [1.0, 0.0]]), False, 0.1)
    first = len(backend.lines)
    overlay.update(np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]), False, 0.1)
    assert len(backend.lines) == first + 2 and backend.draws == 2


def test_accepts_plans_with_extra_columns_and_clears():
    """Verify [T, 3] plans are drawn from x and y only, and clear() removes the drawing."""
    overlay, backend = _overlay(ring_segments=4)
    overlay.update(np.array([[0.0, 0.0, 9.0], [1.0, 0.0, 9.0]]), False, 0.1)
    assert len(backend.lines) == 1 + 4 + 1
    overlay.clear()
    assert backend.lines is None and backend.clears == 1


def test_overlay_fails_soft_when_drawing_breaks():
    """Verify a backend failure disables the overlay instead of raising into the episode loop."""
    overlay, _ = _overlay(backend=FakeBackend(fail_on_draw=True))
    overlay.update(np.empty((0, 2)), False, 0.1)  # must not raise
    assert overlay.enabled is False
    overlay.update(np.empty((0, 2)), False, 0.1)
    overlay.clear()  # also a no-op once disabled


def test_overlay_without_a_gui_disables_itself():
    """Verify constructing the default backend with no viewport (headless/CPU) degrades to a disabled overlay."""
    overlay = DebugOverlay((0.0, 0.0), tolerance=0.4)
    assert overlay.enabled is False
    overlay.update(np.empty((0, 2)), False, 0.1)
    overlay.clear()
