# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for script argument validation (CPU-only)."""

from __future__ import annotations

import pytest

from nav_arena.utils.cli_args import validate_route_args


@pytest.mark.parametrize(
    ("route", "spawn", "goal"),
    [(None, None, None), ("hall_straight", None, None), (None, [0.0, 1.0], [2.0, 3.0])],
)
def test_valid_route_selections(route, spawn, goal):
    """Verify the default route, a named route, and a full custom pair are accepted."""
    validate_route_args(route, spawn, goal)


@pytest.mark.parametrize(
    ("spawn", "goal", "given", "missing"),
    [([0.0, 1.0], None, "--spawn", "--goal"), (None, [2.0, 3.0], "--goal", "--spawn")],
)
def test_one_sided_custom_pose_is_an_error(spawn, goal, given, missing):
    """Verify --spawn without --goal (or the reverse) is rejected instead of silently falling back to a route."""
    with pytest.raises(ValueError, match=f"{given} needs {missing}"):
        validate_route_args(None, spawn, goal)


def test_route_with_custom_pose_is_an_error():
    """Verify a named route cannot be combined with a custom start/goal."""
    with pytest.raises(ValueError, match="cannot be combined"):
        validate_route_args("around_table", [0.0, 1.0], [2.0, 3.0])
