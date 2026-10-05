# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for named scene routes, including checks against the cached occupancy map."""

from __future__ import annotations

import math

import numpy as np
import pytest

from nav_arena.scenes import DEFAULT_ROUTE, INTERIOR_AGENT_ROUTES, get_route, list_routes
from nav_arena.utils import CACHE_DIR

MAP_DIR = CACHE_DIR / "maps" / "kujiale_0003_open_doors" / "cs0.05_z0.01-0.60"


def test_default_route_exists():
    """Verify the default route is defined for the default scene."""
    assert DEFAULT_ROUTE in list_routes("kujiale_0003")
    assert list_routes("kujiale_0003") == ["around_table", "hall_straight", "through_doorway", "to_far_room"]


def test_route_lookup_errors_are_actionable():
    """Verify unknown scenes and routes report what is available."""
    with pytest.raises(KeyError, match="Available routes"):
        get_route("nope", "kujiale_0003")
    with pytest.raises(KeyError, match="Scenes with routes"):
        get_route("hall_straight", "no_such_scene")
    assert list_routes("no_such_scene") == []


def test_spawn_yaw_faces_the_goal_and_quaternion_is_xyzw():
    """Verify the spawn heading is the bearing to the goal and the quaternion is a unit (x, y, z, w) yaw rotation."""
    route = get_route("hall_straight")
    assert route.spawn_yaw == pytest.approx(0.0)
    assert route.spawn_quat_xyzw == pytest.approx((0.0, 0.0, 0.0, 1.0))
    for route in INTERIOR_AGENT_ROUTES["kujiale_0003"].values():
        x, y, z, w = route.spawn_quat_xyzw
        assert (x, y) == (0.0, 0.0)
        assert math.hypot(z, w) == pytest.approx(1.0)
        assert 2 * math.atan2(z, w) == pytest.approx(route.spawn_yaw, abs=1e-9)


def test_reference_paths_are_at_least_euclidean():
    """Verify recorded free-space path lengths are consistent with the straight-line distance."""
    for route in INTERIOR_AGENT_ROUTES["kujiale_0003"].values():
        assert route.reference_path_m >= route.euclidean_m - 0.3, route.name


@pytest.fixture(scope="module")
def occupancy():
    """Cached occupancy map of the open-door scene (skipped when it has not been generated)."""
    pytest.importorskip("scipy")
    yaml = pytest.importorskip("yaml")
    from PIL import Image
    from scipy import ndimage

    if not (MAP_DIR / "map.yaml").is_file():
        pytest.skip(f"occupancy map not generated: {MAP_DIR}")
    meta = yaml.safe_load((MAP_DIR / "map.yaml").read_text())
    image = np.array(Image.open(MAP_DIR / "map.png"))
    resolution, (ox, oy, _) = meta["resolution"], meta["origin"]
    height, width = image.shape
    clearance = ndimage.distance_transform_edt(image > 250) * resolution

    def at(x, y):
        col, row = int(round((x - ox) / resolution)), int(round(height - 1 - (y - oy) / resolution))
        assert 0 <= row < height and 0 <= col < width, f"({x}, {y}) is outside the map"
        return float(clearance[row, col])

    return at


def test_route_endpoints_have_clearance(occupancy):
    """Verify every route's spawn and goal are at least 0.5 m from any obstacle."""
    for route in INTERIOR_AGENT_ROUTES["kujiale_0003"].values():
        assert occupancy(*route.spawn_xy) >= 0.5, f"{route.name} spawn"
        assert occupancy(*route.goal_xy) >= 0.5, f"{route.name} goal"


def test_hall_straight_segment_is_clear_along_its_whole_length(occupancy):
    """Verify the straight route keeps >= 0.7 m clearance everywhere (an earlier version clipped a wall corner)."""
    route = get_route("hall_straight")
    n = int(route.euclidean_m / 0.05) + 1
    clearances = [
        occupancy(
            route.spawn_xy[0] + t * (route.goal_xy[0] - route.spawn_xy[0]),
            route.spawn_xy[1] + t * (route.goal_xy[1] - route.spawn_xy[1]),
        )
        for t in np.linspace(0.0, 1.0, n)
    ]
    assert min(clearances) >= 0.7
