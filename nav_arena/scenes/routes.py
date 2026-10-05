# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Named start/goal routes for InteriorAgent scenes, for repeatable navigation evaluation.

Routes are expressed in the world frame (identical to the occupancy-map frame) and were chosen from the scene's
occupancy map: start and goal have at least 0.5 m of obstacle clearance, and the planned free-space path length gives a
reference for how much detouring each route requires. They are meant for the ``kujiale_0003`` scene with the open-door
conditioning layer (the default). Verify them visually before relying on them for a new robot.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Route:
    """A named start-to-goal navigation route."""

    name: str
    spawn_xy: tuple[float, float]
    goal_xy: tuple[float, float]
    description: str
    reference_path_m: float
    """Length of the shortest obstacle-inflated free-space path (A* on the occupancy map), in meters."""

    @property
    def euclidean_m(self) -> float:
        """Straight-line start-to-goal distance in meters."""
        return math.dist(self.spawn_xy, self.goal_xy)

    @property
    def spawn_yaw(self) -> float:
        """Spawn heading in radians: facing the goal, so every method starts with the same view geometry."""
        return math.atan2(self.goal_xy[1] - self.spawn_xy[1], self.goal_xy[0] - self.spawn_xy[0])

    @property
    def spawn_quat_xyzw(self) -> tuple[float, float, float, float]:
        """Spawn orientation as an Isaac Lab ``(x, y, z, w)`` quaternion (yaw about +Z)."""
        half = self.spawn_yaw / 2.0
        return (0.0, 0.0, math.sin(half), math.cos(half))


INTERIOR_AGENT_ROUTES: dict[str, dict[str, Route]] = {
    "kujiale_0003": {
        route.name: route
        for route in (
            Route(
                "hall_straight",
                (-6.4, 0.5),
                (-0.4, 0.5),
                "Straight run across the open central hall (>= 0.8 m clearance along the whole segment).",
                6.0,
            ),
            Route(
                "around_table",
                (-4.38, 3.32),
                (0.07, 3.32),
                "Detour around the dining table to reach a goal directly behind it.",
                6.3,
            ),
            Route(
                "through_doorway",
                (-2.98, 0.92),
                (-6.28, -1.23),
                "Leave the hall through a ~1 m wall opening into a side room.",
                4.4,
            ),
            Route(
                "to_far_room",
                (-6.48, 1.32),
                (-1.48, -4.98),
                "Long diagonal to the bottom room, passing between furniture.",
                10.1,
            ),
        )
    }
}

DEFAULT_ROUTE = "hall_straight"


def list_routes(scene_id: str = "kujiale_0003") -> list[str]:
    """Names of the routes defined for a scene."""
    return sorted(INTERIOR_AGENT_ROUTES.get(scene_id, {}))


def get_route(name: str, scene_id: str = "kujiale_0003") -> Route:
    """Look up a route.

    Raises:
        KeyError: If the scene has no routes or the route name is unknown.
    """
    routes = INTERIOR_AGENT_ROUTES.get(scene_id)
    if routes is None:
        raise KeyError(f"No routes defined for scene '{scene_id}'. Scenes with routes: {sorted(INTERIOR_AGENT_ROUTES)}")
    if name not in routes:
        raise KeyError(f"Route '{name}' not found for scene '{scene_id}'. Available routes: {sorted(routes)}")
    return routes[name]
