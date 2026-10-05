# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the route/run overlay tool (synthetic occupancy maps; CPU-only)."""

from __future__ import annotations

import json
import math

import numpy as np
from PIL import Image
import pytest
import yaml

from nav_arena.tools import route_map as rm


def _write_map(directory, image, resolution=0.1, origin=(0.0, 0.0)):
    directory.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image.astype(np.uint8)).save(directory / "map.png")
    (directory / "map.yaml").write_text(
        yaml.safe_dump({"image": "map.png", "resolution": resolution, "origin": [origin[0], origin[1], 0.0]})
    )
    return directory


@pytest.fixture
def walled_map(tmp_path):
    """10 m x 10 m free map (0.1 m cells) with a vertical wall at x=5 m that has a gap at y in [8, 9] m."""
    img = np.full((100, 100), 254, dtype=np.uint8)
    img[:, 50:52] = 0
    img[10:20, 50:52] = 254  # rows 10-19 -> y between 8.0 and 9.0 m (row 0 is the top / max y)
    return _write_map(tmp_path / "map", img)


def test_world_cell_round_trip_uses_ros_orientation(walled_map):
    """Verify row 0 is the maximum y and conversions round-trip."""
    grid = rm.OccupancyGrid.load(walled_map)
    assert grid.world_to_cell(0.0, 0.0) == (0, 99)
    assert grid.world_to_cell(0.0, 9.9) == (0, 0)
    for x, y in [(1.5, 2.5), (7.0, 8.4), (9.9, 0.0)]:
        col, row = grid.world_to_cell(x, y)
        assert grid.cell_to_world(col, row) == pytest.approx((x, y), abs=grid.resolution)


def test_clearance_and_free_mask(walled_map):
    """Verify clearance is the distance to the nearest non-free cell."""
    grid = rm.OccupancyGrid.load(walled_map)
    clearance = grid.clearance_map()
    assert grid.clearance_at(clearance, 4.0, 2.0) == pytest.approx(1.0, abs=0.15)  # 1 m from the wall
    assert grid.clearance_at(clearance, 5.05, 2.0) == 0.0  # inside the wall
    assert grid.clearance_at(clearance, -3.0, 2.0) == 0.0  # outside the map


def test_plan_path_goes_through_the_gap(walled_map):
    """Verify A* detours through the wall's gap and the path is longer than the straight line."""
    grid = rm.OccupancyGrid.load(walled_map)
    clearance = grid.clearance_map()
    length, points = rm.plan_path(grid, clearance, (2.0, 2.0), (8.0, 2.0), robot_radius=0.2)
    assert length > 6.0 + 5.0  # must go up to the gap near y=8.5 and back down
    crossing = min(points, key=lambda p: abs(p[0] - 5.05))
    assert 8.0 <= crossing[1] <= 9.0


def test_plan_path_rejects_unsafe_endpoints_and_blocked_maps(walled_map, tmp_path):
    """Verify unsafe start/goal and fully walled maps yield no path."""
    grid = rm.OccupancyGrid.load(walled_map)
    clearance = grid.clearance_map()
    assert rm.plan_path(grid, clearance, (5.05, 2.0), (8.0, 2.0)) is None  # start inside the wall
    assert rm.plan_path(grid, clearance, (2.0, 2.0), (50.0, 2.0)) is None  # goal off the map
    blocked = np.full((100, 100), 254, dtype=np.uint8)
    blocked[:, 50:52] = 0
    g2 = rm.OccupancyGrid.load(_write_map(tmp_path / "blocked", blocked))
    assert rm.plan_path(g2, g2.clearance_map(), (2.0, 2.0), (8.0, 2.0)) is None


def test_find_cached_map_prefers_open_doors_variant(tmp_path):
    """Verify map lookup order and the actionable error."""
    root = tmp_path / "maps"
    _write_map(root / "scene_x" / "cs0.05", np.full((4, 4), 254))
    assert rm.find_cached_map("scene_x", root).parent.name == "scene_x"
    _write_map(root / "scene_x_open_doors" / "cs0.05", np.full((4, 4), 254))
    assert rm.find_cached_map("scene_x", root).parent.name == "scene_x_open_doors"
    assert rm.find_cached_map("scene_x", root, open_doors=False).parent.name == "scene_x"
    with pytest.raises(FileNotFoundError, match="map_generator"):
        rm.find_cached_map("missing", root)


def _write_run(directory, positions, stops, cause="collision"):
    directory.mkdir(parents=True)
    rows = [
        {"step": i * 10, "position": list(p), "stop": i in stops, "path_world": [list(p), [p[0] + 1.0, p[1]]]}
        for i, p in enumerate(positions)
    ]
    (directory / "steps.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (directory / "settings.json").write_text(json.dumps({"policy": "iplanner", "goal_xy": [8.0, 2.0]}))
    (directory / "summary.json").write_text(json.dumps({"terminal_cause": cause, "final_goal_distance_m": 3.2}))
    return directory


def test_load_run_extracts_trajectory_stops_and_outcome(tmp_path):
    """Verify a recorded run's positions, stop points, planned paths, goal, and label."""
    run = rm.load_run(_write_run(tmp_path / "run", [(1, 2), (2, 2), (3, 2)], stops={2}))
    assert run.positions == [(1, 2), (2, 2), (3, 2)]
    assert run.stops == [(3, 2)]
    assert len(run.plans) == 3 and run.goal == (8.0, 2.0)
    assert run.label == "iplanner: collision (3.20 m left)"


def test_render_overlay_marks_routes_and_runs(walled_map, tmp_path):
    """Verify the overlay has the expected size, draws something, and reports clearance warnings."""
    grid = rm.OccupancyGrid.load(walled_map)
    routes = [rm.RouteSpec("ok", (2.0, 2.0), (8.0, 2.0)), rm.RouteSpec("tight", (4.95, 2.0), (8.0, 2.0))]
    run = rm.load_run(_write_run(tmp_path / "run", [(1, 2), (2, 2), (3, 2)], stops={1, 2}))
    image, report = rm.render_overlay(grid, routes, [run], scale=2, show_plans=1)
    assert image.size == (grid.width * 2, grid.height * 2)
    plain = Image.fromarray(grid.image).convert("RGB").resize(image.size, Image.NEAREST)
    assert np.any(np.array(image) != np.array(plain))
    joined = "\n".join(report)
    assert "ok" in joined and "CLEARANCE BELOW ROBOT RADIUS" in joined
    assert any(line.startswith("run iplanner") and "2 with a stop request" in line for line in report)


def test_cli_writes_png_for_adhoc_route(walled_map, tmp_path):
    """Verify the command-line entry point renders an ad-hoc route to a PNG."""
    out = tmp_path / "out" / "routes.png"
    code = rm.main(["--map-dir", str(walled_map), "--spawn", "2", "2", "--goal", "8", "2", "--output", str(out)])
    assert code == 0 and out.is_file()
    assert Image.open(out).size == (300, 300)


def test_cli_reports_missing_map_and_bad_arguments(walled_map, tmp_path):
    """Verify actionable failures: missing cache, lone --spawn, and unknown route names."""
    out = str(tmp_path / "x.png")
    assert rm.main(["--scene", "nope", "--output", out]) == 1
    assert rm.main(["--map-dir", str(walled_map), "--spawn", "1", "1", "--output", out]) == 1
    assert rm.main(["--map-dir", str(walled_map), "--route", "no_such_route", "--output", out]) == 1
