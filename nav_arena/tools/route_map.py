# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render scene routes and recorded runs over a cached 2D occupancy map.

A debugging aid for navigation experiments: see where a route starts and ends, how much clearance it has, what a
free-space (A*) path looks like, and where a robot actually drove, stopped, or collided. Needs only a cached occupancy
map (generate one with ``python -m nav_arena.tools.map_generator``); no simulator is started.

Examples::

    # all routes of the default scene
    python -u -m nav_arena.tools.route_map --scene kujiale_0003

    # one route plus the trajectories of two recorded runs
    python -u -m nav_arena.tools.route_map --route around_table \\
        --run nav_arena/cache/runs/20261004_140729_iplanner_dingo_hall_straight

    # an ad-hoc start/goal
    python -u -m nav_arena.tools.route_map --spawn -3 0.9 --goal -6.3 -1.2
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import heapq
import json
import math
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw
import yaml

from nav_arena.utils import CACHE_DIR, add_logger_args, configure_logging, get_logger

logger = get_logger("route_map")

DEFAULT_ROBOT_RADIUS = 0.33
"""Obstacle inflation used for the reference path: between the Dingo's half-width (0.26 m) and half-diagonal (0.43 m)."""

_ROUTE_COLORS = [
    (230, 40, 40),
    (0, 150, 60),
    (40, 70, 230),
    (190, 0, 190),
    (240, 130, 0),
    (0, 160, 170),
]
_RUN_COLORS = [(0, 0, 0), (120, 70, 20), (90, 90, 90), (20, 110, 20)]


@dataclass
class OccupancyGrid:
    """A ROS-style occupancy map: row 0 is the maximum y, free cells are > 250."""

    image: np.ndarray
    resolution: float
    origin: tuple[float, float]

    @classmethod
    def load(cls, map_dir: Path) -> OccupancyGrid:
        """Load ``map.yaml`` and its image from a map directory."""
        meta = yaml.safe_load((map_dir / "map.yaml").read_text())
        image = np.array(Image.open(map_dir / meta["image"]).convert("L"))
        return cls(image, float(meta["resolution"]), (float(meta["origin"][0]), float(meta["origin"][1])))

    @property
    def height(self) -> int:
        return self.image.shape[0]

    @property
    def width(self) -> int:
        return self.image.shape[1]

    @property
    def free(self) -> np.ndarray:
        """Boolean mask of known-free cells."""
        return self.image > 250

    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        """World (x, y) in meters to ``(col, row)``."""
        col = int(round((x - self.origin[0]) / self.resolution))
        row = int(round(self.height - 1 - (y - self.origin[1]) / self.resolution))
        return col, row

    def cell_to_world(self, col: float, row: float) -> tuple[float, float]:
        """``(col, row)`` to world (x, y) in meters."""
        return (
            self.origin[0] + col * self.resolution,
            self.origin[1] + (self.height - 1 - row) * self.resolution,
        )

    def in_bounds(self, col: int, row: int) -> bool:
        return 0 <= col < self.width and 0 <= row < self.height

    def clearance_map(self) -> np.ndarray:
        """Distance in meters from every cell to the nearest non-free cell."""
        from scipy import ndimage

        return ndimage.distance_transform_edt(self.free) * self.resolution

    def clearance_at(self, clearance: np.ndarray, x: float, y: float) -> float:
        """Clearance at a world point (0 outside the map)."""
        col, row = self.world_to_cell(x, y)
        return float(clearance[row, col]) if self.in_bounds(col, row) else 0.0


def find_cached_map(scene_id: str, cache_root: Path | None = None, open_doors: bool = True) -> Path:
    """Locate the cached occupancy map directory for a scene.

    Args:
        scene_id: Scene identifier, e.g. ``kujiale_0003``.
        cache_root: Map cache root; defaults to ``<cache>/maps``.
        open_doors: Prefer the map generated for the open-door scene variant, which is what the task simulates.

    Raises:
        FileNotFoundError: If no cached map exists, with the command that generates one.
    """
    root = Path(cache_root) if cache_root is not None else CACHE_DIR / "maps"
    names = [f"{scene_id}_open_doors", scene_id] if open_doors else [scene_id, f"{scene_id}_open_doors"]
    for name in names:
        for yaml_path in sorted((root / name).glob("*/map.yaml")):
            return yaml_path.parent
    raise FileNotFoundError(
        f"No cached occupancy map for '{scene_id}' under {root}. Generate one with: "
        f"python -u -m nav_arena.tools.map_generator --scene {scene_id}"
    )


def plan_path(
    grid: OccupancyGrid,
    clearance: np.ndarray,
    start: tuple[float, float],
    goal: tuple[float, float],
    robot_radius: float = DEFAULT_ROBOT_RADIUS,
) -> tuple[float, list[tuple[float, float]]] | None:
    """Shortest 8-connected free-space path with obstacles inflated by ``robot_radius`` (A*).

    Returns:
        ``(length_m, [(x, y), ...])``, or None if the start/goal are not safe or no path exists.
    """
    safe = clearance >= robot_radius
    (c0, r0), (c1, r1) = grid.world_to_cell(*start), grid.world_to_cell(*goal)
    if not (grid.in_bounds(c0, r0) and grid.in_bounds(c1, r1) and safe[r0, c0] and safe[r1, c1]):
        return None
    queue: list[tuple[float, float, tuple[int, int]]] = [(0.0, 0.0, (c0, r0))]
    best = {(c0, r0): 0.0}
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    while queue:
        _, cost, cell = heapq.heappop(queue)
        if cell == (c1, r1):
            cells = [cell]
            while cell in parent:
                cell = parent[cell]
                cells.append(cell)
            return cost * grid.resolution, [grid.cell_to_world(c, r) for c, r in reversed(cells)]
        if cost > best.get(cell, math.inf):
            continue
        for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            nxt = (cell[0] + dc, cell[1] + dr)
            if grid.in_bounds(*nxt) and safe[nxt[1], nxt[0]]:
                new_cost = cost + math.hypot(dc, dr)
                if new_cost < best.get(nxt, math.inf):
                    best[nxt] = new_cost
                    parent[nxt] = cell
                    heapq.heappush(queue, (new_cost + math.hypot(nxt[0] - c1, nxt[1] - r1), new_cost, nxt))
    return None


@dataclass
class RouteSpec:
    """A route to draw."""

    name: str
    spawn: tuple[float, float]
    goal: tuple[float, float]


@dataclass
class RunTrace:
    """A recorded run loaded from a ``verify_baseline`` output directory."""

    label: str
    positions: list[tuple[float, float]]
    stops: list[tuple[float, float]]
    plans: list[list[tuple[float, float]]]
    goal: tuple[float, float] | None
    cause: str | None


def load_run(run_dir: Path) -> RunTrace:
    """Read positions, stop requests, planned paths and the outcome of a recorded run."""
    run_dir = Path(run_dir)
    rows = [json.loads(line) for line in (run_dir / "steps.jsonl").read_text().splitlines() if line.strip()]
    settings = json.loads((run_dir / "settings.json").read_text()) if (run_dir / "settings.json").is_file() else {}
    summary = json.loads((run_dir / "summary.json").read_text()) if (run_dir / "summary.json").is_file() else {}
    goal = tuple(settings["goal_xy"]) if "goal_xy" in settings else None
    cause = summary.get("terminal_cause")
    label = settings.get("policy", run_dir.name)
    if cause:
        label += f": {cause}"
        if "final_goal_distance_m" in summary:
            label += f" ({summary['final_goal_distance_m']:.2f} m left)"
    return RunTrace(
        label=label,
        positions=[tuple(r["position"][:2]) for r in rows],
        stops=[tuple(r["position"][:2]) for r in rows if r.get("stop")],
        plans=[[tuple(p) for p in r["path_world"]] for r in rows],
        goal=goal,
        cause=cause,
    )


def _grid_lines(draw: ImageDraw.ImageDraw, grid: OccupancyGrid, scale: int) -> None:
    x_min, y_min = grid.origin
    x_max = x_min + grid.width * grid.resolution
    y_max = y_min + grid.height * grid.resolution
    for x in range(math.ceil(x_min), math.floor(x_max) + 1):
        px = (x - x_min) / grid.resolution * scale
        major = x % 2 == 0
        draw.line([(px, 0), (px, grid.height * scale)], fill=(255, 150, 150) if major else (255, 215, 215))
        if major:
            draw.text((px + 2, 2), f"x={x}", fill=(190, 0, 0))
    for y in range(math.ceil(y_min), math.floor(y_max) + 1):
        py = (grid.height - 1 - (y - y_min) / grid.resolution) * scale
        major = y % 2 == 0
        draw.line([(0, py), (grid.width * scale, py)], fill=(255, 150, 150) if major else (255, 215, 215))
        if major:
            draw.text((2, py + 2), f"y={y}", fill=(190, 0, 0))


def render_overlay(
    grid: OccupancyGrid,
    routes: Sequence[RouteSpec] = (),
    runs: Sequence[RunTrace] = (),
    scale: int = 3,
    robot_radius: float = DEFAULT_ROBOT_RADIUS,
    show_paths: bool = True,
    show_grid: bool = True,
    show_plans: int = 0,
) -> tuple[Image.Image, list[str]]:
    """Draw routes (and optionally recorded runs) over the occupancy map.

    Args:
        grid: The occupancy map.
        routes: Routes to draw: S/G markers and, unless disabled, the A* reference path.
        runs: Recorded runs to draw: trajectory, stop requests (red x), end point (square).
        scale: Output pixels per map cell.
        robot_radius: Obstacle inflation for the reference path.
        show_paths: Draw the A* reference paths.
        show_grid: Draw a 1 m world grid with labels every 2 m.
        show_plans: If > 0, also draw every Nth planned path from the runs (thin lines).

    Returns:
        ``(image, report_lines)`` where the report summarizes each route's clearance and reference path.
    """
    clearance = grid.clearance_map()
    image = Image.fromarray(grid.image).convert("RGB").resize((grid.width * scale, grid.height * scale), Image.NEAREST)
    draw = ImageDraw.Draw(image)
    if show_grid:
        _grid_lines(draw, grid, scale)

    def px(point: tuple[float, float]) -> tuple[float, float]:
        col, row = grid.world_to_cell(*point)
        return (col + 0.5) * scale, (row + 0.5) * scale

    report: list[str] = []
    legend: list[tuple[tuple[int, int, int], str]] = []
    for route, color in zip(routes, _ROUTE_COLORS * 4):
        euclid = math.dist(route.spawn, route.goal)
        s_clr = grid.clearance_at(clearance, *route.spawn)
        g_clr = grid.clearance_at(clearance, *route.goal)
        planned = plan_path(grid, clearance, route.spawn, route.goal, robot_radius) if show_paths else None
        if planned is not None:
            length, points = planned
            draw.line([px(p) for p in points], fill=color, width=3)
            stats = f"path {length:.1f} m (x{length / euclid:.2f})"
        else:
            stats = "NO SAFE PATH" if show_paths else ""
        for point, tag in ((route.spawn, "S"), (route.goal, "G")):
            cx, cy = px(point)
            r = 6
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=3)
            draw.text((cx + r + 2, cy - r), f"{route.name}:{tag}", fill=color)
        warn = "  CLEARANCE BELOW ROBOT RADIUS" if min(s_clr, g_clr) < robot_radius else ""
        report.append(
            f"{route.name:18s} {route.spawn} -> {route.goal}: straight {euclid:.1f} m, clearance S={s_clr:.2f} "
            f"G={g_clr:.2f} m; {stats}{warn}".rstrip()
        )
        legend.append((color, f"{route.name}  {euclid:.1f} m"))

    for run, color in zip(runs, _RUN_COLORS * 4):
        if show_plans > 0:
            for plan in run.plans[:: show_plans]:
                if len(plan) > 1:
                    draw.line([px(p) for p in plan], fill=(150, 150, 255), width=1)
        if len(run.positions) > 1:
            draw.line([px(p) for p in run.positions], fill=color, width=2)
        for point in run.stops:
            cx, cy = px(point)
            draw.line([cx - 4, cy - 4, cx + 4, cy + 4], fill=(220, 0, 0), width=2)
            draw.line([cx - 4, cy + 4, cx + 4, cy - 4], fill=(220, 0, 0), width=2)
        if run.positions:
            cx, cy = px(run.positions[0])
            draw.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], outline=color, width=2)
            draw.text((cx + 7, cy - 6), "start", fill=color)
            cx, cy = px(run.positions[-1])
            draw.rectangle([cx - 5, cy - 5, cx + 5, cy + 5], outline=color, width=2)
        if run.goal is not None:
            cx, cy = px(run.goal)
            draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=(0, 200, 0))
        legend.append((color, f"run {run.label}"))
        report.append(f"run {run.label}: {len(run.positions)} plans, {len(run.stops)} with a stop request")

    # Legend
    y = 6
    for color, text in legend:
        draw.rectangle([6, y + 2, 18, y + 10], fill=color)
        draw.text((24, y), text, fill=(0, 0, 0))
        y += 14
    return image, report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render routes and recorded runs over a cached occupancy map.")
    parser.add_argument("--scene", default="kujiale_0003", help="Scene ID (selects the cached map and its routes).")
    parser.add_argument("--route", action="append", default=None, help="Named route to draw (repeatable; default: all).")
    parser.add_argument("--spawn", nargs=2, type=float, metavar=("X", "Y"), help="Ad-hoc route start (m).")
    parser.add_argument("--goal", nargs=2, type=float, metavar=("X", "Y"), help="Ad-hoc route goal (m).")
    parser.add_argument("--run", action="append", type=Path, default=[], help="Recorded run directory to overlay.")
    parser.add_argument("--plans-every", type=int, default=0, help="With --run, also draw every Nth planned path.")
    parser.add_argument("--map-dir", type=Path, default=None, help="Occupancy map directory (default: cached map).")
    parser.add_argument("--closed-doors", action="store_true", help="Prefer the closed-door map variant.")
    parser.add_argument("--robot-radius", type=float, default=DEFAULT_ROBOT_RADIUS, help="Obstacle inflation (m).")
    parser.add_argument("--no-path", action="store_true", help="Do not compute/draw the A* reference paths.")
    parser.add_argument("--no-grid", action="store_true", help="Do not draw the metric grid.")
    parser.add_argument("--scale", type=int, default=3, help="Output pixels per map cell.")
    parser.add_argument("--output", type=Path, default=None, help="Output PNG (default: cache/routes/<scene>_routes.png).")
    add_logger_args(parser)
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    try:
        map_dir = args.map_dir or find_cached_map(args.scene, open_doors=not args.closed_doors)
        grid = OccupancyGrid.load(map_dir)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 1
    logger.info(f"Occupancy map: {map_dir} ({grid.width}x{grid.height} cells at {grid.resolution} m)")

    routes: list[RouteSpec] = []
    if args.spawn and args.goal:
        routes.append(RouteSpec("custom", tuple(args.spawn), tuple(args.goal)))
    elif args.spawn or args.goal:
        logger.error("--spawn and --goal must be given together")
        return 1
    if not routes or args.route:
        from nav_arena.scenes.routes import get_route, list_routes

        names = args.route if args.route else ([] if args.run and not args.route else list_routes(args.scene))
        try:
            routes += [RouteSpec(n, get_route(n, args.scene).spawn_xy, get_route(n, args.scene).goal_xy) for n in names]
        except KeyError as exc:
            logger.error(str(exc))
            return 1

    runs = [load_run(path) for path in args.run]
    image, report = render_overlay(
        grid,
        routes,
        runs,
        scale=args.scale,
        robot_radius=args.robot_radius,
        show_paths=not args.no_path,
        show_grid=not args.no_grid,
        show_plans=args.plans_every,
    )
    for line in report:
        logger.info(line)
    output = args.output or (CACHE_DIR / "routes" / f"{args.scene}_routes.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    logger.success(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
