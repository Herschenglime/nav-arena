# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Metric coordinate grid visualizer for occupancy maps and test geometries.

Renders high-resolution metric coordinate overlays with millimeter-accurate world axes,
robot footprint projections, trajectories, and side-by-side regional insets.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence
import yaml
import numpy as np
from PIL import Image

from nav_arena.tools.route_map import find_cached_map
from nav_arena.utils.logger import add_logger_args, configure_logging, get_logger

logger = get_logger("nav_arena.tools.map_visualizer")


def load_occupancy_grid(map_dir: Path) -> tuple[np.ndarray, list[float], float]:
    """Load map image and world extent from a map directory.

    Returns:
        (img_world, extent, resolution) where img_world is oriented with origin='lower'.
    """
    yaml_file = map_dir / "map.yaml"
    if not yaml_file.is_file():
        raise FileNotFoundError(f"Map metadata not found: {yaml_file}")

    meta = yaml.safe_load(yaml_file.read_text())
    img_path = map_dir / meta["image"]
    img_arr = np.array(Image.open(img_path).convert("L"))

    # ROS occupancy map convention: PIL row 0 is top (max Y).
    # Invert vertically so row 0 is at origin_y when using origin="lower".
    img_world = img_arr[::-1, :]

    res = float(meta["resolution"])
    ox = float(meta["origin"][0])
    oy = float(meta["origin"][1])
    h, w = img_world.shape
    extent = [ox, ox + w * res, oy, oy + h * res]

    return img_world, extent, res


def plot_map_visualization(
    map_dir: Path,
    title: str = "Scene Occupancy Grid",
    spawn: tuple[float, float] | None = None,
    goal: tuple[float, float] | None = None,
    robot_radii: dict[str, float] | None = None,
    trajectories: list[dict] | None = None,
    crop_bounds: tuple[float, float, float, float] | None = None,
    output_path: Path | None = None,
    side_by_side: bool = True,
    dpi: int = 200,
) -> Path:
    """Generate a publication-quality coordinate grid plot of an occupancy map.

    Args:
        map_dir: Directory containing map.yaml and map.png.
        title: Main title of the plot.
        spawn: Optional (x, y) spawn position in meters.
        goal: Optional (x, y) goal position in meters.
        robot_radii: Optional dict mapping label to radius in meters, e.g. {'Kaya': 0.15}.
        trajectories: Optional list of trajectory dicts with keys 'x', 'y', 'label', 'color', 'style'.
        crop_bounds: Optional (xmin, xmax, ymin, ymax) in meters for detailed inset.
        output_path: Destination PNG file path.
        side_by_side: Whether to render side-by-side (full map + crop) or single panel.
        dpi: Output resolution.

    Returns:
        The written output file Path.
    """
    import matplotlib.pyplot as plt

    img_world, extent, res = load_occupancy_grid(map_dir)

    if side_by_side and crop_bounds is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9), gridspec_kw={"width_ratios": [1, 1.2]})
        axes = [ax1, ax2]
    else:
        fig, ax = plt.subplots(figsize=(11, 9))
        axes = [ax]

    for i, ax in enumerate(axes):
        ax.imshow(img_world, cmap="gray", origin="lower", extent=extent)
        ax.grid(True, which="both", color="cyan", linestyle="--", linewidth=0.5, alpha=0.6)
        ax.set_xlabel("X (m)", fontsize=11)
        ax.set_ylabel("Y (m)", fontsize=11)

        is_crop = (i == 1) if len(axes) > 1 else (crop_bounds is not None)

        if not is_crop:
            ax.set_title(f"{title} (Full Map)", fontsize=13, fontweight="bold")
            ax.set_xticks(np.arange(np.floor(extent[0]), np.ceil(extent[1]) + 1, 1.0))
            ax.set_yticks(np.arange(np.floor(extent[2]), np.ceil(extent[3]) + 1, 1.0))
        else:
            xmin, xmax, ymin, ymax = crop_bounds  # type: ignore
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_title(f"{title} (Detail: X [{xmin}, {xmax}], Y [{ymin}, {ymax}])", fontsize=13, fontweight="bold")
            ax.set_xticks(np.arange(np.floor(xmin), np.ceil(xmax) + 0.5, 0.5))
            ax.set_yticks(np.arange(np.floor(ymin), np.ceil(ymax) + 0.5, 0.5))

        # Annotate spawn and goal
        if spawn is not None:
            ax.plot(spawn[0], spawn[1], "ro", markersize=8, label=f"Spawn ({spawn[0]:.2f}, {spawn[1]:.2f})")
            if not is_crop:
                ax.annotate(
                    "Spawn",
                    xy=spawn,
                    xytext=(spawn[0], spawn[1] + 0.4),
                    color="red",
                    fontweight="bold",
                    ha="center",
                    arrowprops=dict(arrowstyle="->", color="red"),
                )

        if goal is not None:
            ax.plot(goal[0], goal[1], "go", markersize=8, label=f"Goal ({goal[0]:.2f}, {goal[1]:.2f})")
            if not is_crop:
                ax.annotate(
                    "Goal",
                    xy=goal,
                    xytext=(goal[0], goal[1] + 0.4),
                    color="green",
                    fontweight="bold",
                    ha="center",
                    arrowprops=dict(arrowstyle="->", color="green"),
                )

        # Plot robot footprints at spawn
        if spawn is not None and robot_radii:
            colors = ["cyan", "orange", "magenta", "yellow", "lime"]
            styles = ["-", ":", "--", "-."]
            for idx, (label, radius) in enumerate(robot_radii.items()):
                c = colors[idx % len(colors)]
                s = styles[idx % len(styles)]
                circle = plt.Circle(spawn, radius, color=c, fill=False, linewidth=2, linestyle=s, label=f"{label} (r={radius:.2f}m)")
                ax.add_patch(circle)

        # Plot trajectories
        if trajectories:
            for traj in trajectories:
                ax.plot(
                    traj["x"],
                    traj["y"],
                    color=traj.get("color", "blue"),
                    linestyle=traj.get("style", "-"),
                    linewidth=traj.get("linewidth", 2.0),
                    label=traj.get("label", None),
                )

        ax.legend(loc="upper left" if not is_crop else "lower left", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    if output_path is None:
        output_path = Path("map_visualization.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi)
    plt.close(fig)

    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for map visualizer."""
    parser = argparse.ArgumentParser(description="Visualize occupancy maps with metric coordinate grids and trajectories.")
    parser.add_argument("--scene", default="kujiale_0003", help="Scene ID or map directory.")
    parser.add_argument("--map-dir", type=Path, default=None, help="Direct path to map directory containing map.yaml.")
    parser.add_argument("--closed-doors", action="store_true", help="Prefer the closed-door scene map variant.")
    parser.add_argument("--spawn", nargs=2, type=float, metavar=("X", "Y"), help="Spawn location (m).")
    parser.add_argument("--goal", nargs=2, type=float, metavar=("X", "Y"), help="Goal location (m).")
    parser.add_argument(
        "--robot",
        action="append",
        default=[],
        help="Robot name to query radius for footprint visualization (e.g. --robot kaya --robot dingo).",
    )
    parser.add_argument("--radius", action="append", type=float, default=[], help="Explicit robot radius in meters.")
    parser.add_argument(
        "--crop",
        nargs=4,
        type=float,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
        help="Coordinate bounds for regional crop inset.",
    )
    parser.add_argument(
        "--trajectory",
        nargs=4,
        action="append",
        type=float,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Straight-line trajectory segment to plot.",
    )
    parser.add_argument("--single-panel", action="store_true", help="Render only one panel instead of side-by-side.")
    parser.add_argument("--output", type=Path, default=None, help="Output PNG path.")
    add_logger_args(parser)
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    # Resolve map directory
    if args.map_dir:
        map_dir = args.map_dir
    else:
        try:
            map_dir = find_cached_map(args.scene, open_doors=not args.closed_doors)
        except FileNotFoundError as e:
            logger.error(str(e))
            return 1

    # Resolve robot radii
    radii: dict[str, float] = {}
    for r_val in args.radius:
        radii[f"Radius {r_val:.2f}m"] = r_val

    for r_name in args.robot:
        try:
            from nav_arena.embodiments import get_embodiment

            emb = get_embodiment(r_name)
            radii[emb.name.capitalize()] = getattr(emb, "robot_radius", 0.25)
        except Exception:
            radii[r_name.capitalize()] = 0.25

    # Resolve trajectories
    trajs = []
    if args.trajectory:
        colors = ["green", "red", "magenta", "orange", "blue"]
        for idx, (x1, y1, x2, y2) in enumerate(args.trajectory):
            trajs.append({
                "x": [x1, x2],
                "y": [y1, y2],
                "color": colors[idx % len(colors)],
                "style": "-" if idx == 0 else "--",
                "label": f"Traj {idx+1}: ({x1:.1f}, {y1:.1f}) -> ({x2:.1f}, {y2:.1f})",
            })

    # Default output path
    output_path = args.output
    if output_path is None:
        from nav_arena.config.paths import CACHE_DIR
        output_path = CACHE_DIR / "maps" / f"{args.scene}_visualization.png"

    spawn_pt = tuple(args.spawn) if args.spawn else None  # type: ignore
    goal_pt = tuple(args.goal) if args.goal else None  # type: ignore
    crop = tuple(args.crop) if args.crop else None  # type: ignore

    # Auto-crop around spawn and goal if not specified and spawn/goal are provided
    if crop is None and spawn_pt and goal_pt and not args.single_panel:
        x_min = min(spawn_pt[0], goal_pt[0]) - 3.5
        x_max = max(spawn_pt[0], goal_pt[0]) + 1.5
        y_min = min(spawn_pt[1], goal_pt[1]) - 2.0
        y_max = max(spawn_pt[1], goal_pt[1]) + 2.0
        crop = (x_min, x_max, y_min, y_max)

    logger.info(f"Loading map from {map_dir}...")
    out = plot_map_visualization(
        map_dir=map_dir,
        title=f"Scene '{args.scene}' Grid Map",
        spawn=spawn_pt,
        goal=goal_pt,
        robot_radii=radii,
        trajectories=trajs,
        crop_bounds=crop,
        output_path=output_path,
        side_by_side=not args.single_panel,
    )

    logger.success(f"Generated map visualization: {out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
