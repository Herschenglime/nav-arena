# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The 'doctor', 'routes', and 'map' subcommands."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import subprocess
import sys

import yaml

from nav_arena.cli._common import print_subcommand_help
from nav_arena.cli._common import print_subcommand_help
from nav_arena.utils.logger import get_logger

logger = get_logger("nav_arena.cli")


def handle_doctor(args: argparse.Namespace) -> int:
    """Handle 'doctor' subcommand."""
    from nav_arena.benchmarks.doctor import run_doctor

    return run_doctor(verbose=getattr(args, "verbose", False))


def handle_routes_list(args: argparse.Namespace) -> int:
    """Handle 'routes list' subcommand."""
    from nav_arena.scenes.routes import INTERIOR_AGENT_ROUTES, list_routes

    scene = args.scene
    if scene not in INTERIOR_AGENT_ROUTES:
        logger.error(
            "No registered routes for scene '%s'. Available scenes: %s",
            scene,
            sorted(INTERIOR_AGENT_ROUTES.keys()),
        )
        return 1

    routes = INTERIOR_AGENT_ROUTES[scene]
    if not routes:
        print(f"No routes registered for scene '{scene}'.")
        return 0

    print(f"Registered routes for scene '{scene}':")
    headers = ["Route", "Spawn (x, y)", "Goal (x, y)", "Euclidean (m)", "Ref Path (m)", "Description"]
    rows = []
    for name in list_routes(scene):
        r = routes[name]
        spawn_str = f"({r.spawn_xy[0]:.2f}, {r.spawn_xy[1]:.2f})"
        goal_str = f"({r.goal_xy[0]:.2f}, {r.goal_xy[1]:.2f})"
        dist_str = f"{r.euclidean_m:.2f}"
        ref_str = f"{r.reference_path_m:.2f}"
        rows.append([name, spawn_str, goal_str, dist_str, ref_str, r.description])

    widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(val))

    header_line = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    sep_line = "|-" + "-|-".join("-" * widths[i] for i in range(len(headers))) + "-|"
    print(header_line)
    print(sep_line)
    for row in rows:
        print("| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(row))) + " |")

    return 0


def handle_routes_show(args: argparse.Namespace) -> int:
    """Handle 'routes show' subcommand."""
    from nav_arena.scenes.routes import get_route

    scene = args.scene
    route_name = args.route
    try:
        r = get_route(route_name, scene_id=scene)
    except KeyError as exc:
        logger.error("%s", exc)
        return 1

    print(f"Route:          {r.name}")
    print(f"Scene:          {scene}")
    print(f"Spawn:          ({r.spawn_xy[0]:.2f}, {r.spawn_xy[1]:.2f})")
    print(f"Goal:           ({r.goal_xy[0]:.2f}, {r.goal_xy[1]:.2f})")
    print(f"Spawn Yaw:      {r.spawn_yaw:.4f} rad ({math.degrees(r.spawn_yaw):.1f} deg)")
    print(f"Distance:       {r.euclidean_m:.2f} m (straight-line)")
    print(f"Reference Path: {r.reference_path_m:.2f} m")
    print(f"Description:    {r.description}")
    return 0


def handle_routes(args: argparse.Namespace) -> int:
    """Dispatch 'routes' subcommands."""
    action = getattr(args, "routes_action", None)
    if not action:
        return print_subcommand_help("routes")

    if action == "list":
        return handle_routes_list(args)
    elif action == "show":
        return handle_routes_show(args)
    else:
        logger.error("Unknown routes action '%s'", action)
        return 1


def handle_map_generate(args: argparse.Namespace) -> int:
    """Handle 'map generate' subcommand."""
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "nav_arena.tools.map_generator",
        "--scene",
        args.scene,
        "--cell-size",
        str(args.resolution),
    ]
    if getattr(args, "force", False):
        cmd.append("--force")

    logger.info("Generating occupancy map: %s", " ".join(cmd))
    res = subprocess.run(cmd)
    return res.returncode


def handle_map_show(args: argparse.Namespace) -> int:
    """Handle 'map show' subcommand."""
    from nav_arena.tools.route_map import find_cached_map

    open_doors = not getattr(args, "closed_doors", False)
    try:
        map_dir = find_cached_map(args.scene, open_doors=open_doors)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    yaml_file = map_dir / "map.yaml"
    if not yaml_file.is_file():
        logger.error("map.yaml not found in %s", map_dir)
        return 1

    try:
        with yaml_file.open("r", encoding="utf-8") as f:
            meta = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError) as exc:
        logger.error("Failed to read map config %s: %s", yaml_file, exc)
        return 1

    image_name = meta.get("image", "map.png")
    image_path = map_dir / image_name

    print(f"Scene:            {args.scene}")
    print(f"Map Directory:    {map_dir}")
    print(f"Config:           {yaml_file}")
    print(f"Image:            {image_path}")
    print(f"Resolution:       {meta.get('resolution', '-')} m/cell")
    print(f"Origin:           {meta.get('origin', '-')}")
    print(f"Occupied Thresh:  {meta.get('occupied_thresh', '-')}")
    print(f"Free Thresh:      {meta.get('free_thresh', '-')}")

    if image_path.is_file():
        try:
            from PIL import Image

            with Image.open(image_path) as img:
                res = float(meta.get("resolution", 0.05))
                w_m = img.width * res
                h_m = img.height * res
                print(f"Dimensions:       {img.width} x {img.height} pixels ({w_m:.2f} m x {h_m:.2f} m)")
        except (ImportError, OSError, ValueError) as exc:
            logger.debug("Failed opening map image: %s", exc)

    return 0


def handle_map(args: argparse.Namespace) -> int:
    """Dispatch 'map' subcommands."""
    action = getattr(args, "map_action", None)
    if not action:
        return print_subcommand_help("map")

    if action == "generate":
        return handle_map_generate(args)
    elif action == "show":
        return handle_map_show(args)
    else:
        logger.error("Unknown map action '%s'", action)
        return 1


def add_doctor_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the 'doctor' subcommand."""
    doctor_p = subparsers.add_parser(
        "doctor",
        help="Inspect simulation environment, weights, checkouts, and caches.",
    )
    doctor_p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Show detailed check output.",
    )


def add_routes_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the 'routes' subcommand."""
    routes_parser = subparsers.add_parser(
        "routes",
        help="List and inspect registered PointNav routes.",
    )
    routes_sub = routes_parser.add_subparsers(dest="routes_action", metavar="ACTION")
    routes_list_p = routes_sub.add_parser("list", help="List routes for a scene.")
    routes_list_p.add_argument(
        "--scene",
        type=str,
        default="kujiale_0003",
        help="Scene ID (default: kujiale_0003).",
    )

    routes_show_p = routes_sub.add_parser("show", help="Show route details or preview.")
    routes_show_p.add_argument("route", type=str, help="Route name to inspect.")
    routes_show_p.add_argument(
        "--scene",
        type=str,
        default="kujiale_0003",
        help="Scene ID (default: kujiale_0003).",
    )


def add_map_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the 'map' subcommand."""
    map_parser = subparsers.add_parser(
        "map",
        help="Generate or inspect 2D scene occupancy maps.",
    )
    map_sub = map_parser.add_subparsers(dest="map_action", metavar="ACTION")
    map_gen_p = map_sub.add_parser("generate", help="Generate 2D occupancy grid.")
    map_gen_p.add_argument(
        "--scene",
        type=str,
        default="kujiale_0003",
        help="Scene ID or USD path (default: kujiale_0003).",
    )
    map_gen_p.add_argument(
        "--resolution",
        "--cell-size",
        dest="resolution",
        type=float,
        default=0.05,
        help="Grid resolution in meters per pixel (default: 0.05).",
    )
    map_gen_p.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force map generation even if cached.",
    )

    map_show_p = map_sub.add_parser("show", help="Display 2D occupancy map info.")
    map_show_p.add_argument(
        "--scene",
        type=str,
        default="kujiale_0003",
        help="Scene ID (default: kujiale_0003).",
    )
    map_show_p.add_argument(
        "--closed-doors",
        action="store_true",
        default=False,
        help="Prefer the closed-door map variant.",
    )
