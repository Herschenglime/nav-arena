# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command-line entry point for nav_arena.

Provides subcommands for running single benchmark episodes, running batch sweeps,
inspecting tracked runs, and invoking utility tools.

This package guarantees fast startup (<0.1s) and strictly forbids importing heavy
simulation, deep learning, or ROS frameworks (isaaclab, isaacsim, omni, pxr, rclpy, torch).
"""

from __future__ import annotations

import argparse
import sys

from nav_arena.cli.run import add_run_parser, handle_run
from nav_arena.cli.runs import add_runs_parser, handle_runs
from nav_arena.cli.sweep import add_sweep_parser, handle_sweep
from nav_arena.cli.tools import (
    add_doctor_parser,
    add_map_parser,
    add_robots_parser,
    add_routes_parser,
    handle_doctor,
    handle_map,
    handle_robots,
    handle_routes,
)

_HANDLERS = {
    "run": handle_run,
    "sweep": handle_sweep,
    "runs": handle_runs,
    "doctor": handle_doctor,
    "routes": handle_routes,
    "robots": handle_robots,
    "map": handle_map,
}


def create_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser for the nav_arena CLI."""
    parser = argparse.ArgumentParser(
        prog="nav_arena",
        description="Unified navigation benchmarking and evaluation framework for Isaac Sim.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", metavar="COMMAND")
    add_run_parser(subparsers)
    add_sweep_parser(subparsers)
    add_runs_parser(subparsers)
    add_doctor_parser(subparsers)
    add_routes_parser(subparsers)
    add_robots_parser(subparsers)
    add_map_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point returning an exit code."""
    parser = create_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    handler = _HANDLERS.get(args.subcommand)
    if handler is None:
        parser.print_help()
        return 0
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
