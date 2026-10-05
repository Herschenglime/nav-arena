# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Worker subprocess entry point for isolated navigation benchmark episodes."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from nav_arena.benchmarks.session import RunSession
from nav_arena.benchmarks.spec import RunSpec, validate_spec
from nav_arena.utils import add_logger_args, configure_logging, get_logger

logger = get_logger("benchmarks.worker")


def load_run_spec(spec_input: str) -> RunSpec:
    """Parse and validate a RunSpec from a JSON file path or raw JSON string.

    Args:
        spec_input: Path to a JSON spec file, or raw JSON string content.

    Returns:
        Validated RunSpec instance.

    Raises:
        ValueError: If JSON deserialization or spec validation fails.
    """
    trimmed = spec_input.strip()
    if trimmed.startswith("{"):
        content = trimmed
    else:
        try:
            path = Path(spec_input)
            if path.is_file():
                content = path.read_text(encoding="utf-8")
            else:
                content = spec_input
        except OSError:
            content = spec_input

    try:
        spec = RunSpec.from_json(content)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Failed to deserialize RunSpec JSON: {exc}") from exc

    validate_spec(spec)
    return spec


def parse_worker_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the worker command line: the spec, logging flags, and the Isaac Lab app launcher flags."""
    from isaaclab.app import AppLauncher  # safe before boot; the app itself starts later in RunSession

    parser = argparse.ArgumentParser(description="Worker subprocess executing one isolated navigation episode.")
    parser.add_argument("spec", help="Path to RunSpec JSON file or raw JSON string.")
    add_logger_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Main worker entry point."""
    args = parse_worker_args(argv)
    if args.log_level:
        configure_logging(args.log_level)

    try:
        spec = load_run_spec(args.spec)
    except ValueError as exc:
        logger.error(f"Spec parsing error: {exc}", exc_info=True)
        sys.exit(1)

    # RunSession passes enable_ros2 for the ros2 family to launch_simulation_app, which injects the boot flags.
    try:
        with RunSession(spec, args_cli=args) as session:
            result = session.run_episode()
            code = 0 if getattr(result, "terminal_cause", None) == "goal_reached" else 2
            sys.exit(code)
    except NotImplementedError:
        logger.error("Requested method backend is not implemented.", exc_info=True)
        sys.exit(1)
    except SystemExit:
        raise
    except BaseException as exc:  # process boundary: any failure must become exit code 1 with a logged traceback
        logger.error(f"Worker failure: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
