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


def inject_nav2_boot_flags() -> None:
    """Inject OmniGraph and ROS 2 bridge boot flags into sys.argv if not present."""
    flags = [
        "--enable", "omni.graph",
        "--enable", "omni.graph.action",
        "--enable", "isaacsim.ros2.bridge",
        "--enable", "isaacsim.ros2.nodes",
    ]
    i = 0
    while i < len(flags):
        flag = flags[i]
        if flag == "--enable" and i + 1 < len(flags):
            ext_name = flags[i + 1]
            already_present = any(
                sys.argv[j] == "--enable" and j + 1 < len(sys.argv) and sys.argv[j + 1] == ext_name
                for j in range(len(sys.argv))
            )
            if not already_present:
                sys.argv.extend(["--enable", ext_name])
            i += 2
        else:
            if flag not in sys.argv:
                sys.argv.append(flag)
            i += 1


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
    except Exception as exc:
        raise ValueError(f"Failed to deserialize RunSpec JSON: {exc}") from exc

    validate_spec(spec)
    return spec


def parse_worker_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    """Parse CLI arguments for the benchmark worker process."""
    parser = argparse.ArgumentParser(
        description="Worker subprocess executing one isolated navigation episode."
    )
    parser.add_argument("spec", help="Path to RunSpec JSON file or raw JSON string.")
    add_logger_args(parser)
    return parser.parse_known_args(argv)


def _merge_remaining_args(args: argparse.Namespace, remaining: list[str]) -> argparse.Namespace:
    """Parse extra command-line flags (e.g. AppLauncher options) onto the namespace."""
    i = 0
    while i < len(remaining):
        arg = remaining[i]
        if arg.startswith("--"):
            key_val = arg[2:].split("=", 1)
            key = key_val[0].replace("-", "_")
            if len(key_val) == 2:
                val = key_val[1]
                i += 1
            elif i + 1 < len(remaining) and not remaining[i + 1].startswith("-"):
                val = remaining[i + 1]
                i += 2
            else:
                val = True
                i += 1
            setattr(args, key, val)
        else:
            i += 1
    return args


def main(argv: list[str] | None = None) -> None:
    """Main worker entry point."""
    args, remaining = parse_worker_args(argv)
    if remaining:
        _merge_remaining_args(args, remaining)

    if hasattr(args, "log_level") and args.log_level:
        configure_logging(args.log_level)

    try:
        spec = load_run_spec(args.spec)
    except Exception as exc:
        logger.error(f"Spec parsing error: {exc}", exc_info=True)
        sys.exit(1)

    if spec.method == "nav2" or spec.method_family == "ros2":
        inject_nav2_boot_flags()

    # Pass args to RunSession / AppLauncher (including any additional CLI flags)
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
    except BaseException as exc:
        logger.error(f"Worker failure: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
