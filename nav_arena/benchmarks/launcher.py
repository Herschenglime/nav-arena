# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch one benchmark worker subprocess, stream its output to a log, and enforce a timeout.

Shared by ``nav_arena run`` and ``nav_arena sweep`` so both execute workers identically. Importing this module never
pulls in simulation, deep learning, or ROS frameworks.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
import subprocess
import sys
import threading

from nav_arena.benchmarks.spec import RunSpec
from nav_arena.utils.logger import get_logger
from nav_arena.utils.process import managed_process

logger = get_logger("nav_arena.benchmarks.launcher")


@dataclass
class WorkerOutcome:
    """How a worker subprocess ended."""

    exit_code: int
    timed_out: bool = False
    log_lines: list[str] = field(default_factory=list)


def compute_default_timeout(spec: RunSpec) -> float:
    """Calculate default execution timeout in seconds based on episode bounds.

    Formula: (max_steps / 50.0) + stall_timeout_s + 60.0
    """
    stall_timeout = spec.limits.stall_timeout_s if spec.limits.stall_timeout_s is not None else 10.0
    return (spec.limits.max_steps / 50.0) + stall_timeout + 60.0


def worker_command(spec_path: Path) -> list[str]:
    """Command line that runs the worker on a saved ``RunSpec`` JSON file."""
    return [sys.executable, "-u", "-m", "nav_arena.benchmarks.worker", str(spec_path)]


def run_worker(
    spec_path: Path,
    log_path: Path,
    timeout_s: float,
    quiet: bool = False,
    on_start: Callable[[int], None] | None = None,
) -> WorkerOutcome:
    """Run the worker on ``spec_path`` and wait for it, mirroring its output to ``log_path`` (and stdout unless quiet).

    Args:
        spec_path: Saved ``RunSpec`` JSON the worker should execute.
        log_path: File that receives the worker's combined stdout/stderr.
        timeout_s: Wall-clock limit; the process group is killed when it expires.
        quiet: Suppress live streaming; the log file is still written.
        on_start: Called with the worker's pid right after it starts (used to mark a manifest entry as running).

    Returns:
        The worker's exit code, whether it timed out, and the captured output lines. A launch failure is reported as
        exit code 1.

    Raises:
        KeyboardInterrupt: Propagated so callers can do their own interruption bookkeeping.
    """
    log_lines: list[str] = []
    timed_out = False
    exit_code = 1
    try:
        with open(log_path, "w", encoding="utf-8") as log_file:
            reader: threading.Thread | None = None
            try:
                with managed_process(
                    worker_command(spec_path),
                    timeout=5.0,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                ) as proc:
                    if on_start is not None:
                        on_start(proc.pid)

                    def stream() -> None:
                        if proc.stdout is None:
                            return
                        for line in iter(proc.stdout.readline, ""):
                            log_file.write(line)
                            log_file.flush()
                            log_lines.append(line)
                            if not quiet:
                                sys.stdout.write(line)
                                sys.stdout.flush()

                    reader = threading.Thread(target=stream, daemon=True)
                    reader.start()
                    try:
                        exit_code = proc.wait(timeout=timeout_s)
                    except subprocess.TimeoutExpired:
                        timed_out = True
                        exit_code = 1
            finally:
                if reader is not None:
                    reader.join(timeout=5.0)
    except OSError as exc:
        logger.error("Failed to execute worker process: %s", exc)
        exit_code = 1

    if quiet and (exit_code not in (0, 2) or timed_out):
        sys.stderr.write("".join(log_lines))
        sys.stderr.flush()
    return WorkerOutcome(exit_code=exit_code, timed_out=timed_out, log_lines=log_lines)
