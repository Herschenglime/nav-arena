# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Detect running Isaac Sim sessions before a run starts.

Two Kit sessions on one GPU once froze a user's GUI viewport, so runs refuse to start while another simulator
process is alive (``--force`` bypasses). Detection looks at each process's argument vector rather than at substrings,
so unrelated processes (a pytest run, an idle Python REPL in the Isaac environment, system daemons) never count.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import PurePath
import shlex
import subprocess

from nav_arena.utils.logger import get_logger

logger = get_logger("nav_arena.benchmarks.guard")

_KIT_EXECUTABLES = frozenset({"kit", "isaac-sim", "isaac-sim.sh", "isaaclab.sh"})
_WORKER_MODULE = "nav_arena.benchmarks.worker"


@dataclass(frozen=True)
class ProcessInfo:
    """One row of ``ps``: pid, parent pid, and the command line."""

    pid: int
    ppid: int
    command: str


def parse_ps_output(output: str) -> list[ProcessInfo]:
    """Parse ``ps -eo pid,ppid,args`` output (header line first); malformed rows are skipped."""
    processes: list[ProcessInfo] = []
    for line in output.splitlines()[1:]:
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        try:
            processes.append(ProcessInfo(int(parts[0]), int(parts[1]), parts[2]))
        except ValueError:
            continue
    return processes


def _tokens(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


def is_simulator_command(command: str) -> bool:
    """True if ``command`` starts an Isaac Sim run: a ``verify_*.py`` script, the benchmark worker, or Kit itself."""
    tokens = _tokens(command)
    if not tokens:
        return False
    names = [PurePath(token).name for token in tokens]
    if "pytest" in names:
        return False
    if names[0] in _KIT_EXECUTABLES:
        return True
    if any(name.startswith("verify_") and name.endswith(".py") for name in names):
        return True
    return any(tokens[i] == "-m" and tokens[i + 1] == _WORKER_MODULE for i in range(len(tokens) - 1))


def related_pids(processes: list[ProcessInfo], pid: int) -> set[int]:
    """``pid`` together with its ancestors and its descendants (a run must not block on its own process tree).

    Siblings and other children of an ancestor are deliberately not included: they are separate sessions.
    """
    parent = {p.pid: p.ppid for p in processes}
    related = {pid}
    node = pid
    while parent.get(node, 0) > 0 and parent[node] not in related:
        node = parent[node]
        related.add(node)

    descendants = {pid}
    grew = True
    while grew:
        grew = False
        for proc in processes:
            if proc.ppid in descendants and proc.pid not in descendants:
                descendants.add(proc.pid)
                grew = True
    return related | descendants


def find_simulator_processes(processes: list[ProcessInfo], own_pid: int) -> list[tuple[int, str]]:
    """Simulator processes in ``processes`` that are not part of ``own_pid``'s process tree."""
    ignored = related_pids(processes, own_pid)
    return [(p.pid, p.command) for p in processes if p.pid not in ignored and is_simulator_command(p.command)]


def check_preflight_processes() -> list[tuple[int, str]]:
    """Return ``(pid, command)`` for every other running simulator process (empty if none, or if ``ps`` fails)."""
    try:
        result = subprocess.run(["ps", "-eo", "pid,ppid,args"], capture_output=True, text=True, check=True)
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("Failed to run ps pre-flight check: %s", exc)
        return []
    return find_simulator_processes(parse_ps_output(result.stdout), os.getpid())
