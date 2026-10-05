# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the running-simulator guard (CPU-only, uses recorded ``ps`` output)."""

from __future__ import annotations

from nav_arena.benchmarks.guard import (
    find_simulator_processes,
    is_simulator_command,
    parse_ps_output,
    related_pids,
)

PS_SAMPLE = """    PID    PPID COMMAND
      1       0 /sbin/init
   1496       1 /usr/lib/polkit-1/polkitd --no-debug
   2572       1 /usr/libexec/rtkit-daemon
   4000    3999 /home/robopi/simulation/env_isaaclab/bin/python -m pytest -c nav_arena/pyproject.toml nav_arena/tests/unit
   4100    3999 /home/robopi/simulation/env_isaaclab/bin/python
   4200    1000 /usr/local/bin/python /usr/local/bin/isaacsim-mcp
   5000    1000 /home/robopi/simulation/env_isaaclab/bin/python -u nav_arena/nav_arena/scripts/verify_baseline.py --method iplanner
   5100    5000 /home/robopi/simulation/env_isaaclab/bin/python -m nav_arena.benchmarks.worker /tmp/run_spec.json
   6000    1000 /home/robopi/simulation/env_isaaclab/lib/python3.12/site-packages/isaacsim/kit/kit --ext-folder x
   7000    1000 grep verify_
   7100       1 /home/robopi/simulation/env_isaaclab/bin/python -m nav_arena.cli sweep sweeps/x.yaml
"""


def test_parse_ps_output_skips_header_and_bad_rows():
    processes = parse_ps_output(PS_SAMPLE + "garbage\n")
    assert processes[0].pid == 1 and processes[-1].pid == 7100
    assert len(processes) == 11


def test_unrelated_processes_are_not_simulators():
    """Verify the known false positives (daemons, pytest, an idle REPL in the Isaac env, the MCP server, grep) are ignored."""
    for command in (
        "/usr/lib/polkit-1/polkitd --no-debug",
        "/home/robopi/simulation/env_isaaclab/bin/python -m pytest nav_arena/tests/unit",
        "/home/robopi/simulation/env_isaaclab/bin/python",
        "/usr/local/bin/python /usr/local/bin/isaacsim-mcp",
        "grep verify_",
        "/home/robopi/simulation/env_isaaclab/bin/python -m nav_arena.cli sweep sweeps/x.yaml",
        "pytest tests/unit/test_verify_things.py",
    ):
        assert not is_simulator_command(command), command


def test_simulator_commands_are_detected():
    assert is_simulator_command("python -u nav_arena/nav_arena/scripts/verify_baseline.py --method iplanner")
    assert is_simulator_command("./agy_python.sh nav_arena/nav_arena/scripts/verify_task.py")
    assert is_simulator_command("python -m nav_arena.benchmarks.worker /tmp/run_spec.json")
    assert is_simulator_command("/opt/isaacsim/kit/kit --ext-folder x")


def test_find_simulator_processes_excludes_own_tree():
    processes = parse_ps_output(PS_SAMPLE)
    found = {pid for pid, _ in find_simulator_processes(processes, own_pid=7100)}
    assert found == {5000, 5100, 6000}
    # A worker started by this process (or an ancestor script) is not a conflict.
    assert {pid for pid, _ in find_simulator_processes(processes, own_pid=5000)} == {6000}
    assert {pid for pid, _ in find_simulator_processes(processes, own_pid=5100)} == {6000}


def test_related_pids_includes_ancestors_and_descendants_but_not_siblings():
    processes = parse_ps_output(PS_SAMPLE)
    related = related_pids(processes, 5000)
    assert {5000, 5100, 1000} <= related  # itself, its worker child, its (absent from the table) parent
    assert 6000 not in related and 4200 not in related  # siblings under the same parent are separate sessions
