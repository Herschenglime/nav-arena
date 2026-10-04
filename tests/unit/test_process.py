# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for managed_process context manager."""

import os
import signal
import subprocess
import sys
import time
import pytest

from nav_arena.utils.process import managed_process


@pytest.mark.unit
def test_managed_process_normal_exit():
    """Verify that a process completing normally exits the context manager cleanly."""
    cmd = [sys.executable, "-c", "import sys; sys.exit(0)"]
    with managed_process(cmd) as proc:
        proc.wait(timeout=2.0)
        assert proc.poll() == 0


@pytest.mark.unit
def test_managed_process_terminates_running_process():
    """Verify that a process still running on context exit is terminated via SIGTERM."""
    cmd = [sys.executable, "-c", "import time; time.sleep(10)"]
    with managed_process(cmd) as proc:
        assert proc.poll() is None
        pid = proc.pid
    # Exited context, proc should be terminated
    assert proc.poll() is not None
    assert proc.returncode in (-signal.SIGTERM, signal.SIGTERM, 15, -15)


@pytest.mark.unit
def test_managed_process_kills_entire_process_group():
    """Verify that child and grandchild processes in the process group are terminated."""
    # Spawn a child script that starts a grandchild sleep and signals READY
    script = (
        "import subprocess, time, sys\n"
        "p = subprocess.Popen(['sleep', '10'])\n"
        "sys.stdout.write('READY\\n')\n"
        "sys.stdout.flush()\n"
        "time.sleep(10)\n"
    )
    cmd = [sys.executable, "-u", "-c", script]
    with managed_process(cmd, stdout=subprocess.PIPE, text=True) as proc:
        line = proc.stdout.readline()
        assert "READY" in line
        pgid = os.getpgid(proc.pid)
        assert proc.poll() is None

    # Exited context, main process should be terminated
    assert proc.poll() is not None

    # Process group should no longer exist or have live processes
    with pytest.raises(ProcessLookupError):
        os.killpg(pgid, 0)


@pytest.mark.unit
def test_managed_process_sigkill_escalation():
    """Verify escalation to SIGKILL if SIGTERM is ignored."""
    # Script that ignores SIGTERM and signals READY
    script = (
        "import signal, time, sys\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "sys.stdout.write('READY\\n')\n"
        "sys.stdout.flush()\n"
        "time.sleep(10)\n"
    )
    cmd = [sys.executable, "-u", "-c", script]
    with managed_process(cmd, timeout=0.2, stdout=subprocess.PIPE, text=True) as proc:
        line = proc.stdout.readline()
        assert "READY" in line
        start = time.time()

    elapsed = time.time() - start
    assert proc.poll() is not None
    # Must have waited around timeout before SIGKILL
    assert elapsed >= 0.18
    assert proc.returncode in (-signal.SIGKILL, signal.SIGKILL, 9, -9)



@pytest.mark.unit
def test_managed_process_popen_kwargs():
    """Verify that standard Popen arguments like stdout capture work."""
    cmd = [sys.executable, "-c", "print('hello from managed process')"]
    with managed_process(cmd, stdout=subprocess.PIPE, text=True) as proc:
        stdout, _ = proc.communicate(timeout=2.0)
        assert "hello from managed process" in stdout
