# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Subprocess management helpers ensuring clean process group teardown."""

from __future__ import annotations

from contextlib import contextmanager
import os
import signal
import subprocess
from typing import Any, Generator, Sequence


@contextmanager
def managed_process(
    cmd: Sequence[str] | str,
    timeout: float = 5.0,
    **popen_kwargs: Any,
) -> Generator[subprocess.Popen, None, None]:
    """Launch and manage a child subprocess in a dedicated process group.

    Guarantees that on context exit, all child and grandchild processes in the group
    receive SIGTERM, followed by SIGKILL if they do not terminate within the timeout.

    Args:
        cmd: Command sequence or string to execute.
        timeout: Maximum seconds to wait after SIGTERM before escalating to SIGKILL.
        **popen_kwargs: Additional keyword arguments passed directly to subprocess.Popen.

    Yields:
        The spawned subprocess.Popen instance.
    """
    popen_kwargs.setdefault("start_new_session", True)
    proc = subprocess.Popen(cmd, **popen_kwargs)
    try:
        yield proc
    finally:
        if proc.poll() is None:
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGTERM)
                proc.wait(timeout=timeout)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                try:
                    pgid = os.getpgid(proc.pid)
                    os.killpg(pgid, signal.SIGKILL)
                    proc.wait(timeout=timeout)
                except (ProcessLookupError, OSError, subprocess.TimeoutExpired):
                    pass
            except OSError:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=timeout)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        try:
                            proc.wait(timeout=timeout)
                        except (ProcessLookupError, OSError, subprocess.TimeoutExpired):
                            pass

