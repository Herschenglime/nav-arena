# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for launch_simulation_app exit-status handling (fake AppLauncher; no Kit boot)."""

from __future__ import annotations

import argparse
import sys

import pytest

from nav_arena.core import launch_simulation_app


class FakeApp:
    def __init__(self):
        self.closed_with = []

    def close(self, **kwargs):
        self.closed_with.append(kwargs)


@pytest.fixture
def fake_app(monkeypatch):
    """Replace AppLauncher with a fake whose app records how close() was called."""
    import isaaclab.app

    app = FakeApp()

    class FakeLauncher:
        def __init__(self, args_cli=None, **kwargs):
            self.app = app

    monkeypatch.setattr(isaaclab.app, "AppLauncher", FakeLauncher)
    saved = list(sys.argv)
    yield app
    sys.argv[:] = saved


def _launch():
    return launch_simulation_app(argparse.Namespace(), enable_ros2=False, livestream=False)


def test_normal_exit_closes_with_status_zero(fake_app):
    """Verify a clean block closes the app with exit_code 0."""
    with _launch() as app:
        assert app is fake_app
    assert fake_app.closed_with == [{"exit_code": 0}]


def test_exception_closes_with_failure_status_and_propagates(fake_app):
    """Verify an exception closes with exit_code 1 (close() ends the process) and still propagates."""
    with pytest.raises(RuntimeError, match="boom"):
        with _launch():
            raise RuntimeError("boom")
    assert fake_app.closed_with == [{"exit_code": 1}]


@pytest.mark.parametrize("code,expected", [(2, 2), (0, 0), (None, 0), ("failed", 1)])
def test_sys_exit_status_is_preserved(fake_app, code, expected):
    """Verify sys.exit(n) inside the block is passed through as the process exit status."""
    with pytest.raises(SystemExit):
        with _launch():
            sys.exit(code)
    assert fake_app.closed_with == [{"exit_code": expected}]


def test_keyboard_interrupt_is_a_failure(fake_app):
    """Verify Ctrl+C inside the block closes with a nonzero status."""
    with pytest.raises(KeyboardInterrupt):
        with _launch():
            raise KeyboardInterrupt
    assert fake_app.closed_with == [{"exit_code": 1}]

