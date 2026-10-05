# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure command-line argument checks shared by scripts.

Scripts boot Isaac Sim while importing, so validation that should be unit tested and should fail *before* a slow boot
lives here, with no Isaac imports.
"""

from __future__ import annotations

from collections.abc import Sequence


def validate_route_args(
    route: str | None,
    spawn: Sequence[float] | None,
    goal: Sequence[float] | None,
) -> None:
    """Reject route selections that the runner would otherwise silently ignore.

    A start/goal is either a named route (``--route``, or the scene default when nothing is given) or a custom pair
    (``--spawn`` and ``--goal`` together). ``--spawn-yaw`` is valid with either, so it is not checked here.

    Args:
        route: Value of ``--route`` (``None`` when not given).
        spawn: Value of ``--spawn`` (``None`` when not given).
        goal: Value of ``--goal`` (``None`` when not given).

    Raises:
        ValueError: If only one of ``spawn``/``goal`` is given, or a route is combined with ``spawn``/``goal``.
    """
    if (spawn is None) != (goal is None):
        given, missing = ("--spawn", "--goal") if spawn is not None else ("--goal", "--spawn")
        raise ValueError(f"{given} needs {missing} too: a custom start/goal is a pair (it would otherwise be ignored).")
    if route is not None and spawn is not None:
        raise ValueError(
            f"--route '{route}' cannot be combined with --spawn/--goal: choose a named route or a custom start/goal."
        )
