# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Makes one upstream NavDP baseline importable, enforcing one planner per process.

The upstream baselines import their siblings with unqualified names (``from policy_agent import ...``), and
NavDP and X-NavDP both ship a module called ``policy_agent``. Loading two planners in one process would silently
resolve some imports against the wrong baseline, so a second, different planner is rejected.
"""

from __future__ import annotations

from pathlib import Path
import sys

from .checkpoints import CHECKPOINTS, baselines_root


class PlannerConflictError(RuntimeError):
    """Raised when a different baseline is already loaded in this process."""


_ACTIVE_PLANNER: str | None = None


def active_planner() -> str | None:
    """Name of the baseline loaded in this process, if any."""
    return _ACTIVE_PLANNER


def activate_planner(planner: str, navdp_root: Path | None = None) -> Path:
    """Put a baseline's source directory on ``sys.path`` and mark it as this process's planner.

    Args:
        planner: Baseline key (``iplanner``, ``vint``, ``navdp``, ``viplanner``, ``x_navdp``).
        navdp_root: NavDP checkout; defaults to :data:`nav_arena.utils.paths.NAVDP_ROOT`.

    Returns:
        The baseline's directory.

    Raises:
        PlannerConflictError: If a different baseline was activated earlier in this process.
        FileNotFoundError: If the baseline directory is missing from the NavDP checkout.
    """
    global _ACTIVE_PLANNER
    if planner not in CHECKPOINTS:
        raise KeyError(f"Unknown baseline '{planner}'. Known: {sorted(CHECKPOINTS)}")
    if _ACTIVE_PLANNER is not None and _ACTIVE_PLANNER != planner:
        raise PlannerConflictError(
            f"Baseline '{_ACTIVE_PLANNER}' is already loaded in this process; '{planner}' cannot be loaded "
            "alongside it (upstream modules share names such as 'policy_agent'). Run one planner per process."
        )
    folder = baselines_root(navdp_root) / CHECKPOINTS[planner].folder
    if not folder.is_dir():
        raise FileNotFoundError(
            f"Baseline directory not found: {folder}. Clone the NavDP fork (branch 'nav-arena') into the "
            "workspace or set NAV_ARENA_NAVDP_ROOT."
        )
    paths = [folder]
    if planner == "x_navdp":
        paths.append(folder / "eval" / "src")
    for path in paths:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    _ACTIVE_PLANNER = planner
    return folder


def _reset_active_planner_for_tests() -> None:
    """Forget the active planner (unit tests only; does not unload modules)."""
    global _ACTIVE_PLANNER
    _ACTIVE_PLANNER = None
