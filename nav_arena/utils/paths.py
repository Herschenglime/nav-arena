# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Workspace-relative filesystem locations for assets, datasets, caches, and logs.

Every default resolves relative to the nav_arena checkout (an editable install), so the
workspace can move without code changes. Each location can be overridden individually
through an environment variable.

Default layout::

    <workspace>/                  NAV_ARENA_WORKSPACE (parent of the nav_arena checkout)
    ├── data/                     NAV_ARENA_DATA_DIR
    ├── NavDP/                    NAV_ARENA_NAVDP_ROOT
    └── nav_arena/                PROJECT_ROOT
        └── cache/                NAV_ARENA_CACHE_DIR
"""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
"""The importable ``nav_arena`` package directory (``<workspace>/nav_arena/nav_arena``)."""

PROJECT_ROOT = PACKAGE_ROOT.parent
"""The nav_arena project checkout containing ``pyproject.toml`` (``<workspace>/nav_arena``)."""


def resolve_path(env_var: str, default: str | os.PathLike[str]) -> Path:
    """Resolve a filesystem location, preferring an environment variable override.

    Args:
        env_var: Name of the environment variable that overrides the default.
        default: Path used when the environment variable is unset or empty.

    Returns:
        Absolute, normalized path with ``~`` expanded. Symlinks are preserved rather than
        resolved, so a symlinked dataset directory keeps its workspace-relative path.
    """
    value = os.environ.get(env_var) or default
    return Path(os.path.abspath(Path(value).expanduser()))


WORKSPACE_ROOT = resolve_path("NAV_ARENA_WORKSPACE", PROJECT_ROOT.parent)
"""Simulation workspace holding nav_arena alongside datasets and vendor checkouts."""

DATA_DIR = resolve_path("NAV_ARENA_DATA_DIR", WORKSPACE_ROOT / "data")
"""Root directory for downloaded datasets (e.g. ``InteriorAgent``)."""

CACHE_DIR = resolve_path("NAV_ARENA_CACHE_DIR", PROJECT_ROOT / "cache")
"""Generated runtime caches (git-ignored): conditioned scenes, occupancy maps."""

LOG_DIR = resolve_path("NAV_ARENA_LOG_DIR", WORKSPACE_ROOT)
"""Destination for external process logs (e.g. Nav2 bringup)."""

NAVDP_ROOT = resolve_path("NAV_ARENA_NAVDP_ROOT", WORKSPACE_ROOT / "NavDP")
"""NavDP checkout providing the Dingo asset and learned baseline networks."""
