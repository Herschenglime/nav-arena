# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for workspace-relative path resolution in nav_arena.utils.paths."""

from __future__ import annotations

import importlib
from pathlib import Path
import re

import pytest

import nav_arena.utils.paths as paths

_PATH_ENV_VARS = (
    "NAV_ARENA_WORKSPACE",
    "NAV_ARENA_DATA_DIR",
    "NAV_ARENA_CACHE_DIR",
    "NAV_ARENA_LOG_DIR",
    "NAV_ARENA_RUNS_DIR",
    "NAV_ARENA_NAVDP_ROOT",
)


@pytest.fixture
def reload_paths(monkeypatch):
    """Reload paths with a clean environment, restoring module state afterwards."""
    for var in _PATH_ENV_VARS:
        monkeypatch.delenv(var, raising=False)

    def _reload():
        return importlib.reload(paths)

    yield _reload
    monkeypatch.undo()
    importlib.reload(paths)


def test_package_and_project_roots():
    """Verify PACKAGE_ROOT is the importable package and PROJECT_ROOT holds pyproject.toml."""
    assert (paths.PACKAGE_ROOT / "__init__.py").is_file()
    assert paths.PACKAGE_ROOT.name == "nav_arena"
    assert (paths.PROJECT_ROOT / "pyproject.toml").is_file()


def test_defaults_are_workspace_relative(reload_paths):
    """Verify every default location derives from the nav_arena checkout location."""
    p = reload_paths()
    assert p.WORKSPACE_ROOT == p.PROJECT_ROOT.parent
    assert p.DATA_DIR == p.WORKSPACE_ROOT / "data"
    assert p.CACHE_DIR == p.PROJECT_ROOT / "cache"
    assert p.LOG_DIR == p.WORKSPACE_ROOT
    assert p.RUNS_DIR == p.CACHE_DIR / "runs"
    assert p.NAVDP_ROOT == p.WORKSPACE_ROOT / "NavDP"


def test_workspace_override_propagates(reload_paths, monkeypatch, tmp_path):
    """Verify NAV_ARENA_WORKSPACE relocates every workspace-derived default."""
    monkeypatch.setenv("NAV_ARENA_WORKSPACE", str(tmp_path))
    p = reload_paths()
    assert p.WORKSPACE_ROOT == tmp_path
    assert p.DATA_DIR == tmp_path / "data"
    assert p.NAVDP_ROOT == tmp_path / "NavDP"
    # The cache belongs to the checkout, not the workspace.
    assert p.CACHE_DIR == p.PROJECT_ROOT / "cache"


def test_individual_override_wins(reload_paths, monkeypatch, tmp_path):
    """Verify a specific override takes precedence over the workspace default."""
    monkeypatch.setenv("NAV_ARENA_WORKSPACE", str(tmp_path / "ws"))
    monkeypatch.setenv("NAV_ARENA_DATA_DIR", str(tmp_path / "datasets"))
    p = reload_paths()
    assert p.DATA_DIR == tmp_path / "datasets"
    assert p.NAVDP_ROOT == tmp_path / "ws" / "NavDP"


def test_resolve_path_preserves_symlinks(tmp_path, monkeypatch):
    """Verify symlinked locations keep their link path instead of the link target."""
    target = tmp_path / "real_data"
    target.mkdir()
    link = tmp_path / "data"
    link.symlink_to(target)
    monkeypatch.setenv("NAV_ARENA_TEST_PATH", str(link / "sub" / ".." / "InteriorAgent"))
    assert paths.resolve_path("NAV_ARENA_TEST_PATH", "/unused") == link / "InteriorAgent"


def test_resolve_path_expands_user_and_ignores_empty(monkeypatch):
    """Verify resolve_path expands '~' and falls back to the default for empty values."""
    monkeypatch.setenv("NAV_ARENA_TEST_PATH", "~/somewhere")
    assert paths.resolve_path("NAV_ARENA_TEST_PATH", "/unused") == Path.home() / "somewhere"

    monkeypatch.setenv("NAV_ARENA_TEST_PATH", "")
    assert paths.resolve_path("NAV_ARENA_TEST_PATH", "/fallback") == Path("/fallback")


def test_no_hardcoded_home_paths_in_package():
    """Verify package sources never hardcode absolute home directories."""
    pattern = re.compile(r"/home/[A-Za-z0-9_.-]+/")
    offenders = []
    for path in paths.PACKAGE_ROOT.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        if path.suffix not in {".py", ".yaml", ".yml", ".rviz", ".xml", ".urdf"}:
            continue
        for lineno, line in enumerate(path.read_text(errors="ignore").splitlines(), start=1):
            if pattern.search(line):
                offenders.append(f"{path.relative_to(paths.PACKAGE_ROOT)}:{lineno}")
    assert not offenders, f"Hardcoded home paths (use nav_arena.utils.paths): {offenders}"
