# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Characterization unit tests for InteriorAgent USD scene resolution."""

import os
import pytest

from nav_arena.scenes.interior_agent import resolve_interior_agent_usd


def test_resolve_existing_absolute_path(tmp_path):
    """Verify that an existing absolute path is resolved directly."""
    abs_file = tmp_path / "custom_stage.usd"
    abs_file.write_text("#usda 1.0\n")

    resolved = resolve_interior_agent_usd(str(abs_file))
    assert resolved == str(abs_file)


def test_resolve_usda_candidate(tmp_path):
    """Verify resolution of <base_dir>/<scene_id>/<scene_id>.usda."""
    scene_id = "kujiale_test_01"
    scene_dir = tmp_path / scene_id
    scene_dir.mkdir(parents=True)
    usda_file = scene_dir / f"{scene_id}.usda"
    usda_file.write_text("#usda 1.0\n")

    resolved = resolve_interior_agent_usd(scene_id, base_dir=str(tmp_path))
    assert resolved == str(usda_file)


def test_resolve_usd_candidate(tmp_path):
    """Verify resolution of <base_dir>/<scene_id>/<scene_id>.usd when usda is not present."""
    scene_id = "kujiale_test_02"
    scene_dir = tmp_path / scene_id
    scene_dir.mkdir(parents=True)
    usd_file = scene_dir / f"{scene_id}.usd"
    usd_file.write_text("#usd 1.0\n")

    resolved = resolve_interior_agent_usd(scene_id, base_dir=str(tmp_path))
    assert resolved == str(usd_file)


def test_resolve_usda_precedence_over_usd(tmp_path):
    """Verify that .usda takes precedence over .usd when both exist."""
    scene_id = "kujiale_test_both"
    scene_dir = tmp_path / scene_id
    scene_dir.mkdir(parents=True)
    usda_file = scene_dir / f"{scene_id}.usda"
    usd_file = scene_dir / f"{scene_id}.usd"
    usda_file.write_text("#usda 1.0\n")
    usd_file.write_text("#usd 1.0\n")

    resolved = resolve_interior_agent_usd(scene_id, base_dir=str(tmp_path))
    assert resolved == str(usda_file)


def test_resolve_direct_path_under_base_dir(tmp_path):
    """Verify direct file match under base_dir (e.g. flat scene folder)."""
    file_name = "direct_scene.usda"
    direct_file = tmp_path / file_name
    direct_file.write_text("#usda 1.0\n")

    resolved = resolve_interior_agent_usd(file_name, base_dir=str(tmp_path))
    assert resolved == str(direct_file)


def test_resolve_missing_file_raises_filenotfound(tmp_path):
    """Verify FileNotFoundError is raised with descriptive context when scene does not exist."""
    missing_scene = "nonexistent_scene_9999"
    with pytest.raises(FileNotFoundError) as exc_info:
        resolve_interior_agent_usd(missing_scene, base_dir=str(tmp_path))

    msg = str(exc_info.value)
    assert missing_scene in msg
    assert str(tmp_path) in msg


def test_resolve_nonexistent_absolute_path_raises_filenotfound(tmp_path):
    """Verify nonexistent absolute path falls through and raises FileNotFoundError."""
    nonexistent_abs = str(tmp_path / "does_not_exist" / "scene.usd")
    with pytest.raises(FileNotFoundError):
        resolve_interior_agent_usd(nonexistent_abs, base_dir=str(tmp_path))
