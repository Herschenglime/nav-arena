# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for derived robot USD assets (CPU-only; builds layers in a subprocess)."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from nav_arena.embodiments.assets import derived_asset_path, ensure_derived_asset
from nav_arena.utils.paths import NAVDP_ROOT

pxr = pytest.importorskip("pxr")


def _write_source(path, with_caster=True, meters_per_unit=1.0):
    """Write a minimal Dingo-like USD layer (ground plane + caster material under the default prim)."""
    caster = '    def Material "caster_wheel" {}\n' if with_caster else ""
    path.write_text(
        f'#usda 1.0\n(\n    defaultPrim = "dingo"\n    metersPerUnit = {meters_per_unit}\n    upAxis = "Z"\n)\n'
        'def Xform "dingo"\n{\n    def Xform "GroundPlane" {}\n'
        f'    def Scope "PhysicsMaterials"\n    {{\n{caster}    }}\n}}\n'
    )
    return path


def _compose(path):
    """Open ``path`` as a stage and return it."""
    from pxr import Usd

    return Usd.Stage.Open(str(path))


def test_derived_dingo_asset_bakes_in_both_fixes(tmp_path):
    """Verify the override deactivates the ground plane and sets the caster friction combine mode to 'min'."""
    source = _write_source(tmp_path / "dingo.usda")
    derived = ensure_derived_asset("dingo", source, tmp_path / "cache")

    stage = _compose(derived)
    assert not stage.GetPrimAtPath("/dingo/GroundPlane").IsActive()
    caster = stage.GetPrimAtPath("/dingo/PhysicsMaterials/caster_wheel")
    assert caster.GetAttribute("physxMaterial:frictionCombineMode").Get() == "min"
    assert "PhysxMaterialAPI" in caster.GetMetadata("apiSchemas").GetAddedOrExplicitItems()
    # The upstream file is untouched.
    assert _compose(source).GetPrimAtPath("/dingo/GroundPlane").IsActive()


def test_derived_asset_preserves_stage_units_and_default_prim(tmp_path):
    """Verify unit scale, up axis and default prim carry over; losing metersPerUnit would silently rescale the robot."""
    source = _write_source(tmp_path / "dingo.usda", meters_per_unit=0.01)
    stage = _compose(ensure_derived_asset("dingo", source, tmp_path / "cache"))
    assert stage.GetMetadata("metersPerUnit") == pytest.approx(0.01)
    assert stage.GetMetadata("upAxis") == "Z"
    assert stage.GetDefaultPrim().GetPath().pathString == "/dingo"


def test_derived_asset_is_cached_and_rebuilt_when_the_source_changes(tmp_path):
    """Verify a second call reuses the file, and changing the source yields a new hash-named asset."""
    source = _write_source(tmp_path / "dingo.usda")
    first = ensure_derived_asset("dingo", source, tmp_path / "cache")
    mtime = first.stat().st_mtime_ns
    assert ensure_derived_asset("dingo", source, tmp_path / "cache") == first
    assert first.stat().st_mtime_ns == mtime
    assert first == derived_asset_path("dingo", source, tmp_path / "cache")

    _write_source(source, meters_per_unit=0.5)
    second = ensure_derived_asset("dingo", source, tmp_path / "cache")
    assert second != first


def test_derived_asset_fails_loudly_when_the_upstream_layout_changes(tmp_path):
    """Verify a source without the caster material raises instead of producing a silently unfixed robot."""
    source = _write_source(tmp_path / "dingo.usda", with_caster=False)
    with pytest.raises(RuntimeError, match="caster_wheel"):
        ensure_derived_asset("dingo", source, tmp_path / "cache")
    assert not list((tmp_path / "cache" / "assets").glob("*.usda"))


def test_derived_asset_missing_source(tmp_path):
    """Verify a missing upstream file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        ensure_derived_asset("dingo", tmp_path / "nope.usd", tmp_path / "cache")


@pytest.mark.skipif(not (NAVDP_ROOT / "assets/robots/dingo.usd").is_file(), reason="NavDP Dingo asset not installed")
def test_real_dingo_asset_composes_with_the_fixes(tmp_path):
    """Verify the override applies to the real upstream Dingo asset and keeps its rigid-body hierarchy."""
    source = NAVDP_ROOT / "assets/robots/dingo.usd"
    stage = _compose(ensure_derived_asset("dingo", source, tmp_path / "cache"))
    assert not stage.GetPrimAtPath("/dingo/GroundPlane").IsActive()
    caster = stage.GetPrimAtPath("/dingo/PhysicsMaterials/caster_wheel")
    assert caster.GetAttribute("physxMaterial:frictionCombineMode").Get() == "min"
    assert stage.GetPrimAtPath("/dingo/base_link/collisions/rear_caster_wheel").IsValid()
    original = _compose(source)
    assert stage.GetMetadata("metersPerUnit") == original.GetMetadata("metersPerUnit")
    assert stage.GetMetadata("upAxis") == original.GetMetadata("upAxis")


def test_import_and_nova_carter_succeed_when_ensure_derived_asset_fails(tmp_path):
    """Verify importing embodiments and getting 'nova_carter' succeeds even if ensure_derived_asset fails."""
    import os

    source = _write_source(tmp_path / "dingo_source.usda")
    code = (
        "import sys, types\n"
        "mock_assets = types.ModuleType('nav_arena.embodiments.assets')\n"
        "call_count = 0\n"
        "def _fail(*args, **kwargs):\n"
        "    global call_count\n"
        "    call_count += 1\n"
        "    raise RuntimeError('simulated build failure')\n"
        "mock_assets.ensure_derived_asset = _fail\n"
        "mock_assets.derived_asset_path = lambda *args, **kwargs: None\n"
        "sys.modules['nav_arena.embodiments.assets'] = mock_assets\n"
        "import nav_arena.embodiments as embodiments\n"
        "# 1. Importing embodiments must not call ensure_derived_asset\n"
        "assert call_count == 0\n"
        "# 2. Getting nova_carter must not call ensure_derived_asset\n"
        "carter = embodiments.get_embodiment('nova_carter')\n"
        "assert carter.name == 'nova_carter'\n"
        "assert call_count == 0\n"
        "# 3. Getting dingo must invoke ensure_derived_asset and raise\n"
        "try:\n"
        "    embodiments.get_embodiment('dingo')\n"
        "except RuntimeError as e:\n"
        "    assert 'simulated build failure' in str(e)\n"
        "else:\n"
        "    raise AssertionError('Expected get_embodiment(dingo) to fail with RuntimeError')\n"
        "assert call_count == 1\n"
    )
    env = os.environ.copy()
    env["NAV_ARENA_DINGO_USD"] = str(source)
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert result.returncode == 0, f"Subprocess failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"


def test_get_embodiment_dingo_resolves_asset_on_demand(tmp_path, monkeypatch):
    """Verify that get_embodiment('dingo') builds and resolves the derived USD on demand."""
    import nav_arena.embodiments.dingo as dingo_mod
    from nav_arena.embodiments import get_embodiment

    source = _write_source(tmp_path / "dingo_test.usda")
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(dingo_mod, "DINGO_SOURCE_USD_PATH", source)
    monkeypatch.setattr(dingo_mod, "CACHE_DIR", cache_dir)

    # Cache should not have assets directory before resolution
    assert not (cache_dir / "assets").exists()

    cfg = get_embodiment("dingo")
    usd_path = Path(cfg.articulation_cfg.spawn.usd_path)
    assert usd_path.is_file()
    assert usd_path.parent == cache_dir / "assets"
    assert usd_path.name.startswith("dingo_")


def test_create_dingo_articulation_cfg_explicit_path():
    """Verify create_dingo_articulation_cfg accepts an explicit usd_path override."""
    from nav_arena.embodiments import create_dingo_articulation_cfg

    cfg = create_dingo_articulation_cfg(usd_path="/custom/path/dingo.usd")
    assert cfg.spawn.usd_path == "/custom/path/dingo.usd"

