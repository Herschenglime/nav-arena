# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for derived robot USD assets (CPU-only; builds layers in a subprocess)."""

from __future__ import annotations

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
