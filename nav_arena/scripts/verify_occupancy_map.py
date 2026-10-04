# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Verification script for programmatic 2D occupancy map generation and caching."""

from __future__ import annotations

import argparse
import os
import sys
import numpy as np
from PIL import Image
import yaml

from isaaclab.app import AppLauncher

# Parse CLI arguments
parser = argparse.ArgumentParser(description="Verify 2D occupancy map generation for InteriorAgent scenes.")
parser.add_argument("--scene", type=str, default="kujiale_0003", help="InteriorAgent scene ID or USD path.")
parser.add_argument("--cell-size", type=float, default=0.05, help="Occupancy map grid resolution (meters/pixel).")
parser.add_argument("--output-dir", type=str, default="nav_arena/cache/maps", help="Base directory for cached maps.")
parser.add_argument("--bounds-prim", type=str, default=None, help="Optional prim path to scope bounds calculation.")
parser.add_argument("--force", action="store_true", help="Force regeneration even if cached map exists.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Mandatory boot-time extension flag for Isaac Sim Occupancy Map
sys.argv.extend([
    "--enable", "isaacsim.asset.gen.omap",
    "--enable", "isaacsim.asset.gen.omap.ui",
])

# Launch Omniverse application
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.usd
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils.configclass import configclass

from nav_arena.embodiments.nova_carter import NovaCarterEmbodimentCfg
from nav_arena.tools.map_generator import generate_occupancy_map, get_occupancy_map
from nav_arena.scenes.interior_agent import get_open_door_usd, prepare_interior_agent_stage, resolve_interior_agent_usd


@configclass
class ArchitecturalSceneCfg(InteractiveSceneCfg):
    """Scene containing purely the static architectural geometry without robot embodiment."""

    env_spacing: float = 30.0
    scene_asset: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/Scene",
        spawn=sim_utils.UsdFileCfg(
            usd_path="",  # Populated dynamically
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
    )


def verify_occupancy_map():
    print("=" * 70)
    print(f"[INFO] STARTING OCCUPANCY MAP VERIFICATION FOR SCENE: '{args_cli.scene}'")
    print("=" * 70)

    # 1. Resolve USD path with open doors delta layer
    usd_path = get_open_door_usd(args_cli.scene)
    print(f"[INFO] Resolved USD path: {usd_path}")


    # 2. Derive z-bounds from Nova Carter embodiment sensor height
    embodiment = NovaCarterEmbodimentCfg()
    z_min = 0.01  # Above ground plane
    z_max = embodiment.sensor_height + 0.25  # 0.35 + 0.25 = 0.60 m
    print(f"[INFO] Derived Z-bounds from '{embodiment.name}': z_min={z_min:.2f} m, z_max={z_max:.2f} m")

    # 3. Load stage and ensure physicsScene exists
    print(f"[INFO] Opening stage for '{args_cli.scene}'...")
    omni.usd.get_context().open_stage(usd_path)
    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath("/World/physicsScene").IsValid() and not stage.GetPrimAtPath("/physicsScene").IsValid():
        from pxr import Sdf, UsdPhysics
        UsdPhysics.Scene.Define(stage, Sdf.Path("/World/physicsScene"))

    scene_prim = "/Root" if stage.GetPrimAtPath("/Root").IsValid() else "/World/Scene"

    # 4. Generate or retrieve map
    yaml_path = get_occupancy_map(
        scene_id=args_cli.scene,
        scene_prim_path=scene_prim,
        bounds_prim_path=args_cli.bounds_prim,
        cell_size=args_cli.cell_size,
        z_min=z_min,
        z_max=z_max,
        cache_root=args_cli.output_dir,
        force_generate=args_cli.force,
        stage_preprocessor=prepare_interior_agent_stage,
    )


    # 5. Sanity checks on output files
    print("\n[INFO] Running sanity checks on generated outputs...")
    assert os.path.isfile(yaml_path), f"Expected map.yaml at {yaml_path}, but file not found."
    map_dir = os.path.dirname(yaml_path)
    png_path = os.path.join(map_dir, "map.png")
    assert os.path.isfile(png_path), f"Expected map.png at {png_path}, but file not found."
    print("[SUCCESS] Found both map.yaml and map.png on disk.")

    # 6. Validate YAML metadata
    with open(yaml_path) as f:
        meta = yaml.safe_load(f)

    assert meta.get("image") == "map.png", f"YAML 'image' should be 'map.png', got: {meta.get('image')}"
    assert abs(meta.get("resolution", 0) - args_cli.cell_size) < 1e-4, (
        f"YAML 'resolution' mismatch: {meta.get('resolution')} vs expected {args_cli.cell_size}"
    )
    origin = meta.get("origin")
    assert isinstance(origin, list) and len(origin) == 3, f"YAML 'origin' must be 3-element list, got: {origin}"
    assert meta.get("occupied_thresh") == 0.65, f"Unexpected occupied_thresh: {meta.get('occupied_thresh')}"
    assert meta.get("free_thresh") == 0.196, f"Unexpected free_thresh: {meta.get('free_thresh')}"
    print(f"[SUCCESS] YAML metadata valid: resolution={meta['resolution']}, origin={meta['origin']}")

    # 7. Validate PNG image contents
    img = Image.open(png_path)
    img_arr = np.array(img)
    height, width = img_arr.shape
    total_pixels = width * height
    print(f"[INFO] Map image dimensions: {width} x {height} ({total_pixels:,} total cells)")

    assert width > 10 and height > 10, f"Map dimensions suspiciously small: {width} x {height}"

    occupied_count = int(np.sum(img_arr == 0))
    free_count = int(np.sum(img_arr == 254))
    unknown_count = int(np.sum(img_arr == 205))

    occ_pct = (occupied_count / total_pixels) * 100
    free_pct = (free_count / total_pixels) * 100
    unk_pct = (unknown_count / total_pixels) * 100

    print(f"[INFO] Pixel distribution:")
    print(f"       Occupied (walls):     {occupied_count:8,d} ({occ_pct:5.2f}%)")
    print(f"       Freespace (walkable): {free_count:8,d} ({free_pct:5.2f}%)")
    print(f"       Unknown (exterior):   {unknown_count:8,d} ({unk_pct:5.2f}%)")

    assert occupied_count > 0, "Occupancy map contains 0 occupied pixels (no walls detected)!"
    assert free_count > 0, "Occupancy map contains 0 freespace pixels (no walkable floor detected)!"
    print("[SUCCESS] Map contains valid distribution of occupied, freespace, and unknown cells.")

    # 8. Test cache hit verification
    print("\n[INFO] Testing cache hit behavior...")
    second_yaml = get_occupancy_map(
        scene_id=args_cli.scene,
        scene_prim_path=scene_prim,
        bounds_prim_path=args_cli.bounds_prim,
        cell_size=args_cli.cell_size,
        z_min=z_min,
        z_max=z_max,
        cache_root=args_cli.output_dir,
        force_generate=False,
    )

    assert second_yaml == yaml_path, f"Cache returned different path: {second_yaml} vs {yaml_path}"
    print("[SUCCESS] Caching logic successfully re-used existing map.")

    print("\n" + "=" * 70)
    print("PHASE 1 OCCUPANCY MAP GENERATION VERIFICATION PASSED SUCCESSFULLY!")
    print("=" * 70)


def main():
    try:
        verify_occupancy_map()
    except Exception as e:
        import traceback

        print("[ERROR] Exception occurred during verification:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
