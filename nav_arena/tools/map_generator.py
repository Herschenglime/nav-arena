# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Programmatic 2D occupancy map generation and offline CLI tooling for Isaac Sim scenes."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Callable

import numpy as np
from PIL import Image
import yaml

# Note: pxr and omni modules MUST only be imported after AppLauncher boots to avoid
# Boost.Python converter collisions with pre-installed site-packages.
Sdf = Usd = UsdGeom = UsdPhysics = None  # type: ignore
omni = None  # type: ignore
_omap = None  # type: ignore
compute_coordinates = generate_image = update_location = None  # type: ignore


DEFAULT_CELL_SIZE = 0.05
DEFAULT_Z_MIN = 0.01
DEFAULT_Z_MAX = 0.60
ROS_OCCUPIED_THRESHOLD = 0.65
ROS_FREE_THRESHOLD = 0.196


def _ensure_omni_modules():
    """Dynamically import Omniverse / Isaac Sim modules if not already loaded."""
    global Sdf, Usd, UsdGeom, UsdPhysics, omni, _omap, compute_coordinates, generate_image, update_location
    if omni is None or _omap is None:
        from pxr import Sdf as _Sdf, Usd as _Usd, UsdGeom as _UsdGeom, UsdPhysics as _UsdPhysics
        import omni as _omni
        import omni.usd
        import omni.timeline
        import omni.kit.app

        from isaacsim.asset.gen.omap.bindings import _omap as _omap_module
        from isaacsim.asset.gen.omap.utils import (
            compute_coordinates as _cc,
            generate_image as _gi,
            update_location as _ul,
        )

        Sdf, Usd, UsdGeom, UsdPhysics = _Sdf, _Usd, _UsdGeom, _UsdPhysics
        omni = _omni
        _omap = _omap_module
        compute_coordinates, generate_image, update_location = _cc, _gi, _ul


def prepare_mesh_collisions(scene_prim_path: str = "/World/Scene") -> int:
    """Prepare static meshes under scene_prim_path for accurate PhysX raycasting.

    Strips conflicting RigidBodyAPI schemas from static architectural assets and ensures
    all UsdGeom.Mesh prims have CollisionAPI and MeshCollisionAPI(approximation="none").

    Args:
        scene_prim_path: USD prim path of the scene asset.

    Returns:
        Number of meshes prepared with collision APIs.
    """
    _ensure_omni_modules()
    stage = omni.usd.get_context().get_stage()
    count = 0
    for prim in stage.Traverse():
        if not prim.GetPath().pathString.startswith(scene_prim_path):
            continue
        try:
            # Remove RigidBodyAPI if present on static geometry to allow static trimesh cooking
            if prim.HasAPI(UsdPhysics.CollisionAPI) and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                prim.RemoveAPI(UsdPhysics.RigidBodyAPI)

            if prim.IsA(UsdGeom.Mesh):
                points_attr = UsdGeom.Mesh(prim).GetPointsAttr().Get()
                if points_attr is None or len(points_attr) == 0:
                    continue
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI.Apply(prim)
                mc = UsdPhysics.MeshCollisionAPI.Apply(prim)
                attr = mc.GetApproximationAttr()
                if not attr.IsValid():
                    attr = mc.CreateApproximationAttr()
                attr.Set(UsdPhysics.Tokens.none)
                count += 1
        except Exception:
            continue
    return count


def compute_scene_bounds(
    scene_prim_path: str = "/World/Scene",
    bounds_prim_path: str | None = None,
    z_min: float | None = None,
    z_max: float | None = None,
    padding: float = 0.5,
) -> tuple[float, float, float, float]:
    """Compute axis-aligned XY world bounding box of scene geometry.

    If `bounds_prim_path` is given and valid, computes bounds directly on that prim.
    Otherwise, computes the union bounding box across all UsdGeom.Mesh prims under `scene_prim_path`,
    inherently ignoring lights (DomeLight/DistantLight), cameras, and non-mesh helpers.

    Args:
        scene_prim_path: Prim path to the root scene prim.
        bounds_prim_path: Optional prim path to scope bounds calculation.
        z_min: Optional minimum Z threshold to filter subterranean meshes.
        z_max: Optional maximum Z threshold to filter high ceiling meshes.
        padding: Padding in meters added around the bounding box.

    Returns:
        (min_x, min_y, max_x, max_y) bounding box in world meters.
    """
    _ensure_omni_modules()
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No active USD stage found.")

    if bounds_prim_path:
        prim = stage.GetPrimAtPath(bounds_prim_path)
        if prim.IsValid():
            imageable = UsdGeom.Imageable(prim)
            bbox = imageable.ComputeWorldBound(Usd.TimeCode.Default(), UsdGeom.Tokens.default_)
            box_range = bbox.ComputeAlignedRange()
            min_pt = box_range.GetMin()
            max_pt = box_range.GetMax()
            return (
                float(min_pt[0]) - padding,
                float(min_pt[1]) - padding,
                float(max_pt[0]) + padding,
                float(max_pt[1]) + padding,
            )
        else:
            print(
                f"[WARNING] Specified bounds_prim_path '{bounds_prim_path}' does not exist on stage. "
                "Falling back to mesh bounds..."
            )

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    min_x, min_y = float("inf"), float("inf")
    max_x, max_y = float("-inf"), float("-inf")
    mesh_found = False

    for prim in stage.Traverse():
        if not prim.GetPath().pathString.startswith(scene_prim_path):
            continue
        if not prim.IsA(UsdGeom.Mesh):
            continue

        bound = bbox_cache.ComputeWorldBound(prim)
        box_range = bound.ComputeAlignedRange()
        b_min = box_range.GetMin()
        b_max = box_range.GetMax()

        # If z slice filtering is active, skip meshes completely outside the slice
        if z_min is not None and z_max is not None:
            if b_max[2] < z_min or b_min[2] > z_max + 2.5:
                continue

        min_x = min(min_x, float(b_min[0]))
        min_y = min(min_y, float(b_min[1]))
        max_x = max(max_x, float(b_max[0]))
        max_y = max(max_y, float(b_max[1]))
        mesh_found = True

    if not mesh_found:
        raise RuntimeError(f"No valid UsdGeom.Mesh prims found under '{scene_prim_path}'.")

    return min_x - padding, min_y - padding, max_x + padding, max_y + padding


def generate_occupancy_map(
    scene_prim_path: str = "/World/Scene",
    bounds_prim_path: str | None = None,
    cell_size: float = DEFAULT_CELL_SIZE,
    z_min: float = DEFAULT_Z_MIN,
    z_max: float = DEFAULT_Z_MAX,
    output_dir: str = "nav_arena/cache/maps/default",
    warmup_steps: int = 30,
    stage_preprocessor: Callable[[Usd.Stage], None] | None = None,
) -> str:
    """Generate Nav2-compatible 2D occupancy map (map.png and map.yaml) from stage geometry.

    Args:
        scene_prim_path: USD prim path of the scene asset.
        bounds_prim_path: Optional USD prim path to calculate scene bounds from.
        cell_size: Resolution of the map in meters per pixel.
        z_min: Lower bound for Z raycasting slice in meters.
        z_max: Upper bound for Z raycasting slice in meters.
        output_dir: Directory where map.png and map.yaml will be saved.
        warmup_steps: Number of simulation steps before triggering raycast.
        stage_preprocessor: Optional scene-specific conditioning hook executed on the USD stage.

    Returns:
        Path to the generated map.yaml file.
    """
    _ensure_omni_modules()
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No active USD stage found in current context.")

    # 0. Execute scene-specific stage conditioning hook if provided
    if stage_preprocessor is not None:
        print("[INFO] Executing scene-specific stage preprocessor hook...")
        stage_preprocessor(stage)

    # Validate and ensure stage units are in meters to prevent scale distortion
    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
    if abs(meters_per_unit - 1.0) >= 1e-6:
        print(
            f"[WARNING] Stage units are not meters (metersPerUnit={meters_per_unit}). "
            "Setting stage metersPerUnit to 1.0 for meter-scale occupancy map generation."
        )
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # 1. Prepare static mesh collisions
    prepared_count = prepare_mesh_collisions(scene_prim_path)
    print(f"[INFO] Prepared {prepared_count} collision meshes under '{scene_prim_path}'")

    # 2. Compute scene bounds automatically
    min_x, min_y, max_x, max_y = compute_scene_bounds(
        scene_prim_path=scene_prim_path,
        bounds_prim_path=bounds_prim_path,
        z_min=z_min,
        z_max=z_max,
    )
    print(f"[INFO] Computed scene bounds for '{scene_prim_path}':")
    print(f"       X: [{min_x:.3f}, {max_x:.3f}], Y: [{min_y:.3f}, {max_y:.3f}] m")
    print(f"       Z raycast slice: [{z_min:.3f}, {z_max:.3f}] m, cell size: {cell_size:.3f} m")

    os.makedirs(output_dir, exist_ok=True)

    om = _omap.acquire_omap_interface()
    try:
        om.set_cell_size(cell_size)
        update_location(
            om,
            start_location=(0.0, 0.0, 0.0),
            lower_bound=(min_x, min_y, z_min),
            upper_bound=(max_x, max_y, z_max),
        )

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()

        app = omni.kit.app.get_app()
        # Warmup physics scene so all collision approximations and geometry are active
        for _ in range(warmup_steps):
            app.update()

        print("[INFO] Executing 2D occupancy raycast generation...")
        om.generate()
        app.update()

        timeline.stop()

        dims = om.get_dimensions()
        width, height = int(dims[0]), int(dims[1])
        if width <= 0 or height <= 0:
            raise RuntimeError(f"Generated occupancy map has empty dimensions: ({width}, {height})")

        print(f"[INFO] Raw occupancy map dimensions: {width} x {height} cells")

        # Generate RGBA image representation:
        # Occupied (buffer==1.0): 0 (black)
        # Unknown (buffer==0.5): 205 (light grey)
        # Freespace (buffer==0.0): 254 (white)
        rgba_flat = generate_image(
            om,
            occupied_col=[0, 0, 0, 255],
            unknown_col=[205, 205, 205, 255],
            freespace_col=[254, 254, 254, 255],
        )

        im = Image.frombytes("RGBA", (width, height), bytes(rgba_flat))
        # 180-degree rotation matches standard ROS coordinate layout and NVIDIA omap UI behavior
        im_rotated = im.rotate(180, expand=True)
        im_gray = im_rotated.convert("L")

        image_filename = "map.png"
        image_path = os.path.join(output_dir, image_filename)
        im_gray.save(image_path)
        print(f"[INFO] Saved occupancy map image to: {image_path}")

        # Compute bottom-left origin in world coordinates per NVIDIA omap specification
        # Under 180-deg rotation, bottom-left is top_right from compute_coordinates
        _top_left, top_right, _bottom_left, _bottom_right, _ = compute_coordinates(om, cell_size)
        origin_x = float(top_right[0])
        origin_y = float(top_right[1])
        origin_yaw = 0.0

        yaml_content = {
            "image": image_filename,
            "resolution": float(cell_size),
            "origin": [origin_x, origin_y, origin_yaw],
            "negate": 0,
            "occupied_thresh": ROS_OCCUPIED_THRESHOLD,
            "free_thresh": ROS_FREE_THRESHOLD,
        }

        yaml_path = os.path.join(output_dir, "map.yaml")
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f, sort_keys=False)
        print(f"[INFO] Saved occupancy map config to: {yaml_path}")

        return os.path.abspath(yaml_path)

    finally:
        _omap.release_omap_interface(om)


def get_occupancy_map(
    scene_id: str,
    scene_prim_path: str = "/World/Scene",
    bounds_prim_path: str | None = None,
    cell_size: float = DEFAULT_CELL_SIZE,
    z_min: float = DEFAULT_Z_MIN,
    z_max: float = DEFAULT_Z_MAX,
    cache_root: str = "nav_arena/cache/maps",
    output_dir: str | None = None,
    force_generate: bool = False,
    warmup_steps: int = 30,
    stage_preprocessor: Callable[[Usd.Stage], None] | None = None,
) -> str:
    """Retrieve cached occupancy map for a scene or generate it if not present.

    Args:
        scene_id: Identifier of the scene (e.g. 'kujiale_0003').
        scene_prim_path: USD prim path of the scene asset.
        bounds_prim_path: Optional USD prim path to calculate bounds from.
        cell_size: Resolution in meters per pixel.
        z_min: Lower bound for Z raycast slice.
        z_max: Upper bound for Z raycast slice.
        cache_root: Base directory for cached maps when output_dir is None.
        output_dir: Explicit output directory override. If specified, maps will be saved directly here.
        force_generate: If True, re-generate map even if cache exists.
        warmup_steps: Number of simulation steps before raycasting.
        stage_preprocessor: Optional scene-specific conditioning hook executed on the USD stage.

    Returns:
        Absolute path to the map.yaml file.
    """
    if output_dir is not None:
        target_dir = output_dir
    else:
        cache_key = f"cs{cell_size:.2f}_z{z_min:.2f}-{z_max:.2f}"
        target_dir = os.path.join(cache_root, scene_id, cache_key)

    yaml_path = os.path.join(target_dir, "map.yaml")
    png_path = os.path.join(target_dir, "map.png")

    if not force_generate and os.path.isfile(yaml_path) and os.path.isfile(png_path):
        print(f"[INFO] [CACHE HIT] Found cached occupancy map at: {yaml_path}")
        return os.path.abspath(yaml_path)

    print(f"[INFO] [CACHE MISS] Generating occupancy map for '{scene_id}' ({target_dir})...")
    generate_occupancy_map(
        scene_prim_path=scene_prim_path,
        bounds_prim_path=bounds_prim_path,
        cell_size=cell_size,
        z_min=z_min,
        z_max=z_max,
        output_dir=target_dir,
        warmup_steps=warmup_steps,
        stage_preprocessor=stage_preprocessor,
    )
    return os.path.abspath(yaml_path)


def main():
    """CLI entrypoint for offline 2D occupancy map generation."""
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(
        description="Offline 2D occupancy grid map generator for Isaac Sim scenes."
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="kujiale_0003",
        help="InteriorAgent scene ID or path to USD/USDA stage file (default: kujiale_0003).",
    )
    parser.add_argument(
        "--cell-size",
        type=float,
        default=DEFAULT_CELL_SIZE,
        help=f"Occupancy map grid resolution in meters per pixel (default: {DEFAULT_CELL_SIZE}).",
    )
    parser.add_argument(
        "--z-min",
        type=float,
        default=DEFAULT_Z_MIN,
        help=f"Lower Z bound for raycast slice in meters (default: {DEFAULT_Z_MIN}).",
    )
    parser.add_argument(
        "--z-max",
        type=float,
        default=DEFAULT_Z_MAX,
        help=f"Upper Z bound for raycast slice in meters (default: {DEFAULT_Z_MAX}).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory to save map.yaml and map.png (defaults to nav_arena/cache/maps/<scene_id>/<cache_key>).",
    )
    parser.add_argument(
        "--cache-root",
        type=str,
        default="nav_arena/cache/maps",
        help="Base cache directory when --output-dir is not specified (default: nav_arena/cache/maps).",
    )
    parser.add_argument(
        "--bounds-prim",
        type=str,
        default=None,
        help="Optional USD prim path to constrain bounds calculation.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=30,
        help="Physics warmup steps before raycasting (default: 30).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force map generation even if cached map already exists.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    # Mandatory boot-time extension flag for Isaac Sim Occupancy Map
    sys.argv.extend([
        "--enable", "isaacsim.asset.gen.omap",
        "--enable", "isaacsim.asset.gen.omap.ui",
    ])

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    try:
        _ensure_omni_modules()

        # Resolve scene USD path
        stage_preprocessor = None
        try:
            from nav_arena.scenes.interior_agent import get_open_door_usd, prepare_interior_agent_stage

            usd_path = get_open_door_usd(args_cli.scene)
            stage_preprocessor = prepare_interior_agent_stage
            scene_id = os.path.splitext(os.path.basename(usd_path))[0]
            if scene_id.endswith("_open_doors"):
                scene_id = scene_id[:-len("_open_doors")]
        except Exception:
            usd_path = os.path.abspath(args_cli.scene) if os.path.isfile(args_cli.scene) else args_cli.scene
            scene_id = os.path.splitext(os.path.basename(usd_path))[0]

        print(f"[INFO] Opening USD stage: {usd_path}")
        omni.usd.get_context().open_stage(usd_path)
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError(f"Failed to open stage at: {usd_path}")

        # Ensure physicsScene exists
        if (
            not stage.GetPrimAtPath("/World/physicsScene").IsValid()
            and not stage.GetPrimAtPath("/physicsScene").IsValid()
        ):
            UsdPhysics.Scene.Define(stage, Sdf.Path("/World/physicsScene"))

        if stage.GetPrimAtPath("/Root").IsValid():
            scene_prim = "/Root"
        elif stage.GetPrimAtPath("/World/Scene").IsValid():
            scene_prim = "/World/Scene"
        elif stage.GetDefaultPrim().IsValid():
            scene_prim = stage.GetDefaultPrim().GetPath().pathString
        else:
            scene_prim = "/"

        yaml_path = get_occupancy_map(
            scene_id=scene_id,
            scene_prim_path=scene_prim,
            bounds_prim_path=args_cli.bounds_prim,
            cell_size=args_cli.cell_size,
            z_min=args_cli.z_min,
            z_max=args_cli.z_max,
            cache_root=args_cli.cache_root,
            output_dir=args_cli.output_dir,
            force_generate=args_cli.force,
            warmup_steps=args_cli.warmup_steps,
            stage_preprocessor=stage_preprocessor,
        )

        print("=" * 70)
        print(f"[SUCCESS] Occupancy map generated successfully: {yaml_path}")
        print("=" * 70)
    except Exception as e:
        import traceback

        print(f"[ERROR] Exception during occupancy map generation: {e}", file=sys.stderr)
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
