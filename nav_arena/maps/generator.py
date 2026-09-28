# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Programmatic 2D occupancy map generation and caching for Isaac Sim scenes."""

from __future__ import annotations

import os
import numpy as np
from PIL import Image
import yaml

from pxr import Sdf, Usd, UsdGeom, UsdPhysics
import omni.usd
import omni.timeline
import omni.kit.app

from isaacsim.asset.gen.omap.bindings import _omap
from isaacsim.asset.gen.omap.utils import compute_coordinates, generate_image, update_location


DEFAULT_CELL_SIZE = 0.05
DEFAULT_Z_MIN = 0.01
DEFAULT_Z_MAX = 0.60
ROS_OCCUPIED_THRESHOLD = 0.65
ROS_FREE_THRESHOLD = 0.196


def prepare_mesh_collisions(scene_prim_path: str = "/World/Scene") -> int:
    """Prepare static meshes under scene_prim_path for accurate PhysX raycasting.

    Strips conflicting RigidBodyAPI schemas from static architectural assets and ensures
    all UsdGeom.Mesh prims have CollisionAPI and MeshCollisionAPI(approximation="none").

    Args:
        scene_prim_path: USD prim path of the scene asset.

    Returns:
        Number of meshes prepared with collision APIs.
    """
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

    Returns:
        Path to the generated map.yaml file.
    """
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No active USD stage found in current context.")

    # Validate stage units are in meters to prevent scale distortion
    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
    assert abs(meters_per_unit - 1.0) < 1e-6, (
        f"Stage units are not meters (metersPerUnit={meters_per_unit}). "
        "Occupancy map generation requires meter-scale stages."
    )

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
    force_generate: bool = False,
    warmup_steps: int = 30,
) -> str:
    """Retrieve cached occupancy map for a scene or generate it if not present.

    Args:
        scene_id: Identifier of the scene (e.g. 'kujiale_0003').
        scene_prim_path: USD prim path of the scene asset.
        bounds_prim_path: Optional USD prim path to calculate bounds from.
        cell_size: Resolution in meters per pixel.
        z_min: Lower bound for Z raycast slice.
        z_max: Upper bound for Z raycast slice.
        cache_root: Base directory for cached maps.
        force_generate: If True, re-generate map even if cache exists.
        warmup_steps: Number of simulation steps before raycasting.

    Returns:
        Absolute path to the map.yaml file.
    """
    cache_key = f"cs{cell_size:.2f}_z{z_min:.2f}-{z_max:.2f}"
    cache_dir = os.path.join(cache_root, scene_id, cache_key)
    yaml_path = os.path.join(cache_dir, "map.yaml")
    png_path = os.path.join(cache_dir, "map.png")

    if not force_generate and os.path.isfile(yaml_path) and os.path.isfile(png_path):
        print(f"[INFO] [CACHE HIT] Found cached occupancy map at: {yaml_path}")
        return os.path.abspath(yaml_path)

    print(f"[INFO] [CACHE MISS] Generating occupancy map for '{scene_id}' ({cache_key})...")
    generate_occupancy_map(
        scene_prim_path=scene_prim_path,
        bounds_prim_path=bounds_prim_path,
        cell_size=cell_size,
        z_min=z_min,
        z_max=z_max,
        output_dir=cache_dir,
        warmup_steps=warmup_steps,
    )
    return os.path.abspath(yaml_path)
