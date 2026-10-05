# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation / trajectory sanitizers shared by the NavDP-family adapters.

Ported from the NavDP Isaac Sim integration (``isaac_policies.py``).
"""

from __future__ import annotations

import numpy as np


def as_path(value) -> np.ndarray:
    """Validate and convert a planner trajectory ``[1, T, >=2]`` (array or tensor) to ``[T, >=2]`` float32.

    Raises:
        ValueError: If the trajectory is not a single nonempty finite path.
    """
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    path = np.asarray(value, dtype=np.float32)
    if path.ndim != 3 or path.shape[0] != 1 or path.shape[1] == 0 or path.shape[2] < 2:
        raise ValueError(f"Expected a single nonempty [1,T,2+] trajectory, got {path.shape}")
    if not np.isfinite(path).all():
        raise ValueError("Planner returned a nonfinite trajectory")
    return path[0]


def prepare_observation(rgb, depth, *, sanitize_depth: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Batch an RGB-D observation for the upstream agents.

    Args:
        rgb: uint8 ``[H, W, 3]`` image.
        depth: Metric image-plane depth ``[H, W]`` aligned with ``rgb``.
        sanitize_depth: Replace non-finite / non-positive depth with 0. iPlanner and VIPlanner mask invalid values
            *after* resizing, so they receive the raw depth.

    Returns:
        ``(images [1,H,W,3] uint8, depths [1,H,W,1] float32)``. The depth buffer is copied: upstream preprocessing
        writes into it, and the simulator's buffer must not be modified.

    Raises:
        ValueError: On wrong dtype or shape.
    """
    rgb = np.asarray(rgb)
    depth = np.asarray(depth, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Expected RGB uint8 [H,W,3]")
    if depth.shape != rgb.shape[:2]:
        raise ValueError("Depth must be [H,W], aligned with RGB")
    depth = (
        np.where(np.isfinite(depth) & (depth > 0), depth, 0).astype(np.float32)
        if sanitize_depth
        else depth.copy()
    )
    return np.ascontiguousarray(rgb[None]), depth[None, :, :, None]
