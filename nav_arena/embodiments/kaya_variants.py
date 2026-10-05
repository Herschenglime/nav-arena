# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kaya's variants: the single source of their data.

Pure data (no torch or Isaac Lab), so the light registry can register Kaya with its complete variants without
importing the embodiment. ``kaya.py`` reuses these constants for its defaults.
"""

from __future__ import annotations

import math

from .base import EmbodimentVariant

# Camera rotation quaternions (x, y, z, w) in Isaac Lab's world convention (+X forward, +Y left, +Z up).
ROT_LEVEL = (0.0, 0.0, 0.0, 1.0)
"""Level camera facing +x."""
ROT_PITCHED_DOWN_20 = (0.0, math.sin(math.radians(10.0)), 0.0, math.cos(math.radians(10.0)))
"""Pitched 20 degrees down: a positive rotation about +Y tips the optical axis toward the ground."""

BASE_LINK_HEIGHT = 0.091
"""Height of ``base_link``'s origin above the floor when Kaya rests on its wheels (from the USD)."""

MAST_CAMERA_OFFSET = (0.0, 0.0, 0.30 - BASE_LINK_HEIGHT)
"""Virtual mast: camera 0.30 m above the floor."""
NATIVE_CAMERA_OFFSET = (0.06, 0.0, 0.07)
"""The RealSense lens as mounted in the USD: about 0.16 m above the floor."""

KAYA_MAST = EmbodimentVariant(
    name="mast",
    description="Virtual camera mast at 0.30 m (same viewpoint as Nova Carter / Dingo).",
    overrides={"camera_offset": MAST_CAMERA_OFFSET, "camera_rot": ROT_LEVEL},
)
KAYA_NATIVE = EmbodimentVariant(
    name="native",
    description="Native RealSense D435 pose from the Kaya USD: ~0.16 m high, pitched 20° down.",
    overrides={"camera_offset": NATIVE_CAMERA_OFFSET, "camera_rot": ROT_PITCHED_DOWN_20},
)
KAYA_VARIANTS = (KAYA_MAST, KAYA_NATIVE)
KAYA_DEFAULT_VARIANT = "mast"
