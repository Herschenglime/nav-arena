# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Occupancy map generation and caching utilities for nav_arena."""

from .generator import generate_occupancy_map, get_occupancy_map

__all__ = ["generate_occupancy_map", "get_occupancy_map"]
