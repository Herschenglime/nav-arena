# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Offline developer tools for nav_arena."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .map_generator import generate_occupancy_map, get_occupancy_map

__all__ = ["generate_occupancy_map", "get_occupancy_map"]


def __getattr__(name: str):
    if name in __all__:
        from . import map_generator
        return getattr(map_generator, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
