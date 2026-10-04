# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scene loaders and environment definitions."""

from typing import TYPE_CHECKING

from .routes import DEFAULT_ROUTE, INTERIOR_AGENT_ROUTES, Route, get_route, list_routes

if TYPE_CHECKING:
    from .interior_agent import (
        DEFAULT_INTERIOR_AGENT_CACHE_DIR,
        DEFAULT_INTERIOR_AGENT_DIR,
        DEFAULT_INTERIOR_AGENT_SCENE_ID,
        DEFAULT_INTERIOR_AGENT_USD,
        InteriorAgentSceneCfg,
        create_interior_agent_scene_cfg,
        get_open_door_usd,
        get_preprocessed_usd,
        prepare_interior_agent_stage,
        resolve_interior_agent_usd,
    )

_INTERIOR_AGENT_EXPORTS = {
    "DEFAULT_INTERIOR_AGENT_CACHE_DIR",
    "DEFAULT_INTERIOR_AGENT_DIR",
    "DEFAULT_INTERIOR_AGENT_SCENE_ID",
    "DEFAULT_INTERIOR_AGENT_USD",
    "InteriorAgentSceneCfg",
    "create_interior_agent_scene_cfg",
    "get_open_door_usd",
    "get_preprocessed_usd",
    "prepare_interior_agent_stage",
    "resolve_interior_agent_usd",
}

__all__ = [
    "DEFAULT_ROUTE",
    "INTERIOR_AGENT_ROUTES",
    "Route",
    "get_route",
    "list_routes",
    *_INTERIOR_AGENT_EXPORTS,
]


def __getattr__(name: str):
    if name in _INTERIOR_AGENT_EXPORTS:
        from . import interior_agent

        return getattr(interior_agent, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)


