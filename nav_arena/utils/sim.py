# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simulation environment utilities and mock helpers."""

from __future__ import annotations

import types
from typing import Any


def create_mock_env(scene: Any, sim: Any) -> types.SimpleNamespace:
    """Create a mock environment wrapper for ActionManager and components requiring env context.

    Args:
        scene: InteractiveScene instance or mock with num_envs attribute.
        sim: SimulationContext instance or mock with device and get_physics_dt() method.

    Returns:
        SimpleNamespace containing scene, sim, device, num_envs, physics_dt, step_dt.
    """
    get_dt_fn = getattr(sim, "get_physics_dt", None)
    physics_dt = get_dt_fn() if callable(get_dt_fn) else getattr(sim, "physics_dt", 0.0)
    return types.SimpleNamespace(
        scene=scene,
        sim=sim,
        device=getattr(sim, "device", "cpu"),
        num_envs=getattr(scene, "num_envs", 1),
        physics_dt=physics_dt,
        step_dt=physics_dt,
    )
