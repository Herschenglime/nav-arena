# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for simulation utilities and mock helpers."""

import types
import pytest

from nav_arena.utils.sim import create_mock_env


class DummyScene:
    def __init__(self, num_envs: int = 4):
        self.num_envs = num_envs


class DummySim:
    def __init__(self, device: str = "cuda:0", physics_dt: float = 0.01):
        self.device = device
        self._physics_dt = physics_dt

    def get_physics_dt(self) -> float:
        return self._physics_dt


@pytest.mark.unit
def test_create_mock_env_structure():
    """Verify that create_mock_env constructs a SimpleNamespace with all expected fields."""
    scene = DummyScene(num_envs=2)
    sim = DummySim(device="cuda:0", physics_dt=0.02)

    mock_env = create_mock_env(scene, sim)

    assert isinstance(mock_env, types.SimpleNamespace)
    assert mock_env.scene is scene
    assert mock_env.sim is sim
    assert mock_env.device == "cuda:0"
    assert mock_env.num_envs == 2
    assert mock_env.physics_dt == 0.02
    assert mock_env.step_dt == 0.02


@pytest.mark.unit
def test_create_mock_env_fallback_attributes():
    """Verify fallback values when optional attributes or methods are missing."""
    empty_scene = types.SimpleNamespace()
    empty_sim = types.SimpleNamespace(physics_dt=0.005)

    mock_env = create_mock_env(empty_scene, empty_sim)

    assert mock_env.scene is empty_scene
    assert mock_env.sim is empty_sim
    assert mock_env.device == "cpu"
    assert mock_env.num_envs == 1
    assert mock_env.physics_dt == 0.005
    assert mock_env.step_dt == 0.005


@pytest.mark.unit
def test_create_mock_env_non_callable_get_physics_dt():
    """Verify that a non-callable get_physics_dt attribute does not cause TypeError."""
    scene = types.SimpleNamespace(num_envs=1)
    sim = types.SimpleNamespace(get_physics_dt=0.033, physics_dt=0.033)

    mock_env = create_mock_env(scene, sim)

    assert mock_env.physics_dt == 0.033
    assert mock_env.step_dt == 0.033
