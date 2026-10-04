# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for multi-embodiment PointNav configuration and MDP terms (CPU-only; no Kit boot)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from nav_arena.embodiments import dingo_stage_patch
from nav_arena.scenes import create_interior_agent_scene_cfg
from nav_arena.tasks import PointNavTask, create_point_nav_env_cfg
from nav_arena.tasks.point_nav import apply_embodiment_stage_patch, lateral_contact


@pytest.fixture
def scene_usd(tmp_path):
    """A placeholder scene file; configs only record its path."""
    path = tmp_path / "scene.usda"
    path.write_text("#usda 1.0\n")
    return str(path)


def _cfg(scene_usd, **kwargs):
    return create_point_nav_env_cfg(scene_id_or_path=scene_usd, open_doors=False, **kwargs)


def test_default_cfg_is_unchanged_nova_carter(scene_usd):
    """Verify the default config still targets the Nova Carter with chassis-link contact sensing."""
    cfg = _cfg(scene_usd)
    assert cfg.robot_name == "nova_carter"
    assert cfg.scene.robot.prim_path == "{ENV_REGEX_NS}/Robot"
    assert cfg.actions.robot_action.wheel_radius == 0.14
    assert cfg.scene.contact_forces.prim_path == "{ENV_REGEX_NS}/Robot/chassis_link"
    assert cfg.scene.lidar.prim_path == "{ENV_REGEX_NS}/Robot/chassis_link"
    assert cfg.terminations.collision.func.__name__ == "illegal_contact"
    assert cfg.events.embodiment_stage_patch is None
    assert cfg.scene.replicate_physics is True
    assert cfg.scene.camera is None and cfg.scene.goal_camera is None
    assert cfg.sim.render_interval == 3


def test_dingo_cfg_binds_embodiment(scene_usd):
    """Verify the Dingo config uses its USD, wheel geometry, base_link sensing, and lateral-force collisions."""
    cfg = _cfg(scene_usd, robot_name="dingo")
    assert cfg.robot_name == "dingo"
    assert cfg.scene.robot.spawn.usd_path.endswith("dingo.usd")
    assert cfg.actions.robot_action.wheel_radius == 0.1225
    assert cfg.actions.robot_action.asset_name == "robot"
    assert cfg.scene.contact_forces.prim_path == "{ENV_REGEX_NS}/Robot/base_link"
    assert cfg.scene.lidar.prim_path == "{ENV_REGEX_NS}/Robot/base_link"
    assert tuple(cfg.scene.lidar.offset.pos) == (0.0, 0.0, 0.30)
    assert cfg.terminations.collision.func is lateral_contact
    assert cfg.terminations.collision.params["threshold"] == 1.0


def test_dingo_stage_patch_runs_as_prestartup_event(scene_usd):
    """Verify the Dingo patch is registered as a prestartup event and physics replication is disabled for it."""
    cfg = _cfg(scene_usd, robot_name="dingo")
    event = cfg.events.embodiment_stage_patch
    assert event.mode == "prestartup"
    assert event.func is apply_embodiment_stage_patch
    assert event.params["patch_fn"] is dingo_stage_patch
    # Isaac Lab rejects prestartup events while physics replication is enabled.
    assert cfg.scene.replicate_physics is False


def test_stage_patch_event_receives_the_sim_stage():
    """Verify the prestartup event passes the simulation stage to the embodiment patch."""
    calls = []
    env = SimpleNamespace(sim=SimpleNamespace(stage="STAGE"))
    apply_embodiment_stage_patch(env, None, patch_fn=calls.append)
    assert calls == ["STAGE"]


def test_camera_enabled_defaults_to_5hz_render_interval(scene_usd):
    """Verify enabling the camera renders at 5 Hz (interval 20 at dt=0.01) unless overridden."""
    cfg = _cfg(scene_usd, robot_name="dingo", enable_camera=True)
    assert cfg.scene.camera.prim_path == "{ENV_REGEX_NS}/Robot/base_link/front_camera"
    assert cfg.sim.dt == 0.01
    assert cfg.sim.render_interval == 20
    assert cfg.sim.render_interval * cfg.sim.dt == pytest.approx(0.2)
    assert _cfg(scene_usd, enable_camera=True, render_interval=5).sim.render_interval == 5
    assert _cfg(scene_usd, render_interval=7).sim.render_interval == 7


def test_goal_camera_is_free_standing_rgb(scene_usd):
    """Verify the goal camera is a global RGB-only prim at /World/GoalCamera."""
    cfg = _cfg(scene_usd, robot_name="dingo", enable_goal_camera=True)
    assert cfg.scene.goal_camera.prim_path == "/World/GoalCamera"
    assert cfg.scene.goal_camera.data_types == ["rgb"]
    assert cfg.scene.camera is None


def test_goal_camera_requires_single_env(scene_usd):
    """Verify a goal camera cannot be combined with multiple environments."""
    with pytest.raises(ValueError, match="num_envs == 1"):
        create_interior_agent_scene_cfg(scene_usd, open_doors=False, num_envs=2, enable_goal_camera=True)


def test_unknown_robot_lists_available_embodiments(scene_usd):
    """Verify an unregistered robot name fails with the registry's helpful message."""
    with pytest.raises(KeyError, match="Available embodiments"):
        _cfg(scene_usd, robot_name="no_such_robot")


def test_tilt_limit_is_configurable(scene_usd):
    """Verify the tipped termination defaults to 0.6 rad and can be overridden."""
    assert _cfg(scene_usd).terminations.tipped.params["limit_angle"] == 0.6
    assert _cfg(scene_usd, max_tilt=0.3).terminations.tipped.params["limit_angle"] == 0.3


def _sensor_env(forces):
    """Fake env whose contact sensor reports `forces` with shape [envs, history, bodies, 3]."""
    sensor = SimpleNamespace(data=SimpleNamespace(net_forces_w_history=SimpleNamespace(torch=forces)))
    return SimpleNamespace(scene=SimpleNamespace(sensors={"contact_forces": sensor}))


def _entity():
    return SimpleNamespace(name="contact_forces", body_ids=slice(None))


def test_lateral_contact_ignores_vertical_support_force():
    """Verify a large vertical force (floor support) is not a collision but a horizontal one is."""
    forces = torch.zeros(2, 3, 1, 3)
    forces[0, :, 0, 2] = 50.0  # env 0: resting on a caster, vertical force only
    forces[1, 1, 0, :] = torch.tensor([0.9, 0.9, 0.0])  # env 1: |F_xy| = 1.27 N at one history step
    result = lateral_contact(_sensor_env(forces), threshold=1.0, sensor_cfg=_entity())
    assert result.tolist() == [False, True]


def test_lateral_contact_threshold_and_history():
    """Verify forces below the threshold pass, and any history frame above it triggers."""
    forces = torch.zeros(1, 3, 2, 3)
    forces[0, 0, 1, 0] = 0.99
    assert not lateral_contact(_sensor_env(forces), 1.0, _entity()).item()
    forces[0, 2, 0, 1] = -1.01  # sign-independent, earlier-or-later history frame
    assert lateral_contact(_sensor_env(forces), 1.0, _entity()).item()


def test_terminal_cause_priority_and_none():
    """Verify get_terminal_cause reports goal over collision over tipped over time_out, else None."""
    task = PointNavTask.__new__(PointNavTask)
    state = {"goal": False, "collision": False, "tipped": False, "time_out": False}

    class Terms:
        def get_term(self, name):
            return torch.tensor([state[name]])

    task.termination_manager = Terms()
    assert task.get_terminal_cause() is None
    state["time_out"] = True
    assert task.get_terminal_cause() == "time_out"
    state["tipped"] = True
    assert task.get_terminal_cause() == "tipped"
    state["collision"] = True
    assert task.get_terminal_cause() == "collision"
    state["goal"] = True
    assert task.get_terminal_cause() == "goal_reached"


def test_camera_helpers_explain_missing_sensors():
    """Verify camera accessors raise an actionable error when the env was built without cameras."""
    task = PointNavTask.__new__(PointNavTask)
    task.scene = {}
    with pytest.raises(RuntimeError, match="enable_camera=True"):
        task.get_camera_frame()
    with pytest.raises(RuntimeError, match="enable_goal_camera=True"):
        task.render_goal_image((1.0, 2.0))


def test_goal_marker_is_visible_by_default_but_can_be_hidden(scene_usd):
    """Verify the goal arrow (real geometry that cameras see) can be disabled for camera-driven policies."""
    assert _cfg(scene_usd).commands.pose_2d_command.debug_vis is True
    assert _cfg(scene_usd, show_goal_marker=False).commands.pose_2d_command.debug_vis is False


def test_scene_queries_are_opt_in(scene_usd):
    """Verify PhysX scene queries (needed by the follow camera's occlusion rays) are off unless requested."""
    assert _cfg(scene_usd).sim.enable_scene_query_support is False
    assert _cfg(scene_usd, scene_queries=True).sim.enable_scene_query_support is True
