# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unitree Go2 quadruped embodiment, walked by NVIDIA's pretrained flat-terrain locomotion policy.

The body twist command goes to Isaac Lab's ``PreTrainedPolicyAction`` unchanged: it feeds the twist and the robot's
proprioception to the policy at 50 Hz and turns the policy output into joint position targets. Everything the policy
depends on is taken from the Isaac Lab task it was trained in (``UnitreeGo2FlatEnvCfg_PLAY`` with its PhysX presets
resolved, observation noise off): the robot (actuators, armature, default pose), the timing, the observations and the
actions. Nothing here can drift from training:

- physics ``dt = 0.005`` with the policy running every 4 physics steps;
- observations: base linear and angular velocity, projected gravity, command, joint positions and velocities, last action;
- actions: joint position offsets scaled by 0.25 around the default pose;
- commands up to 1 m/s forward and sideways and 1 rad/s yaw (the training range).

At a zero command the policy stands crouched (base about 0.22 m up, legs off the default pose and not symmetric). That is
the policy's own stance: it stands the same way in Isaac Lab's training env (checked 2026-10-05).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from isaaclab.assets.articulation import ArticulationCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import UnitreeGo2FlatEnvCfg_PLAY
from isaaclab_tasks.manager_based.navigation.mdp import PreTrainedPolicyActionCfg
from isaaclab_tasks.utils.hydra import resolve_presets

from .base import RobotEmbodimentCfg
from .policies import GO2_FLAT_POLICY, ensure_policy
from .registry import register_embodiment

def training_env_cfg() -> UnitreeGo2FlatEnvCfg_PLAY:
    """A fresh copy of the Isaac Lab task the policy was trained in, with its PhysX presets resolved.

    Its robot (actuators, armature, default pose), timing, observations and actions are reused verbatim. A fresh copy per
    call matters: ``PreTrainedPolicyAction`` rewrites parts of its observation config when it is constructed.
    """
    return resolve_presets(UnitreeGo2FlatEnvCfg_PLAY())


def create_go2_action_cfg() -> PreTrainedPolicyActionCfg:
    """The locomotion-policy action term. Downloads (once) and verifies the pinned policy file."""
    training = training_env_cfg()
    low_level_decimation = 4
    if training.decimation != low_level_decimation:
        raise ValueError(f"Go2 policy was trained at decimation {training.decimation}, expected {low_level_decimation}")
    return PreTrainedPolicyActionCfg(
        asset_name="robot",
        policy_path=str(ensure_policy(GO2_FLAT_POLICY)),
        low_level_decimation=low_level_decimation,
        low_level_actions=training.actions.joint_pos,
        low_level_observations=training.observations.policy,
        # The default draws velocity arrows into the scene, which the depth camera would see.
        debug_vis=False,
    )


def create_go2_articulation_cfg() -> ArticulationCfg:
    """The Go2 exactly as in the training env (UNITREE_GO2_CFG plus its armature = 0 preset)."""
    return training_env_cfg().scene.robot


@dataclass
class Go2EmbodimentCfg(RobotEmbodimentCfg):
    """Unitree Go2 walking on NVIDIA's flat-terrain locomotion policy."""

    name: str = "go2"
    drive_type: str = "quadruped"
    articulation_cfg: ArticulationCfg = field(default_factory=create_go2_articulation_cfg)
    action_cfg: PreTrainedPolicyActionCfg = field(default_factory=create_go2_action_cfg)

    # Within the policy's training range (+-1 m/s, +-1 rad/s).
    max_linear_speed: float = 1.0
    max_lateral_speed: float = 1.0
    max_angular_speed: float = 1.0

    # Training timing: the policy only walks at the rate it was trained for. Control step 0.02 s (50 Hz).
    sim_dt: float = 0.005  # training_env_cfg().sim.dt; checked by the unit tests
    decimation: int = 4
    spawn_height: float = 0.4  # the training spawn height; the robot drops onto its feet and stands

    base_frame: str = "base_link"
    chassis_frame: str = "base_link"
    body_link: str = "base"
    # Body, hips, thighs and calves count as collisions; the feet touch the floor at every step, so they are left out.
    # Contact forces are normal forces only, so a calf grazing the floor reads vertical and is ignored with
    # ground_contact_on_body (horizontal forces only), while hitting a wall reads horizontal.
    contact_bodies: str = "(base|.*_hip|.*_thigh|.*_calf)"
    ground_contact_on_body: bool = True

    # The base pitches and rolls while walking: keep the 2D scan level.
    lidar_ray_alignment: str = "yaw"
    sensor_height: float = 0.15
    lidar_offset: tuple[float, float, float] = (0.0, 0.0, 0.15)
    # Front of the head, as in NavDP's Go2 config (about 0.5 m above the floor while standing).
    camera_offset: tuple[float, float, float] = (0.32, 0.0, 0.20)
    camera_rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

    # Standing footprint about 0.70 x 0.31 m (with legs).
    footprint: tuple[tuple[float, float], ...] = ((0.35, 0.155), (0.35, -0.155), (-0.35, -0.155), (-0.35, 0.155))
    chassis_size: tuple[float, float, float] = (0.70, 0.31, 0.40)
    chassis_offset: tuple[float, float, float] = (0.0, 0.0, -0.10)
    robot_radius: float = 0.40
    robot_height: float = 0.45


register_embodiment("go2", Go2EmbodimentCfg, drive_type="quadruped")
