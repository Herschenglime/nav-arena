# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contract for navigation policies that run inside the simulation process.

In-process policies take camera observations and return a short local path; a shared path follower
(:mod:`nav_arena.methods.in_process.controller`) turns that path into wheel commands at the control rate.
Planning is therefore decoupled from control: policies replan at a few Hz while the follower re-tracks the most
recent path every control step.

This module must stay free of Isaac Sim / Omniverse / ROS 2 imports so it can be unit-tested on the CPU.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import numpy as np


@dataclass
class PolicyObservation:
    """Sensor and goal inputs for one planning step.

    Conventions: RGB is uint8 ``[H, W, 3]``; depth is metric image-plane distance in meters ``[H, W]`` (invalid
    pixels may be ``inf``/NaN/0); body-frame vectors use X forward, Y left, Z up; quaternions are ``(x, y, z, w)``.
    """

    rgb: np.ndarray
    depth: np.ndarray
    goal_body: np.ndarray | None = None
    """Point goal in the robot body frame, shape ``[1, 3]`` (meters). Ignored by image-goal-only policies."""
    goal_image: np.ndarray | None = None
    """Goal RGB image uint8 ``[H, W, 3]``, required by image-goal tasks."""
    robot_position: np.ndarray | None = None
    """World position ``[1, 3]``; required by policies that use pose-based guidance (X-NavDP)."""
    robot_quaternion: np.ndarray | None = None
    """World orientation ``[1, 4]`` as unit ``(x, y, z, w)``; required together with ``robot_position``."""


@dataclass
class Plan:
    """A local plan produced by a policy.

    Attributes:
        path: Waypoints in the robot body frame at planning time, shape ``[T, >=2]``; columns are x (forward) and
            y (left) in meters.
        stop: If True the policy requests the robot to stop (e.g. predicted collision risk, native stop mask).
        diagnostics: Policy-specific scalars for logging (fear, critic range, ...).
    """

    path: np.ndarray
    stop: bool = False
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class InProcessPolicyCfg:
    """Settings shared by all in-process policies."""

    device: str = "cuda:0"
    """Torch device for model inference."""
    checkpoint: Path | None = None
    """Checkpoint override; ``None`` uses the policy's default location."""
    task: str = "pointgoal"
    """Goal modality: ``"pointgoal"`` or ``"imagegoal"``."""
    plan_hz: float | None = None
    """Replanning rate; ``None`` uses the policy's native default (``default_plan_hz``)."""
    seed: int | None = 0
    """Seed for stochastic planners (diffusion sampling); ``None`` leaves RNG state untouched."""


class InProcessPolicy(ABC):
    """A navigation policy executed directly in the simulation process."""

    name: ClassVar[str]
    """Registry name."""
    supported_tasks: ClassVar[tuple[str, ...]] = ("pointgoal",)
    """Goal modalities the policy can consume."""
    default_plan_hz: ClassVar[float] = 5.0
    """Native replanning rate."""
    requires_pose: ClassVar[bool] = False
    """Whether ``PolicyObservation.robot_position/robot_quaternion`` must be provided."""

    def __init__(self, cfg: InProcessPolicyCfg) -> None:
        if cfg.task not in self.supported_tasks:
            raise ValueError(f"{self.name} supports tasks {self.supported_tasks}, got '{cfg.task}'")
        self.cfg = cfg

    @property
    def plan_hz(self) -> float:
        """Effective replanning rate in Hz."""
        return self.cfg.plan_hz if self.cfg.plan_hz is not None else self.default_plan_hz

    def reset(self) -> None:
        """Clear per-episode history (observation memory, stuck detection). Stateless policies do nothing."""

    @abstractmethod
    def step(self, obs: PolicyObservation) -> Plan:
        """Plan from the current observation.

        Raises:
            ValueError: If the observation is malformed or lacks inputs this policy requires.
        """
