# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Specification and validation logic for single navigation episodes."""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Any

from nav_arena.utils.cli_args import validate_route_args

VALID_METHODS: tuple[str, ...] = ("iplanner", "vint", "navdp", "viplanner", "x_navdp", "nav2")
VALID_METHOD_FAMILIES: tuple[str, ...] = ("in_process", "ros2")


def _normalize_for_serialization(val: Any) -> Any:
    """Recursively convert Path objects to strings and ensure JSON-serializable containers."""
    if isinstance(val, Path):
        return str(val)
    if isinstance(val, dict):
        return {str(k): _normalize_for_serialization(v) for k, v in val.items()}
    if isinstance(val, (list, tuple, set)):
        return [_normalize_for_serialization(v) for v in val]
    return val


@dataclass
class VizCfg:
    """Visualization settings for simulation rendering and debug displays."""

    gui: bool = False
    follow_camera: bool = False
    goal_overlay: bool = True

    def __post_init__(self) -> None:
        self.gui = bool(self.gui)
        self.follow_camera = bool(self.follow_camera)
        self.goal_overlay = bool(self.goal_overlay)


@dataclass
class EpisodeLimits:
    """Termination bounds and constraints for a single episode."""

    max_steps: int = 1500
    goal_tolerance: float = 0.4
    max_speed: float = 0.3
    stall_timeout_s: float = 10.0

    def __post_init__(self) -> None:
        self.max_steps = int(self.max_steps)
        self.goal_tolerance = float(self.goal_tolerance)
        self.max_speed = float(self.max_speed)
        self.stall_timeout_s = float(self.stall_timeout_s)


@dataclass
class RunSpec:
    """Complete, serializable specification of one benchmark episode."""

    method: str
    method_family: str = ""
    robot: str = "dingo"
    scene: str = "kujiale_0003"
    route: str | None = None
    spawn: tuple[float, float] | None = None
    goal: tuple[float, float] | None = None
    spawn_yaw: float | None = None
    seed: int = 0
    method_params: dict[str, Any] = field(default_factory=dict)
    limits: EpisodeLimits = field(default_factory=EpisodeLimits)
    viz: VizCfg = field(default_factory=VizCfg)
    output_dir: Path | None = None

    def __post_init__(self) -> None:
        if not self.method_family:
            self.method_family = "ros2" if self.method == "nav2" else "in_process"
        if self.limits is None:
            self.limits = EpisodeLimits()
        elif isinstance(self.limits, dict):
            self.limits = EpisodeLimits(**self.limits)
        if self.viz is None:
            self.viz = VizCfg()
        elif isinstance(self.viz, dict):
            self.viz = VizCfg(**self.viz)
        if self.method_params is None:
            self.method_params = {}
        if self.spawn is not None:
            self.spawn = tuple(float(x) for x in self.spawn)
        if self.goal is not None:
            self.goal = tuple(float(x) for x in self.goal)
        if self.spawn_yaw is not None:
            self.spawn_yaw = float(self.spawn_yaw)
        if self.output_dir is not None and not isinstance(self.output_dir, Path):
            self.output_dir = Path(self.output_dir)

    @property
    def spec_hash(self) -> str:
        """SHA-256 hex digest of canonical spec fields defining the benchmark run.

        Canonical fields include method, method_family, robot, scene, route,
        spawn, goal, spawn_yaw, seed, method_params, and limits.
        Runtime output destination (output_dir) and visualizer settings (viz)
        do not alter the benchmark experiment definition and are excluded.
        """
        canonical_data = {
            "goal": [float(x) for x in self.goal] if self.goal is not None else None,
            "limits": asdict(self.limits),
            "method": self.method,
            "method_family": self.method_family,
            "method_params": _normalize_for_serialization(self.method_params),
            "robot": self.robot,
            "route": self.route,
            "scene": self.scene,
            "seed": int(self.seed),
            "spawn": [float(x) for x in self.spawn] if self.spawn is not None else None,
            "spawn_yaw": float(self.spawn_yaw) if self.spawn_yaw is not None else None,
        }
        canonical_json = json.dumps(
            canonical_data, sort_keys=True, separators=(",", ":"), default=str
        )
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "method": self.method,
            "method_family": self.method_family,
            "robot": self.robot,
            "scene": self.scene,
            "route": self.route,
            "spawn": list(self.spawn) if self.spawn is not None else None,
            "goal": list(self.goal) if self.goal is not None else None,
            "spawn_yaw": self.spawn_yaw,
            "seed": self.seed,
            "method_params": _normalize_for_serialization(self.method_params),
            "limits": asdict(self.limits),
            "viz": asdict(self.viz),
            "output_dir": str(self.output_dir) if self.output_dir is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunSpec:
        """Construct a RunSpec instance from a dictionary."""
        d = dict(data)
        if "limits" in d and isinstance(d["limits"], dict):
            d["limits"] = EpisodeLimits(**d["limits"])
        if "viz" in d and isinstance(d["viz"], dict):
            d["viz"] = VizCfg(**d["viz"])
        if "spawn" in d and d["spawn"] is not None:
            d["spawn"] = tuple(float(x) for x in d["spawn"])
        if "goal" in d and d["goal"] is not None:
            d["goal"] = tuple(float(x) for x in d["goal"])
        if "output_dir" in d and d["output_dir"] is not None:
            d["output_dir"] = Path(d["output_dir"])
        return cls(**d)

    def to_json(self, indent: int | None = 2) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> RunSpec:
        """Construct a RunSpec instance from a JSON string."""
        return cls.from_dict(json.loads(json_str))


def validate_spec(spec: RunSpec) -> None:
    """Validate a RunSpec for consistency, valid method, bounds, and route parameters.

    Zero Isaac Sim, Torch, or GPU dependencies; suitable for fast pre-flight checking.

    Raises:
        ValueError: If any field fails validation constraints.
    """
    if spec.method not in VALID_METHODS:
        raise ValueError(f"Invalid method '{spec.method}'. Expected one of: {VALID_METHODS}")

    if spec.method_family not in VALID_METHOD_FAMILIES:
        raise ValueError(
            f"Invalid method_family '{spec.method_family}'. Expected one of: {VALID_METHOD_FAMILIES}"
        )

    expected_family = "ros2" if spec.method == "nav2" else "in_process"
    if spec.method_family != expected_family:
        raise ValueError(
            f"Method '{spec.method}' requires method_family '{expected_family}', but got '{spec.method_family}'"
        )

    if not isinstance(spec.robot, str) or not spec.robot.strip():
        raise ValueError(f"robot must be a non-empty string, got '{spec.robot}'")

    if not isinstance(spec.scene, str) or not spec.scene.strip():
        raise ValueError(f"scene must be a non-empty string, got '{spec.scene}'")

    if spec.route is not None and (not isinstance(spec.route, str) or not spec.route.strip()):
        raise ValueError(f"route must be a non-empty string or None, got '{spec.route}'")

    # Route / spawn / goal consistency
    validate_route_args(spec.route, spec.spawn, spec.goal)

    if spec.spawn is not None:
        if len(spec.spawn) != 2 or not all(
            isinstance(x, (int, float)) and math.isfinite(x) for x in spec.spawn
        ):
            raise ValueError(f"spawn must be a coordinate pair (x, y) of finite floats, got {spec.spawn}")

    if spec.goal is not None:
        if len(spec.goal) != 2 or not all(
            isinstance(x, (int, float)) and math.isfinite(x) for x in spec.goal
        ):
            raise ValueError(f"goal must be a coordinate pair (x, y) of finite floats, got {spec.goal}")

    if spec.spawn_yaw is not None:
        if (
            isinstance(spec.spawn_yaw, bool)
            or not isinstance(spec.spawn_yaw, (int, float))
            or not math.isfinite(spec.spawn_yaw)
        ):
            raise ValueError(f"spawn_yaw must be a finite float, got {spec.spawn_yaw}")

    if isinstance(spec.seed, bool) or not isinstance(spec.seed, int) or spec.seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {spec.seed}")

    # Limits validation (> 0 for steps/tolerance/speed, >= 0 for stall timeout)
    if not isinstance(spec.limits, EpisodeLimits):
        raise ValueError(f"limits must be an EpisodeLimits instance, got {spec.limits}")

    if (
        isinstance(spec.limits.max_steps, bool)
        or not isinstance(spec.limits.max_steps, int)
        or spec.limits.max_steps <= 0
    ):
        raise ValueError(f"limits.max_steps must be a positive integer, got {spec.limits.max_steps}")

    if (
        isinstance(spec.limits.goal_tolerance, bool)
        or not isinstance(spec.limits.goal_tolerance, (int, float))
        or not math.isfinite(spec.limits.goal_tolerance)
        or spec.limits.goal_tolerance <= 0
    ):
        raise ValueError(
            f"limits.goal_tolerance must be a positive finite float, got {spec.limits.goal_tolerance}"
        )

    if (
        isinstance(spec.limits.max_speed, bool)
        or not isinstance(spec.limits.max_speed, (int, float))
        or not math.isfinite(spec.limits.max_speed)
        or spec.limits.max_speed <= 0
    ):
        raise ValueError(f"limits.max_speed must be a positive finite float, got {spec.limits.max_speed}")

    if (
        isinstance(spec.limits.stall_timeout_s, bool)
        or not isinstance(spec.limits.stall_timeout_s, (int, float))
        or not math.isfinite(spec.limits.stall_timeout_s)
        or spec.limits.stall_timeout_s < 0
    ):
        raise ValueError(
            f"limits.stall_timeout_s must be a non-negative finite float, got {spec.limits.stall_timeout_s}"
        )


def apply_overrides(
    spec: RunSpec,
    overrides: dict[str, Any],
    logger: logging.Logger | None = None,
) -> RunSpec:
    """Return a new RunSpec with overridden fields applied, logging each override.

    Logs a WARNING if method, scene, or robot is overridden.
    Logs an INFO for all other field overrides.

    Args:
        spec: Base RunSpec providing defaults.
        overrides: Dictionary mapping field or CLI flag names to override values.
        logger: Optional logger instance. Defaults to logging.getLogger("nav_arena.benchmarks.spec").

    Returns:
        A new RunSpec with overrides applied.
    """
    if logger is None:
        logger = logging.getLogger("nav_arena.benchmarks.spec")

    data = spec.to_dict()

    for raw_key, new_val in overrides.items():
        flag = raw_key if raw_key.startswith("--") else f"--{raw_key.replace('_', '-')}"
        clean_key = raw_key.lstrip("-").replace("-", "_")

        if clean_key == "method":
            old_val = data["method"]
            data["method"] = new_val
            if "method_family" not in overrides and "--method-family" not in overrides:
                data["method_family"] = "ros2" if new_val == "nav2" else "in_process"
            logger.warning("%s overrides spec.method: %s -> %s", flag, old_val, new_val)

        elif clean_key in ("robot", "scene"):
            old_val = data[clean_key]
            data[clean_key] = new_val
            logger.warning("%s overrides spec.%s: %s -> %s", flag, clean_key, old_val, new_val)

        elif clean_key in ("route", "spawn_yaw", "method_family"):
            old_val = data[clean_key]
            data[clean_key] = new_val
            logger.info("%s overrides spec.%s: %s -> %s", flag, clean_key, old_val, new_val)

        elif clean_key in ("seed", "seeds"):
            old_val = data["seed"]
            resolved_seed = new_val
            if isinstance(new_val, (list, tuple)):
                if len(new_val) == 1:
                    resolved_seed = new_val[0]
                else:
                    raise ValueError(f"Single episode spec cannot accept multiple seeds: {new_val}")
            data["seed"] = int(resolved_seed)
            logger.info("%s overrides spec.seed: %s -> %s", flag, old_val, data["seed"])

        elif clean_key in ("output", "output_dir"):
            old_val = data["output_dir"]
            data["output_dir"] = str(new_val) if new_val is not None else None
            logger.info("%s overrides spec.output_dir: %s -> %s", flag, old_val, data["output_dir"])

        elif clean_key in ("spawn", "goal"):
            old_val = data[clean_key]
            data[clean_key] = [float(x) for x in new_val] if new_val is not None else None
            logger.info("%s overrides spec.%s: %s -> %s", flag, clean_key, old_val, data[clean_key])

        elif clean_key == "limits":
            old_val = data["limits"]
            lim_dict = asdict(new_val) if isinstance(new_val, EpisodeLimits) else dict(new_val)
            data["limits"].update(lim_dict)
            logger.info("%s overrides spec.limits: %s -> %s", flag, old_val, data["limits"])

        elif clean_key in (
            "max_steps",
            "goal_tolerance",
            "max_speed",
            "stall_timeout_s",
            "goal_dist",
            "stall_timeout",
        ):
            mapped_key = (
                "goal_tolerance"
                if clean_key == "goal_dist"
                else ("stall_timeout_s" if clean_key == "stall_timeout" else clean_key)
            )
            old_val = data["limits"][mapped_key]
            data["limits"][mapped_key] = new_val
            logger.info("%s overrides spec.limits.%s: %s -> %s", flag, mapped_key, old_val, new_val)

        elif clean_key == "viz":
            old_val = data["viz"]
            viz_dict = asdict(new_val) if isinstance(new_val, VizCfg) else dict(new_val)
            data["viz"].update(viz_dict)
            logger.info("%s overrides spec.viz: %s -> %s", flag, old_val, data["viz"])

        elif clean_key in ("gui", "follow_camera", "goal_overlay"):
            old_val = data["viz"][clean_key]
            data["viz"][clean_key] = bool(new_val)
            logger.info("%s overrides spec.viz.%s: %s -> %s", flag, clean_key, old_val, data["viz"][clean_key])

        elif clean_key in ("no_gui", "no_follow_camera", "no_goal_overlay"):
            target_key = clean_key[3:]  # 'gui', 'follow_camera', 'goal_overlay'
            old_val = data["viz"][target_key]
            data["viz"][target_key] = not bool(new_val)
            logger.info("%s overrides spec.viz.%s: %s -> %s", flag, target_key, old_val, data["viz"][target_key])

        elif clean_key in ("policy_arg", "policy_args"):
            items = new_val if isinstance(new_val, (list, tuple)) else [new_val]
            for item in items:
                if isinstance(item, dict):
                    data["method_params"].update(_normalize_for_serialization(item))
                    logger.info("%s overrides spec.method_params: %s", flag, item)
                elif isinstance(item, str):
                    k, sep, v = item.partition("=")
                    if not sep:
                        raise ValueError(f"--policy-arg expects KEY=VALUE, got '{item}'")
                    try:
                        parsed_v = ast.literal_eval(v)
                    except (ValueError, SyntaxError):
                        parsed_v = v
                    old_v = data["method_params"].get(k)
                    data["method_params"][k] = _normalize_for_serialization(parsed_v)
                    logger.info("%s overrides spec.method_params.%s: %s -> %s", flag, k, old_v, parsed_v)

        elif clean_key == "method_params":
            old_val = data["method_params"]
            data["method_params"].update(_normalize_for_serialization(new_val))
            logger.info("%s overrides spec.method_params: %s -> %s", flag, old_val, data["method_params"])

        else:
            old_val = data["method_params"].get(clean_key)
            norm_val = _normalize_for_serialization(new_val)
            data["method_params"][clean_key] = norm_val
            logger.info("%s overrides spec.method_params.%s: %s -> %s", flag, clean_key, old_val, norm_val)

    return RunSpec.from_dict(data)
