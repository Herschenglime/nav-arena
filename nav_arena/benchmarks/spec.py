# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Specification and validation logic for single navigation episodes."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Any, NamedTuple

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


class ResolvedViz(NamedTuple):
    """Visualization settings with every default resolved against ``gui``."""

    gui: bool
    follow_camera: bool
    goal_overlay: bool


@dataclass
class VizCfg:
    """Visualization settings for simulation rendering and debug displays.

    ``follow_camera`` and ``goal_overlay`` default to ``None`` ("auto"), meaning they follow ``gui``: on with a GUI,
    off when headless. Use :meth:`resolved` to get concrete values; it is the only place that rule lives.
    """

    gui: bool = False
    follow_camera: bool | None = None
    goal_overlay: bool | None = None
    follow_distance: float = 1.6
    follow_height: float = 1.2
    show_goal_marker: bool = False
    """Draw Isaac Lab's goal arrow as scene geometry. The policy's cameras see it as an obstacle."""

    def __post_init__(self) -> None:
        self.gui = bool(self.gui)
        self.follow_camera = None if self.follow_camera is None else bool(self.follow_camera)
        self.goal_overlay = None if self.goal_overlay is None else bool(self.goal_overlay)
        self.follow_distance = float(self.follow_distance)
        self.follow_height = float(self.follow_height)
        self.show_goal_marker = bool(self.show_goal_marker)

    def resolved(self) -> ResolvedViz:
        """Concrete follow-camera and overlay flags: unset values follow ``gui``."""
        return ResolvedViz(
            gui=self.gui,
            follow_camera=self.gui if self.follow_camera is None else self.follow_camera,
            goal_overlay=self.gui if self.goal_overlay is None else self.goal_overlay,
        )

    @classmethod
    def from_options(cls, options: Mapping[str, Any]) -> VizCfg:
        """Build from a flat options mapping (sweep ``options``, batch.yaml); absent keys keep their defaults."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in options.items() if k in known and v is not None})


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

    @classmethod
    def from_options(cls, options: Mapping[str, Any]) -> EpisodeLimits:
        """Build from a flat options mapping, accepting the CLI aliases ``goal_dist`` and ``stall_timeout``."""
        aliases = {"goal_dist": "goal_tolerance", "stall_timeout": "stall_timeout_s"}
        known = {f.name for f in fields(cls)}
        values: dict[str, Any] = {}
        for key, value in options.items():
            name = aliases.get(key, key)
            if name in known and value is not None:
                values.setdefault(name, value)
        return cls(**values)


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
        canonical_data: dict[str, Any] = {
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
        if self.viz.show_goal_marker:
            # It changes what the policy's cameras see, so it is part of the experiment (omitted when off so hashes
            # of runs recorded before this field existed stay valid).
            canonical_data["show_goal_marker"] = True
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


_WARN_KEYS = ("robot", "scene")
_PLAIN_KEYS = ("route", "spawn_yaw", "method_family")
_LIMIT_KEYS = {
    "max_steps": "max_steps",
    "goal_tolerance": "goal_tolerance",
    "goal_dist": "goal_tolerance",
    "max_speed": "max_speed",
    "stall_timeout_s": "stall_timeout_s",
    "stall_timeout": "stall_timeout_s",
}
_VIZ_KEYS = ("gui", "follow_camera", "goal_overlay", "follow_distance", "follow_height", "show_goal_marker")
_PARAM_ALIASES = {"planner_device": "device"}
"""CLI-style names that are stored under a different policy-config field."""


def _parse_policy_arg(item: str) -> tuple[str, Any]:
    """Parse ``KEY=VALUE`` into a key and a Python literal (or the raw string)."""
    key, sep, value = item.partition("=")
    if not sep:
        raise ValueError(f"--policy-arg expects KEY=VALUE, got '{item}'")
    try:
        return key, ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return key, value


def _override_method(data: dict[str, Any], flag: str, value: Any, overrides: dict[str, Any], log: logging.Logger) -> None:
    old = data["method"]
    data["method"] = value
    if "method_family" not in overrides and "--method-family" not in overrides:
        data["method_family"] = "ros2" if value == "nav2" else "in_process"
    log.warning("%s overrides spec.method: %s -> %s", flag, old, value)


def _override_seed(data: dict[str, Any], flag: str, value: Any, log: logging.Logger) -> None:
    old = data["seed"]
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"Single episode spec cannot accept multiple seeds: {value}")
        value = value[0]
    data["seed"] = int(value)
    log.info("%s overrides spec.seed: %s -> %s", flag, old, data["seed"])


def _override_policy_args(data: dict[str, Any], flag: str, value: Any, log: logging.Logger) -> None:
    for item in value if isinstance(value, (list, tuple)) else [value]:
        if isinstance(item, dict):
            data["method_params"].update(_normalize_for_serialization(item))
            log.info("%s overrides spec.method_params: %s", flag, item)
        elif isinstance(item, str):
            key, parsed = _parse_policy_arg(item)
            old = data["method_params"].get(key)
            data["method_params"][key] = _normalize_for_serialization(parsed)
            log.info("%s overrides spec.method_params.%s: %s -> %s", flag, key, old, parsed)


def _override_viz(data: dict[str, Any], flag: str, key: str, value: Any, log: logging.Logger) -> None:
    """Override one visualization field; ``no_<name>`` flags negate a boolean field."""
    if key.startswith("no_"):
        key, value = key[3:], not bool(value)
    old = data["viz"][key]
    data["viz"][key] = value if key in ("follow_distance", "follow_height") else bool(value)
    log.info("%s overrides spec.viz.%s: %s -> %s", flag, key, old, data["viz"][key])


def _apply_override(
    data: dict[str, Any], raw_key: str, value: Any, overrides: dict[str, Any], log: logging.Logger
) -> None:
    flag = raw_key if raw_key.startswith("--") else f"--{raw_key.replace('_', '-')}"
    key = raw_key.lstrip("-").replace("-", "_")

    if key == "method":
        _override_method(data, flag, value, overrides, log)
    elif key in _WARN_KEYS:
        old, data[key] = data[key], value
        log.warning("%s overrides spec.%s: %s -> %s", flag, key, old, value)
    elif key in _PLAIN_KEYS:
        old, data[key] = data[key], value
        log.info("%s overrides spec.%s: %s -> %s", flag, key, old, value)
    elif key in ("seed", "seeds"):
        _override_seed(data, flag, value, log)
    elif key in ("output", "output_dir"):
        old, data["output_dir"] = data["output_dir"], (str(value) if value is not None else None)
        log.info("%s overrides spec.output_dir: %s -> %s", flag, old, data["output_dir"])
    elif key in ("spawn", "goal"):
        old, data[key] = data[key], ([float(x) for x in value] if value is not None else None)
        log.info("%s overrides spec.%s: %s -> %s", flag, key, old, data[key])
    elif key == "limits":
        old = dict(data["limits"])
        data["limits"].update(asdict(value) if isinstance(value, EpisodeLimits) else dict(value))
        log.info("%s overrides spec.limits: %s -> %s", flag, old, data["limits"])
    elif key in _LIMIT_KEYS:
        field_name = _LIMIT_KEYS[key]
        old, data["limits"][field_name] = data["limits"][field_name], value
        log.info("%s overrides spec.limits.%s: %s -> %s", flag, field_name, old, value)
    elif key == "viz":
        old = dict(data["viz"])
        data["viz"].update(asdict(value) if isinstance(value, VizCfg) else dict(value))
        log.info("%s overrides spec.viz: %s -> %s", flag, old, data["viz"])
    elif key in _VIZ_KEYS or (key.startswith("no_") and key[3:] in _VIZ_KEYS):
        _override_viz(data, flag, key, value, log)
    elif key in ("policy_arg", "policy_args"):
        _override_policy_args(data, flag, value, log)
    elif key == "method_params":
        old = dict(data["method_params"])
        data["method_params"].update(_normalize_for_serialization(value))
        log.info("%s overrides spec.method_params: %s -> %s", flag, old, data["method_params"])
    else:
        param = _PARAM_ALIASES.get(key, key)
        old = data["method_params"].get(param)
        data["method_params"][param] = _normalize_for_serialization(value)
        log.info("%s overrides spec.method_params.%s: %s -> %s", flag, param, old, data["method_params"][param])


def apply_overrides(
    spec: RunSpec,
    overrides: dict[str, Any],
    logger: logging.Logger | None = None,
) -> RunSpec:
    """Return a new RunSpec with overridden fields applied, logging each override.

    Logs a WARNING if method, scene, or robot is overridden, and an INFO for every other field.
    Unrecognized keys are treated as policy config fields (``method_params``).

    Args:
        spec: Base RunSpec providing defaults.
        overrides: Dictionary mapping field or CLI flag names to override values.
        logger: Optional logger instance. Defaults to logging.getLogger("nav_arena.benchmarks.spec").

    Returns:
        A new RunSpec with overrides applied.
    """
    log = logger if logger is not None else logging.getLogger("nav_arena.benchmarks.spec")
    data = spec.to_dict()
    for raw_key, value in overrides.items():
        _apply_override(data, raw_key, value, overrides, log)
    return RunSpec.from_dict(data)
