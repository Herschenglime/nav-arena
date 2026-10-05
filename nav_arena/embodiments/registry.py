# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""In-memory registry for robot embodiments."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .base import RobotEmbodimentCfg

_EMBODIMENT_REGISTRY: dict[str, Any] = {}


def _normalize_embodiment_target(target: Any) -> Any:
    """Ensure classes inheriting from RobotEmbodimentCfg are properly converted to dataclasses."""
    if isinstance(target, type):
        from dataclasses import dataclass
        from .base import RobotEmbodimentCfg

        # Subclasses inherit __dataclass_fields__, so is_dataclass() is True even when the
        # subclass itself was never processed; check the class's own namespace instead.
        if issubclass(target, RobotEmbodimentCfg) and "__dataclass_fields__" not in vars(target):
            return dataclass(target)
    return target


def register_embodiment(name: str, cfg_factory_or_cfg: Any = None) -> Any:
    """Register an embodiment configuration or factory.

    Supports direct registration:
        register_embodiment("dingo", DingoEmbodimentCfg)
    or decorator syntax:
        @register_embodiment("dingo")
        class DingoEmbodimentCfg(RobotEmbodimentCfg): ...

    Args:
        name: Unique string identifier for the embodiment (e.g. 'nova_carter', 'dingo').
        cfg_factory_or_cfg: Optional RobotEmbodimentCfg instance, class, or zero-argument callable returning one.
            If omitted, returns a decorator.

    Returns:
        The registered cfg_factory_or_cfg, or a decorator function.
    """
    if cfg_factory_or_cfg is None:
        def decorator(cls_or_fn: Any) -> Any:
            processed = _normalize_embodiment_target(cls_or_fn)
            _EMBODIMENT_REGISTRY[name] = processed
            return processed

        return decorator

    processed = _normalize_embodiment_target(cfg_factory_or_cfg)
    _EMBODIMENT_REGISTRY[name] = processed
    return cfg_factory_or_cfg


def get_embodiment(name: str) -> RobotEmbodimentCfg:
    """Retrieve an embodiment configuration by name.

    If the registered item is a callable (class or factory function), it is invoked
    to return a fresh instance. If it is already an instance, a deepcopy is returned
    to prevent in-place mutation of the registry state.

    Args:
        name: Name of the registered embodiment.

    Returns:
        The resolved RobotEmbodimentCfg instance.

    Raises:
        KeyError: If the embodiment is not found in the registry.
    """
    if name not in _EMBODIMENT_REGISTRY:
        raise KeyError(
            f"Embodiment '{name}' not found in registry. "
            f"Available embodiments: {list_embodiments()}"
        )
    item = _EMBODIMENT_REGISTRY[name]
    if callable(item):
        cfg = item()
    else:
        import copy

        cfg = copy.deepcopy(item)
    if hasattr(cfg, "name") and not cfg.name:
        cfg.name = name
    return cfg


def list_embodiments() -> list[str]:
    """List all registered embodiment names in sorted order.

    Returns:
        List of registered embodiment names.
    """
    return sorted(list(_EMBODIMENT_REGISTRY.keys()))


def clear_registry() -> None:
    """Clear all registered embodiments (useful for isolated unit testing)."""
    _EMBODIMENT_REGISTRY.clear()


def register_default_embodiments() -> None:
    """Register built-in framework embodiments ('nova_carter', 'dingo')."""
    from .dingo import DingoEmbodimentCfg
    from .nova_carter import NovaCarterEmbodimentCfg

    register_embodiment("nova_carter", NovaCarterEmbodimentCfg)
    register_embodiment("dingo", DingoEmbodimentCfg)
