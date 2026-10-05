# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""In-memory registry of in-process navigation policies."""

from __future__ import annotations

from typing import Any, Callable

from .base import InProcessPolicy

_POLICY_REGISTRY: dict[str, Callable[..., InProcessPolicy]] = {}
_DEFAULTS_LOADED = False


def _normalize(name: str) -> str:
    """Policy names use underscores; accept the hyphenated upstream spellings (e.g. ``x-navdp``)."""
    return name.strip().lower().replace("-", "_")


def register_policy(name: str, factory: Callable[..., InProcessPolicy] | None = None) -> Any:
    """Register a policy class or factory under ``name``; usable directly or as a decorator.

    The factory is called as ``factory(intrinsic, **cfg_kwargs)``.
    """
    if factory is None:

        def decorator(target: Callable[..., InProcessPolicy]) -> Callable[..., InProcessPolicy]:
            _POLICY_REGISTRY[_normalize(name)] = target
            return target

        return decorator
    _POLICY_REGISTRY[_normalize(name)] = factory
    return factory


def register_default_policies() -> None:
    """Register the built-in learned baselines (importing the adapter registers them)."""
    global _DEFAULTS_LOADED
    from .navdp_adapter import policies as _policies  # noqa: F401  (import registers the classes)

    for policy_cls in _policies.ADAPTER_POLICIES:
        _POLICY_REGISTRY.setdefault(_normalize(policy_cls.name), policy_cls)
    _DEFAULTS_LOADED = True


def _ensure_defaults() -> None:
    if not _DEFAULTS_LOADED:
        register_default_policies()


def list_policies() -> list[str]:
    """Sorted names of all registered policies."""
    _ensure_defaults()
    return sorted(_POLICY_REGISTRY)


def get_policy(name: str, intrinsic: Any, **cfg_kwargs: Any) -> InProcessPolicy:
    """Instantiate a registered policy.

    Args:
        name: Policy name (``iplanner``, ``vint``, ``navdp``, ``viplanner``, ``x_navdp``); hyphens are accepted.
        intrinsic: Camera intrinsic matrix ``[3, 3]`` the policy will receive images from.
        **cfg_kwargs: Fields of the policy's config dataclass (``device``, ``checkpoint``, ``task``, ...).

    Raises:
        KeyError: If the name is not registered.
    """
    _ensure_defaults()
    key = _normalize(name)
    if key not in _POLICY_REGISTRY:
        raise KeyError(f"Policy '{name}' not found in registry. Available policies: {list_policies()}")
    return _POLICY_REGISTRY[key](intrinsic, **cfg_kwargs)
