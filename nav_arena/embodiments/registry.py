# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""In-memory registry for robot embodiments."""

from __future__ import annotations

import dataclasses
import importlib
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .base import EmbodimentVariant, RobotEmbodimentCfg

VARIANT_SEPARATOR = "."
"""Joins a robot and its variant (``kaya.native``). A dot, because names become run directory names."""

_EMBODIMENT_REGISTRY: dict[str, Any] = {}
_VARIANTS: dict[str, dict[str, EmbodimentVariant]] = {}
_DEFAULT_VARIANT: dict[str, str] = {}
_DRIVE_TYPE: dict[str, str] = {}


def _normalize_embodiment_target(target: Any) -> Any:
    """Ensure classes inheriting from RobotEmbodimentCfg are properly converted to dataclasses.

    A ``"package.module:Name"`` string is a lazy target: it is imported on first use, so the registry (and everything
    that only needs robot names and variants, like the CLI) never imports torch or Isaac Lab.
    """
    if isinstance(target, type):
        from dataclasses import dataclass
        from .base import RobotEmbodimentCfg

        # Subclasses inherit __dataclass_fields__, so is_dataclass() is True even when the
        # subclass itself was never processed; check the class's own namespace instead.
        if issubclass(target, RobotEmbodimentCfg) and "__dataclass_fields__" not in vars(target):
            return dataclass(target)
    return target


def _field_names(target: Any) -> set[str] | None:
    """Field names of a registered dataclass class or instance; None for a factory (checked when resolved)."""
    if isinstance(target, type) or dataclasses.is_dataclass(target):
        return {f.name for f in dataclasses.fields(target)}
    return None


def _check_variants(name: str, target: Any, variants: Sequence[EmbodimentVariant], default_variant: str | None) -> None:
    """Reject malformed variants at registration, so a typo can never silently do nothing."""
    if VARIANT_SEPARATOR in name:
        raise ValueError(f"Embodiment name '{name}' must not contain '{VARIANT_SEPARATOR}' (it separates the variant)")
    if not variants:
        if default_variant is not None:
            raise ValueError(f"'{name}' sets default_variant '{default_variant}' but has no variants")
        return
    names = [v.name for v in variants]
    if len(set(names)) != len(names):
        raise ValueError(f"'{name}' has duplicate variant names: {names}")
    for v in variants:
        if not v.name or VARIANT_SEPARATOR in v.name:
            raise ValueError(f"Variant name '{v.name}' of '{name}' must be non-empty and contain no '{VARIANT_SEPARATOR}'")
    if default_variant not in names:
        raise ValueError(f"'{name}' needs default_variant to be one of {names}, got {default_variant!r}")
    fields = _field_names(target)
    if fields is not None:
        reserved = {"name", "variant"}
        for v in variants:
            bad = sorted((set(v.overrides) - fields) | (set(v.overrides) & reserved))
            if bad:
                raise ValueError(f"Variant '{v.name}' of '{name}' overrides unknown or reserved fields: {bad}")


def register_embodiment(
    name: str,
    cfg_factory_or_cfg: Any = None,
    *,
    variants: Sequence[EmbodimentVariant] | None = None,
    default_variant: str | None = None,
    drive_type: str | None = None,
) -> Any:
    """Register an embodiment configuration or factory.

    Supports direct registration:
        register_embodiment("dingo", DingoEmbodimentCfg)
    or decorator syntax:
        @register_embodiment("dingo")
        class DingoEmbodimentCfg(RobotEmbodimentCfg): ...

    Args:
        name: Unique string identifier for the embodiment (e.g. 'nova_carter', 'dingo'); no dots.
        cfg_factory_or_cfg: Optional RobotEmbodimentCfg instance, class, zero-argument callable returning one, or a lazy
            ``"package.module:Name"`` string. If omitted, returns a decorator.
        variants: Known configurations of the robot (sensor placement, available sensors, limits). Each is addressed
            as ``<name>.<variant>``. ``None`` keeps the variants already registered under ``name`` (so a module that
            registers itself on import cannot wipe them); ``()`` clears them.
        default_variant: The variant the bare ``name`` resolves to. Required when ``variants`` is given.
        drive_type: Light copy of the embodiment's drive type, so listings need not import the embodiment.

    Returns:
        The registered cfg_factory_or_cfg, or a decorator function.
    """
    def store(target: Any) -> Any:
        processed = _normalize_embodiment_target(target)
        if variants is not None:
            _check_variants(name, processed, variants, default_variant)
        elif VARIANT_SEPARATOR in name:
            raise ValueError(f"Embodiment name '{name}' must not contain '{VARIANT_SEPARATOR}'")
        _EMBODIMENT_REGISTRY[name] = processed
        if variants is not None:
            _VARIANTS[name] = {v.name: v for v in variants}
            if variants:
                _DEFAULT_VARIANT[name] = default_variant
            else:
                _DEFAULT_VARIANT.pop(name, None)
        if drive_type is not None:
            _DRIVE_TYPE[name] = drive_type
        return processed

    if cfg_factory_or_cfg is None:
        def decorator(cls_or_fn: Any) -> Any:
            return store(cls_or_fn)

        return decorator

    store(cfg_factory_or_cfg)
    return cfg_factory_or_cfg


def _split_name(name: str) -> tuple[str, str | None]:
    base, sep, variant = name.partition(VARIANT_SEPARATOR)
    return base, (variant if sep else None)


def resolve_embodiment_name(name: str) -> str:
    """Canonical name for a robot: the bare name of a robot without variants, ``<robot>.<variant>`` otherwise.

    ``"kaya"`` resolves to its default variant (``"kaya.mast"``); ``"dingo"`` stays ``"dingo"``. Recording the
    canonical name keeps results unambiguous even if a robot's default variant changes later.

    Raises:
        KeyError: If the robot or variant is not registered.
    """
    base, variant = _split_name(name)
    if base not in _EMBODIMENT_REGISTRY:
        raise KeyError(f"Embodiment '{name}' not found in registry. Available embodiments: {list_embodiments()}")
    variants = _VARIANTS.get(base, {})
    if not variants:
        if variant is not None:
            raise KeyError(f"Embodiment '{base}' has no variants, so '{name}' is not valid")
        return base
    variant = variant or _DEFAULT_VARIANT[base]
    if variant not in variants:
        raise KeyError(f"Embodiment '{base}' has no variant '{variant}'. Available variants: {sorted(variants)}")
    return f"{base}{VARIANT_SEPARATOR}{variant}"


def get_embodiment(name: str) -> RobotEmbodimentCfg:
    """Retrieve an embodiment configuration by name.

    If the registered item is a callable (class or factory function), it is invoked
    to return a fresh instance. If it is already an instance, a deepcopy is returned
    to prevent in-place mutation of the registry state.

    Args:
        name: Name of the registered embodiment: ``"kaya"`` (the default variant) or ``"kaya.native"``.

    Returns:
        The resolved RobotEmbodimentCfg instance, with the variant's overrides applied.

    Raises:
        KeyError: If the embodiment or variant is not found in the registry.
        ValueError: If a variant overrides a field the embodiment does not have.
    """
    base, variant = _split_name(resolve_embodiment_name(name))
    item = _EMBODIMENT_REGISTRY[base]
    if isinstance(item, str):
        module_name, _, attr = item.partition(":")
        item = _normalize_embodiment_target(getattr(importlib.import_module(module_name), attr))
        _EMBODIMENT_REGISTRY[base] = item
    if callable(item):
        cfg = item()
    else:
        import copy

        cfg = copy.deepcopy(item)
    if hasattr(cfg, "name") and not cfg.name:
        cfg.name = base
    if variant:
        selected = _VARIANTS[base][variant]
        try:
            cfg = dataclasses.replace(cfg, variant=variant, **selected.overrides)
        except TypeError as exc:
            raise ValueError(f"Variant '{variant}' of '{base}' has invalid overrides: {exc}") from exc
    return cfg


def list_embodiments() -> list[str]:
    """List all registered embodiment names in sorted order.

    Returns:
        List of registered embodiment names (robots, without variant suffixes).
    """
    return sorted(list(_EMBODIMENT_REGISTRY.keys()))


def list_variants(name: str) -> list[EmbodimentVariant]:
    """Variants of a robot in registration order (empty for a robot without variants)."""
    if name not in _EMBODIMENT_REGISTRY:
        raise KeyError(f"Embodiment '{name}' not found in registry. Available embodiments: {list_embodiments()}")
    return list(_VARIANTS.get(name, {}).values())


def default_variant(name: str) -> str | None:
    """The variant the bare robot name resolves to, or None for a robot without variants."""
    return _DEFAULT_VARIANT.get(name)


def drive_type_of(name: str) -> str | None:
    """Drive type recorded at registration (without importing the embodiment), or None if it was not given."""
    return _DRIVE_TYPE.get(name)


def clear_registry() -> None:
    """Clear all registered embodiments (useful for isolated unit testing)."""
    _EMBODIMENT_REGISTRY.clear()
    _VARIANTS.clear()
    _DEFAULT_VARIANT.clear()
    _DRIVE_TYPE.clear()


def register_default_embodiments() -> None:
    """Register built-in framework embodiments ('nova_carter', 'dingo', 'kaya') lazily, without importing them."""
    from .base import EmbodimentVariant

    register_embodiment(
        "nova_carter", "nav_arena.embodiments.nova_carter:NovaCarterEmbodimentCfg", variants=(), drive_type="diff"
    )
    register_embodiment("dingo", "nav_arena.embodiments.dingo:DingoEmbodimentCfg", variants=(), drive_type="diff")
    register_embodiment(
        "kaya",
        "nav_arena.embodiments.kaya:KayaEmbodimentCfg",
        variants=[
            EmbodimentVariant(
                name="mast",
                description="Virtual camera mast at 0.30 m (same viewpoint as Nova Carter / Dingo).",
            ),
            EmbodimentVariant(
                name="native",
                description="Native RealSense D435 pose from the Kaya USD: ~0.16 m high, pitched 20° down.",
            ),
        ],
        default_variant="mast",
        drive_type="holonomic",
    )


register_default_embodiments()
