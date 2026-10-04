# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmarks package for nav_arena runs, sweeps, and tracking."""

from nav_arena.benchmarks.manifest import (
    VALID_RUN_STATUSES,
    BatchManifest,
    RunRecord,
    RunStatus,
)
from nav_arena.benchmarks.spec import (
    VALID_METHODS,
    VALID_METHOD_FAMILIES,
    EpisodeLimits,
    RunSpec,
    VizCfg,
    apply_overrides,
    validate_spec,
)

_SESSION_EXPORTS = {
    "EpisodeSpec",
    "RunSession",
    "SessionKey",
    "print_result_summary",
}

__all__ = [
    "BatchManifest",
    "EpisodeLimits",
    "EpisodeSpec",
    "RunRecord",
    "RunSession",
    "RunSpec",
    "RunStatus",
    "SessionKey",
    "VALID_METHODS",
    "VALID_METHOD_FAMILIES",
    "VALID_RUN_STATUSES",
    "VizCfg",
    "apply_overrides",
    "print_result_summary",
    "validate_spec",
]


def __getattr__(name: str):
    if name in _SESSION_EXPORTS:
        from nav_arena.benchmarks import session
        return getattr(session, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

