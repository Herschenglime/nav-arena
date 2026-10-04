# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tolerant readers for the small JSON/YAML files written by runs and batches.

Run directories can be partially written or corrupted (a killed worker, a manual edit). These helpers return ``None``
for a missing or unreadable file, log why at debug level, and let any other exception propagate.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("nav_arena.benchmarks.io")


def load_json(path: Path | str) -> Any | None:
    """Parse a JSON file, or return ``None`` if it is missing, unreadable, or not valid JSON."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.debug("Could not read %s: %s", path, exc)
        return None


def load_yaml(path: Path | str) -> Any | None:
    """Parse a YAML file, or return ``None`` if it is missing, unreadable, or not valid YAML."""
    try:
        return yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        logger.debug("Could not read %s: %s", path, exc)
        return None
