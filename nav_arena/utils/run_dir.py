# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""What an episode recorder writes into a run directory, and helpers to detect or remove it.

Shared by the episode recorder (which refuses to overwrite a recorded run) and the benchmark orchestrator (which
clears a run explicitly before retrying it). Stdlib only.
"""

from __future__ import annotations

from pathlib import Path

RECORDER_FILES = ("settings.json", "steps.jsonl", "summary.json", "goal_rgb.png")
"""Fixed-name files written by the recorder."""
RECORDER_GLOBS = ("rgb_*.png", "depth_m_*.npy")
"""Snapshot files written by the recorder."""


def recorded_artifacts(run_dir: Path | str) -> list[Path]:
    """Files in ``run_dir`` that a recorder wrote (empty if the directory is missing or holds only other files)."""
    directory = Path(run_dir)
    if not directory.is_dir():
        return []
    found = [directory / name for name in RECORDER_FILES if (directory / name).exists()]
    for pattern in RECORDER_GLOBS:
        found.extend(sorted(directory.glob(pattern)))
    return found


def clear_recorded_artifacts(run_dir: Path | str) -> int:
    """Delete everything the recorder wrote in ``run_dir`` (keeps ``run_spec.json``, logs, and other files).

    Returns:
        The number of files removed.
    """
    artifacts = recorded_artifacts(run_dir)
    for path in artifacts:
        path.unlink()
    return len(artifacts)
