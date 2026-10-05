# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Checkpoint locations, provenance, and integrity checks for the NavDP-family baselines."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

from nav_arena.utils.paths import NAVDP_ROOT


@dataclass(frozen=True)
class CheckpointInfo:
    """A released checkpoint and where it comes from."""

    planner: str
    folder: str
    """Baseline directory under ``NavDP/baselines``."""
    filename: str
    source_url: str
    sha256: str
    """SHA-256 of the file the baselines load (after any documented conversion)."""
    note: str = ""


# Local hashes are of the files as loaded here. ViNT's and VIPlanner's Mask2Former checkpoints are converted from the
# authors' training checkpoints into plain state dicts using restricted deserialization (tensors unchanged).
CHECKPOINTS: dict[str, CheckpointInfo] = {
    "iplanner": CheckpointInfo(
        "iplanner",
        "iplanner",
        "iplanner.pth",
        "https://github.com/leggedrobotics/iPlanner",
        "b801f444697b238dcc667fa16d2f0b1fec000608b21e56b448123b67a2c83c1d",
        "original module checkpoint converted to a plain state dict; hash is of the converted local file",
    ),
    "vint": CheckpointInfo(
        "vint",
        "vint",
        "vint.pth",
        "https://drive.google.com/file/d/1ckrceGb5m_uUtq3pD8KHwnqtJgPl6kF5/view",
        "38fe46147136192862b918b12172b16b2d938ffeeb27ff7ff3c2f87c92b5d5e9",
        "converted from the authors' training checkpoint (source sha256 155fd72d...)",
    ),
    "navdp": CheckpointInfo(
        "navdp",
        "navdp",
        "navdp_pretrain.ckpt",
        "https://huggingface.co/InternRobotics/X-NavDP/blob/main/navdp_pretrain.ckpt",
        "3bb3ad4ab241e857bb57a4021cc6aab76d5263e81fbf80298d579053ef011947",
        "NavDP pretraining checkpoint (published in the X-NavDP asset release)",
    ),
    "x_navdp": CheckpointInfo(
        "x_navdp",
        "x-navdp",
        "x-navdp_posttrain.ckpt",
        "https://huggingface.co/InternRobotics/X-NavDP/resolve/main/x-navdp_posttrain.ckpt",
        "267089a81bbbe7a913debda6603f3f1b66a79520370ce953b2d888d793b89f24",
    ),
    "viplanner": CheckpointInfo(
        "viplanner",
        "viplanner",
        "viplanner.pt",
        "https://drive.google.com/file/d/1PY7XBkyIGESjdh1cMSiJgwwaIT0WaxIc/view",
        "2fd5219cfb160e5035d43319632b3d975637a0e770c4d455a26d3124a15ca87b",
    ),
    "viplanner_mask2former": CheckpointInfo(
        "viplanner_mask2former",
        "viplanner",
        "mask2former.pth",
        "https://drive.google.com/file/d/1DZoaLbXA1qPtg-gUKRUWS2rOH2tvDOOl/view",
        "bcc22c0b4acfb675f85dd3ef6fac2ca1b6813913a9a72b6897f0fdc784b57130",
        "converted: unchanged state_dict tensors, COCO metadata keys lowercased for MMDetection 3",
    ),
}


def baselines_root(navdp_root: Path | None = None) -> Path:
    """The ``baselines/`` directory of the NavDP checkout."""
    return Path(navdp_root or NAVDP_ROOT) / "baselines"


def default_checkpoint(planner: str, navdp_root: Path | None = None) -> Path:
    """Default on-disk path of a planner's checkpoint (``NavDP/baselines/<folder>/checkpoints/<file>``)."""
    info = CHECKPOINTS[planner]
    return baselines_root(navdp_root) / info.folder / "checkpoints" / info.filename


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            digest.update(block)
    return digest.hexdigest()
