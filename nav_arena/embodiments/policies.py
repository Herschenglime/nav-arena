# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pretrained low-level locomotion policies: fetched once, pinned by SHA-256, cached locally.

A legged embodiment turns the body twist into joint targets with a locomotion policy trained in Isaac Lab. The policy
file is part of the experiment: a different file is a different robot. So it is pinned by its SHA-256 (from NVIDIA's
release manifest), downloaded once into the cache, and verified every time it is used.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import tempfile
import urllib.request

from nav_arena.utils.paths import CACHE_DIR

ISAAC_ASSET_ROOT = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0"
"""Isaac Sim 6.0 asset root (the value of ``persistent.isaac.asset_root.default``)."""


@dataclass(frozen=True)
class PolicyArtifact:
    """A pinned policy file."""

    name: str
    url: str
    sha256: str
    env_override: str
    """Environment variable that points at a local copy (offline use). The copy must have the same SHA-256."""

    @property
    def cache_path(self) -> Path:
        return CACHE_DIR / "policies" / self.name / Path(self.url).name


GO2_FLAT_POLICY = PolicyArtifact(
    name="go2_flat",
    url=f"{ISAAC_ASSET_ROOT}/Isaac/Samples/Policies/go2/physx_policy.pt",
    # From isaacsim.robot.policy.examples' bundled Go2 spec (trusted release manifest).
    sha256="984c802b8acd68c3199605ff36603f91ac9f3192eb1ebcfd80b0e764831abe0f",
    env_override="NAV_ARENA_GO2_POLICY",
)
"""Unitree Go2 flat-terrain velocity policy trained in Isaac Lab (48 observations -> 12 joint position offsets)."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified(path: Path, artifact: PolicyArtifact) -> Path:
    actual = _sha256(path)
    if actual != artifact.sha256:
        raise ValueError(
            f"Policy '{artifact.name}' at {path} has SHA-256 {actual}, expected {artifact.sha256}. "
            "It is not the pinned policy; delete it (cache) or point the override at the right file."
        )
    return path


def ensure_policy(artifact: PolicyArtifact, timeout_s: float = 60.0) -> Path:
    """Local path of a pinned policy, downloading it into the cache on first use.

    Resolution order: the artifact's environment override, then the cache, then a download into the cache. Every
    path returned has the pinned SHA-256; a mismatch raises instead of silently running a different policy.

    Raises:
        FileNotFoundError: If the override points at a missing file.
        ValueError: If a file does not match the pinned SHA-256.
        OSError: If the download fails (offline): set the override to a local copy.
    """
    override = os.environ.get(artifact.env_override)
    if override:
        path = Path(override).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"{artifact.env_override}={override} does not exist")
        return _verified(path, artifact)

    target = artifact.cache_path
    if target.is_file():
        return _verified(target, artifact)

    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False, suffix=".part") as handle:
        partial = Path(handle.name)
    try:
        try:
            with urllib.request.urlopen(artifact.url, timeout=timeout_s) as response, partial.open("wb") as out:
                while chunk := response.read(1 << 20):
                    out.write(chunk)
        except OSError as exc:
            raise OSError(
                f"Could not download policy '{artifact.name}' from {artifact.url}: {exc}. "
                f"Offline? Set {artifact.env_override} to a local copy."
            ) from exc
        _verified(partial, artifact)
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    return target
