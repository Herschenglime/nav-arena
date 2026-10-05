# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for pinned locomotion-policy files (download, cache, override, SHA-256 verification); no network."""

from __future__ import annotations

import hashlib
import io

import pytest

from nav_arena.embodiments import policies
from nav_arena.embodiments.policies import PolicyArtifact, ensure_policy

PAYLOAD = b"pretend torchscript policy"


@pytest.fixture
def artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(policies, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.delenv("TEST_POLICY_OVERRIDE", raising=False)
    return PolicyArtifact(
        name="test_policy",
        url="https://example.invalid/policies/policy.pt",
        sha256=hashlib.sha256(PAYLOAD).hexdigest(),
        env_override="TEST_POLICY_OVERRIDE",
    )


def _serve(monkeypatch, payload=PAYLOAD):
    calls = []

    def fake_urlopen(url, timeout):
        calls.append(url)
        return io.BytesIO(payload)

    monkeypatch.setattr(policies.urllib.request, "urlopen", fake_urlopen)
    return calls


def test_downloads_once_then_uses_the_verified_cache(artifact, monkeypatch):
    """Verify the first call downloads into the cache and later calls reuse it without the network."""
    calls = _serve(monkeypatch)
    first = ensure_policy(artifact)
    second = ensure_policy(artifact)
    assert first == second == artifact.cache_path
    assert first.read_bytes() == PAYLOAD
    assert calls == [artifact.url]


def test_a_download_with_the_wrong_hash_is_rejected_and_not_cached(artifact, monkeypatch):
    """Verify a tampered or changed upstream file never lands in the cache."""
    _serve(monkeypatch, payload=b"some other policy")
    with pytest.raises(ValueError, match="SHA-256"):
        ensure_policy(artifact)
    assert not artifact.cache_path.exists()
    assert list(artifact.cache_path.parent.glob("*.part")) == []


def test_a_corrupted_cache_is_an_error_not_a_silent_redownload(artifact, monkeypatch):
    """Verify a cached file that no longer matches the pin is reported (someone replaced it)."""
    artifact.cache_path.parent.mkdir(parents=True)
    artifact.cache_path.write_bytes(b"edited")
    with pytest.raises(ValueError, match="SHA-256"):
        ensure_policy(artifact)


def test_override_is_used_and_verified(artifact, monkeypatch, tmp_path):
    """Verify the offline override is used as-is when it matches the pin, and rejected when it does not."""
    calls = _serve(monkeypatch)
    local = tmp_path / "local.pt"
    local.write_bytes(PAYLOAD)
    monkeypatch.setenv("TEST_POLICY_OVERRIDE", str(local))
    assert ensure_policy(artifact) == local
    assert calls == []
    local.write_bytes(b"different")
    with pytest.raises(ValueError, match="SHA-256"):
        ensure_policy(artifact)
    monkeypatch.setenv("TEST_POLICY_OVERRIDE", str(tmp_path / "missing.pt"))
    with pytest.raises(FileNotFoundError):
        ensure_policy(artifact)


def test_offline_download_failure_names_the_override(artifact, monkeypatch):
    """Verify a network failure tells the user how to run offline."""

    def offline(url, timeout):
        raise OSError("no route to host")

    monkeypatch.setattr(policies.urllib.request, "urlopen", offline)
    with pytest.raises(OSError, match="TEST_POLICY_OVERRIDE"):
        ensure_policy(artifact)


def test_go2_artifact_points_at_the_isaac_sim_6_policy():
    """Verify the Go2 pin matches the policy verified for this work (Isaac Sim 6.0 sample, flat terrain)."""
    assert policies.GO2_FLAT_POLICY.url.endswith("/Isaac/6.0/Isaac/Samples/Policies/go2/physx_policy.pt")
    assert policies.GO2_FLAT_POLICY.sha256.startswith("984c802b")
