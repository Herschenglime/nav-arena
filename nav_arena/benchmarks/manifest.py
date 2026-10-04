# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manifest state machine and record tracking for benchmark batches and runs.

This module provides data models and state transitions for orchestrating batch
navigation evaluations. It guarantees zero simulation, deep learning, or ROS imports,
ensuring fast and lightweight manifest operations in orchestrators and CLI tools.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
import os
from pathlib import Path
from typing import Any


class RunStatus(str, Enum):
    """Execution status for an individual benchmark run."""

    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


VALID_RUN_STATUSES: tuple[str, ...] = tuple(s.value for s in RunStatus)


def _utcnow_iso() -> str:
    """Return current UTC time formatted as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RunRecord:
    """Record of an individual episode run within a batch or single evaluation."""

    id: str
    status: str
    method: str
    method_family: str
    method_params: dict[str, Any] = field(default_factory=dict)
    robot: str = "dingo"
    scene: str = "kujiale_0003"
    route: str | None = None
    seed: int = 0
    pid: int | None = None
    started: str | None = None
    ended: str | None = None
    exit_code: int | None = None
    terminal_cause: str | None = None
    run_dir: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.status, RunStatus):
            self.status = self.status.value
        elif isinstance(self.status, str):
            self.status = self.status.lower()
        if self.status not in VALID_RUN_STATUSES:
            raise ValueError(f"Invalid status '{self.status}'. Must be one of {VALID_RUN_STATUSES}.")
        if self.method_params is None:
            self.method_params = {}
        if self.robot is None:
            self.robot = "dingo"
        if self.scene is None:
            self.scene = "kujiale_0003"
        if self.method_family is None:
            self.method_family = "in_process"
        if self.pid is not None:
            self.pid = int(self.pid)
        if self.seed is not None:
            self.seed = int(self.seed)
        else:
            self.seed = 0
        if self.exit_code is not None:
            self.exit_code = int(self.exit_code)

    def to_dict(self) -> dict[str, Any]:
        """Convert record to a JSON-serializable dictionary."""
        return {
            "id": self.id,
            "status": self.status,
            "method": self.method,
            "method_family": self.method_family,
            "method_params": copy.deepcopy(self.method_params),
            "robot": self.robot,
            "scene": self.scene,
            "route": self.route,
            "seed": self.seed,
            "pid": self.pid,
            "started": self.started,
            "ended": self.ended,
            "exit_code": self.exit_code,
            "terminal_cause": self.terminal_cause,
            "run_dir": self.run_dir,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunRecord:
        """Construct a RunRecord from a dictionary."""
        d = dict(data)
        return cls(
            id=str(d["id"]),
            status=str(d.get("status") or RunStatus.QUEUED.value),
            method=str(d["method"]),
            method_family=str(d["method_family"]) if d.get("method_family") is not None else "in_process",
            method_params=dict(d.get("method_params") or {}),
            robot=str(d["robot"]) if d.get("robot") is not None else "dingo",
            scene=str(d["scene"]) if d.get("scene") is not None else "kujiale_0003",
            route=str(d["route"]) if d.get("route") is not None else None,
            seed=int(d["seed"]) if d.get("seed") is not None else 0,
            pid=int(d["pid"]) if d.get("pid") is not None else None,
            started=str(d["started"]) if d.get("started") is not None else None,
            ended=str(d["ended"]) if d.get("ended") is not None else None,
            exit_code=int(d["exit_code"]) if d.get("exit_code") is not None else None,
            terminal_cause=str(d["terminal_cause"]) if d.get("terminal_cause") is not None else None,
            run_dir=str(d["run_dir"]) if d.get("run_dir") is not None else None,
            error=str(d["error"]) if d.get("error") is not None else None,
        )


@dataclass
class BatchManifest:
    """State machine and tracker for a batch sweep containing multiple runs."""

    batch_id: str
    runs: list[RunRecord] = field(default_factory=list)
    created_at: str = field(default_factory=_utcnow_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = _utcnow_iso()
        if self.metadata is None:
            self.metadata = {}
        if self.runs:
            parsed_runs: list[RunRecord] = []
            seen_ids: set[str] = set()
            for r in self.runs:
                rec = RunRecord.from_dict(r) if isinstance(r, dict) else r
                if not isinstance(rec, RunRecord):
                    raise TypeError(f"Expected RunRecord or dict, got {type(r).__name__}")
                if rec.id in seen_ids:
                    raise ValueError(
                        f"Duplicate run record ID '{rec.id}' in batch manifest '{self.batch_id}'."
                    )
                seen_ids.add(rec.id)
                parsed_runs.append(rec)
            self.runs = parsed_runs

    def get_run(self, run_id: str) -> RunRecord:
        """Find a RunRecord by ID.

        Args:
            run_id: Identifier of the run.

        Returns:
            The matching RunRecord.

        Raises:
            KeyError: If no run with the given ID exists.
        """
        for r in self.runs:
            if r.id == run_id:
                return r
        raise KeyError(f"Run record '{run_id}' not found in batch manifest '{self.batch_id}'.")

    def add_run(self, run: RunRecord) -> None:
        """Add a new RunRecord to the manifest.

        Args:
            run: The RunRecord to add.

        Raises:
            ValueError: If a run with the same ID already exists.
        """
        if any(r.id == run.id for r in self.runs):
            raise ValueError(f"Run record with ID '{run.id}' already exists in batch manifest.")
        self.runs.append(run)

    # -------------------------------------------------------------------------
    # State transitions
    # -------------------------------------------------------------------------

    def mark_running(
        self,
        run_id: str,
        pid: int,
        started: str | None = None,
        run_dir: str | None = None,
    ) -> None:
        """Transition a run to RUNNING state.

        Allowed source states:
            - QUEUED (normal progression)
            - RUNNING (re-entry or pid update)
            - FAILED, TIMEOUT (resume / retry)

        Disallowed source states:
            - DONE, SKIPPED
        """
        run = self.get_run(run_id)
        if run.status in (RunStatus.DONE.value, RunStatus.SKIPPED.value):
            raise ValueError(
                f"Cannot transition run '{run_id}' from terminal state '{run.status}' to '{RunStatus.RUNNING.value}'."
            )

        run.status = RunStatus.RUNNING.value
        run.pid = int(pid)
        run.started = started or _utcnow_iso()
        run.ended = None
        run.exit_code = None
        run.terminal_cause = None
        run.error = None
        if run_dir is not None:
            run.run_dir = str(run_dir)

    def mark_done(
        self,
        run_id: str,
        exit_code: int,
        terminal_cause: str,
        ended: str | None = None,
        run_dir: str | None = None,
    ) -> None:
        """Transition a run to DONE state.

        Allowed source states:
            - RUNNING
        """
        run = self.get_run(run_id)
        if run.status != RunStatus.RUNNING.value:
            raise ValueError(
                f"Cannot transition run '{run_id}' from '{run.status}' to '{RunStatus.DONE.value}'. Expected '{RunStatus.RUNNING.value}'."
            )

        run.status = RunStatus.DONE.value
        run.exit_code = int(exit_code)
        run.terminal_cause = str(terminal_cause)
        run.ended = ended or _utcnow_iso()
        if run_dir is not None:
            run.run_dir = str(run_dir)

    def mark_failed(
        self,
        run_id: str,
        exit_code: int,
        error: str | None = None,
        ended: str | None = None,
        run_dir: str | None = None,
    ) -> None:
        """Transition a run to FAILED state.

        Allowed source states:
            - RUNNING (execution failure)
            - QUEUED (pre-execution launch failure)
            - FAILED (idempotent / error update)
        """
        run = self.get_run(run_id)
        if run.status not in (RunStatus.RUNNING.value, RunStatus.QUEUED.value, RunStatus.FAILED.value):
            raise ValueError(
                f"Cannot transition run '{run_id}' from '{run.status}' to '{RunStatus.FAILED.value}'."
            )

        run.status = RunStatus.FAILED.value
        run.exit_code = int(exit_code)
        run.error = error
        run.ended = ended or _utcnow_iso()
        if run_dir is not None:
            run.run_dir = str(run_dir)

    def mark_timeout(
        self,
        run_id: str,
        ended: str | None = None,
        error: str | None = None,
        exit_code: int | None = None,
        run_dir: str | None = None,
    ) -> None:
        """Transition a run to TIMEOUT state.

        Allowed source states:
            - RUNNING
            - TIMEOUT (idempotent update)
        """
        run = self.get_run(run_id)
        if run.status not in (RunStatus.RUNNING.value, RunStatus.TIMEOUT.value):
            raise ValueError(
                f"Cannot transition run '{run_id}' from '{run.status}' to '{RunStatus.TIMEOUT.value}'. Expected '{RunStatus.RUNNING.value}'."
            )

        run.status = RunStatus.TIMEOUT.value
        run.exit_code = int(exit_code) if exit_code is not None else -1
        run.terminal_cause = "timeout"
        run.error = error or "Episode execution timed out"
        run.ended = ended or _utcnow_iso()
        if run_dir is not None:
            run.run_dir = str(run_dir)

    def mark_skipped(
        self,
        run_id: str,
        reason: str | None = None,
        ended: str | None = None,
    ) -> None:
        """Transition a run to SKIPPED state.

        Allowed source states:
            - QUEUED
        """
        run = self.get_run(run_id)
        if run.status != RunStatus.QUEUED.value:
            raise ValueError(
                f"Cannot transition run '{run_id}' from '{run.status}' to '{RunStatus.SKIPPED.value}'. Expected '{RunStatus.QUEUED.value}'."
            )

        run.status = RunStatus.SKIPPED.value
        run.terminal_cause = "skipped"
        run.error = reason
        run.ended = ended or _utcnow_iso()

    def reset_run(self, run_id: str) -> None:
        """Reset a run back to QUEUED state for retrying.

        Allowed source states:
            - RUNNING, FAILED, TIMEOUT, QUEUED
        """
        run = self.get_run(run_id)
        if run.status in (RunStatus.DONE.value, RunStatus.SKIPPED.value):
            raise ValueError(
                f"Cannot reset run '{run_id}' from terminal state '{run.status}' to '{RunStatus.QUEUED.value}'."
            )

        run.status = RunStatus.QUEUED.value
        run.pid = None
        run.started = None
        run.ended = None
        run.exit_code = None
        run.terminal_cause = None
        run.error = None

    # -------------------------------------------------------------------------
    # Queries & Helpers
    # -------------------------------------------------------------------------

    def counts_by_status(self) -> dict[str, int]:
        """Return a mapping of status values to run counts."""
        counts = {s.value: 0 for s in RunStatus}
        for r in self.runs:
            counts[r.status] = counts.get(r.status, 0) + 1
        return counts

    def is_finished(self) -> bool:
        """Return True if all runs are in a terminal state (done, failed, timeout, skipped)."""
        terminal_statuses = {
            RunStatus.DONE.value,
            RunStatus.FAILED.value,
            RunStatus.TIMEOUT.value,
            RunStatus.SKIPPED.value,
        }
        return all(r.status in terminal_statuses for r in self.runs)

    def summary_dict(self) -> dict[str, Any]:
        """Return a high-level summary dictionary of the batch progress."""
        return {
            "batch_id": self.batch_id,
            "total": len(self.runs),
            "counts": self.counts_by_status(),
            "is_finished": self.is_finished(),
            "created_at": self.created_at,
        }

    # -------------------------------------------------------------------------
    # Serialization and atomic persistence
    # -------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Convert BatchManifest to a JSON-serializable dictionary."""
        return {
            "batch_id": self.batch_id,
            "created_at": self.created_at,
            "metadata": copy.deepcopy(self.metadata),
            "runs": [r.to_dict() for r in self.runs],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize BatchManifest to a formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BatchManifest:
        """Construct BatchManifest from a dictionary."""
        d = dict(data)
        runs_data = d.get("runs") or []
        runs = [RunRecord.from_dict(r) if isinstance(r, dict) else r for r in runs_data]
        return cls(
            batch_id=str(d["batch_id"]),
            runs=runs,
            created_at=str(d.get("created_at") or ""),
            metadata=dict(d.get("metadata") or {}),
        )

    @classmethod
    def from_json(cls, json_str: str) -> BatchManifest:
        """Deserialize BatchManifest from a JSON string."""
        return cls.from_dict(json.loads(json_str))

    def save(self, path: Path | str) -> None:
        """Atomically persist manifest to path using a temporary file, fsync, and os.replace.

        Args:
            path: Target JSON file path (e.g. `<batch_dir>/manifest.json`).
        """
        target_path = Path(path).resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target_path.with_suffix(".tmp")

        payload = self.to_json(indent=2)
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, target_path)

    @classmethod
    def load(cls, path: Path | str) -> BatchManifest:
        """Load and parse BatchManifest from a JSON file path.

        Args:
            path: Path to the JSON manifest file.

        Returns:
            Parsed BatchManifest instance.

        Raises:
            FileNotFoundError: If the manifest file does not exist.
            ValueError: If parsing or deserialization fails.
        """
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Manifest file not found: {p}")
        try:
            content = p.read_text(encoding="utf-8")
            return cls.from_json(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Corrupt manifest JSON file '{p}': {exc}") from exc
        except Exception as exc:
            raise ValueError(f"Failed to load manifest from '{p}': {exc}") from exc
