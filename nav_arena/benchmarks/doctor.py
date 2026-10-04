# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""System health checks for nav_arena simulation, baselines, and caches."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Sequence

from nav_arena.benchmarks.sweep import check_preflight_processes
from nav_arena.methods.in_process.navdp_adapter.checkpoints import (
    CHECKPOINTS,
    default_checkpoint,
)
from nav_arena.utils.logger import get_logger
from nav_arena.utils.paths import CACHE_DIR, NAVDP_ROOT

logger = get_logger("doctor")


@dataclass
class CheckResult:
    """Outcome of an individual health check."""

    name: str
    status: str  # "PASS", "WARN", "FAIL"
    message: str
    details: list[str] = field(default_factory=list)

    @property
    def is_failure(self) -> bool:
        return self.status == "FAIL"


def check_python_environment() -> CheckResult:
    """Check Python virtual environment and interpreter version."""
    py_version = sys.version_info
    ver_str = f"{py_version[0]}.{py_version[1]}.{py_version[2]}"
    in_venv = bool(os.environ.get("VIRTUAL_ENV")) or (sys.prefix != sys.base_prefix)

    details = [
        f"Executable: {sys.executable}",
        f"Prefix: {sys.prefix}",
        f"Version: {ver_str}",
    ]
    if os.environ.get("VIRTUAL_ENV"):
        details.append(f"VIRTUAL_ENV: {os.environ.get('VIRTUAL_ENV')}")

    if py_version < (3, 10):
        return CheckResult(
            name="Python Environment",
            status="FAIL",
            message=f"Python {ver_str} is unsupported (< 3.10)",
            details=details,
        )

    if not in_venv:
        return CheckResult(
            name="Python Environment",
            status="WARN",
            message=f"Running in Python {ver_str} without an active virtualenv",
            details=details,
        )

    return CheckResult(
        name="Python Environment",
        status="PASS",
        message=f"Python {ver_str} in virtualenv ({Path(sys.prefix).name})",
        details=details,
    )


def check_gpu_device() -> CheckResult:
    """Check GPU availability and driver status using nvidia-smi."""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return CheckResult(
            name="GPU Device",
            status="FAIL",
            message="nvidia-smi not found in PATH; GPU drivers or CUDA may be missing",
            details=[],
        )

    try:
        res = subprocess.run(
            [nvidia_smi, "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, OSError) as exc:
        return CheckResult(
            name="GPU Device",
            status="FAIL",
            message=f"nvidia-smi query failed: {exc}",
            details=[],
        )

    lines = [line.strip() for line in res.stdout.strip().splitlines() if line.strip()]
    if not lines:
        return CheckResult(
            name="GPU Device",
            status="WARN",
            message="nvidia-smi reported no active GPUs",
            details=[],
        )

    gpu_summaries: list[str] = []
    details: list[str] = []
    for i, line in enumerate(lines):
        parts = [p.strip() for p in line.split(",")]
        name = parts[0] if len(parts) > 0 else "Unknown GPU"
        total_mem = parts[1] if len(parts) > 1 else "N/A"
        free_mem = parts[2] if len(parts) > 2 else "N/A"
        driver = parts[3] if len(parts) > 3 else "N/A"
        mem_str = f"{free_mem}/{total_mem} MB free" if total_mem not in ("[N/A]", "N/A", "") else "unified memory"
        gpu_summaries.append(f"{name} ({mem_str})")
        details.append(f"GPU {i}: {name} (Driver: {driver}, Memory: {mem_str})")

    return CheckResult(
        name="GPU Device",
        status="PASS",
        message=f"Found {len(lines)} GPU(s): {', '.join(gpu_summaries)}",
        details=details,
    )


def check_running_processes() -> CheckResult:
    """Check for conflicting running simulator or verification processes."""
    conflicts = check_preflight_processes()
    if conflicts:
        details = [f"[PID {pid}] {cmd}" for pid, cmd in conflicts]
        return CheckResult(
            name="Running Processes",
            status="WARN",
            message=f"{len(conflicts)} active simulator process(es) detected",
            details=details,
        )

    return CheckResult(
        name="Running Processes",
        status="PASS",
        message="No conflicting simulator processes running",
        details=[],
    )


def check_navdp_repository(navdp_root: Path | None = None) -> CheckResult:
    """Check NavDP checkout path and repository git status."""
    root = Path(navdp_root) if navdp_root is not None else NAVDP_ROOT
    if not root.is_dir():
        return CheckResult(
            name="NavDP Repository",
            status="FAIL",
            message=f"NavDP checkout missing at {root}",
            details=[f"Expected path: {root}"],
        )

    git_bin = shutil.which("git")
    details = [f"Path: {root}"]
    if not git_bin:
        return CheckResult(
            name="NavDP Repository",
            status="PASS",
            message=f"NavDP directory present at {root} (git not found)",
            details=details,
        )

    try:
        sha_res = subprocess.run(
            [git_bin, "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        sha = sha_res.stdout.strip()
        status_res = subprocess.run(
            [git_bin, "-C", str(root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
        is_dirty = bool(status_res.stdout.strip())
        status_suffix = " (dirty)" if is_dirty else " (clean)"
        details.append(f"Commit: {sha}{status_suffix}")

        return CheckResult(
            name="NavDP Repository",
            status="PASS",
            message=f"NavDP checkout present at {root} ({sha}{status_suffix})",
            details=details,
        )
    except (subprocess.CalledProcessError, OSError):
        return CheckResult(
            name="NavDP Repository",
            status="PASS",
            message=f"NavDP directory present at {root} (not a git repo or detached)",
            details=details,
        )


def check_model_checkpoints(navdp_root: Path | None = None) -> CheckResult:
    """Check availability of pretrained model weights for registered baselines."""
    missing: list[str] = []
    found: list[str] = []
    details: list[str] = []

    for name, info in sorted(CHECKPOINTS.items()):
        ckpt_path = default_checkpoint(name, navdp_root=navdp_root)
        if ckpt_path.is_file():
            found.append(name)
            size_mb = ckpt_path.stat().st_size / (1024 * 1024)
            details.append(f"[FOUND] {name}: {ckpt_path.name} ({size_mb:.1f} MB)")
        else:
            missing.append(name)
            details.append(f"[MISSING] {name}: {ckpt_path}")

    total = len(CHECKPOINTS)
    if not found:
        return CheckResult(
            name="Model Checkpoints",
            status="FAIL",
            message=f"No baseline checkpoints found (0/{total}) under NavDP/baselines",
            details=details,
        )

    if missing:
        return CheckResult(
            name="Model Checkpoints",
            status="WARN",
            message=f"{len(missing)} of {total} checkpoints missing ({', '.join(missing)})",
            details=details,
        )

    return CheckResult(
        name="Model Checkpoints",
        status="PASS",
        message=f"All {total} registered baseline checkpoints available",
        details=details,
    )


def check_map_cache(cache_dir: Path | None = None) -> CheckResult:
    """Check map cache status under CACHE_DIR / 'maps'."""
    maps_dir = (cache_dir or CACHE_DIR) / "maps"
    if not maps_dir.is_dir():
        return CheckResult(
            name="Map Cache",
            status="WARN",
            message=f"Map cache directory does not exist ({maps_dir})",
            details=[f"Expected path: {maps_dir}"],
        )

    map_yamls = sorted(maps_dir.rglob("map.yaml"))
    if not map_yamls:
        return CheckResult(
            name="Map Cache",
            status="WARN",
            message=f"No cached maps found in {maps_dir}. Generate one with 'nav_arena map generate --scene <scene>'",
            details=[f"Directory: {maps_dir}"],
        )

    scenes: list[str] = []
    details: list[str] = []
    for my in map_yamls:
        rel_parts = my.relative_to(maps_dir).parts
        scene_name = rel_parts[0] if len(rel_parts) > 1 else my.parent.name
        if scene_name not in scenes:
            scenes.append(scene_name)
        details.append(f"{scene_name}: {my}")

    return CheckResult(
        name="Map Cache",
        status="PASS",
        message=f"Found {len(map_yamls)} cached map(s): {', '.join(scenes)}",
        details=details,
    )


def run_doctor_checks(
    navdp_root: Path | None = None,
    cache_dir: Path | None = None,
) -> list[CheckResult]:
    """Run all system health checks."""
    return [
        check_python_environment(),
        check_gpu_device(),
        check_running_processes(),
        check_navdp_repository(navdp_root=navdp_root),
        check_model_checkpoints(navdp_root=navdp_root),
        check_map_cache(cache_dir=cache_dir),
    ]


def format_doctor_results(results: Sequence[CheckResult], verbose: bool = False) -> str:
    """Format doctor check results for terminal display."""
    lines = [
        "=" * 70,
        "nav_arena doctor - System Health Report",
        "=" * 70,
    ]

    has_fail = False
    has_warn = False

    for res in results:
        status_tag = f"[{res.status}]"
        lines.append(f"{status_tag:<8} {res.name}: {res.message}")
        if res.status == "FAIL":
            has_fail = True
        elif res.status == "WARN":
            has_warn = True

        if verbose or res.status in ("FAIL", "WARN"):
            for d in res.details:
                lines.append(f"         - {d}")

    lines.append("-" * 70)
    if has_fail:
        lines.append("Doctor check FAILED: resolve [FAIL] items before running simulations.")
    elif has_warn:
        lines.append("Doctor check passed with WARNINGS: some optional components or baselines are missing.")
    else:
        lines.append("All health checks PASSED. System is ready.")
    lines.append("=" * 70)

    return "\n".join(lines)


def run_doctor(
    navdp_root: Path | None = None,
    cache_dir: Path | None = None,
    verbose: bool = False,
) -> int:
    """Execute all health checks, print results, and return exit code (0 for pass, 1 for fail)."""
    results = run_doctor_checks(navdp_root=navdp_root, cache_dir=cache_dir)
    print(format_doctor_results(results, verbose=verbose))
    if any(r.is_failure for r in results):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(run_doctor())
