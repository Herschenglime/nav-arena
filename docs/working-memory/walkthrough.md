# Walkthrough: Unified `nav_arena` Entry Point & Batch Sweeps

All phases (Phases 0 through 4) of the unified entry point design have been implemented, tested, and validated end-to-end.

---

## 1. Summary of Changes

### Phase 0: Lazy Dingo Asset Resolution
- **Files**: [`nav_arena/embodiments/dingo.py`](file:///home/robopi/simulation/nav_arena/nav_arena/embodiments/dingo.py), [`nav_arena/embodiments/__init__.py`](file:///home/robopi/simulation/nav_arena/nav_arena/embodiments/__init__.py)
- Replaced eager top-level `_resolve_dingo_usd()` with `create_dingo_articulation_cfg()`.
- Guaranteed that importing `nav_arena.embodiments` never fails or builds derived assets before simulator startup.

### Phase 1: `RunSpec`, `RunSession`, Worker Subprocess & Single Run
- **Files**: [`nav_arena/benchmarks/spec.py`](file:///home/robopi/simulation/nav_arena/nav_arena/benchmarks/spec.py), [`session.py`](file:///home/robopi/simulation/nav_arena/nav_arena/benchmarks/session.py), [`worker.py`](file:///home/robopi/simulation/nav_arena/nav_arena/benchmarks/worker.py), [`scripts/verify_baseline.py`](file:///home/robopi/simulation/nav_arena/nav_arena/scripts/verify_baseline.py), [`cli.py`](file:///home/robopi/simulation/nav_arena/nav_arena/cli.py)
- Introduced `RunSpec` with canonical hashing, serialization round-trip, pure validation, and override logging.
- Decoupled `RunSession` and `EpisodeSpec`, and introduced `nav_arena.benchmarks.worker` returning exit codes:
  - `0`: Goal reached
  - `2`: Terminated without reaching goal (collision, stalled, timeout)
  - `1`: Infrastructure failure
- Refactored `verify_baseline.py` into a thin shim delegating to the unified benchmark library.
- Added `[project.scripts] nav_arena = "nav_arena.cli:main"` in [`pyproject.toml`](file:///home/robopi/simulation/nav_arena/pyproject.toml).

### Phase 2: Batch Sweeps & Manifest State Machine
- **Files**: [`nav_arena/benchmarks/manifest.py`](file:///home/robopi/simulation/nav_arena/nav_arena/benchmarks/manifest.py), [`sweep.py`](file:///home/robopi/simulation/nav_arena/nav_arena/benchmarks/sweep.py), [`sweeps/baselines_kujiale_0003.yaml`](file:///home/robopi/simulation/nav_arena/sweeps/baselines_kujiale_0003.yaml)
- Implemented `BatchManifest` with atomic writes (`fsync` + replace) and strict lifecycle status transitions (`queued` -> `running` -> `done`/`failed`/`timeout`/`skipped`).
- Implemented Cartesian matrix expansion (`robots x methods x routes x seeds`).
- Implemented sequential sweep execution under `managed_process` with pre-flight process checks, per-run timeout, live streaming, Ctrl-C safety, and `--resume` support.

### Phase 3: Run Tracking & Metrics Comparison
- **Files**: [`nav_arena/benchmarks/tracking.py`](file:///home/robopi/simulation/nav_arena/nav_arena/benchmarks/tracking.py)
- Implemented `nav_arena runs list [--batch ID]` (discovering both batch sweeps and standalone runs).
- Implemented `nav_arena runs show <run-or-batch-id>`.
- Implemented `nav_arena runs compare <batch-id> [--by method|route|robot]` computing success rate, mean ± std time-to-goal, mean path length, collision counts, and mean inference ms.
- Implemented `nav_arena runs rerun <run-id>`.

### Phase 4: System Diagnostics & Route/Map Tools
- **Files**: [`nav_arena/benchmarks/doctor.py`](file:///home/robopi/simulation/nav_arena/nav_arena/benchmarks/doctor.py), [`nav_arena/scenes/__init__.py`](file:///home/robopi/simulation/nav_arena/nav_arena/scenes/__init__.py)
- Implemented `nav_arena doctor` checking Python virtualenv, GPU, active processes, NavDP repo, model checkpoints, and map cache.
- Implemented `nav_arena routes list` and `routes show <route>`.
- Implemented `nav_arena map show` and `map generate`.
- Decoupled `nav_arena.scenes.__init__` with lazy imports to ensure read-only queries never import Isaac Lab or PyTorch.

---

## 2. Verification Results

### Tier 1 CPU Unit Tests
```bash
./agy_python.sh -m pytest -c nav_arena/pyproject.toml nav_arena/tests/unit -q
```
**Result**: **413 passed**, 0 failures in 11.46s.

### Import Isolation Audit
```bash
./agy_python.sh -c "
import sys
import nav_arena.cli
import nav_arena.benchmarks.spec
import nav_arena.benchmarks.manifest
import nav_arena.benchmarks.tracking
import nav_arena.benchmarks.doctor
forbidden = ('isaacsim', 'isaaclab', 'omni', 'pxr', 'rclpy', 'torch')
assert not any(m == p or m.startswith(f'{p}.') for p in forbidden for m in sys.modules)
print('PASS')
"
```
**Result**: Clean pass with 0 leaked simulation or heavy modules. Startup time < 0.15s.

### Developer Compatibility Check
```bash
./agy_python.sh nav_arena/nav_arena/scripts/verify_baseline.py --help
```
**Result**: Exits cleanly with code 0.

### End-to-End Headless Simulation Run
```bash
./agy_python.sh -m nav_arena.cli run --method iplanner --robot dingo --scene kujiale_0003 --route hall_straight
```
**Simulation Output**:
```text
[INFO] t=  0.0s pos=(-6.40, 0.50) goal_distance=6.00 m v=0.00 m/s w=+0.00 rad/s
[INFO] t=  5.0s pos=(-4.95, 0.51) goal_distance=4.55 m v=0.30 m/s w=-0.04 rad/s
[INFO] t= 10.0s pos=(-3.45, 0.49) goal_distance=3.05 m v=0.30 m/s w=+0.01 rad/s
[INFO] t= 15.0s pos=(-1.95, 0.57) goal_distance=1.56 m v=0.30 m/s w=-0.03 rad/s
[INFO] Episode finished: goal_reached after 18.9 s sim time (95 plans, 8 ms/plan, 5.61 m travelled)
[INFO] === RESULT ===
[INFO] Terminal cause:       goal_reached
[INFO] Time to goal:         18.90 s
[INFO] Distance remaining:   0.40 m (started at 6.00 m)
[INFO] Path length driven:   5.61 m over 18.9 s sim time
[INFO] Plans / stop requests: 95 / 0; 8 ms per plan
[INFO] Run artifacts:        /home/robopi/simulation/nav_arena/cache/runs/20261004_185134_iplanner_dingo_hall_straight_b98e
[SUCCESS] [PASS] Goal reached - cause=goal_reached
```
**Result**: Exit code 0. Generated full artifacts (`settings.json`, `run_spec.json`, `steps.jsonl`, `summary.json`, `worker.log`, `results.csv`, RGB/depth snapshots).

### CLI Query Verification
```bash
./agy_python.sh -m nav_arena.cli runs show 20261004_185134_iplanner_dingo_hall_straight_b98e
```
**Output**:
```text
Run: 20261004_185134_iplanner_dingo_hall_straight_b98e
Directory: /home/robopi/simulation/nav_arena/cache/runs/20261004_185134_iplanner_dingo_hall_straight_b98e
Status: done
Outcome:
  Success:        True
  Terminal Cause: goal_reached
  Exit Code:      0
Metrics:
  Time to Goal:     18.90 s
  Path Length:      5.61 m
  Mean Inference:   7.8 ms
  Steps:            945
  Final Goal Dist:  0.40 m
Settings:
  method: iplanner
  robot: dingo
  scene: kujiale_0003
  route: hall_straight
  seed: 0
Log: /home/robopi/simulation/nav_arena/cache/runs/20261004_185134_iplanner_dingo_hall_straight_b98e/worker.log
```

---

## 3. Git History (Unpushed Local Commits)

```text
52799e0 fix(benchmarks): support pre-created run dirs and standalone runs in runs list
7800579 feat(cli): introduce doctor, routes, and map subcommands
dec6416 feat(benchmarks): introduce run tracking and query subcommands
30cd5da feat(benchmarks): introduce sweep matrix planner and nav_arena sweep subcommand
c157a80 feat(benchmarks): introduce BatchManifest state machine and atomic persistence
3f57f2a feat(cli): introduce nav_arena run subcommand and console script
2a31a62 feat(benchmarks): introduce RunSession, worker subprocess, and verify_baseline shim
5673592 feat(benchmarks): introduce RunSpec, validation, and override engine
1ff5d26 refactor(embodiments): resolve Dingo USD asset lazily on demand
```
