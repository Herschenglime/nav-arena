# Embodiment expansion: progress checklist

Handoff file. The design and rationale are in [`embodiment-expansion-plan.md`](embodiment-expansion-plan.md).
Update this file in the same commit as the work it describes, so it always matches the branch.

- **Branch:** `feat/embodiment-expansion` (from `main` after PR #1)
- **Rules:** commit locally, never push (the user pushes). Headless sims via `./agy_python.sh` are OK; GUI checks
  and the Nav2 regression are the user's. Install packages only into `env_isaaclab` with `uv pip --no-deps`.
- **Last updated:** 2026-10-04
- **Next action:** Phase 1 step 7 (`tests/integration/test_integration_kaya.py`)

## Phase 0: drive-agnostic base (diff-drive results must not change)

- [x] 0. Baseline recorded BEFORE changing code: see `embodiment-expansion-baseline.md` (runs are deterministic, so
  the post-refactor comparison is exact; nova_carter's camera check already FAILS on `main`, which is expected)
  - [x] unit tests pass (441)
  - [x] `verify_embodiment --camera` for nova_carter (camera check fails, pre-existing) and dingo
  - [x] `verify_task --camera` for both
  - [x] `verify_baseline --method iplanner --route hall_straight` for both
- [x] 1. Body-twist `[vx, vy, wz]` command everywhere (committed; dingo verify_embodiment/verify_task match baseline)
  - [x] `DifferentialDriveAction`: `action_dim = 3`, 3-tuple scale/offset, `vy` ignored
  - [x] `runner.py`: warm-up and step actions width 3, width read from `PointNavTask.action_dim`
  - [x] `ros2/adapters/action_adapter.py`: `TwistActionAdapter` fills `linear.x, linear.y, angular.z`
  - [x] `verify_embodiment.py`, `verify_task.py`: action shapes (`run_ros2_nav.py`/`verify_nav2.py` pass the adapter's tensor straight through, no change)
- [x] 2. `RobotEmbodimentCfg`: `drive_type` (`diff | holonomic | quadruped`, validated), `max_lateral_speed`, `is_policy_driven()` (committed)
- [x] 3. Embodiment-owned `sim_dt`, `decimation`, `spawn_height` (+ `step_dt`); render intervals derived from rates (33 Hz / 5 Hz); `session.py` step dt from the embodiment; scripts pass 2-tuple spawn so z comes from the embodiment (committed)
- [x] 4. `contact_bodies` separate from `body_link` (`contact_body_expr`) (committed)
- [x] 5. Drive-aware `follow_path -> (vx, vy, wz)`; bit-identical when `max_lateral_speed == 0` (tested on 500 random paths); session passes `min(embodiment.max_lateral_speed, max_speed)` (committed)
- [x] 6. Embodiment variants (committed). Implemented as:
  - `EmbodimentVariant(name, description, overrides)` in `embodiments/base.py`; `register_embodiment(name, target, variants=, default_variant=, drive_type=)`;
    `get_embodiment("kaya")` = default variant, `"kaya.native"` = another; `resolve_embodiment_name` pins the canonical `kaya.mast`
  - registration validates override field names (typo = error), default variant, duplicates, dots in names
  - `sensors` field (`lidar`, `camera`); `create_embodiment_camera_cfg` raises for a variant without a camera, the scene drops the lidar
    when absent
  - **Light registry:** targets may be lazy `"module:Class"` strings and `nav_arena.embodiments` exports heavy names lazily (same
    pattern as `scenes/`), so the CLI can list/validate robots without importing torch or Isaac Lab (CONTRIBUTING invariant)
  - re-registering a name with `variants=None` keeps its variants (modules self-register on import); `variants=()` clears them
  - CLI: `nav_arena robots list` (top-level, beside `routes`; NOT `tools robots list` as the plan said). `run`/`sweep` validate
    `--robot` up front and record the canonical name
  - Deviation from the plan: the camera requirement is enforced when the camera is built (before the scene loads), not in
    `validate_spec`, because `spec.py` must stay free of simulator/ML imports. All current policies need the camera
  - Built-in robots have no variants yet; Kaya adds `kaya.mast` / `kaya.native` in Phase 1. `kaya.py` must NOT self-register with
    bare `register_embodiment("kaya", ...)` unless it passes the variants too (it can; `variants=None` keeps them)
- [x] 7. Unit tests and docs updated (`adding-a-robot.md` incl. variants section, CONTRIBUTING, README) (committed)
- [x] Exit: 475 unit tests pass; nova_carter and dingo match the recorded baseline EXACTLY (summary.json fields and every steps.jsonl row identical; verify_embodiment/verify_task numbers identical, nova_carter camera check fails as before)
- [x] Phase 0 committed

## Phase 1: holonomic drive (Kaya)

- [x] 1. Inspect the Kaya USD with `pxr`: findings in `embodiment-expansion-kaya-notes.md` (wheel geometry + `isaacmecanumwheel:*` attrs authored in the USD; 30 passive roller joints; camera ~0.16 m high pitched 20 deg; ground plane outside the default prim)
- [x] 2. `holonomic_matrix` / `holonomic_ik` / `holonomic_fk` in `kinematics.py` + 15 unit tests (`test_holonomic_kinematics.py`, Kaya + 4-wheel mecanum) (committed)
- [x] 2b. `WheelGeometry` dataclass + `read_wheel_geometry()` in `wheel_geometry.py`; reads `isaacmecanumwheel:*` attrs and joint frames from the live USD stage — no hand-copied numbers; 9 unit tests in `test_wheel_geometry.py` (committed)
- [x] 3. `HolonomicDriveAction` / `HolonomicDriveActionCfg` in `actions.py`: resolves wheel joints, reads USD geometry at construction (or takes a `WheelGeometry` from config), clips per-axis, writes joint velocity targets (committed)
- [x] 4. `embodiments/kaya.py` with variants `kaya.mast` (default) and `kaya.native`; register and export (committed)
- [x] 5. Roller-jitter check in `verify_embodiment`: zero-command drift < 0.004 m over 1.0s; tuned wheel damping to USD authored 174.5 (committed)
- [x] 6. Per-axis drive check in `verify_embodiment.py` (forward, strafe, rotate) (committed)
- [x] 7. `tests/integration/test_integration_kaya.py` (includes the `HolonomicController` cross-check)
- [x] 8. Closed loop `verify_baseline` for kaya vs nova_carter
- [x] Docs updated; Phase 1 committed

## Phase 2: quadruped (Unitree Go2)

- [ ] 1. `embodiments/policies.py`: fetch and cache `physx_policy.pt`, verify SHA-256 `984c802b…abe0f`, env override
- [ ] 2. `embodiments/go2.py` with `PreTrainedPolicyActionCfg` (`debug_vis=False`, `low_level_decimation=4`)
- [ ] 3. `lidar_ray_alignment` field (`yaw` for Go2)
- [ ] 4. Checks on `PreTrainedPolicyAction` as-is (joint order, lambdas not serialized by the worker)
- [ ] 5. Quadruped checks in `verify_embodiment.py` (10 s stand, per-axis tracking)
- [ ] 6. `tests/integration/test_integration_go2.py`
- [ ] 7. Closed loop `verify_baseline` for go2 vs nova_carter
- [ ] Docs updated; Phase 2 committed

## Notes and decisions log

- 2026-10-04: baseline finding: nova_carter's camera sees its own chassis (nearest depth 0.061 m, depth max 0.12 m) on `main`.
  Not part of this work; consider a separate fix. Check whether `hall_straight` results for nova_carter are meaningful.

- 2026-10-04: Go2 policy verified to exist; no training needed. Nav2 for new robots is out of scope (future work).
- 2026-10-04: Kaya selected over O3dyn (too large) and Ridgeback (arm only). Lab mecanum robot is future work.
