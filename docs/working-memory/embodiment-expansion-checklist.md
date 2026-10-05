# Embodiment expansion: progress checklist

Handoff file. The design and rationale are in [`embodiment-expansion-plan.md`](embodiment-expansion-plan.md).
Update this file in the same commit as the work it describes, so it always matches the branch.

- **Branch:** `feat/embodiment-expansion` (from `main` after PR #1)
- **Rules:** commit locally, never push (the user pushes). Headless sims via `./agy_python.sh` are OK; GUI checks
  and the Nav2 regression are the user's. Install packages only into `env_isaaclab` with `uv pip --no-deps`.
- **Last updated:** 2026-10-04
- **Next action:** Phase 0 step 6 (embodiment variants, `sensors` field, `tools robots list`)

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
- [ ] 6. Embodiment variants (`kaya.mast`, bare name = default), `sensors` field, `nav_arena tools robots list`
- [ ] 7. Unit tests and docs updated (`adding-a-robot.md` drive-type section)
- [ ] Exit: unit tests pass; nova_carter and dingo match the recorded baseline
- [ ] Phase 0 committed

## Phase 1: holonomic drive (Kaya)

- [ ] 1. Inspect the Kaya USD with `pxr` (joints, wheel geometry, `metersPerUnit`, camera pose); notes into this file
- [ ] 2. `holonomic_matrix` / `holonomic_ik` / `holonomic_fk` in `kinematics.py` + unit tests
- [ ] 3. `HolonomicDriveAction` / `HolonomicDriveActionCfg`
- [ ] 4. `embodiments/kaya.py` with variants `kaya.mast` (default) and `kaya.native`; register and export
- [ ] 5. Roller-jitter check in `verify_embodiment`; derived USD only if needed
- [ ] 6. Per-axis drive check in `verify_embodiment.py` (forward, strafe, rotate)
- [ ] 7. `tests/integration/test_integration_kaya.py` (includes the `HolonomicController` cross-check)
- [ ] 8. Closed loop `verify_baseline` for kaya vs nova_carter
- [ ] Docs updated; Phase 1 committed

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
