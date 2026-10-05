# Embodiment expansion: progress checklist

Handoff file. The design and rationale are in [`embodiment-expansion-plan.md`](embodiment-expansion-plan.md).
Update this file in the same commit as the work it describes, so it always matches the branch.

- **Branch:** `feat/embodiment-expansion` (from `main` after PR #1)
- **Rules:** commit locally, never push (the user pushes). Headless sims via `./agy_python.sh` are OK; GUI checks
  and the Nav2 regression are the user's. Install packages only into `env_isaaclab` with `uv pip --no-deps`.
- **Last updated:** 2026-10-05
- **Next action:** none planned; Phases 0-2 complete. Open follow-ups are under "Future work" in the plan and in the notes below

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
- [x] 5. Settle + zero-command hold in `verify_embodiment` (review fix, `0e92598`). Kaya settles in ~3.5-4.5 s after the
  spawn drop (wheels turn one after another as its weight settles onto the rollers; independent of drop height, tested at
  0.15 m and 0.10 m), then holds still to 0.1 mm. Damping 174.5 matches the USD. NOTE: the original claim of a drift check
  here was false; it was added in the review.
- [x] 6. Per-axis drive checks (review fix, `0e92598`): rules in `nav_arena/utils/drive_check.py` (unit tested). Each axis must
  reach 70% of the command in the body frame with limited off-axis motion; rotation must keep a fixed turning centre. Kaya:
  vx 96%, vy 97%, wz 78%; Dingo vx 101%, wz 93%; Nova Carter vx 100%, wz 81% (after a 1.5 s spin-up for its casters).
  The original rotation check read the quaternion as (w, x, y, z) and never measured yaw.
- [x] 7. `tests/integration/test_integration_kaya.py`; the `HolonomicController` cross-check was added in the review
  (`90a4616`): our wheel speeds match Isaac Sim's controller (built from Isaac's own USD reader) to 4.8e-7 rad/s.
- [x] 8. Closed loop `verify_baseline` for kaya vs nova_carter (see `embodiment-expansion-baseline.md`; mast variant only)
- [x] Review fixes: Kaya variant data has one source (`kaya_variants.py`, `3330df0`; before, `kaya.native` silently became the
  mast camera after a registry reset); LiDAR moved to the 0.30 m mast height; `verify_task` free-drive force covers all bodies
- [x] Docs updated; Phase 1 committed. Verified after the review: 520 unit tests; integration tests for kaya (4, incl. iPlanner
  closed loop), dingo and task: 8 passed

## Phase 2: quadruped (Unitree Go2)

- [x] 0. Settle-until-still episode start: `settle_until_still` in `nav_arena/utils/drive_check.py`, used by the runner
  (`EpisodeCfg.settle`; raises if the episode ends or the robot never settles) and by `verify_embodiment`. Settle time is
  recorded in `settings.json`. Baseline re-recorded: Dingo now STALLS on hall_straight (iPlanner fear 0.71 vs threshold 0.7
  at x = -3.55; it was 0.63 before) -> the old success was marginal; see the baseline doc.
- [x] 1. `embodiments/policies.py`: `ensure_policy` downloads `physx_policy.pt` once into `cache/policies/go2_flat/`, verifies
  SHA-256 `984c802b...abe0f` on every use; `NAV_ARENA_GO2_POLICY` override for offline use (same hash required); 6 unit tests
- [x] 2. `embodiments/go2.py`: Isaac Lab's `PreTrainedPolicyActionCfg` used as-is (`debug_vis=False`, `low_level_decimation=4`).
  Robot, observations, actions and dt all come from `training_env_cfg()` = `resolve_presets(UnitreeGo2FlatEnvCfg_PLAY())`
  (presets must be resolved, as Isaac Lab's scripts do; it also sets armature 0). Registered lazily as `go2`
- [x] 3. `lidar_ray_alignment` field (`base` | `yaw`; `yaw` for Go2), wired through `create_2d_lidar_cfg`, the scene and
  `verify_embodiment`; `verify_embodiment` now runs physics at the embodiment's `sim_dt`
- [x] 4. `PreTrainedPolicyAction` as-is: joint order OK (tracks 96-104% of commands); env cfg is never serialized (only specs).
  It needs **inference mode**: the policy output carries autograd state, which the warp backend rejects. `PointNavTask.step` and
  `.reset` now run under `torch.inference_mode()` (as Isaac Lab's play scripts do; reset too, or buffers created in a step
  cannot be updated in place). No numerical change for the other robots (per-plan logs identical)
- [x] 5. Quadruped checks in `verify_embodiment.py`: 10 s zero-command stand (drift 0.6 mm, base height steady), per-axis
  tracking: vx 96-97%, vy 104%, wz 75-93% (varies run to run), turning centre within 2 cm
- [x] 6. `tests/integration/test_integration_go2.py` (stand + axes + camera; task; iPlanner closed loop)
- [x] 7. Closed loop `verify_baseline` iplanner hall_straight: go2 goal_reached (1001 steps, path 5.62 m); see baseline doc
- [x] 8. End-to-end through the CLI (`nav_arena run --method iplanner --robot go2 --route hall_straight`): goal_reached in
  20.0 s, 101 plans, 5.62 m (same as verify_baseline). Found and fixed a stale progress-log line (printed v=0 since Phase 0)
- [x] Docs updated (guide: legged-robot section; README); Phase 2 committed. Verified: 536 unit tests; full integration suite
  13 passed (go2 3, kaya 4, dingo, task, occupancy, tf_tree)

## Notes and decisions log

- 2026-10-05: open follow-ups: (1) single-start results are fragile (Dingo flips goal_reached -> stalled on a 1.5 cm start
  change; iPlanner fear 0.63 vs 0.71 at the 0.7 threshold) -> compare methods over several starts/seeds; (2) nova_carter's
  camera sees its own chassis (pre-existing); (3) Nav2 for kaya/go2; (4) lab mecanum robot via URDF.

- 2026-10-05: Go2 stands crouched at a zero command (base 0.22 m, legs asymmetric). Checked against the same policy in Isaac
  Lab's own training env: identical (0.222 m), so it is the policy's stance, not our setup. Actuators match training
  (DCMotor Kp 25, Kd 0.5, 23.5 N m).

- 2026-10-05: review of the Phase 1 handoff found false checklist claims (drift check, HolonomicController cross-check), a
  variant-data bug, and a broken rotation check; all fixed and verified (see Phase 1 items 5-7).
- 2026-10-05: Kaya contact sensing covers all bodies (`contact_bodies=".*"`). This is correct because Isaac Lab's contact sensor
  reports normal forces only (no friction): driving gives 0 N horizontal on all 34 bodies, while a wheel hitting a wall
  gives a horizontal normal. Caveat: sloped floors or door sills would tilt the floor normal and could false-trigger.
- 2026-10-05: `verify_task` start/goal moved to y = -0.25 for all robots so small Kaya hits the same wall when reversing.
  Diff-drive verify_task numbers changed accordingly (baseline doc updated).
- 2026-10-05: twist reference point. Isaac Sim's holonomic controller commands the chassis centre of mass; ours commands the
  `base_link` origin (Kaya's COM is 3 cm ahead of it), because `base_link` is the pose navigation tracks.

- 2026-10-04: baseline finding: nova_carter's camera sees its own chassis (nearest depth 0.061 m, depth max 0.12 m) on `main`.
  Not part of this work; consider a separate fix. Check whether `hall_straight` results for nova_carter are meaningful.

- 2026-10-04: Go2 policy verified to exist; no training needed. Nav2 for new robots is out of scope (future work).
- 2026-10-04: Kaya selected over O3dyn (too large) and Ridgeback (arm only). Lab mecanum robot is future work.
