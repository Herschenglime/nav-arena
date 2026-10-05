# Embodiment expansion plan: holonomic drive (Kaya) + quadruped (Unitree Go2)

## Context

nav_arena compares a novel learned navigation method against baselines. Today every layer assumes a
differential-drive `[v, ω]` command (see "Adding a new drive type" in `docs/guides/adding-a-robot.md`). We want
two more embodiments, both chosen because they are well supported with existing examples:

- **Holonomic:** NVIDIA Kaya (3 omni wheels, ships with Isaac Sim, used in Isaac Sim's holonomic examples/tests).
- **Quadruped:** Unitree Go2, driven by Isaac Lab's `PreTrainedPolicyAction` **as-is** (no fork), with NVIDIA's
  published flat-terrain locomotion policy. **No policy training is in scope.**

Scope: in-process methods only. Nav2 / ROS 2 support for the new robots is future work (only keep the existing
ROS 2 diff-drive path working).

### Findings that shape the plan

- **Go2 policy exists (verified).** `{ISAAC_NUCLEUS_DIR}/Samples/Policies/go2/physx_policy.pt` (+ `physx_env.yaml`),
  referenced by Isaac Sim's bundled spec (`isaacsim.robot.policy.examples/.../bundled/go2.py`). Downloaded bytes
  match the pinned SHA-256 `984c802b…abe0f`. TorchScript MLP: **48 inputs → 12 outputs** = Isaac Lab flat velocity
  obs (lin vel 3, ang vel 3, gravity 3, command 3, joint pos 12, joint vel 12, last action 12; no height scan).
  Training timing: `sim.dt = 0.005`, `decimation = 4` (50 Hz policy). Actions: `JointPositionAction`, scale 0.25,
  `use_default_offset`. Actuators: DCMotor, Kp 25, Kd 0.5, effort 23.5. Command ranges ±1.0 for vx, vy, ωz.
  All of this matches Isaac Lab's `UNITREE_GO2_CFG` (`isaaclab_assets/robots/unitree.py`) and
  `UnitreeGo2FlatEnvCfg_PLAY` (`isaaclab_tasks/.../locomotion/velocity/config/go2/flat_env_cfg.py`; PLAY disables
  observation noise).
- **`PreTrainedPolicyAction`** (`isaaclab_tasks/manager_based/navigation/mdp/`): `action_dim = 3` (vx, vy, ωz);
  runs the policy every `low_level_decimation` *physics* steps; rewires the low-level obs `velocity_commands` and
  `actions` terms; `debug_vis` defaults **True** (draws arrows the depth camera would see → must set False).
- **nav_arena hardcodes** `sim.dt = 0.01, decimation = 2` (`tasks/point_nav.py:222-224`), a camera
  `render_interval` of 20 tied to dt=0.01, spawn z=0.25 (`benchmarks/session.py:239`), a `[[v, w]]` action
  (`methods/in_process/runner.py:187,312`), a 2-tuple `follow_path`, a 2-wide `TwistActionAdapter`, and one
  `body_link` for both sensor mounting and contact sensing.
- **Holonomic candidates surveyed** in the Isaac Sim 6.0 asset library: Fraunhofer O3dyn (real mecanum, has Isaac
  Sim strafe/rotate tests) is ~1.9 × 1.2 m, too large for the InteriorAgent home scenes; Clearpath Ridgeback ships
  only with an arm; AgileX Limo is not modelled as mecanum. **Kaya is the only well-supported, appropriately
  small holonomic asset.** The holonomic kinematics below are written for general roller angles, so a mecanum
  robot (e.g. the lab robot, imported from its URDF) can be added later as just another embodiment.

## Phase 0 — Refactor to a drive-agnostic base (diff drive results must not change)

1. **Canonical command = body twist `[vx, vy, ωz]`** for every embodiment.
   - `embodiments/actions.py`: `DifferentialDriveAction.action_dim = 3`; `vy` is ignored (clamped to 0).
     `scale`/`offset` become 3-tuples.
   - `methods/in_process/runner.py`: warm-up zero action and the step action become width 3
     (`task.step(torch.tensor([[vx, vy, wz]]))`); take the width from `task` / the action manager, not a literal.
   - `ros2/adapters/action_adapter.py`: `TwistActionAdapter` fills `[linear.x, linear.y, angular.z]`.
   - `scripts/verify_embodiment.py`, `scripts/run_ros2_nav.py`, `scripts/verify_nav2.py`: update action shapes.
2. **Drive limits on the embodiment** (`embodiments/base.py`): add `drive_type: str` (`"diff" | "holonomic" |
   "quadruped"`) and `max_lateral_speed: float = 0.0` alongside the existing `max_linear_speed` /
   `max_angular_speed`. Diff-drive geometry fields stay, optional.
   - Quadruped gets its own name because bipeds may be added later. A biped would be a separate
     `"biped"` type that reuses the same `PreTrainedPolicyAction` path; any shared code checks a helper such as
     `is_policy_driven(drive_type)` rather than comparing against a single string.
3. **Embodiment-owned timing and spawn height** (`embodiments/base.py`, `tasks/point_nav.py`,
   `benchmarks/session.py`, `scenes/interior_agent.py`):
   - New fields `sim_dt = 0.01`, `decimation = 2`, `spawn_height = 0.25` (defaults = today's values).
   - `PointNavEnvCfg` takes `sim.dt` / `decimation` from the embodiment.
   - Express the camera render interval as a rate (5 Hz) and convert with `sim_dt` (20 steps at 0.01 → 40 at 0.005).
   - The session and scripts use `embodiment.spawn_height` instead of a literal 0.25.
4. **Separate sensor mounting from contact sensing:** add `contact_bodies: str | None` (a regex relative to the
   robot root; defaults to `body_link`), used by `PointNavSceneCfg.contact_forces` in `create_point_nav_env_cfg`.
5. **Drive-aware follower** (`methods/in_process/controller.py`): `follow_path(...) -> (vx, vy, wz)`, given the
   embodiment's limits through `FollowerCfg` (`max_lateral_speed`, default 0).
   - With `max_lateral_speed == 0` the output must be **bit-identical** to today's `(v, ω)` plus `vy = 0`.
   - With lateral speed available: keep tracking heading toward the lookahead point, because the learned
     policies need the forward camera to face the direction of travel. Add `vy` proportional to the target's
     lateral offset, clipped. Replace "stop and turn in place" with "turn while translating" when the heading
     error exceeds the threshold.
   - The runner builds `FollowerCfg` from the embodiment's limits.
6. **Embodiment variants**, so known configurations such as Kaya mast and native are explicit and selectable
   from the CLI.
   - **Naming:** a variant is addressed as `<robot>.<variant>`, e.g. `kaya.mast` / `kaya.native`. It is one
     string, so `--robot`, sweep `robots:` lists, `RunSpec.robot`, manifests, session keys and run directory names
     (`cli/run.py:48`, `benchmarks/sweep.py:432`, `benchmarks/session.py:327`) carry it unchanged. A dot is used
     because those names become directory names, where a `:` is awkward and a `/` would create a subfolder.
   - **Definition:** `EmbodimentVariant(name, description, overrides: dict[str, Any])` in `embodiments/base.py`.
     `register_embodiment(name, cfg, variants=[...], default_variant=...)` in `embodiments/registry.py`.
     Overrides are plain field replacements applied with `dataclasses.replace`. An unknown field raises at
     registration, so a typo can't silently do nothing. Variants change data (sensor offsets, sensor set,
     limits), never code.
   - **Default variant by bare name:** every robot with variants declares a default, and the bare robot name
     addresses it everywhere: `--robot kaya`, `robots: [kaya]` in a sweep, `create_point_nav_env_cfg(robot_name="kaya")`.
     This works the same as `--robot kaya.mast`. Only non-default variants need the suffix (`--robot kaya.native`).
     `tools robots list` marks the default.
   - **Resolution:** `get_embodiment("kaya")` resolves to the default variant. `get_embodiment` /
     `RunSpec` canonicalize the name to the full `kaya.mast`, so a result never records an ambiguous name.
     A robot without variants keeps its plain name (`nova_carter`, `dingo`: no change to existing results).
   - **Sensor availability:** new embodiment field `sensors: tuple[str, ...] = ("lidar", "camera")`, which
     variants may override. A method that needs the RGB-D camera fails at spec validation, not mid-run, when the
     selected variant lacks one.
   - **Discovery:** `nav_arena tools robots list` (next to the existing `tools routes list` in `cli/tools.py`)
     prints each robot with its variants (default marked), drive type, sensors, and the variant description.
     The `--robot` help text points to it.
   - **Tests:** registry resolution and canonicalization, unknown-override rejection, default selection,
     spec validation of sensor requirements, and the CLI listing (`test_embodiment_registry.py`,
     `test_cli_tools.py`, `test_benchmark_spec.py`).
7. **Tests/docs:** update `test_in_process_controller.py` (add an equivalence test against the old diff
   outputs), `test_in_process_runner.py`, `test_point_nav_cfg.py`, `test_embodiment_registry.py`,
   `tests/ros2/test_action_adapter.py`. Update the drive-type section of `docs/guides/adding-a-robot.md`.

**Exit criterion:** the unit tests pass. For Nova Carter and Dingo, `verify_embodiment`, `verify_task` and a
`verify_baseline` run (iplanner, `hall_straight`) produce the same outcome and metrics as a run recorded before
the refactor; record that baseline first.

## Phase 1 — Holonomic drive: Kaya

1. **Inspect the USD** per guide step 1 (CPU `pxr`; Kaya at `{ISAAC_NUCLEUS_DIR}/Robots/NVIDIA/Kaya/kaya.usd`).
   Record:
   - the default prim, the 3 axle joint names (URDF: `axle_{0,1,2}_joint`), the roller joints (~90), and the
     wheel positions/axes
   - wheel radius (~0.04 m), chassis radius (~0.12 m)
   - `metersPerUnit`; the URDF meshes are scaled 0.1, so check this carefully
2. **Kinematics** (`embodiments/kinematics.py`): `holonomic_matrix(wheel_positions, wheel_axes,
   roller_angles, wheel_radius) -> (N, 3)` plus `holonomic_ik` / `holonomic_fk` (pseudo-inverse). Same math
   as Isaac Sim's `HolonomicController` (`u_i = a_i · (v + ω × r_i)`, `φ̇_i = u_i / (r cos γ)`), valid for
   omni (90°) and mecanum (45°).
   - **Why not call `HolonomicController` directly:** it is built for one robot at a time, uses NumPy/Warp, and
     returns Isaac Sim's experimental `RobotState`; it needs a running Kit app. An Isaac Lab action term needs
     batched torch on the GPU across `num_envs`, writing to Isaac Lab's `Articulation`. The same applies to
     Isaac Sim's `DifferentialController`, which is why nav_arena's diff drive is its own short torch function.
     Its geometry builder is private (`_build_kinematics`), so depending on it would be fragile.
   - **What we do reuse:** the wheel geometry (positions, orientations, mecanum angles) is read from the
     holonomic attributes authored in the Kaya USD, the same data `HolonomicRobotUsdSetup` reads. We don't
     hand-copy numbers.
   - The integration test asserts that our matrix equals `HolonomicController`'s wheel speeds for the same
     geometry and a set of twists. This test runs in Kit, where that controller can be imported.
   - Unit tests: IK→FK round trip, a known Kaya case, and a 4-wheel mecanum case.
3. **Action term** (`embodiments/actions.py`): `HolonomicDriveAction` / `HolonomicDriveActionCfg`.
   - It resolves N wheel joints (and raises on a count mismatch), clips the command to the limits, and writes
     joint velocity targets.
   - It follows the structure of `DifferentialDriveAction`.
   - Roller joints get no actuator; they are left free.
4. **Embodiment** `embodiments/kaya.py`, registered in `embodiments/registry.py` and exported from
   `embodiments/__init__.py`.
   - Fields: `drive_type="holonomic"`, `max_lateral_speed`, a small footprint and radius, and
     `ground_contact_on_body` as the inspection dictates.
   - **Two explicit variants** (Phase 0 variant system):
     - `kaya.mast` (default): RGB-D camera on a virtual mast at the standard ~0.30 m, so the policies see a
       comparable viewpoint.
     - `kaya.native`: the real RealSense pose (~0.1 m, taken from the USD).
     Both are run and reported separately.
   - Use a derived USD (`embodiments/assets.py`) only if inspection finds junk to fix.
5. **Risk to check first:** roller-contact jitter in PhysX. Check it in `verify_embodiment` before tuning
   anything else. If needed, tune solver iterations or roller friction in the derived USD.

## Phase 2 — Quadruped: Unitree Go2 via `PreTrainedPolicyAction`

1. **Policy artifact** (new `embodiments/policies.py`, following the `nav_arena.utils.paths` / `cache/` pattern
   of `assets.py`):
   - Resolve `{ISAAC_NUCLEUS_DIR}/Samples/Policies/go2/physx_policy.pt`.
   - Cache it under `cache/policies/go2/` and verify the pinned SHA-256; fail loudly on a mismatch.
   - Allow an env-var override (`NAV_ARENA_GO2_POLICY`) for offline use.
2. **Embodiment** `embodiments/go2.py`:
   - `articulation_cfg = UNITREE_GO2_CFG` (unchanged gains and default pose, so it matches training).
   - `action_cfg = PreTrainedPolicyActionCfg`, configured as:
     - `policy_path` = the cached file
     - `low_level_decimation = 4`
     - `low_level_actions = UnitreeGo2FlatEnvCfg_PLAY().actions.joint_pos`
     - `low_level_observations = UnitreeGo2FlatEnvCfg_PLAY().observations.policy` (noise off)
     - `debug_vis = False`
   - Timing: `sim_dt = 0.005`; `decimation = 4` (50 Hz nav command).
   - Spawn: `spawn_height = 0.4`.
   - Command limits: `drive_type="quadruped"`, `max_linear_speed`/`max_lateral_speed`/`max_angular_speed` ≤ 1.0
     (inside the training range).
   - Sensors and contact: `body_link = "base"`; `contact_bodies = "(base|.*_hip|.*_thigh|.*_calf)"` (feet
     excluded, as in x-navdp's `GO2_COLLISION_LINK_REGEX`). Camera at about base + (0.32, 0, 0.20)
     (x-navdp's Go2 mount).
   - Footprint ≈ 0.70 × 0.31 m. Keep the existing `tipped` termination as the fall check.
3. **LiDAR alignment:** add `lidar_ray_alignment` to the embodiment (default `"base"`). Go2 uses `"yaw"` so the
   scan stays level while the body pitches (`embodiments/sensors.py`, `scenes/interior_agent.py`).
4. **Things to check on `PreTrainedPolicyAction` as-is:**
   - Its `last_action` reset relies on `episode_length_buf`. PointNav is a `ManagerBasedRLEnv`, so that's fine.
   - The joint order in the 6.0 `go2.usd` must match the order used in training (that USD came from the 4.5
     staging bucket); the velocity-tracking check below catches a mismatch.
   - The cfg holds lambdas after construction, so confirm the benchmark worker never serializes the env cfg
     afterwards.
5. **Risks:**
   - The policy saw a zero command in only 2 % of its training environments, so verify that it can stand still
     at a zero command.
   - Small commands (< 0.1 m/s) may track poorly. If so, set a minimum speed in the follower only for
     `drive_type="quadruped"`.

## Validation (extend `scripts/verify_embodiment.py`)

- **Per-axis drive check**, driven by `drive_type`:
  - Command +vx, +vy and +ωz separately for ~3 s each and measure the resulting body-frame displacement and yaw.
  - Pass: the commanded axis moves in the correct direction at ≥ 70 % of the commanded speed, with small
    cross-axis drift.
  - Diff drive skips the vy check (existing > 0.05 m forward check retained).
- **Quadruped:** 10 s at a zero command, no fall, drift < 0.1 m; velocity-tracking error reported per axis.
- **Then** `verify_task --camera` and a closed-loop `verify_baseline` run (iplanner, `hall_straight`) for kaya
  and go2, compared against nova_carter on the same route.
- **Unit tests** (CPU): kinematics round trips; registry fields for kaya and go2; sensor prim paths and the
  contact regex; follower equivalence (diff) plus holonomic behaviour; the policy-cache SHA check (mocked
  download).
- **Integration tests** (required, `@pytest.mark.integration`), modelled on
  `tests/integration/test_integration_dingo.py`:
  - `test_integration_kaya.py`: spawns in the scene; checks per-axis motion (forward, strafe, rotate) and valid
    depth; cross-checks our wheel speeds against `HolonomicController`.
  - `test_integration_go2.py`: spawns; stands 10 s at a zero command without falling; tracks forward and
    lateral commands; produces valid depth; a contact on the feet alone does not end the episode as a collision.
  - Phase 0 adds a regression check to the existing diff-drive integration tests: the shape-3 action produces
    the same motion as before.

Commands (headless runs by the agent use `./agy_python.sh`; the user's form is shown):
```bash
source setup.env
pytest -c nav_arena/pyproject.toml nav_arena/tests/unit -q
python -u nav_arena/nav_arena/scripts/verify_embodiment.py --robot kaya --camera
python -u nav_arena/nav_arena/scripts/verify_embodiment.py --robot go2 --camera
python -u nav_arena/nav_arena/scripts/verify_task.py --robot go2 --camera
python -u nav_arena/nav_arena/scripts/verify_baseline.py --method iplanner --robot go2 --route hall_straight
```

## Workflow

- **All implementation happens on a new branch** `feat/embodiment-expansion`, created from an up-to-date
  `main`. `feat/unified-entry-point` was merged on the remote; local `main` is behind, at `4d42904`. Steps:
  `git fetch origin main`, `git switch main`, `git merge --ff-only origin/main`,
  `git switch -c feat/embodiment-expansion`.
- Commit each phase locally (the user pushes). Phase 0 must pass its exit criterion before Phase 1 starts.
- Update `docs/guides/adding-a-robot.md` (drive types, quadruped section) and the README's embodiment list at the
  end of each phase.

## Future work (out of scope)

- Nav2 for holonomic and legged robots (OmniMotionModel, DWB vy sampling, 3-axis velocity smoother,
  per-embodiment param overlays).
- A mecanum embodiment for the lab robot via URDF import, reusing `holonomic_ik` and `HolonomicDriveAction`.
- Rough-terrain or stair locomotion policies.
