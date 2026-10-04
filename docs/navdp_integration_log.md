# NavDP / learned-baseline integration — decision & progress log

Maintained by the agent while working autonomously on branch `feat/navdp-baselines`.
**Review the "Needs your review" items first.** Nothing here has been pushed. Delete this file once
you've read it (it is committed on the feature branch only).

Plan of record: `~/.claude/plans/1-it-s-fine-to-goofy-gizmo.md` (supersedes
`baseline_integration_plan.md`).

## Needs your review

_(none yet)_

## Decisions made (with rationale)

| # | Decision | Why | Reversible by |
|---|---|---|---|
| D1 | NavDP fork pinned at `8b9ee13` (`Herschenglime/NavDP`, branch `nav-arena`); labmate's untracked harness NOT put on the fork | Fork holds only upstream-file fixes; harness is ported into nav_arena | — |
| D2 | Dependencies installed into `env_isaaclab` via `uv pip --no-deps`, incl. labmate's prebuilt mmcv wheel copied from `alex-spark`; pinned in `requirements/baselines.txt` | Keeps Isaac's torch/numpy untouched; wheel imports and `mmcv.ops` loads here | `uv pip uninstall` |
| D3 | Added `RobotEmbodimentCfg.body_link` (USD link carrying colliders/sensors): `chassis_link` Carter, `base_link` Dingo. `chassis_frame` stays a TF/URDF frame name | Dingo USD has a single rigid body; hardcoded `/Robot/chassis_link` paths would be wrong for it. (Plan called this `contact_body`.) | — |
| D4 | `dingo_stage_patch` rewritten in plain USD (no `pxr.PhysxSchema`) and raises if no caster material found | PhysxSchema only exists inside Kit; old code silently skipped the friction fix | — |
| D5 | Camera uses `IsaacRtxRendererCfg` explicitly | Labmate validated this; Lab's default is the generic `RendererCfg` | one line |
| D6 | Timing: keep nav_arena's dt=0.01 / decimation=2 (50 Hz control); only make `render_interval` configurable (20 → 5 Hz camera for baselines) | User: perfect replication not required | — |
| D7 | Only InteriorAgent scenes (no Nucleus office) | User decision | — |
| D8 | `InProcessPolicy.step(PolicyObservation) -> Plan(path, stop, diagnostics)` — returns a path, not `(v, w)`; observation is a dataclass, not a dict | Planner (3-5 Hz) and follower (control rate) are decoupled, as in the labmate's design; typed fields catch bad inputs | contract in `methods/in_process/base.py` |
| D9 | Dropped `InProcessPolicy.is_goal_reached()` from the original plan | Goal checking belongs to the task (`PointNavTask.is_goal_reached`), not the policy | — |
| D10 | Labmate's `follow_path` ported verbatim (as `FollowerCfg` + `follow_path`); his `wheel_speeds` dropped in favor of `diff_drive_ik` / `DifferentialDriveAction` | Avoids duplicate kinematics; his follower is what produced his working runs | — |
| D11 | Registry names use underscores (`x_navdp`); hyphenated upstream spellings accepted | Matches the original plan and Python identifiers | — |
| D12 | Policy `seed` defaults to 0 (torch + numpy seeded at construction) | NavDP diffusion sampling is stochastic; makes runs repeatable | `seed=None` |
| D14 | Collision detection for robots with `ground_contact_on_body` (Dingo) uses lateral (XY) contact force only (`lateral_contact`) | Measured: Dingo `base_link` reads ~17.7 N net force just resting on its caster, which would trip the Carter-style any-force check at t=0. Free-drive lateral force is 0.000 N; wall impact still detected | set `ground_contact_on_body=False` |
| D15 | Dingo's stage patch runs as a `prestartup` event, so `scene.replicate_physics=False` is set for patched embodiments | Isaac Lab raises if a prestartup event is used with replication on. Irrelevant for single-env benchmarks; multi-env Dingo scenes lose physics replication (slower setup) | — |
| D16 | Added a `tipped` termination (`mdp.bad_orientation`, 0.6 rad) to all envs, incl. Carter | Port of the labmate's `fell_or_tipped` stop; also adds `get_terminal_cause()` | `max_tilt` |
| D17 | `render_interval` defaults to 20 (5 Hz at dt=0.01) when a camera is enabled, else 3 | Each render also pays for RTX camera rendering; baselines plan at 5 Hz. GUI users can pass a smaller value | `render_interval=` |
| D13 | Moved Nav2 to `methods/ros2/nav2`; also updated `verify_tf_tree.py`, which the original plan missed | It referenced the old path | — |

## Progress

- [x] Phase 1 (embodiment registry + Dingo) — `6fa7f73`, follow-up `e929347`
- [x] Paths refactor — `4d42904`
- [x] Dependencies + pinned requirements — `e8dc139`
- [x] Phase 2 — RGB-D camera + multi-robot verify_embodiment (headless run: Dingo+camera PASS, Carter PASS)
- [x] Phase 3 — methods restructure + NavDP adapter port (129 unit tests; real-model smoke tests on all five)
- [x] Phase 4 — generalized PointNavTask (headless: Carter + Dingo/camera PASS; 3 existing integration tests PASS)
- [ ] Phase 5 — episode runner, verify_baseline, Dingo integration test

## Findings / surprises

- **Phase 2 / boot-order trap:** `verify_embodiment.py --robot` cannot use argparse `choices=list_embodiments()`:
  importing `nav_arena.embodiments` at module level pulls in `pxr` before `AppLauncher` boots and the app crashes
  (`SystemError: Missing frame when calling profile function`). The name is validated after boot instead.
  Any future script that needs the registry for CLI choices has the same constraint.
- **Phase 2 / camera sanity:** Dingo nearest ground return 1.12 m, depth valid fraction 0.43, RGB uint8
  `(1,360,640,3)`, depth `(1,360,640,1)` float32. Matches the expected geometry (camera ~0.42 m above floor,
  20.6 deg half-VFOV). The Dingo drove 0.139 m in 60 steps at v=0.5 (Carter 0.159 m) — acceleration-limited
  at start, not a caster-drag signal; no ground-truth friction test exists yet (see Phase 5 integration test).

- **Phase 3 / fidelity vs. labmate's recordings** (real checkpoints, GB10, his recorded frames): iPlanner reproduces his
  logged paths and fear values exactly (0.0 m deviation, 6 frames). ViNT and VIPlanner match exactly. NavDP matches at
  step 0 and within 0.16 m afterwards (stochastic + frame history absent in my isolated-frame replay). X-NavDP matches at
  step 0, ~2.4 m apart later (stateful pose guidance, replay has no history). Not a regression: expected for stateful
  stochastic planners; the real check is closed-loop behavior in Phase 5.
- **Phase 3 / checkpoint hashes:** all recorded SHA-256 values match the files on disk. iPlanner had no recorded hash in
  the labmate's JSON; I recorded the hash of the local converted file (`b801f444...`), whose original source is unknown.
- **Phase 3 / spec drift:** the pinned dependency list also covers VIPlanner's stack, which is already installed
  (mmcv wheel copied from `alex-spark`, `mmcv.ops` loads).

- **Phase 4 / measured Dingo contact behavior:** `base_link` net force while driving backward into a wall read 17-18 N
  (mostly the caster's vertical support); lateral force during free driving 0.000 N. The wall impact was detected at
  env step 116. This validated D14.
- **Phase 4 / pre-existing quirk (not fixed):** `verify_task.py` prints `final_dist=1.500 m` / `impact_force=0.00 N`
  after a termination because the env has already auto-reset by the time those values are read. The pass/fail logic is
  unaffected.
- **Phase 4 / `verify_nav2.py` not re-run:** it needs the full ROS 2 + Nav2 stack and is on your list. What I did verify:
  the relocated `nav2.launch.py` builds its launch description with the new relative config paths, and the
  `test_integration_tf_tree` test (which uses the moved `robot_description.launch.py`) passes.
