# NavDP / learned-baseline integration — decision & progress log

Maintained by the agent while working autonomously on branch `feat/navdp-baselines`.
**Review the "Needs your review" items first.** Nothing here has been pushed. Delete this file once
you've read it (it is committed on the feature branch only).

Plan of record: `~/.claude/plans/1-it-s-fine-to-goofy-gizmo.md` (supersedes
`baseline_integration_plan.md`).

## Needs your review

1. **Pre-existing bug fixed in `nav_arena.core.launch_simulation_app`.** `SimulationApp.close()` ends the process, and
   the manager never passed an exit code, so *every* failure inside the `with` block exited 0 (and exceptions never
   reached the script's `except`). Consequence: `test_integration_task` had been passing hollowly: it ran `verify_task.py
   --num-steps 30` but reaching the goal takes ~108 steps, so its own assertion failed unnoticed. I fixed the manager
   (nonzero status for exceptions, `sys.exit(n)` honored, traceback logged) and the test (150 steps). Other scripts keep
   their now-dead outer `except` blocks. Please sanity-check this against how you run scripts interactively.
2. **Routes need your eyes in the GUI.** Chosen from the occupancy map; start/goal have >= 0.5 m clearance and A* paths
   exist, but I only checked them headless. My first `hall_straight` clipped a wall corner (spawn 0.08 m from a wall
   face; the Dingo's corner poked into it), so it was replaced by a segment with >= 0.8 m clearance along its whole
   length. `through_doorway` passes a ~1 m opening and is hard for any method.
   `python -u nav_arena/nav_arena/scripts/verify_baseline.py --method iplanner --route <name> --viz kit`
3. **`verify_nav2.py` was not re-run by me** (full ROS 2 + Nav2 stack; on your list). Verified instead: the relocated
   launch file builds, the TF-tree integration test (uses the moved `robot_description.launch.py`) and the L2 ROS tests
   pass.
4. **Baseline behavior vs. expectations (see results table).** iPlanner/VIPlanner *stop and never resume* when predicted
   fear >= 0.7 (the labmate's design, kept as-is); in `around_table`/`through_doorway` iPlanner halts at the first plan.
   NavDP often ends 0.4-0.5 m short of a 0.4 m goal tolerance (his own recorded run did the same); `--goal-dist 0.5`
   would count it. Do you want a recovery behavior (e.g. rotate in place when stopped) or looser tolerance as defaults?
5. **Unpushed work:** nav_arena commits on `feat/navdp-baselines` are local. NavDP fork is pinned at `8b9ee13` (pushed by
   you).
6. Delete this log (and decide whether to keep `requirements/baselines-mmcv.md`) before merging.

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
- [x] Phase 5 — episode runner, verify_baseline, routes, Dingo integration tests (6/6 integration, 169 unit tests pass)

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

- **Phase 5 / results matrix** (Dingo unless noted; kujiale_0003 open doors; 30 s budget; goal tolerance 0.4 m; follower
  max 0.3 m/s; headless; seed 0):

  | Method | hall_straight | around_table | through_doorway |
  |---|---|---|---|
  | iPlanner | goal, 18.9 s | max_steps (stopped by fear at plan 1) | max_steps (stopped by fear at plan 1) |
  | NavDP | max_steps, 0.47 m short | collision (2.2 m driven) | collision (1.8 m driven) |
  | X-NavDP | goal, 19.0 s | collision (3.3 m driven) | goal, 12.4 s |
  | ViNT (image goal) | collision at 3.6 m | not run | not run |
  | VIPlanner | max_steps, 2.6 m left (89/150 stop requests) | not run | not run |
  | iPlanner on Nova Carter | goal, 19.0 s | - | - |

  Collisions were checked against the occupancy map: each had a footprint point at 0.0-0.1 m from an obstacle (table
  edge, door frame), i.e. genuine contacts, not detector artifacts. Inference per plan: iPlanner ~8 ms, ViNT 28 ms,
  VIPlanner 151 ms, NavDP 151-177 ms, X-NavDP 219-281 ms (GB10).
- **Phase 5 / flaky Kit startup crash (seen once):** a boot-time SIGSEGV in `libcarb.tasking.plugin.so`
  (exit 139, 13 ms into startup) in one run of a scratch script; 4 immediate repeats were fine. Not related to these changes.
- **Phase 5 / planning cadence:** the runner calls `task.refresh_camera_frame()` (render + read) at each plan instead of
  relying on `render_interval` phase alignment, so ViNT's 3 Hz (33-step period) works with the 5 Hz render cadence.
