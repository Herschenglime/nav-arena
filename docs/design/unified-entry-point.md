# Design: a unified `nav_arena` entry point with batch runs and run tracking

Status: **design only, nothing here is built.** Written to be handed to a fresh implementation session, so the first
section is everything that session needs to know. Decisions marked **(decided)** were settled with the project owner;
anything under "Open items" is still undecided.

## Read this first (handoff context)

**Repo and branch.** nav_arena lives at `~/simulation/nav_arena` (git root; the Python package is
`nav_arena/nav_arena/`). Work happens on `feat/navdp-baselines` (or a branch from it). The owner pushes manually
(passkey-protected SSH key on a shared machine), so commit locally and never push. The NavDP fork
(`git@github.com:Herschenglime/NavDP.git`, branch `nav-arena`, pinned `8b9ee13`) is checked out at `~/simulation/NavDP`.
Commit messages end with the attribution trailer the session specifies, and no model identifiers.

**Running things.** From `~/simulation`: `source setup.env` (activates the uv env `env_isaaclab`; install packages only
there, with `uv pip install --no-deps`, never globally). Isaac Lab is headless by default; `--viz kit` opens the GUI.
Agents run scripts through `./agy_python.sh <script.py> [args]` (sets up the env, unbuffered output). CPU unit tests:
`pytest -c nav_arena/pyproject.toml nav_arena/tests/unit -q` (about 200 tests, about 8 s). Integration tests
(`-m integration`) boot Isaac and need the GPU free.

**Install.** The `nav_arena` command exists only after the package is (re)installed in editable mode, because
console scripts are generated at install time. From `~/simulation` with `setup.env` sourced:
`uv pip install --python "$VIRTUAL_ENV/bin/python" --no-deps -e nav_arena` (nothing is installed globally; `--no-deps` keeps
Isaac's pinned torch/numpy). Re-run it after adding the `[project.scripts]` entry, and mention it in the README's
setup section. `python -m nav_arena` works without it.

**Workspace conventions.** `~/simulation/AGENTS.md` (outside this repo) is the source of truth for how commands are run
(`source setup.env`, `python -u nav_arena/nav_arena/scripts/...`, `./agy_python.sh` for agents, `--viz kit`); read it
first. Its "~2 s" unit-test timing is stale (about 8 s now).

**Process rules.** Plan before implementing; record decisions in a reviewable doc. Never run GPU jobs in the background
and check `ps aux | grep -E 'verify_|isaaclab'` first: a background batch once hung the owner's GUI session.

**Existing pieces to reuse.**

| Piece | Path | Notes |
|---|---|---|
| Episode loop and recorder | `nav_arena/methods/in_process/runner.py` | `run_episode(task, policy, EpisodeCfg) -> EpisodeResult`; `_Recorder` writes the run directory |
| Policy contract and registry | `methods/in_process/{base,registry,controller}.py`, `navdp_adapter/` | `get_policy(name, intrinsic, **cfg)`; policies return a path, a shared follower tracks it |
| The logic to move into the worker | `nav_arena/scripts/verify_baseline.py` | argument parsing, route resolution, env/task/viewer/overlay/policy construction, result report |
| Task and env cfg | `nav_arena/tasks/point_nav.py` | `create_point_nav_env_cfg`, `PointNavTask` |
| Routes | `nav_arena/scenes/routes.py` | `Route`, `get_route`, `list_routes` |
| Map and route tools | `nav_arena/tools/{map_generator,route_map}.py` | `find_cached_map`, `load_run`, `render_overlay` |
| App lifecycle | `nav_arena/core/app.py` | `launch_simulation_app`; exit-code behavior |
| Subprocess helper | `nav_arena/utils/process.py` | `managed_process` (process-group-safe) |
| Paths, logging | `nav_arena/utils/{paths,logger}.py` | `RUNS_DIR`, `CACHE_DIR`, `ArenaLogger` |
| Pre-boot arg checks | `nav_arena/utils/cli_args.py` | pure functions usable before the simulator boots |

**Hard constraints.**

- One Isaac app per process, and `SimulationApp.close()` ends the process. Code after the `with launch_simulation_app`
  block never runs; return results from inside the block (`sys.exit(n)`).
- Do not import `isaaclab`, `isaacsim`, `omni` or `pxr`-dependent modules before `AppLauncher` boots (the app crashes).
  This is why `verify_baseline.py` validates robot and route names after boot and why argparse `choices` cannot come
  from the embodiment registry. The orchestrator process in this design must never import Isaac at all.
- One learned baseline per Python process (upstream NavDP modules share names such as `policy_agent`).
- `_Recorder` creates the run directory with `mkdir(exist_ok=False)`, so run ids must be unique.
- Cameras and `render_interval`, the scene, the robot and the policy are fixed when the env is built.
- The derived Dingo USD is currently built when `nav_arena.embodiments` is imported (phase 0 below makes it lazy);
  until then a failed build breaks the import of the whole package, and therefore the CLI's worker.

**Known quirks (not blockers).** GUI playback can look jerky although simulated speed is constant (uneven wall-clock
step cost; see `docs/baseline_integration_results.md`). The goal arrow is hidden from cameras on purpose
(`--show-goal-marker` off). iPlanner/VIPlanner stop permanently when predicted fear reaches 0.7 (reported as
`stalled`). Only 14 of 20 planned baseline runs have clean results.

**Decisions already made.**

- **(decided)** Console script `nav_arena` (no `python -u`); `python -m nav_arena` also works.
- **(decided)** Method-agnostic interface: every method, including Nav2 later, is used and tracked the same way.
- **(decided)** The core entry point never calls a `verify_*` script. The run logic becomes a library; `verify_baseline.py`
  becomes a thin shim over it. Other `verify_*` scripts stay as manual engineering checks and are not exposed through
  the CLI.
- **(decided)** Run ids inside a batch: `<batch_id>/<NNN>_<method>_<robot>_<route>_s<seed>`.
- **(decided)** Sequential execution only for now; parallelism is later engineering work.
- **(decided)** Isolation first (one process per run), with the worker API designed for in-place resets later.
- **(decided)** One scene per sweep file; flags override YAML with a logged message; both flags and YAML are supported.
- **(decided)** Failure semantics: see "Exit codes".
- **(decided)** Nav2 is postponed but must fit the same schema and tracking.

---

## 1. Goals and non-goals

Goals: one command to run a method on a route and record it; a batch command that runs a method x robot x route x seed
matrix and tracks progress; a method-agnostic results schema; reproducibility (the resolved spec and environment are
saved with every batch); reuse of the existing runner and recorder.

Non-goals (this design): parallel runs, Nav2 backend implementation, a results dashboard, new baselines, changing the
policy contract.

## 2. Command surface

With `setup.env` sourced and the package installed (`pip install -e nav_arena`, already the documented setup):

```bash
nav_arena run   --method iplanner --robot dingo --scene kujiale_0003 --route hall_straight
nav_arena run   --spec runs/one.yaml --seeds 5            # flags override the spec, with a logged message
nav_arena sweep sweeps/baselines_kujiale_0003.yaml        # a batch from a spec file
nav_arena sweep --scene kujiale_0003 --methods iplanner,navdp --routes hall_straight,around_table --seeds 0,1,2
nav_arena sweep ... --resume                              # skip runs already done in the batch
nav_arena runs  list [--batch ID]                         # table of batches or runs with status and outcome
nav_arena runs  show <run-or-batch>                       # one run's summary, or a batch's progress
nav_arena runs  compare <batch> [--by method|route|robot] # success rate, TTG, path length, collisions, inference ms
nav_arena runs  rerun <run-id>                            # re-execute a run with its saved spec
nav_arena routes list|show [--scene]                      # registered routes (+ draws via tools.route_map)
nav_arena map   generate|show --scene ...                 # thin wrappers over tools.map_generator / route_map
nav_arena doctor                                          # env, checkpoints, NavDP checkout, map cache, GPU-busy check
```

`verify_*` scripts are intentionally absent: they remain `python -u nav_arena/nav_arena/scripts/verify_*.py`.

Implementation notes: `nav_arena/cli.py` (argparse subcommands; stdlib only plus PyYAML 6.0.3, already installed) with
`[project.scripts] nav_arena = "nav_arena.cli:main"` in `pyproject.toml`. The CLI process only parses, plans, spawns
workers and reads files, so it starts in milliseconds and imports no Isaac. Output is unbuffered: workers are started
with `PYTHONUNBUFFERED=1` and the CLI line-flushes.

## 3. `RunSpec`: one description of one episode

A serializable dataclass shared by the CLI, YAML files and the worker (JSON on the worker's command line):

| Field | Meaning |
|---|---|
| `method` | `iplanner`, `vint`, `navdp`, `viplanner`, `x_navdp`, or reserved `nav2` |
| `method_params` | policy config overrides (today `--policy-arg KEY=VALUE`, `--plan-hz`, `--task`, planner device) |
| `robot`, `scene` | embodiment name; InteriorAgent scene id or USD path |
| `route` or (`spawn`, `goal`, optional `spawn_yaw`) | named route, or a custom pair (never both; a lone spawn or goal is an error) |
| `goal_tolerance`, `max_speed`, `max_steps`, `stall_timeout_s` | episode limits |
| `seed` | policy RNG seed |
| `viz` | `{gui, follow_camera, goal_overlay}`; GUI is never used in batches by default |
| `output_dir` | where the run is recorded |

`method_family` (`in_process` or `ros2`) is derived from `method`. Validation is a pure function (extending
`nav_arena/utils/cli_args.py`, which already holds the spawn/goal/route check) so bad specs fail before any boot.

### Flags and YAML

A spec file provides defaults; any flag given on the command line overrides the matching field and logs one line per
override at INFO (WARNING if it changes the method, scene or robot):

```
[WARNING] --scene overrides spec.scene: kujiale_0003 -> kujiale_0004
[INFO]    --seeds overrides sweep.seeds: [0, 1, 2] -> [5]
```

The resolved spec (final values) is saved in the run or batch directory.

## 4. Worker: the run logic as a library

Today `verify_baseline.py::run_verification` builds the env cfg, boots the app, constructs the task, viewer, overlay
and policy, calls `run_episode`, and prints a report. That logic moves into a new package (name open, `nav_arena/benchmarks/`
here), in two layers so that in-place resets (section 7) are possible later:

- **`RunSession`**: the expensive, reusable part. Boots the app (`launch_simulation_app`), builds the env cfg and
  `PointNavTask`, creates the viewer and overlay, loads the policy. Keyed by a **session key**: scene, robot, method,
  method params, camera/sensor config and viz settings (anything baked into the env or the loaded network).
- **`session.run_episode(EpisodeSpec) -> RunResult`**: the cheap, repeatable part. `EpisodeSpec` is spawn, goal,
  tolerance, seed, limits and output directory. It calls the existing `run_episode` from
  `methods/in_process/runner.py` and writes the same artifacts.

Phase 1 executes exactly one episode per session (isolation). Subprocess entry point:
`python -m nav_arena.benchmarks.worker <runspec.json>`; it boots, runs, writes artifacts, and exits with the run's exit
code (0 goal reached, 2 not reached, 1 error). `verify_baseline.py` becomes a shim: parse flags, build a `RunSpec`,
call the same library, keep its printed report. There is then one implementation of a baseline run.

## 5. Method backends (the path to Nav2)

A `MethodBackend` seam sits under the session: it decides **who produces the command each step**.

- `in_process` (exists): `InProcessPolicy.step(obs) -> Plan`, shared follower, `run_episode`.
- `ros2_nav2` (later): launch the Nav2 stack (`methods/ros2/nav2/`) with `managed_process`, run the OmniGraph ROS 2 bridge,
  send the goal, collect the same `EpisodeResult`. It needs the ROS 2 boot flags (`enable_ros2=True`) and isolation of
  ROS domain/ports so two runs cannot cross-talk.

Both share scene and task construction, route handling, terminal-cause detection, metrics and recording; only the
command source differs. Until the Nav2 backend exists, `method: nav2` is a **valid reserved value**: the schema,
manifest and `results.csv` carry it, specs validate, and running it fails with a clear "not implemented yet". Results
from `verify_nav2.py` could later be imported into the same schema.

## 6. Tracking: batches, manifest, results

Layout, under `cache/runs/` (`RUNS_DIR`, overridable with `NAV_ARENA_RUNS_DIR`):

```
cache/runs/<batch_id>/              # batch_id = <YYYYmmdd-HHMM>_<spec name>
  batch.yaml                        # resolved spec, git SHA + dirty flag, NavDP SHA, env versions, host, start/end
  manifest.json                     # per-run state, rewritten atomically after every state change
  results.csv                       # one row per finished run
  001_iplanner_dingo_hall_straight_s0/   # run directories: existing recorder layout
    settings.json  steps.jsonl  summary.json  rgb_*.png  depth_m_*.npy  worker.log
  002_iplanner_dingo_around_table_s0/
```

A single `nav_arena run` writes `cache/runs/<YYYYmmdd_HHMMSS>_<method>_<robot>_<route>_<4 hex>/` (timestamp plus a
random suffix, so same-second runs cannot collide).

**`manifest.json`** (method-agnostic; no policy-only fields):

```json
{
  "batch_id": "20261004-1530_baselines_kujiale_0003",
  "runs": [
    {"id": "001_iplanner_dingo_hall_straight_s0", "status": "done",
     "method": "iplanner", "method_family": "in_process", "method_params": {},
     "robot": "dingo", "scene": "kujiale_0003", "route": "hall_straight", "seed": 0,
     "pid": 12345, "started": "...", "ended": "...", "exit_code": 0,
     "terminal_cause": "goal_reached", "run_dir": "001_iplanner_dingo_hall_straight_s0"}
  ]
}
```

Statuses: `queued`, `running`, `done` (the episode finished, whatever the outcome), `failed` (infrastructure error),
`timeout` (killed by the per-run limit), `skipped` (invalid spec or unsupported method). Outcome is separate from
status: a `done` run may have `terminal_cause: collision`.

**`results.csv` columns** (one row per finished run, from `summary.json` plus the spec): `batch_id, run_id, scene,
robot, method, method_family, route, seed, terminal_cause, success, time_to_goal_s, sim_time_s, path_length_m,
initial_goal_distance_m, final_goal_distance_m, plans, stop_requests, mean_inference_ms, wall_time_s, goal_tolerance,
max_speed, spec_hash`. `scene` is a first-class column so results from different layouts can never be mixed silently.

**Execution rules.**

- Sequential. A `--jobs` flag is reserved and fixed at 1.
- Before starting, refuse (or wait with `--wait`) if another Isaac session is running (the `ps` guard).
- Each run is a worker subprocess under `managed_process`, with a per-run timeout (`timeout_s`, default derived from
  `max_steps`) that kills the process group.
- Ctrl-C stops the current worker, marks it `failed`, leaves the manifest consistent, and exits.
- `--resume` re-reads the manifest and skips `done` runs; `queued` and `running`/`failed` runs are retried.
- The orchestrator never reads Isaac state; everything it knows comes from the worker's exit code, `summary.json` and
  `worker.log`.

**Exit codes.** `nav_arena run`: 0 goal reached, 2 episode ended without the goal, 1 infrastructure error (same as
`verify_baseline.py`). `nav_arena sweep`: 0 if every run executed (regardless of goal success), 1 if any run had an
infrastructure failure or timeout; per-run outcomes live in the manifest.

**Reporting.** `runs list` and `runs compare` read only `manifest.json`/`results.csv`: success rate per method and
route, mean and standard deviation of time to goal and path length across seeds, collisions, mean inference time,
split by scene. `route_map --run <run_dir>` is reused for plots.

## 7. Isolation first, with room for in-place resets

Phases 1 and 2 use **complete isolation**: one worker process, one Isaac boot, one episode. No state is shared between
runs, which makes results easy to trust.

Booting Isaac, loading the scene, initializing RTX and loading a policy dominate the cost of a short episode, so the
design must allow reusing a booted session. The `RunSession`/`EpisodeSpec` split exists for that. Reuse needs a way to
reset the environment to a new start and goal using Isaac Lab's own paradigms rather than a reboot:

- Today the spawn pose is baked into the env cfg (`robot_spawn_pos`/`robot_spawn_rot` in `create_point_nav_env_cfg`)
  and the reset event is `mdp.reset_scene_to_default`, which always returns to that one default.
- Needed: `PointNavTask.reset_to(spawn_xy, yaw, goal_xy, goal_tolerance)`, built on the manager API: a reset event
  (a `reset_root_state_*` event term, or direct `write_root_pose_to_sim` plus zero velocities and the articulation's
  default joint state), writing the goal into the `UniformPose2dCommand` term (command manager), and resetting the
  termination, contact and tilt state and the action term (`DifferentialDriveAction.reset`).
- Spike before building: confirm that `env.reset(options=...)` or event-term parameters can carry a per-episode pose
  (versus overriding the asset's default root state before `env.reset()`), and that camera sensors and the contact
  sensor clear their history on reset.
- The viewer, overlay, recorder, stall tracker and policy state (`policy.reset()`) are cleared per episode, and RNGs
  are re-seeded.

What can share a session is bounded: the **method** cannot change (one baseline per process); the scene, robot, sensor
set, `render_interval` and stage fixes are fixed at env build. The sweep planner therefore groups runs by session key
and runs each group in one worker, ordered by route and seed. `--isolate` (the default until verified) forces one
process per run; `--reuse-sessions` enables grouping.

Contamination check to build in: the same `EpisodeSpec` run first in a session and after N other episodes must agree
on success and terminal cause and roughly on path length (physics warm-up differs slightly). If it does not, isolation
stays the default.

## 8. Sweep specs

One scene per sweep file: routes, maps and the open/closed-door variant are all per scene, and each numbered kujiale
scene is a different layout. A multi-scene study is several sweep files (or a top-level `scenes:` list whose entries
each carry their own `routes`), never an implicit cross product.

```yaml
# nav_arena/sweeps/baselines_kujiale_0003.yaml
name: baselines_kujiale_0003
scene: kujiale_0003
robots: [dingo]
methods: [iplanner, navdp, x_navdp, viplanner]
routes: [hall_straight, around_table, through_doorway, to_far_room]   # names registered for this scene
seeds: [0, 1, 2]
options: {goal_tolerance: 0.4, max_speed: 0.3, max_steps: 1500, stall_timeout_s: 10}
method_params:
  iplanner: {fear_threshold: 0.8}
timeout_s: 180
```

Routes may also be inline (`{name: custom1, spawn: [-3, 0.9], goal: [-6.3, -1.2]}`). The matrix is
`robots x methods x routes x seeds`; methods that cannot do a requested option (e.g. image goal) fail validation before
any run starts. Sweep files live in `nav_arena/sweeps/` (versioned and reviewable; named `<purpose>_<scene>.yaml`),
and the resolved copy is always saved in the batch directory.

## 9. Phasing

Each phase is independently shippable and leaves `verify_baseline.py` working.

| Phase | Deliverable | Done when |
|---|---|---|
| 0 | **Lazy Dingo asset.** `embodiments/dingo.py` calls `ensure_derived_asset` at import time (`DINGO_USD_PATH = _resolve_dingo_usd()`), so a failed build (unwritable cache, upstream layout change, `pxr` missing) raises from `import nav_arena.embodiments` and breaks every script, even for other robots. Build on first use instead: resolve the path when the Dingo's `ArticulationCfg` is constructed or its embodiment is first requested (e.g. a factory for `DINGO_CFG`, or `DingoEmbodimentCfg.articulation_cfg` built lazily), keep the source path fallback when the NavDP checkout is absent, and keep the build in a subprocess (no `pxr` import in the parent before `AppLauncher`) | importing `nav_arena.embodiments` never builds or fails; `get_embodiment("nova_carter")` works with the Dingo build broken; the first Dingo spawn builds and caches the layer; `tests/unit/test_embodiment_assets.py` and the Dingo integration tests still pass |
| 1 | `RunSpec`, `RunSession`/`EpisodeSpec` split (one episode per process), `benchmarks/worker.py`, `verify_baseline.py` as a shim, `nav_arena run`, console script, `results.csv` row per run, method-agnostic schema with reserved `nav2` | `nav_arena run ...` reproduces a current `verify_baseline` result (iPlanner `hall_straight` reaches the goal in about 19 s); the shim and CLI share one implementation; unit tests for spec parsing/validation and override logging |
| 2 | `nav_arena sweep`, spec files, batch directory, manifest state machine, per-run timeout, Ctrl-C safety, `--resume`, Isaac-busy guard | a small sweep completes; killing it mid-run and `--resume` finishes the rest; manifest state-machine tests with a fake worker (no Isaac) |
| 3 | `runs list/show/compare/rerun` | `compare` reproduces the results table in `docs/baseline_integration_results.md` from a batch |
| 4 | `doctor`, `routes`, `map` subcommands | `doctor` catches a missing checkpoint/NavDP checkout and a busy GPU |
| 5 | In-place resets: `PointNavTask.reset_to` spike and implementation, `--reuse-sessions`, contamination regression test | grouped sweep is faster and passes the contamination check; otherwise isolation stays default |
| 6 | Nav2 backend under the same interface | `nav_arena run --method nav2 ...` writes the same artifacts and appears in `runs compare` |
| 7 | Drive-aware follower hooks (ties to the new-drive guide in `docs/guides/adding-a-robot.md`) | an ackermann or holonomic robot runs a baseline through the same CLI |

Tests to add per phase: CPU unit tests for spec parsing, YAML/flag override logging, matrix expansion, run-id
generation, manifest transitions and resume (with a fake worker that writes a canned `summary.json`); one integration
run (Dingo, `hall_straight`, headless) per phase that touches the worker.

## 10. Open items

- Package name for the worker library (`nav_arena/benchmarks/` is a placeholder).
- Whether `nav_arena run` streams the worker's log live or only writes `worker.log` (recommendation: stream, with
  `--quiet`).
- The default `timeout_s` formula and whether a global batch time budget is needed.
- Whether `routes`/`map` subcommands should also create files (e.g. scaffold a new `Route`), or only list and draw.
- Parallelism: the cause of the earlier GUI hang under concurrent load is unknown (the GB10 has up to 96 GB addressable
  GPU memory, so plain memory exhaustion is unlikely); investigate before any `--jobs N`.
- Importing historical `verify_nav2.py` results and the existing 14 baseline runs into the new schema.
- Stop recovery for iPlanner/VIPlanner (a permanent stop on one fear spike) is deliberately postponed; it belongs to
  future work, not to this design.
