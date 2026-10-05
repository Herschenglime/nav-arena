# Contributing & Development Guidelines

Welcome to `nav_arena`. This document outlines the architectural boundaries, coding standards, developer primitives, and testing tiers required when developing new features or extending existing modules.

---

## 1. Architectural Boundaries (Where Code Lives)

The codebase strictly enforces separation of concerns across nine dedicated modules:

```text
nav_arena/
├── cli/          # Main entry point package (run, sweep, runs, doctor, routes, robots, map) - ZERO simulator imports
├── benchmarks/   # Unified benchmark engine: RunSpec, RunSession, worker, manifest, sweep, tracking, doctor
├── core/         # SimulationApp lifecycle & boot-time extension injection
├── embodiments/  # Robot kinematics, sensor configs, in-memory URDF synthesis (ZERO ROS 2 dependencies)
├── tasks/        # RL/benchmark environments (ManagerBasedRLEnv), MDP terms, episode resets
├── scenes/       # Non-destructive USD loaders, delta layer conditioning, BVH preprocessors
├── methods/      # Autonomy baselines: in_process/ (policies run in the sim process) and ros2/ (external stacks, e.g. Nav2)
├── ros2/         # ROS 2 middleware bridges (OmniGraph builders, action adapters, async publisher nodes)
├── tools/        # Standalone offline utilities (CLI 2D occupancy grid generator, route overlay)
├── utils/        # Cross-cutting primitives (ArenaLogger, managed_process, paths, create_mock_env)
├── scripts/      # Thin executable runners and verifiers (NO domain logic definitions)
├── tests/        # 3-tier verification suite (unit/, ros2/, integration/)
└── docs/         # (repo level) guides/, design/, and integration notes
```

### Module Responsibilities & Invariants

- **`nav_arena.cli`**: Top-level command-line dispatcher for `run`, `sweep`, `runs`, `doctor`, `routes`, `robots`, and `map`.
  - **Fast Startup Invariant**: Strictly **ZERO simulator or ML imports** (`isaacsim`, `isaaclab`, `omni`, `pxr`, `rclpy`, `torch`). The CLI process only parses arguments, validates specs, and spawns isolated worker subprocesses via `managed_process`.
- **`nav_arena.benchmarks`**: Evaluation harness and run tracking.
  - **`spec.py`**: Pure, pre-boot `RunSpec` validation and CLI/YAML override resolution.
  - **`session.py` & `worker.py`**: Subprocess isolation boundary for running simulation episodes.
  - **`manifest.py`**: Atomic manifest tracking (`BatchManifest`) with crash-resilient `fsync` + replace file writes.
  - **`sweep.py`**: Cartesian matrix planner and sequential execution loop with `--resume` support.
  - **`tracking.py`**: Run indexing and metrics comparison (`results.csv`).
  - **`doctor.py`**: Diagnostic environment and dependency auditor.
- **`nav_arena.core`**: Handles `AppLauncher` bootstrapping. Only put initialization and lifecycle logic here; do not introduce robot or task domain logic.
- **`nav_arena.embodiments`**: Defines physical properties, sensor mount points, and kinematic math.
  - **Light registry:** `registry.py` and `base.py` import no simulator or ML modules, and the package exports its heavy names lazily (like `scenes/`). Register built-in robots with lazy `"module:Class"` strings so `nav_arena robots list` and `--robot` validation stay free of `torch`/`isaaclab`.
  - **Variants:** known configurations of a robot (`kaya.mast`, `kaya.native`) are data-only field overrides declared at registration; see the [robot guide](docs/guides/adding-a-robot.md#variants).
  - **Critical Invariant**: Strictly **ZERO ROS 2 imports**. Embodiments must remain purely mathematical and Isaac Lab-native. All ROS 2 bridging belongs exclusively in `nav_arena.ros2`.
  - Put pure kinematics calculations in [`kinematics.py`](file:///home/robopi/simulation/nav_arena/nav_arena/embodiments/kinematics.py) (e.g. `diff_drive_ik`, `diff_drive_fk`).
- **`nav_arena.tasks`**: Inherit from Isaac Lab's `ManagerBasedRLEnv`. Define MDP command terms, observation groups, termination conditions, and contact metrics here.
- **`nav_arena.scenes`**: Load USD stages and apply non-destructive delta layers. Never mutate source USD assets directly on disk.
- **`nav_arena.methods`**: Autonomy algorithms and baselines, split by execution model.
  - **`methods/ros2/`**: External ROS 2 stacks run as subprocesses (Nav2 bringup, launch files, parameter YAMLs under `ros2/nav2/`).
  - **`methods/in_process/`**: Policies executed directly in the simulation process. Implement [`InProcessPolicy`](file:///home/robopi/simulation/nav_arena/nav_arena/methods/in_process/base.py) (`step(PolicyObservation) -> Plan`) and register it with `register_policy`; policies return a local path and the shared follower ([`controller.py`](file:///home/robopi/simulation/nav_arena/nav_arena/methods/in_process/controller.py)) turns it into velocities at the control rate.
  - **Critical Invariant**: `methods/in_process` must not import `isaacsim`/`omni`/ROS 2 at module level, so policies stay unit-testable on the CPU; import heavy model frameworks lazily in constructors.
  - The NavDP-family adapters wrap an upstream NavDP checkout (see `NAV_ARENA_NAVDP_ROOT`) and enforce **one baseline per process**, because upstream modules share names (e.g. `policy_agent`).
- **`nav_arena.ros2`**: The boundary between Isaac Sim and ROS 2 middleware. Contains OmniGraph generators ([`graph_builder.py`](file:///home/robopi/simulation/nav_arena/nav_arena/ros2/graph_builder.py)), action adapters ([`action_adapter.py`](file:///home/robopi/simulation/nav_arena/nav_arena/ros2/adapters/action_adapter.py)), and publisher nodes ([`sensors.py`](file:///home/robopi/simulation/nav_arena/nav_arena/ros2/sensors.py), [`state_publisher.py`](file:///home/robopi/simulation/nav_arena/nav_arena/ros2/state_publisher.py)).
- **`nav_arena.tools`**: Offline CLI utilities (e.g. [`map_generator.py`](file:///home/robopi/simulation/nav_arena/nav_arena/tools/map_generator.py)). Utilities should be self-contained and callable via `python -m nav_arena.tools.<tool_name>`.
- **`nav_arena.utils`**: Shared developer infrastructure ([`logger.py`](file:///home/robopi/simulation/nav_arena/nav_arena/utils/logger.py), [`process.py`](file:///home/robopi/simulation/nav_arena/nav_arena/utils/process.py), [`paths.py`](file:///home/robopi/simulation/nav_arena/nav_arena/utils/paths.py), [`sim.py`](file:///home/robopi/simulation/nav_arena/nav_arena/utils/sim.py)).
  - **Filesystem paths**: Never hardcode absolute paths. Derive asset, dataset, cache, and log locations from [`nav_arena.utils.paths`](file:///home/robopi/simulation/nav_arena/nav_arena/utils/paths.py) (`WORKSPACE_ROOT`, `DATA_DIR`, `CACHE_DIR`, `LOG_DIR`, `NAVDP_ROOT`), each overridable via a `NAV_ARENA_*` environment variable. Add new locations there with `resolve_path()`.
- **`nav_arena.scripts`**: Thin CLI entrypoints ([`verify_*.py`](file:///home/robopi/simulation/nav_arena/nav_arena/scripts/), [`run_ros2_nav.py`](file:///home/robopi/simulation/nav_arena/nav_arena/scripts/run_ros2_nav.py)). Scripts must only configure arguments, instantiate modular components, and orchestrate execution.

---

## 2. Mandatory Developer Primitives

Always use the standardized framework utilities instead of ad-hoc implementations:

### 1. Structured Logging with `ArenaLogger`
Do **NOT** use raw `print()` statements. Use [`ArenaLogger`](file:///home/robopi/simulation/nav_arena/nav_arena/utils/logger.py) for clean ANSI color formatting, timestamps, severity filtering, and Carbonite bridge integration.

```python
import argparse
from nav_arena.utils import add_logger_args, configure_logging, get_logger

# 1. Bind CLI arguments
parser = argparse.ArgumentParser()
add_logger_args(parser)  # Adds --log-level {DEBUG,INFO,SUCCESS,WARNING,ERROR,CRITICAL}
args = parser.parse_args()

# 2. Configure global logging level
configure_logging(args.log_level)
logger = get_logger("my_component")

# 3. Use structured verification helpers
logger.section("INITIALIZING BENCHMARK")
logger.info("Setting up simulation scene...")
logger.check("Sensor active", sensor is not None, "360 rays initialized")
logger.success("Benchmark completed successfully!")
```

### 2. SimulationApp Lifecycle Management (`launch_simulation_app`)
Omniverse Kit registers ActionGraph schemas and ROS 2 bridge extensions **only** when declared before `AppLauncher` boots. Never instantiate `AppLauncher` directly or mutate `sys.argv` manually. Always use the [`launch_simulation_app`](file:///home/robopi/simulation/nav_arena/nav_arena/core/app.py) context manager:

```python
from nav_arena.core import launch_simulation_app

# Automatically injects --enable omni.graph, isaacsim.ros2.bridge, etc.
with launch_simulation_app(args_cli, enable_ros2=True, livestream=True) as simulation_app:
    # 1. Imports from isaaclab/omni must happen INSIDE or AFTER this block
    from isaaclab.sim import SimulationContext
    sim = SimulationContext()
    ...
    # Lifecycle teardown (simulation_app.close()) is handled automatically upon exit
```

### 3. Subprocess Management (`managed_process`)
When launching background ROS 2 daemons (e.g. `robot_state_publisher`, `nav2_bringup`), do **NOT** use bare `subprocess.Popen()`. Always use [`managed_process`](file:///home/robopi/simulation/nav_arena/nav_arena/utils/process.py) to guarantee process-group cleanup (`SIGTERM` with escalation to `SIGKILL` after 5 seconds):

```python
from nav_arena.utils import managed_process

cmd = ["ros2", "launch", "nav2_bringup", "bringup_launch.py", "use_sim_time:=True"]
with managed_process(cmd, description="Nav2 Bringup"):
    # Run test or step simulation
    ...
# Process group is safely terminated without orphaned background zombies
```

### 4. ROS 2 Concurrency (`BackgroundRos2Executor`)
Do **NOT** call `rclpy.spin()` or synchronous blocking `spin_once(timeout_sec=0.0)` inside simulation physics loops. Single-stepping waitsets under heavy rendering loads starves service clients and drops velocity callbacks. Always spin nodes asynchronously on a dedicated worker thread via [`BackgroundRos2Executor`](file:///home/robopi/simulation/nav_arena/nav_arena/ros2/executor.py):

```python
from nav_arena.ros2 import BackgroundRos2Executor

# Spin nodes concurrently without stalling the PhysX loop
with BackgroundRos2Executor(nodes=[action_adapter, state_publisher, scan_publisher]):
    while simulation_app.is_running():
        sim.step()
```

---

## 3. Physics, Geometry & USD Invariants

- **Quaternion Ordering**: Isaac Lab's `AssetBaseCfg.InitialStateCfg.rot` strictly expects **`(x, y, z, w)`**, NOT `(w, x, y, z)`.
  - Identity rotation: `(0.0, 0.0, 0.0, 1.0)`.
  - Inverted roll (180° around X-axis): `(1.0, 0.0, 0.0, 0.0)`.
- **Scripts and boot order**: Scripts must not import `nav_arena.embodiments`, `nav_arena.tasks`, or anything else that pulls in Isaac Lab/`pxr` at module level: `AppLauncher` must boot first (the app crashes otherwise). Do not populate argparse `choices` from the registries; validate names after boot.
- **Spawn Ground Clearance**: Mobile robots must never spawn at `pos=(0.0, 0.0, 0.0)` on a `GroundPlane` at $z=0$. Always provide clearance (e.g. `pos=(0.0, 0.0, 0.25)`) to prevent explosive PhysX depenetration impulses on step 0.
- **USD Delta Layers & Raycaster Static BVH**: Isaac Lab's `MultiMeshRayCaster` bakes its static triangle BVH (`merge_prim_meshes=True`) at environment load. Calling `prim.SetActive(False)` *after* stage load drops PhysX collision, but **does not rebuild the raycaster BVH**. To open doors or alter scene geometry for sensors, pre-condition scenes prior to stage load using USD composition delta layers (`subLayerPaths` via `get_preprocessed_usd`).
- **Standard Coordinate Frames (REP-105)**: The bridge strictly maintains the standard transform tree: `map -> odom -> base_link -> chassis_link -> lidar_link`. Odometry publishes `odom -> base_link`, and AMCL or static state publisher provides `map -> odom`.
- **Latched Topic QoS**: Task state topics that publish once per episode (e.g., `/goal_pose`, `/goal_reached`) must use `TRANSIENT_LOCAL` durability with depth 1 so late-joining nodes immediately receive state updates.

### Embodiment, Sensor, and Learned-Baseline Gotchas

Each item below cost real debugging time; the "why" is the part to remember. How-to guides: [`docs/guides/`](docs/guides/).

**Embodiments and robot assets**
- **Fix assets statically, not at runtime.** Express USD fixes as a derived asset ([`embodiments/assets.py`](nav_arena/embodiments/assets.py)): a small layer sublayering the untouched upstream file. A runtime `stage_patch_fn` runs as a `prestartup` event, which Isaac Lab only allows with `replicate_physics=False` (replicated physics parses just `env_0`, so per-clone edits would be silently ignored); see the README's "Robot Assets and USD Fixes".
- **`pxr.PhysxSchema` only exists inside Kit.** Author PhysX attributes as plain USD (`apiSchemas` metadata plus the attribute) so code works in CPU tests, and raise if the target prim is missing instead of silently skipping.
- **Derived USDs must carry `metersPerUnit` and `upAxis`.** Sublayer metadata does not propagate to the referencing stage; a missing `metersPerUnit` silently rescales the robot.
- **Lazy derived asset instantiation**: Derived asset configurations must be created via factory functions (e.g. `create_dingo_articulation_cfg()`) rather than module-level globals (`DINGO_CFG`). Eager USD generation during module imports breaks pure-Python tools and CPU unit tests before `AppLauncher` boots or when cache files are missing.
- **Single-rigid-body robots (Dingo).** Colliders, caster and sensors all live on `base_link` (`body_link`), and the resting caster reads ~17.7 N on that link. `ground_contact_on_body=True` switches collision detection to lateral (XY) force; a plain force-norm check would terminate at t=0.
- **Camera orientation.** Cameras use `convention="world"` (+X forward, +Z up) with the XYZW quaternion `(0, 0, 0, 1)`; copying a WXYZ quaternion from older Isaac Lab code rotates the camera.
- **Occupancy maps are ROS-oriented**: image row 0 is the maximum y. Reading a map picture mirrored led to unsafe routes once.
- **Current command space is `[v, omega]`** everywhere (action term, follower, runner). New drive types (ackermann, holonomic) must change all of them; see the "new drive type" section of the adding-a-robot guide.

**Sensors, rendering, and the viewport**
- **Never add visual aids as scene geometry.** Isaac Lab's goal arrow is real geometry that depth/RGB cameras see: planners treat it as an obstacle on their own goal and stop short (it erased a real run). It is off by default; the viewport overlay is an `omni.ui.scene` layer. `isaacsim.util.debug_draw` was measured to leak into camera RGB, so do not use it for anything a policy could see.
- **`render_interval`** defaults to 20 (5 Hz at dt=0.01) when a camera is enabled; the runner calls `task.refresh_camera_frame()` at each plan rather than relying on render-phase alignment.
- **GUI playback can look jerky** although simulated speed is constant (uneven wall-clock step cost). Not a physics bug; see the results doc's future work.

**Learned baselines**
- **One baseline per Python process.** Upstream NavDP modules share names (`policy_agent`), so a process can load only one planner. Strict checkpoint loading and the SHA-256 table in `navdp_adapter/checkpoints.py` are deliberate; do not relax them.
- **Install baseline dependencies only into `env_isaaclab`**, with `uv pip install --no-deps` (Isaac's pinned torch/numpy must not move). Never install globally.
- **iPlanner/VIPlanner stop permanently** when predicted fear reaches the threshold; a stopped robot's view never changes, so nothing recovers. The runner reports this as a `stalled` episode and logs the stop.
- **Argument errors must come before the slow boot.** `verify_baseline.py` rejects ignored combinations (`--spawn` without `--goal`, `--route` with `--spawn/--goal`) through [`nav_arena.utils.cli_args`](nav_arena/utils/cli_args.py); put new pre-boot checks there so they are unit testable.

**Lifecycle and processes**
- **`launch_simulation_app` ends the process on exit.** `SimulationApp.close()` terminates it, so code after the `with` block never runs; return results by calling `sys.exit(n)` inside the block (an unhandled exception exits 1 with a logged traceback). Before this was fixed, every failure exited 0 and hid a failing integration test.
- **One Isaac app per process**, so multi-run work needs one subprocess per run.
- **Run directories are never overwritten**: single runs default to `<timestamp>_<method>_<robot>_<route>_<hex4>` (unique even within one second). The recorder accepts an empty or spec-only directory (the orchestrator pre-creates it with `run_spec.json`) but raises `FileExistsError` if it already holds recorder output; `sweep --resume` and `runs rerun` clear the previous attempt explicitly first (`nav_arena.utils.run_dir`).
- **Visualization settings are tri-state**: `VizCfg.follow_camera`/`goal_overlay` default to `None` ("auto", follow `gui`) and are resolved only by `VizCfg.resolved()`. Do not default them in a CLI handler: that is how `nav_arena run --gui` once lost the follow camera.
- **`method_params` holds policy config fields only.** Viewer and scene knobs belong in `VizCfg`; `device`, `plan_hz` and `task` are policy config keys passed straight through.

---

## 4. Testing Framework & Marker Tiers

All contributions must be accompanied by appropriate test coverage across the 3-tier testing hierarchy:

| Tier | Directory | Scope | Target Runtime | Marker |
|---|---|---|---|---|
| **L1: Unit** | [`tests/unit/`](file:///home/robopi/simulation/nav_arena/tests/unit/) | Fast CPU-only algorithmic logic, math, URDF parsing, mock tests. | ~11 seconds (410+ tests) | `unit` (default) |
| **L2: ROS 2** | [`tests/ros2/`](file:///home/robopi/simulation/nav_arena/tests/ros2/) | Subprocess and ROS 2 middleware node interaction. | ~30 seconds | `@pytest.mark.ros2` |
| **L3: Simulation** | [`tests/integration/`](file:///home/robopi/simulation/nav_arena/tests/integration/) | Full in-process or subprocess Isaac Sim PhysX and sensor tests. | ~30 seconds | `@pytest.mark.integration` |

### Testing Rules
1. **CPU-Only Isolation for L1**: Unit tests in `tests/unit/` must never import `isaacsim` or boot Omniverse Kit. All ROS launch testing autoloading is disabled in `pyproject.toml` (`-p no:launch-testing-ros -p no:launch_testing`).
2. **Pre-Commit Verification**: Run the fast L1 test suite before submitting changes:
   ```bash
   source setup.env
   pytest -c nav_arena/pyproject.toml nav_arena/tests/unit -v
   ```
3. **Targeted Runs**: To run the entire test suite excluding heavy GPU simulation:
   ```bash
   pytest -c nav_arena/pyproject.toml -m "not integration" -v
   ```

---

## 5. Command Execution Conventions

- **Base Directory**: All user commands run from `/home/robopi/simulation`.
- **User Instruction Format**: Always format execution commands for users as:
  ```bash
  source setup.env
  nav_arena <subcommand> [args...]
  ```
  or for internal developer verification scripts:
  ```bash
  source setup.env
  python -u nav_arena/nav_arena/scripts/<script.py> [args...]
  ```
  Ensure all paths prepend `nav_arena/` when referencing internal package scripts.
- **Agent Self-Execution**: In automated agent terminals, run `./agy_python.sh <script.py> [args...]` from `/home/robopi/simulation` to encapsulate `setup.env` and provide unbuffered stdout (`-u`).
- **GPU etiquette**: Do not run Isaac jobs in the background while someone is using the GUI on the same machine: a long headless batch hung a live viewport (black screen). Check for running sessions first (`ps aux | grep -E 'verify_|isaaclab'`), and run integration tests (L3) only when the GPU is free.
- **Interactive GUI**: Isaac Lab runs headless by default even if `--headless` is omitted (`--viz none`). To open the interactive viewport window on the user's active display, pass `--gui` in `nav_arena run` (or `--viz kit` in raw scripts).
