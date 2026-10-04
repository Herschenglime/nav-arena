# Contributing & Development Guidelines

Welcome to `nav_arena`. This document outlines the architectural boundaries, coding standards, developer primitives, and testing tiers required when developing new features or extending existing modules.

---

## 1. Architectural Boundaries (Where Code Lives)

The codebase strictly enforces separation of concerns across nine dedicated modules:

```text
nav_arena/
├── core/         # SimulationApp lifecycle & boot-time extension injection
├── embodiments/  # Robot kinematics, sensor configs, in-memory URDF synthesis (ZERO ROS 2 dependencies)
├── tasks/        # RL/benchmark environments (ManagerBasedRLEnv), MDP terms, episode resets
├── scenes/       # Non-destructive USD loaders, delta layer conditioning, BVH preprocessors
├── methods/      # Autonomy baselines and upstream stack configs (Nav2 parameters and bringup)
├── ros2/         # ROS 2 middleware bridges (OmniGraph builders, action adapters, async publisher nodes)
├── tools/        # Standalone offline utilities (CLI 2D occupancy grid generator)
├── utils/        # Cross-cutting primitives (ArenaLogger, managed_process, create_mock_env)
├── scripts/      # Thin executable runners and verifiers (NO domain logic definitions)
└── tests/        # 3-tier verification suite (unit/, ros2/, integration/)
```

### Module Responsibilities & Invariants

- **`nav_arena.core`**: Handles `AppLauncher` bootstrapping. Only put initialization and lifecycle logic here; do not introduce robot or task domain logic.
- **`nav_arena.embodiments`**: Defines physical properties, sensor mount points, and kinematic math.
  - **Critical Invariant**: Strictly **ZERO ROS 2 imports**. Embodiments must remain purely mathematical and Isaac Lab-native. All ROS 2 bridging belongs exclusively in `nav_arena.ros2`.
  - Put pure kinematics calculations in [`kinematics.py`](file:///home/robopi/simulation/nav_arena/nav_arena/embodiments/kinematics.py) (e.g. `diff_drive_ik`, `diff_drive_fk`).
- **`nav_arena.tasks`**: Inherit from Isaac Lab's `ManagerBasedRLEnv`. Define MDP command terms, observation groups, termination conditions, and contact metrics here.
- **`nav_arena.scenes`**: Load USD stages and apply non-destructive delta layers. Never mutate source USD assets directly on disk.
- **`nav_arena.methods`**: Autonomy algorithms and baselines. Nav2 bringup, launch files, parameter YAMLs, and alternative planning methods belong here.
- **`nav_arena.ros2`**: The boundary between Isaac Sim and ROS 2 middleware. Contains OmniGraph generators ([`graph_builder.py`](file:///home/robopi/simulation/nav_arena/nav_arena/ros2/graph_builder.py)), action adapters ([`action_adapter.py`](file:///home/robopi/simulation/nav_arena/nav_arena/ros2/adapters/action_adapter.py)), and publisher nodes ([`sensors.py`](file:///home/robopi/simulation/nav_arena/nav_arena/ros2/sensors.py), [`state_publisher.py`](file:///home/robopi/simulation/nav_arena/nav_arena/ros2/state_publisher.py)).
- **`nav_arena.tools`**: Offline CLI utilities (e.g. [`map_generator.py`](file:///home/robopi/simulation/nav_arena/nav_arena/tools/map_generator.py)). Utilities should be self-contained and callable via `python -m nav_arena.tools.<tool_name>`.
- **`nav_arena.utils`**: Shared developer infrastructure ([`logger.py`](file:///home/robopi/simulation/nav_arena/nav_arena/utils/logger.py), [`process.py`](file:///home/robopi/simulation/nav_arena/nav_arena/utils/process.py), [`sim.py`](file:///home/robopi/simulation/nav_arena/nav_arena/utils/sim.py)).
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
- **Spawn Ground Clearance**: Mobile robots must never spawn at `pos=(0.0, 0.0, 0.0)` on a `GroundPlane` at $z=0$. Always provide clearance (e.g. `pos=(0.0, 0.0, 0.25)`) to prevent explosive PhysX depenetration impulses on step 0.
- **USD Delta Layers & Raycaster Static BVH**: Isaac Lab's `MultiMeshRayCaster` bakes its static triangle BVH (`merge_prim_meshes=True`) at environment load. Calling `prim.SetActive(False)` *after* stage load drops PhysX collision, but **does not rebuild the raycaster BVH**. To open doors or alter scene geometry for sensors, pre-condition scenes prior to stage load using USD composition delta layers (`subLayerPaths` via `get_preprocessed_usd`).
- **Standard Coordinate Frames (REP-105)**: The bridge strictly maintains the standard transform tree: `map -> odom -> base_link -> chassis_link -> lidar_link`. Odometry publishes `odom -> base_link`, and AMCL or static state publisher provides `map -> odom`.
- **Latched Topic QoS**: Task state topics that publish once per episode (e.g., `/goal_pose`, `/goal_reached`) must use `TRANSIENT_LOCAL` durability with depth 1 so late-joining nodes immediately receive state updates.

---

## 4. Testing Framework & Marker Tiers

All contributions must be accompanied by appropriate test coverage across the 3-tier testing hierarchy:

| Tier | Directory | Scope | Target Runtime | Marker |
|---|---|---|---|---|
| **L1: Unit** | [`tests/unit/`](file:///home/robopi/simulation/nav_arena/tests/unit/) | Fast CPU-only algorithmic logic, math, URDF parsing, mock tests. | < 3 seconds | `unit` (default) |
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
  python -u nav_arena/nav_arena/scripts/<script.py> [args...]
  ```
  Ensure all paths prepend `nav_arena/` when referencing internal package scripts.
- **Agent Self-Execution**: In automated agent terminals, run `./agy_python.sh <script.py> [args...]` from `/home/robopi/simulation` to encapsulate `setup.env` and provide unbuffered stdout (`-u`).
- **Interactive GUI**: Isaac Lab runs headless by default even if `--headless` is omitted (`--viz none`). To open the interactive viewport window on the user's active display, pass `--viz kit`.
