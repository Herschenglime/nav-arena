# Nav Arena

A lightweight, modular multi-embodiment navigation simulation and evaluation framework built on native **NVIDIA Isaac Sim 6.1.0** and **Isaac Lab 3.0.0** on `aarch64` Linux (Ubuntu 24.04 / NVIDIA GB10).

---

## 1. Overview & Architecture

### Project Goal
The primary goal of `nav_arena` is scaffolding **Isaac Lab Arena** to work natively for autonomous navigation tasks. It acts as a robust, production-grade bridge connecting high-fidelity PhysX physics and photorealistic USD simulation directly to standard ROS 2 navigation stacks (such as Nav2) without friction or performance degradation.

### Separation of Concerns
The repository enforces a strict decoupling across modular subsystems:
- **Core Simulation ([`nav_arena.core`](file:///home/robopi/simulation/nav_arena/nav_arena/core/app.py))**: Centralized SimulationApp lifecycle management ([`launch_simulation_app`](file:///home/robopi/simulation/nav_arena/nav_arena/core/app.py)), boot-time OmniGraph/ROS 2 extension flag injection, and livestreaming configuration.
- **Embodiments ([`nav_arena.embodiments`](file:///home/robopi/simulation/nav_arena/nav_arena/embodiments/))**: Robot physical properties, kinematic configurations (`DifferentialDriveAction`), pure kinematics math ([`diff_drive_ik`](file:///home/robopi/simulation/nav_arena/nav_arena/embodiments/kinematics.py)), programmatic sensor rigging (planar 360° LiDAR), and programmatic in-memory URDF synthesis ([`generate_robot_urdf`](file:///home/robopi/simulation/nav_arena/nav_arena/embodiments/urdf.py)).
- **ROS 2 Bridges ([`nav_arena.ros2`](file:///home/robopi/simulation/nav_arena/nav_arena/ros2/))**: Zero-latency OmniGraph nodes for simulation clock (`/clock`), odometry (`/odom`), and TF (`map -> odom -> base_link`), coupled with asynchronous Python nodes ([`LaserScanPublisherNode`](file:///home/robopi/simulation/nav_arena/nav_arena/ros2/sensors.py), [`TaskStatePublisherNode`](file:///home/robopi/simulation/nav_arena/nav_arena/ros2/state_publisher.py)), action adapters ([`TwistActionAdapter`](file:///home/robopi/simulation/nav_arena/nav_arena/ros2/adapters/action_adapter.py)), and background executors ([`BackgroundRos2Executor`](file:///home/robopi/simulation/nav_arena/nav_arena/ros2/executor.py)).
- **Scenes ([`nav_arena.scenes`](file:///home/robopi/simulation/nav_arena/nav_arena/scenes/interior_agent.py))**: Non-destructive USD scene conditioning using composition delta layers (`subLayerPaths`), automatic doorway clearing, and static triangle BVH generation for InteriorAgent assets.
- **Tasks ([`nav_arena.tasks`](file:///home/robopi/simulation/nav_arena/nav_arena/tasks/point_nav.py))**: RL and benchmark task definitions ([`PointNavTask`](file:///home/robopi/simulation/nav_arena/nav_arena/tasks/point_nav.py) extending Isaac Lab's `ManagerBasedRLEnv`), goal sampling, timeout handling, and chassis-isolated contact metrics.
- **Methods ([`nav_arena.methods`](file:///home/robopi/simulation/nav_arena/nav_arena/methods/nav2/))**: Upstream navigation stack baselines and integration launch infrastructure, including Nav2 lifecycle orchestrators and tuned parameter configurations.
- **Tools ([`nav_arena.tools`](file:///home/robopi/simulation/nav_arena/nav_arena/tools/map_generator.py))**: Standalone offline utilities, including programmatic 2D occupancy grid generation from USD collision geometry.
- **Utilities ([`nav_arena.utils`](file:///home/robopi/simulation/nav_arena/nav_arena/utils/))**: Shared cross-cutting infrastructure: unified structured logging ([`ArenaLogger`](file:///home/robopi/simulation/nav_arena/nav_arena/utils/logger.py)), robust subprocess lifecycle management ([`managed_process`](file:///home/robopi/simulation/nav_arena/nav_arena/utils/process.py)), and simulation mock helpers ([`create_mock_env`](file:///home/robopi/simulation/nav_arena/nav_arena/utils/sim.py)).

---

## 2. Repository Layout

```text
nav_arena/
├── pyproject.toml                     # Package specification (editable pip install)
├── README.md                          # Project documentation and architecture guide
├── cache/                             # Generated runtime caches (git-ignored)
│   ├── maps/                          # Cached 2D occupancy grids (PNG + YAML)
│   └── scenes/                        # Conditioned USD delta layers (e.g. open_doors)
├── nav_arena/
│   ├── config/                        # RViz visualization layouts and displays
│   ├── core/                          # SimulationApp lifecycle & boot-time extension injection
│   │   ├── __init__.py                # Exports launch_simulation_app
│   │   └── app.py                     # launch_simulation_app context manager
│   ├── embodiments/                   # Robot kinematics, sensor factories, in-memory URDF
│   │   ├── actions.py                 # DifferentialDriveActionCfg & ActionAdapter
│   │   ├── kinematics.py              # Pure math differential drive kinematics (FK & IK)
│   │   ├── nova_carter.py             # Nova Carter differential base configuration
│   │   ├── sensors.py                 # Planar 2D LiDAR raycaster configuration
│   │   └── urdf.py                    # Programmatic URDF string synthesis
│   ├── methods/                       # Autonomy baselines and external stacks
│   │   └── nav2/                      # Nav2 bringup launch scripts, parameters, URDF bridge
│   ├── ros2/                          # Core ROS 2 Integration Architecture
│   │   ├── adapters/                  # Action adapters (TwistActionAdapter for /cmd_vel)
│   │   ├── executor.py                # BackgroundRos2Executor (dedicated worker thread)
│   │   ├── graph_builder.py           # OmniGraph builder for clock, TF, and odometry
│   │   ├── sensors.py                 # LaserScanPublisherNode (360° RayCaster to /scan)
│   │   └── state_publisher.py         # TaskStatePublisherNode (static TF, /goal_pose, /goal_reached)
│   ├── scenes/                        # USD loaders and delta layer conditioning (InteriorAgent)
│   │   └── interior_agent.py          # USD scene loader, delta layer conditioning, BVH preprocessor
│   ├── tasks/                         # PointNavTask (ManagerBasedRLEnv), MDP terms, metrics
│   │   └── point_nav.py               # PointNavTask environment & MDP configuration
│   ├── tools/                         # Offline utilities (CLI 2D map generator)
│   │   └── map_generator.py           # Programmatic 2D occupancy grid generation tool
│   ├── utils/                         # Cross-cutting primitives and helpers
│   │   ├── logger.py                  # ArenaLogger framework, ANSI colors, Carbonite bridge
│   │   ├── process.py                 # managed_process subprocess context manager
│   │   └── sim.py                     # Testing mocks and scene path resolution
│   └── scripts/                       # Executable verification and benchmark runners
│       ├── run_ros2_nav.py            # Main execution orchestrator for ROS 2 bridge
│       ├── verify_embodiment.py       # Validates robot kinematics and 2D LiDAR ranges
│       ├── verify_scene.py            # Validates scene loading and sensor raycasting
│       ├── verify_task.py             # Validates PointNavTask MDP metrics and resets
│       ├── verify_occupancy_map.py    # Generates and validates 2D occupancy maps
│       ├── verify_tf_tree.py          # Validates robot_state_publisher and complete TF tree
│       └── verify_nav2.py             # End-to-end Nav2 autonomous navigation benchmark
└── tests/
    ├── unit/                          # Tier 1: Fast CPU-only unit tests (~2s via pytest)
    ├── ros2/                          # Tier 2: Subprocess & ROS 2 integration tests
    └── integration/                   # Tier 3: Full Isaac Sim simulation integration tests
```

---

## 3. Installation & Setup

Ensure your Isaac Lab virtual environment (`env_isaaclab`) and ROS 2 Jazzy workspace are configured:

```bash
cd /home/robopi/simulation
source setup.env
pip install -e nav_arena
```

> **Execution Rule for Users**: Always execute commands directly from `/home/robopi/simulation` using `source setup.env && python -u nav_arena/nav_arena/scripts/<script.py> [args...]`.

---

## 4. Core Frameworks & Abstractions

### SimulationApp Lifecycle Management (`nav_arena.core`)
Omniverse Kit requires ActionGraph schemas and ROS 2 bridge extensions to be declared in `sys.argv` *before* `AppLauncher` is instantiated. The [`launch_simulation_app`](file:///home/robopi/simulation/nav_arena/nav_arena/core/app.py) context manager encapsulates this requirement:

```python
from nav_arena.core import launch_simulation_app

# Automatically injects --enable omni.graph, isaacsim.ros2.bridge, etc.
with launch_simulation_app(args_cli, enable_ros2=True, livestream=True) as simulation_app:
    # Build stage, construct ActionGraphs, step simulation
    ...
```

### Structured Logging Framework (`ArenaLogger`)
The [`ArenaLogger`](file:///home/robopi/simulation/nav_arena/nav_arena/utils/logger.py) framework standardizes logging across all scripts and tests:
- Custom `SUCCESS` severity level ($25$, between `INFO` and `WARNING`).
- Clean visual formatting: ANSI colors, elapsed execution timestamps, and suppression of noisy third-party loggers (e.g. `rclpy`, `urdf_parser_py`).
- Automatic bridging to Omniverse Carbonite log handlers (`carb.log_info`, `carb.log_warn`, `carb.log_error`) when running inside Isaac Sim.
- Verification assertion helpers: `logger.section(title)`, `logger.check(name, passed, detail)`, and `logger.success(msg)`.

```python
import argparse
from nav_arena.utils import add_logger_args, configure_logging, get_logger

parser = argparse.ArgumentParser()
add_logger_args(parser)  # Adds --log-level {DEBUG,INFO,SUCCESS,WARNING,ERROR,CRITICAL}
args = parser.parse_args()

configure_logging(args.log_level)
logger = get_logger("my_component")

logger.section("INITIALIZING BENCHMARK")
logger.info("Setting up simulation environment...")
logger.check("Sensor initialized", lidar is not None, "360 rays active")
logger.success("Benchmark completed successfully!")
```

### Offline 2D Occupancy Map Generation (`nav_arena.tools.map_generator`)
Generates ROS 2-standard 2D occupancy grid maps (`map.yaml` and `map.png`) directly from USD collision meshes across an obstacle height slice (default: $0.01\,\text{m}$ to $0.60\,\text{m}$) without requiring manual SLAM:

```bash
source setup.env
# Run map generator on default scene (kujiale_0003) with 0.05m resolution
python -u -m nav_arena.tools.map_generator --scene kujiale_0003 --cell-size 0.05

# Force regeneration with custom height bounds
python -u -m nav_arena.tools.map_generator --scene kujiale_0003 --z-min 0.02 --z-max 0.80 --force
```

Maps are automatically cached to `cache/maps/<scene_id>/cs<cell_size>_z<z_min>-<z_max>/` and dynamically located by Nav2 bringup.

---

## 5. Testing Framework Tiers

The test suite is organized into three distinct verification tiers:

| Tier | Directory | Description | Typical Runtime | Target Environment |
|---|---|---|---|---|
| **L1: Unit Tests** | [`tests/unit/`](file:///home/robopi/simulation/nav_arena/tests/unit/) | Fast, CPU-only algorithmic & component tests (kinematics, LaserScan math, logger, process management, scene resolution, URDF synthesis). | ~2.2 seconds (52 tests) | Pure Python / CPU |
| **L2: ROS 2 Tests** | [`tests/ros2/`](file:///home/robopi/simulation/nav_arena/tests/ros2/) | Subprocess & ROS 2 middleware tests (Action adapters, OmniGraph builders, state publisher QoS, closed-loop driving). Marked with `@pytest.mark.ros2`. | ~30 seconds (4 tests) | ROS 2 Jazzy & Subprocess |
| **L3: Simulation Tests** | [`tests/integration/`](file:///home/robopi/simulation/nav_arena/tests/integration/) | In-process Isaac Sim tests (occupancy grid generation, full `PointNavTask` stepping and resets, TF tree continuity). Marked with `@pytest.mark.integration`. | ~26 seconds (3 tests) | GPU / Isaac Sim PhysX |

### Running Tests

```bash
source setup.env

# 1. Run all L1 Unit Tests (fast CPU verification)
pytest -c nav_arena/pyproject.toml nav_arena/tests/unit -v

# 2. Run L2 ROS 2 Integration Tests
pytest -c nav_arena/pyproject.toml nav_arena/tests/ros2 -v

# 3. Run L3 Simulation Integration Tests
pytest -c nav_arena/pyproject.toml nav_arena/tests/integration -v

# 4. Run entire suite excluding heavy GPU simulation
pytest -c nav_arena/pyproject.toml -m "not integration" -v
```

---

## 6. Execution & Verification Scripts

All user commands run from `/home/robopi/simulation`.

### Embodiment Verification
Validates Nova Carter kinematics, exercises `DifferentialDriveAction`, reads 2D LiDAR range arrays, and validates displacement:
```bash
source setup.env
python -u nav_arena/nav_arena/scripts/verify_embodiment.py
```

### Scene Verification (InteriorAgent)
Loads an InteriorAgent USD scene, settles the robot on floor geometry, drives forward, and verifies 360° LiDAR raycasting:
```bash
source setup.env
# Headless execution
python -u nav_arena/nav_arena/scripts/verify_scene.py --scene kujiale_0003

# Interactive GUI window
python -u nav_arena/nav_arena/scripts/verify_scene.py --viz kit --loop
```

### Occupancy Map Verification
Generates and validates 2D occupancy map bounds and verifies YAML metadata:
```bash
source setup.env
python -u nav_arena/nav_arena/scripts/verify_occupancy_map.py --scene kujiale_0003
```

### TF Tree Verification
Spawns the robot, launches `robot_state_publisher` using programmatic URDF, and asserts transform continuity (`map -> odom -> base_link -> chassis_link -> lidar_link`):
```bash
source setup.env
python -u nav_arena/nav_arena/scripts/verify_tf_tree.py
```

### PointNav Task & Metrics Verification
Instantiates `PointNavTask`, exercises goal tracking (`UniformPose2dCommandCfg`), drives the robot forward to verify goal reach termination, checks state restoration upon episode reset, and tests chassis-isolated collision detection:
```bash
source setup.env
# Headless task test
python -u nav_arena/nav_arena/scripts/verify_task.py

# Interactive GUI inspection
python -u nav_arena/nav_arena/scripts/verify_task.py --viz kit
```

### Nav2 Autonomous Navigation Benchmark
Launches Isaac Lab alongside the complete Nav2 navigation stack (Map Server, AMCL, Costmaps, NavfnPlanner, DWBLocalPlanner, BT Navigator), feeds simulated LiDAR to `/scan`, synchronizes `/clock` and TF, and autonomously navigates to goal poses:
```bash
source setup.env
# Run headless benchmark (default open living room route: (-2.5, 0.0) -> (-1.0, 0.0))
python -u nav_arena/nav_arena/scripts/verify_nav2.py

# Run with interactive Omniverse Kit GUI and RViz2 visualization
python -u nav_arena/nav_arena/scripts/verify_nav2.py --viz kit --rviz

# Run with custom start and goal coordinates
python -u nav_arena/nav_arena/scripts/verify_nav2.py --spawn-x -2.5 --spawn-y 0.0 --goal-x 1.5 --goal-y 0.0
```

### ROS 2 Bridge Runner
Main execution orchestrator booting the simulation, establishing the OmniGraph ROS 2 bridge, and listening to `/cmd_vel`:
```bash
source setup.env
# Terminal 1: Launch simulation bridge
python -u nav_arena/nav_arena/scripts/run_ros2_nav.py

# Terminal 2: Visualize in RViz2
rviz2 -d nav_arena/nav_arena/config/nav_arena.rviz
```

---

## 7. Implementation Invariants & Technical Notes

- **Quaternion Ordering**: Isaac Lab's `AssetBaseCfg.InitialStateCfg.rot` strictly expects **`(x, y, z, w)`**. Identity rotation is `(0.0, 0.0, 0.0, 1.0)`. Passing `(1.0, 0.0, 0.0, 0.0)` rolls the asset 180° around the X-axis (inverting it).
- **Spawn Ground Clearance**: Mobile robots must spawn with clearance (e.g. `pos=(0.0, 0.0, 0.25)`) to avoid violent PhysX collision depenetration impulses on step 0.
- **OmniGraph Extension Booting**: Any script creating OmniGraphs or ROS 2 bridge nodes must declare extensions before `AppLauncher` boots. Use [`launch_simulation_app`](file:///home/robopi/simulation/nav_arena/nav_arena/core/app.py) to manage this automatically.
- **USD Delta Layers & Raycaster Static BVH**: Isaac Lab's `MultiMeshRayCaster` bakes its static triangle BVH (`merge_prim_meshes=True`) at environment initialization. Calling `prim.SetActive(False)` *after* stage load drops PhysX collision, but **does not rebuild the raycaster BVH**. To open doors or alter geometry for sensors, scenes are pre-conditioned before load using non-destructive USD delta layers (`subLayerPaths` via `get_preprocessed_usd`).
- **ROS 2 Concurrency & Worker Threads**: Coupling asynchronous DDS waitsets to a synchronous physics loop via single-step `spin_once(timeout_sec=0.0)` starves service futures and drops `/cmd_vel` callbacks under rendering load. ROS 2 callback dispatch runs on a dedicated worker thread via [`BackgroundRos2Executor`](file:///home/robopi/simulation/nav_arena/nav_arena/ros2/executor.py).
- **Standard Coordinate Frames (REP-105)**: The bridge strictly maintains the REP-105 transform tree `map -> odom -> base_link -> chassis_link -> lidar_link`. Odometry publishes `odom -> base_link`, and AMCL or the static state publisher provides `map -> odom`.
- **Latched Topic QoS**: Task state topics that publish once per episode (e.g., `/goal_pose` and `/goal_reached`) use `TRANSIENT_LOCAL` durability with depth 1 to prevent late-joining subscribers from missing signals.
- **Doorway Costmap Buffers**: In residential scenes with $0.65 - 0.80\,\text{m}$ doorways, Nav2 costmaps configure tight footprint padding ($0.01\,\text{m}$) and an inflation radius ($0.35\,\text{m}$) with decay $\alpha = 3.0 / 4.0$ to provide a clean zero-cost corridor through narrow passages.
