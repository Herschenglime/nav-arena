# Nav Arena

A lightweight, modular multi-embodiment navigation simulation and evaluation framework built natively on **NVIDIA Isaac Sim 6.1.0** and **Isaac Lab 3.0.0** on `aarch64` Linux (Ubuntu 24.04 / NVIDIA GB10).

---

## 1. Overview & Project Goals

### Benchmarking Navigation Models
The primary mission of `nav_arena` is providing a unified, reproducible testbed to **benchmark, evaluate, and compare diverse navigation models**—including reinforcement learning policies, learned end-to-end agents, heuristic methods, and classical navigation stacks.

### Conceptual Model, Not Arena APIs
Rather than importing or binding to Isaac Lab Arena software APIs, `nav_arena` adopts its clean **conceptual design model**: strictly separating **Tasks** (MDP terms, metrics, episode resets), **Embodiments** (physical kinematics, sensor configurations, URDF synthesis), and **Scenes** (USD environments, semantic collision geometry). This modular decoupling keeps the simulation runtime lightweight, portable, and free of unnecessary framework overhead.

### Role of ROS 2 & Nav2: A Verification Bridge
While `nav_arena` provides production-grade ROS 2 bridging, **ROS 2 is not the central objective of the project**. Instead, the complete Nav2 navigation stack was implemented as a reference external baseline to rigorously stress-test and verify simulation bridging capabilities—confirming that zero-latency clock synchronization (`/clock`), OmniGraph odometry (`/odom`), 360° RayCaster LiDAR publishing (`/scan`), and REP-105 transform trees operate seamlessly without performance degradation or physics stalls.

---

## 2. Quickstart & Installation

### Environment Setup
Ensure your Isaac Lab virtual environment and ROS 2 Jazzy workspace are activated:

```bash
cd /home/robopi/simulation
source setup.env
pip install -e nav_arena
```

> **Execution Rule for Users**: Always execute commands directly from `/home/robopi/simulation` using `source setup.env && python -u nav_arena/nav_arena/scripts/<script.py> [args...]`.

### Common Quickstart Commands

```bash
source setup.env

# 1. Run the autonomous Nav2 navigation benchmark (headless)
python -u nav_arena/nav_arena/scripts/verify_nav2.py

# 2. Run Nav2 benchmark with interactive Kit GUI and RViz2 visualization
python -u nav_arena/nav_arena/scripts/verify_nav2.py --viz kit --rviz

# 3. Inspect PointNav task execution interactively in Omniverse Kit
python -u nav_arena/nav_arena/scripts/verify_task.py --viz kit

# 4. Generate an offline 2D occupancy grid for scene kujiale_0003
python -u -m nav_arena.tools.map_generator --scene kujiale_0003 --cell-size 0.05

# 5. Run the fast CPU-only unit test suite (< 2.5s)
pytest -c nav_arena/pyproject.toml nav_arena/tests/unit -v
```

### Learned Baseline Setup (NavDP family)

The in-process learned baselines (iPlanner, ViNT, NavDP, VIPlanner, X-NavDP) keep their network code upstream in a NavDP checkout at `<workspace>/NavDP` (override with `NAV_ARENA_NAVDP_ROOT`). Use the maintained fork, which carries the device-handling and Python 3.12 fixes the baselines need:

```bash
cd ~/simulation
git clone git@github.com:Herschenglime/NavDP.git NavDP
git -C NavDP checkout nav-arena          # tested with commit 8b9ee13
```

Install the extra Python packages into the existing Isaac Lab uv environment **without dependency resolution**, so Isaac's pinned torch/numpy stay in place (nothing is installed globally):

```bash
source setup.env
uv pip install --python "$VIRTUAL_ENV/bin/python" --no-deps -r nav_arena/requirements/baselines.txt
# VIPlanner additionally needs mmcv with CUDA ops: see nav_arena/requirements/baselines-mmcv.md
```

Checkpoints live in `NavDP/baselines/<planner>/checkpoints/`; their expected SHA-256 hashes and sources are recorded in [`navdp_adapter/checkpoints.py`](file:///home/robopi/simulation/nav_arena/nav_arena/methods/in_process/navdp_adapter/checkpoints.py). Only one baseline can be loaded per Python process (upstream modules share names), so run one method per invocation.

---

## 3. Execution & Verification Scripts

All executable runners are located in [`nav_arena/nav_arena/scripts/`](file:///home/robopi/simulation/nav_arena/nav_arena/scripts/) and are launched from `/home/robopi/simulation`.

### Autonomous Nav2 Navigation Benchmark
Launches Isaac Lab alongside the complete Nav2 autonomy stack (Map Server, AMCL, Costmaps, NavfnPlanner, DWBLocalPlanner, BT Navigator), publishes simulated LiDAR to `/scan`, synchronizes `/clock` and TF, and autonomously navigates the Nova Carter base to goal poses:
```bash
source setup.env

# Headless benchmark (default open living room route: (-2.5, 0.0) -> (-1.0, 0.0))
python -u nav_arena/nav_arena/scripts/verify_nav2.py

# Interactive Kit GUI + RViz2 visualization
python -u nav_arena/nav_arena/scripts/verify_nav2.py --viz kit --rviz

# Custom spawn and goal coordinates
python -u nav_arena/nav_arena/scripts/verify_nav2.py --spawn-x -2.5 --spawn-y 0.0 --goal-x 1.5 --goal-y 0.0
```

### Embodiment Verification
Validates a registered embodiment's differential kinematics, exercises `DifferentialDriveAction`, reads 2D LiDAR range arrays, and validates displacement. `--robot` selects any registered embodiment (`nova_carter` by default, or `dingo`); `--camera` additionally mounts the RGB-D camera and checks the RGB / metric-depth tensors:
```bash
source setup.env
python -u nav_arena/nav_arena/scripts/verify_embodiment.py
python -u nav_arena/nav_arena/scripts/verify_embodiment.py --robot dingo --camera
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

### PointNav Task & Metrics Verification
Instantiates `PointNavTask`, exercises goal tracking (`UniformPose2dCommandCfg`), drives the robot forward to verify goal reach termination, checks state restoration upon episode reset, and tests chassis-isolated collision detection:
```bash
source setup.env

# Headless task verification
python -u nav_arena/nav_arena/scripts/verify_task.py

# Clearpath Dingo with the RGB-D camera and goal-image rendering
python -u nav_arena/nav_arena/scripts/verify_task.py --robot dingo --camera

# Interactive GUI inspection
python -u nav_arena/nav_arena/scripts/verify_task.py --viz kit
```

`create_point_nav_env_cfg(robot_name=...)` configures the task for any registered embodiment: its articulation, action term, LiDAR/contact/camera sensors (on the embodiment's USD `body_link`), and any USD stage patch. `enable_camera=True` mounts the RGB-D camera (rendered at 5 Hz by default); `enable_goal_camera=True` adds a free-standing camera for `PointNavTask.render_goal_image(...)`, used by image-goal policies. Episodes end with `goal_reached`, `collision`, `tipped`, or `time_out` (`PointNavTask.get_terminal_cause()`).

### TF Tree Verification
Spawns the robot, launches `robot_state_publisher` using programmatic URDF, and asserts transform continuity (`map -> odom -> base_link -> chassis_link -> lidar_link`):
```bash
source setup.env
python -u nav_arena/nav_arena/scripts/verify_tf_tree.py
```

### Occupancy Map Verification
Generates and validates 2D occupancy map bounds and verifies YAML metadata:
```bash
source setup.env
python -u nav_arena/nav_arena/scripts/verify_occupancy_map.py --scene kujiale_0003
```

### ROS 2 Bridge Runner
Standalone execution orchestrator booting the simulation, establishing the OmniGraph ROS 2 bridge, and listening to `/cmd_vel`:
```bash
source setup.env

# Terminal 1: Launch simulation bridge
python -u nav_arena/nav_arena/scripts/run_ros2_nav.py

# Terminal 2: Visualize in RViz2
rviz2 -d nav_arena/nav_arena/config/nav_arena.rviz
```

---

## 4. Testing Framework Tiers

The test suite is organized into three distinct verification tiers:

| Tier | Directory | Description | Typical Runtime | Target Environment |
|---|---|---|---|---|
| **L1: Unit Tests** | [`tests/unit/`](file:///home/robopi/simulation/nav_arena/tests/unit/) | Fast, CPU-only algorithmic & component tests (kinematics, LaserScan math, logger, process management, scene resolution, URDF synthesis, embodiment registry, sensor configs, paths, in-process policy adapters and path follower). | ~5 seconds (129 tests) | Pure Python / CPU |
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

## 5. Architecture & Separation of Concerns

The codebase is organized into cleanly decoupled subsystems:

- **Core Simulation ([`nav_arena.core`](file:///home/robopi/simulation/nav_arena/nav_arena/core/app.py))**: Centralized SimulationApp lifecycle management ([`launch_simulation_app`](file:///home/robopi/simulation/nav_arena/nav_arena/core/app.py)), boot-time OmniGraph/ROS 2 extension flag injection, and livestreaming configuration.
- **Embodiments ([`nav_arena.embodiments`](file:///home/robopi/simulation/nav_arena/nav_arena/embodiments/))**: Robot physical properties, kinematic configurations (`DifferentialDriveAction`), pure kinematics math ([`diff_drive_ik`](file:///home/robopi/simulation/nav_arena/nav_arena/embodiments/kinematics.py)), programmatic sensor rigging (planar 360° LiDAR), and in-memory URDF synthesis ([`generate_robot_urdf`](file:///home/robopi/simulation/nav_arena/nav_arena/embodiments/urdf.py)). Completely decoupled from ROS 2 middleware dependencies.
- **Tasks ([`nav_arena.tasks`](file:///home/robopi/simulation/nav_arena/nav_arena/tasks/point_nav.py))**: RL and benchmark task definitions ([`PointNavTask`](file:///home/robopi/simulation/nav_arena/nav_arena/tasks/point_nav.py) extending Isaac Lab's `ManagerBasedRLEnv`), goal sampling, timeout handling, and chassis-isolated contact metrics.
- **Scenes ([`nav_arena.scenes`](file:///home/robopi/simulation/nav_arena/nav_arena/scenes/interior_agent.py))**: Non-destructive USD scene conditioning using composition delta layers (`subLayerPaths`), automatic doorway clearing, and static triangle BVH generation for InteriorAgent assets.
- **Methods ([`nav_arena.methods`](file:///home/robopi/simulation/nav_arena/nav_arena/methods/))**: Navigation baselines, split by how they execute. [`in_process/`](file:///home/robopi/simulation/nav_arena/nav_arena/methods/in_process/) holds policies run directly in the simulation process (the `InProcessPolicy` contract, a shared path follower, a policy registry, and adapters for the NavDP-family learned baselines: iPlanner, ViNT, NavDP, VIPlanner, X-NavDP). [`ros2/`](file:///home/robopi/simulation/nav_arena/nav_arena/methods/ros2/) holds external ROS 2 stacks, including the Nav2 bringup launch files and tuned parameters.
- **ROS 2 Bridges ([`nav_arena.ros2`](file:///home/robopi/simulation/nav_arena/nav_arena/ros2/))**: Zero-latency OmniGraph nodes for simulation clock (`/clock`), odometry (`/odom`), and TF (`map -> odom -> base_link`), coupled with asynchronous Python nodes ([`LaserScanPublisherNode`](file:///home/robopi/simulation/nav_arena/nav_arena/ros2/sensors.py), [`TaskStatePublisherNode`](file:///home/robopi/simulation/nav_arena/nav_arena/ros2/state_publisher.py)), action adapters ([`TwistActionAdapter`](file:///home/robopi/simulation/nav_arena/nav_arena/ros2/adapters/action_adapter.py)), and background executors ([`BackgroundRos2Executor`](file:///home/robopi/simulation/nav_arena/nav_arena/ros2/executor.py)).
- **Tools ([`nav_arena.tools`](file:///home/robopi/simulation/nav_arena/nav_arena/tools/map_generator.py))**: Standalone offline utilities, including programmatic 2D occupancy grid generation from USD collision geometry.
- **Utilities ([`nav_arena.utils`](file:///home/robopi/simulation/nav_arena/nav_arena/utils/))**: Shared cross-cutting infrastructure: unified structured logging ([`ArenaLogger`](file:///home/robopi/simulation/nav_arena/nav_arena/utils/logger.py)), robust subprocess lifecycle management ([`managed_process`](file:///home/robopi/simulation/nav_arena/nav_arena/utils/process.py)), workspace-relative path resolution ([`paths.py`](file:///home/robopi/simulation/nav_arena/nav_arena/utils/paths.py)), and simulation mock helpers ([`create_mock_env`](file:///home/robopi/simulation/nav_arena/nav_arena/utils/sim.py)).

---

## 6. Repository Layout

```text
nav_arena/
├── pyproject.toml                     # Package specification (editable pip install)
├── README.md                          # Project documentation and architecture guide
├── CONTRIBUTING.md                    # Developer guidelines, primitives, and invariants
├── requirements/                      # Pinned, --no-deps dependency lists for the learned baselines
├── docs/                              # Integration notes and decision logs
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
│   │   ├── sensors.py                 # Planar 2D LiDAR raycaster and RGB-D pinhole camera configuration
│   │   └── urdf.py                    # Programmatic URDF string synthesis
│   ├── methods/                       # Autonomy baselines and external stacks
│   │   ├── in_process/                # Policies run inside the simulation process (no ROS 2)
│   │   │   ├── base.py                # InProcessPolicy contract, PolicyObservation, Plan
│   │   │   ├── controller.py          # Frame transforms and shared lookahead path follower
│   │   │   ├── registry.py            # get_policy / list_policies / register_policy
│   │   │   └── navdp_adapter/         # iPlanner, ViNT, NavDP, VIPlanner, X-NavDP adapters over NavDP
│   │   └── ros2/
│   │       └── nav2/                  # Nav2 bringup launch scripts, parameters, URDF bridge
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
│   │   ├── paths.py                   # Workspace-relative asset/data/cache paths (NAV_ARENA_* overrides)
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

## 7. Development & Contributing

For guidelines on adding new components, architectural boundaries, developer primitives, and test standards, refer to [`CONTRIBUTING.md`](CONTRIBUTING.md):
- **Developer Primitives**: Lifecycle management ([`launch_simulation_app`](file:///home/robopi/simulation/nav_arena/nav_arena/core/app.py)), structured logging ([`ArenaLogger`](file:///home/robopi/simulation/nav_arena/nav_arena/utils/logger.py)), subprocess groups ([`managed_process`](file:///home/robopi/simulation/nav_arena/nav_arena/utils/process.py)), and background concurrency ([`BackgroundRos2Executor`](file:///home/robopi/simulation/nav_arena/nav_arena/ros2/executor.py)).
- **Physics & USD Invariants**: Quaternion conventions `(x, y, z, w)`, minimum spawn clearance, and non-destructive USD composition delta layers.
- **Testing Rules**: L1 CPU-only test isolation standards and pytest marker conventions.

