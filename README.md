# Nav Arena

A lightweight, modular multi-embodiment navigation simulation and evaluation framework built on native **NVIDIA Isaac Sim 6.1.0** and **Isaac Lab 3.0.0** on `aarch64` Linux (Ubuntu 24.04 / NVIDIA GB10).

---

## 1. Overview

### Project Goal
The primary goal of this project is scaffolding **Isaac Lab Arena** to work natively for autonomous navigation tasks. It acts as a robust, fully-featured bridge connecting high-fidelity physics and photorealistic simulation directly to standard ROS 2 navigation stacks (such as Nav2) without friction.

`nav_arena` is designed to benchmark multi-embodiment navigation policies and Nav2 stacks in photorealistic, physics-rich environments. The framework decouples robot kinematics, sensor rigging, and scene definitions through modular Python configurations, interfacing seamlessly with external autonomy stacks over standard ROS 2 topics.

### Key Capabilities
- **Modular Embodiment Factory**: Declarative robot configurations with kinematic abstractions (`DifferentialDriveAction` via Isaac Lab's `ActionManager`).
- **Python-Rigged Sensors**: Programmatically attaches sensors (e.g. 2D planar LiDAR via `RayCasterCfg` / `MultiMeshRayCasterCfg`) to mobile base links without modifying upstream USD assets.
- **USD Delta Layer Pipeline**: Non-destructive scene conditioning using USD composition (`subLayerPaths`). Deactivates obstacles (e.g., closed doors) ahead of stage load, ensuring sensor and collision parity from step 0.
- **2D Occupancy Grid Generation**: Automatic programmatic generation and disk caching of 2D binary and trinary occupancy grids directly from USD collision meshes at configurable heights and resolutions.
- **Native ROS 2 & Nav2 Integration**: Built-in OmniGraph bridges for simulation clock synchronization (`/clock`), odometry (`/odom`), transform frames (`/tf`), laser scans (`/scan`), and command velocity reception (`/cmd_vel`), paired with in-memory URDF synthesis and parameterized Nav2 bringup.
- **Multi-Modal Visual Execution**: Supports headless evaluation, remote WebRTC livestreaming, and interactive local Omniverse Kit GUI (`--viz kit`).

---

## 2. Repository Structure

```text
nav_arena/
├── pyproject.toml                     # Package specification (editable pip install)
├── README.md                          # Project documentation and usage guide
├── cache/                             # Generated runtime caches (git-ignored)
│   ├── maps/                          # Cached 2D occupancy grids (PNG + YAML)
│   └── scenes/                        # Conditioned USD delta layers (e.g. open_doors)
├── nav_arena/
│   ├── embodiments/                   # Robot kinematics, sensor factories, in-memory URDF
│   ├── mapping/                       # 2D occupancy map generation and bounds analysis
│   ├── methods/                       # Autonomy baselines and external stacks
│   │   └── nav2/                      # Nav2 bringup launch scripts, URDF publisher, parameters
│   ├── scenes/                        # USD loaders and delta layer conditioning (InteriorAgent)
│   ├── tasks/                         # PointNavTask, MDP metrics, goals, contacts
│   ├── ros2/                          # Core ROS 2 Integration Architecture
│   │   ├── adapters/                  # Action adapters (e.g., TwistActionAdapter)
│   │   ├── graph_builder.py           # OmniGraph builder for clock, TF, and odometry
│   │   └── state_publisher.py         # TaskStatePublisherNode (TF, /goal_pose, /goal_reached)
│   └── scripts/                       
│       ├── run_ros2_nav.py            # Main execution orchestrator for the ROS 2 bridge
│       ├── verify_embodiment.py       # Validates robot kinematics and 2D LiDAR ranges
│       ├── verify_scene.py            # Validates scene mesh loading and sensor raycasting
│       ├── verify_task.py             # Validates PointNavTask MDP metrics and resets
│       ├── verify_occupancy_map.py    # Generates and validates 2D occupancy maps
│       ├── verify_tf_tree.py          # Validates robot_state_publisher and complete TF tree
│       └── verify_nav2.py             # End-to-end Nav2 autonomous navigation benchmark
└── tests/
    └── ros2/                          # Automated PyTest integration suite
        ├── test_state_publisher.py    # Validates static TF and latched QoS
        └── test_closed_loop.py        # End-to-end simulated driving and ROS 2 bridging test
```

---

## 3. Installation

Ensure your Isaac Lab virtual environment (`env_isaaclab`) and ROS 2 Jazzy workspace are configured:

```bash
source setup.env
pip install -e nav_arena
```

---

## 4. Usage & Verification

All user commands are executed from the base workspace directory (`~/simulation`).

### Embodiment Verification
Runs 60 simulation steps, exercises the `DifferentialDriveAction` controller, reads 2D LiDAR range arrays, and validates displacement:

```bash
source setup.env
python -u nav_arena/nav_arena/scripts/verify_embodiment.py
```

### Scene & Navigation Verification (InteriorAgent)
Loads an InteriorAgent USD scene (defaults to `kujiale_0003`, or pass `--scene <name_or_path>`), settles Nova Carter on the floor geometry, drives forward, and verifies 360° LiDAR raycasting:

```bash
source setup.env
# Run headless verification on default scene
python -u nav_arena/nav_arena/scripts/verify_scene.py

# Run on a different scene (e.g. kujiale_0004)
python -u nav_arena/nav_arena/scripts/verify_scene.py --scene kujiale_0004

# Run with interactive GUI window
python -u nav_arena/nav_arena/scripts/verify_scene.py --viz kit --loop
```

### 2D Occupancy Grid Generation & Verification
Generates a 2D occupancy grid from USD collision geometry across an obstacle height slice (default: $0.01\,\text{m}$ to $0.60\,\text{m}$), saves standard ROS `map.yaml` and `map.png` artifacts to `cache/maps/<scene_id>/`, and verifies bounds:

```bash
source setup.env
# Generate or verify occupancy map for default scene (kujiale_0003)
python -u nav_arena/nav_arena/scripts/verify_occupancy_map.py

# Force regeneration with custom resolution (e.g. 0.025 m/pixel)
python -u nav_arena/nav_arena/scripts/verify_occupancy_map.py --scene kujiale_0003 --cell-size 0.025 --force
```

### TF Tree & Robot Description Verification
Spawns the embodiment in simulation, launches `robot_state_publisher` using the generated in-memory URDF, and asserts complete transform continuity across `/clock`, `/odom`, and `/tf` (`map -> odom -> base_link -> chassis_link -> lidar_link`):

```bash
source setup.env
python -u nav_arena/nav_arena/scripts/verify_tf_tree.py
```

### PointNav Task & Metrics Verification
Instantiates the full `ManagerBasedRLEnv`-derived `PointNavTask`, exercises goal generation and tracking (`UniformPose2dCommandCfg`), drives the robot forward to verify goal reach termination, checks state restoration upon episode reset, and tests wall impact contact detection (`ContactSensorCfg` / `illegal_contact`).

#### Headless Verification
```bash
source setup.env
# Run automated verification suite on default scene
python -u nav_arena/nav_arena/scripts/verify_task.py

# Run on a different scene (e.g. kujiale_0004)
python -u nav_arena/nav_arena/scripts/verify_task.py --scene kujiale_0004
```

#### Visual Inspection (Interactive Viewport Window)
```bash
source setup.env
python -u nav_arena/nav_arena/scripts/verify_task.py --viz kit
```

*(Note: On the first launch on GB10 / aarch64, allow ~60–90 seconds for Vulkan shader compilation before the viewport window renders).*

### Nav2 Autonomous Navigation Benchmark
Launches Isaac Lab alongside the full Nav2 navigation stack (Map Server, AMCL, Costmaps, NavfnPlanner, DWBLocalPlanner, BT Navigator), feeds simulated LiDAR to `/scan`, synchronizes `/clock` and TF, and autonomously navigates between distant rooms:

```bash
source setup.env
# Run headless benchmark on default long-distance route (West bedroom -> East bedroom)
python -u nav_arena/nav_arena/scripts/verify_nav2.py

# Run with interactive Omniverse Kit GUI and RViz2 visualization
python -u nav_arena/nav_arena/scripts/verify_nav2.py --viz kit --rviz

# Run with custom start and goal coordinates
python -u nav_arena/nav_arena/scripts/verify_nav2.py --spawn-x -6.42 --spawn-y 0.64 --goal-x 5.70 --goal-y -1.52
```

### ROS 2 Bridge & Closed-Loop Integration
The primary orchestrator boots the simulation, establishes the ROS 2 bridge (Clock, TF, Odom), and listens to `/cmd_vel` to drive the robot.

**Interactive Execution with RViz2:**
```bash
source setup.env
# Terminal 1: Launch the simulation bridge
python -u nav_arena/nav_arena/scripts/run_ros2_nav.py

# Terminal 2: Launch RViz to visualize relative map and odom TF poses, and the goal
rviz2 -d nav_arena/nav_arena/config/nav_arena.rviz
```

**Automated End-to-End Driving Test:**
```bash
source setup.env
python -u nav_arena/tests/ros2/test_closed_loop.py
```

---

## 5. Implementation Notes & Invariants

- **Quaternion Ordering**: Isaac Lab's `AssetBaseCfg.InitialStateCfg.rot` expects `(x, y, z, w)`. Identity rotation is `(0.0, 0.0, 0.0, 1.0)`. Passing `(1.0, 0.0, 0.0, 0.0)` inverts the asset 180° around the X-axis.
- **Ground Clearance**: Mobile robots must spawn with ground clearance (e.g. `pos=(0.0, 0.0, 0.25)`) to avoid PhysX collision penetration depenetration impulses on step 0.
- **OmniGraph Extension Booting**: Scripts using ActionGraphs must inject `--enable omni.graph --enable omni.graph.action --enable isaacsim.ros2.bridge --enable isaacsim.ros2.nodes` into `sys.argv` before `AppLauncher` is instantiated.
- **USD Delta Layers & Raycaster Static BVH**: Isaac Lab's `MultiMeshRayCaster` bakes its static triangle BVH (`merge_prim_meshes=True`) at environment initialization. Calling `prim.SetActive(False)` *after* the stage loads drops PhysX collision, but **does not rebuild the raycaster BVH**. To open doors or alter geometry for sensors, scenes must be conditioned *before* stage load using non-destructive USD delta layers (`subLayerPaths` via `get_preprocessed_usd` / `get_open_door_usd`).
- **ROS 2 Concurrency & Nav2 Lifecycle**: Coupling asynchronous DDS waitsets to a synchronous physics loop via single-step `spin_once(timeout_sec=0.0)` starves service futures and drops `/cmd_vel` callbacks under rendering load. ROS 2 callback dispatch must run on a dedicated worker thread (`threading.Thread(target=executor.spin)`), cleanly decoupling middleware processing from physics stepping.
- **Doorway Costmap Buffers**: In tight residential scenes with $0.65 - 0.80\,\text{m}$ doorways, large footprint padding ($0.05\,\text{m}$) leaves $< 1\,\text{cm}$ clearance margin to inscribed obstacles ($253$). Nav2 costmaps configure tight footprint padding ($0.01\,\text{m}$) and an inflation radius ($0.35\,\text{m}$) with decay $\alpha = 3.0 / 4.0$ to provide a clean zero-cost corridor through narrow passages.
- **Global Scene & Multi-Mesh Raycasting**: Indoor USD environments (like InteriorAgent) are loaded as a single global stage asset at `/World/Scene`. 2D planar LiDAR uses Isaac Lab's `MultiMeshRayCasterCfg` (`merge_prim_meshes=True`, `track_mesh_transforms=False`) to unify all room and obstacle sub-meshes for real-time GPU raycasting.
- **MDP Task & Metric Architecture**: `PointNavTask` inherits from Isaac Lab's `ManagerBasedRLEnv`. Goal poses are generated and tracked with `UniformPose2dCommandCfg` with visual arrow markers in the viewport. Collision metrics specifically bind a `ContactSensorCfg` to `{ENV_REGEX_NS}/Robot/chassis_link` so wheel-ground contacts do not trigger false-positive collisions while chassis impacts immediately register `illegal_contact`.
- **ConfigClass Import Convention**: Always import `configclass` as `from isaaclab.utils.configclass import configclass` to prevent namespace collisions where submodule imports overwrite the function handle.
- **Standard Coordinate Frames (REP-105)**: While Isaac Sim defaults its root to `world`, standard ROS 2 navigation packages strictly expect the `map -> odom -> base_link` tree. Our bridge dynamically queries the robot spawn and publishes a static `map -> odom` transform to bridge the two ecosystems seamlessly.
- **Latched Topic QoS**: Task state topics that publish once per episode (e.g., `/goal_pose` and `/goal_reached`) must use `TRANSIENT_LOCAL` durability with depth 1. This prevents late-joining test runners or metric recorders from missing the termination signals due to race conditions.
