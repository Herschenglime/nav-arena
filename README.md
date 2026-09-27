# Nav Arena

A lightweight, modular multi-embodiment navigation simulation and evaluation framework built on native **NVIDIA Isaac Sim 6.1.0** and **Isaac Lab 3.0.0** on `aarch64` Linux (Ubuntu 24.04 / NVIDIA GB10).

---

## 1. Overview

### Project Goal
The primary goal of this project is scaffolding **Isaac Lab Arena** to work natively for autonomous navigation tasks. It acts as a robust, fully-featured bridge to connect high-fidelity physics and photorealistic simulation directly to standard ROS 2 navigation stacks (such as Nav2) without friction.

`nav_arena` is designed to benchmark multi-embodiment navigation policies and Nav2 stacks in photorealistic, physics-rich environments. The framework decouples robot kinematics, sensor rigging, and scene definitions through modular Python configurations, interfacing seamlessly with external autonomy stacks over standard ROS 2 topics.

### Key Capabilities
- **Modular Embodiment Factory**: Declarative robot configurations with kinematic abstractions (`DifferentialDriveAction` via Isaac Lab's `ActionManager`).
- **Python-Rigged Sensors**: Programmatically attaches sensors (e.g. 2D planar LiDAR via `RayCasterCfg`) to mobile base links without modifying upstream USD assets.
- **Native ROS 2 Integration**: Built-in OmniGraph bridges for simulation clock synchronization (`/clock`), odometry (`/odom`), transform frames (`/tf`), and command velocity reception (`/cmd_vel`).
- **Multi-Modal Visual Execution**: Supports headless evaluation, remote WebRTC livestreaming, and interactive local Omniverse Kit GUI (`--viz kit`).

---

## 2. Repository Structure

```text
nav_arena/
├── pyproject.toml                     # Package specification (editable pip install)
├── README.md                          # Project documentation and usage guide
├── nav_arena/
│   ├── embodiments/                   # Robot kinematics and sensor factories (Nova Carter)
│   ├── scenes/                        # Photorealistic scene loaders (InteriorAgent)
│   ├── tasks/                         # PointNavTask, MDP metrics, goals, contacts
│   ├── ros2/                          # Core ROS 2 Integration Architecture
│   │   ├── adapters/                  # Action adapters (e.g., TwistActionAdapter)
│   │   ├── graph_builder.py           # OmniGraph builder for clock, TF, and odometry
│   │   └── state_publisher.py         # TaskStatePublisherNode (TF, /goal_pose, /goal_reached)
│   └── scripts/                       
│       ├── run_ros2_nav.py            # Main execution orchestrator for the ROS 2 bridge
│       └── verify_*.py                # Standalone verification tools (scene, task, embodiment)
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

All commands are executed from the base workspace directory (`~/simulation`).

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

### PointNav Task & Metrics Verification
Instantiates the full `ManagerBasedRLEnv`-derived `PointNavTask`, exercises goal generation and tracking (`UniformPose2dCommandCfg`), drives the robot forward to verify goal reach termination, checks state restoration upon episode reset, and tests wall impact contact detection (`ContactSensorCfg` / `illegal_contact`).

#### Headless Verification
Runs the automated test suite (goal tracking, forward locomotion, goal reach termination, episode reset, and wall collision impact):

```bash
source setup.env
# Run automated verification suite on default scene
python -u nav_arena/nav_arena/scripts/verify_task.py

# Run on a different scene (e.g. kujiale_0004)
python -u nav_arena/nav_arena/scripts/verify_task.py --scene kujiale_0004
```

#### Visual Inspection (Interactive Viewport Window)
Launches the native Omniverse Kit GUI window on your active monitor (`DISPLAY`), frames the living room corridor and target green arrow goal marker with an external camera, and visually renders the complete verification sequence (goal reach, reset, and wall collision impact):

```bash
source setup.env
python -u nav_arena/nav_arena/scripts/verify_task.py --viz kit
```

*(Note: On the first launch on GB10 / aarch64, allow ~60–90 seconds for Vulkan shader compilation before the viewport window renders).*

### Interactive GUI Visualization (Embodiment)
Launches the native Omniverse Kit viewport on your active monitor (`DISPLAY`), tracks Nova Carter with an external camera, and continuously runs test maneuvers:

```bash
source setup.env
python -u nav_arena/nav_arena/scripts/verify_embodiment.py --viz kit --loop
```

*(Note: On the first launch on GB10 / aarch64, allow ~60–90 seconds for Vulkan shader compilation before the viewport window renders).*

### ROS 2 Bridge & Closed-Loop Integration
The primary orchestrator boots the simulation, establishes the ROS 2 bridge (Clock, TF, Odom), and listens to `/cmd_vel` to drive the robot.

**Interactive Execution with RViz2:**
```bash
source setup.env
# Terminal 1: Launch the simulation bridge
python -u nav_arena/nav_arena/scripts/run_ros2_nav.py

# Terminal 2: Launch RViz to visualize the relative map and odom TF poses, and the location of the goal
rviz2 -d nav_arena/nav_arena/config/nav_arena.rviz
```

**Automated End-to-End Verification:**
To run the closed-loop driving test (verifying that a script can publish `/cmd_vel` and successfully reach the goal state):
```bash
source setup.env
python -u nav_arena/tests/ros2/test_closed_loop.py
```

---

## 5. Implementation Notes & Invariants

- **Quaternion Ordering**: Isaac Lab's `AssetBaseCfg.InitialStateCfg.rot` expects `(x, y, z, w)`. Identity rotation is `(0.0, 0.0, 0.0, 1.0)`.
- **Ground Clearance**: Mobile robots must spawn with ground clearance (e.g. `pos=(0.0, 0.0, 0.25)`) to avoid PhysX collision penetration depenetration impulses.
- **OmniGraph Extension Booting**: Scripts using ActionGraphs must inject `--enable omni.graph --enable omni.graph.action --enable isaacsim.ros2.bridge --enable isaacsim.ros2.nodes` into `sys.argv` before `AppLauncher` is instantiated.
- **Global Scene & Multi-Mesh Raycasting**: Indoor USD environments (like InteriorAgent) are loaded as a single global stage asset at `/World/Scene`. 2D planar LiDAR uses Isaac Lab's `MultiMeshRayCasterCfg` (`merge_prim_meshes=True`, `track_mesh_transforms=False`) to unify all room and obstacle sub-meshes for real-time GPU raycasting.
- **MDP Task & Metric Architecture**: `PointNavTask` inherits from Isaac Lab's `ManagerBasedRLEnv`. Goal poses are generated and tracked with `UniformPose2dCommandCfg` with visual arrow markers in the viewport. Collision metrics specifically bind a `ContactSensorCfg` to `{ENV_REGEX_NS}/Robot/chassis_link` so wheel-ground contacts do not trigger false-positive collisions while chassis impacts immediately register `illegal_contact`.
- **ConfigClass Import Convention**: Always import `configclass` as `from isaaclab.utils.configclass import configclass` to prevent namespace collisions where submodule imports overwrite the function handle.
- **Standard Coordinate Frames (REP-105)**: While Isaac Sim defaults its root to `world`, standard ROS 2 navigation packages strictly expect the `map -> odom -> base_link` tree. Our bridge dynamically queries the robot spawn and publishes a static `map -> odom` transform to bridge the two ecosystems seamlessly.
- **Latched Topic QoS**: Task state topics that publish once per episode (e.g., `/goal_pose` and `/goal_reached`) must use `TRANSIENT_LOCAL` durability with depth 1. This prevents late-joining test runners or metric recorders from missing the termination signals due to race conditions.
