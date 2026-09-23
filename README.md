# Nav Arena

A lightweight, modular multi-embodiment navigation simulation and evaluation framework built on native **NVIDIA Isaac Sim 6.1.0** and **Isaac Lab 3.0.0** on `aarch64` Linux (Ubuntu 24.04 / NVIDIA GB10).

---

## 1. Overview

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
├── nav_arena/                         # Core RL environments and scenes (pure Python)
│   ├── embodiments/                   # Robot kinematics, sensors, and ROS 2 bridges
│   │   ├── __init__.py                # Embodiment package exports
│   │   ├── actions.py                 # DifferentialDriveAction ActionTerm ([v, w] -> wheel speeds)
│   │   ├── nova_carter.py             # NVIDIA Nova Carter articulation and actuator config
│   │   ├── sensors.py                 # 2D planar LiDAR sensor factory (RayCasterCfg)
│   │   └── ros2_bridge.py             # ROS 2 OmniGraphs (Clock, Odometry, TF) and Twist receiver
│   ├── scenes/                        # Photorealistic scene loaders (e.g. InteriorAgent OpenUSD)
│   │   ├── __init__.py                # Scene package exports
│   │   └── interior_agent.py          # Dynamic InteriorAgent USD scene loader & configs
│   ├── tasks/                         # Navigation tasks, MDP terms, and evaluation metrics
│   │   ├── __init__.py                # Tasks package exports
│   │   └── point_nav.py               # PointNavTask & PointNavEnvCfg (MDP metrics, goals, contacts)
│   └── scripts/                       # Verification tools and standalone tests
│       ├── verify_embodiment.py       # Standalone Nova Carter embodiment verification script
│       ├── verify_scene.py            # InteriorAgent scene & robot embodiment verification script
│       └── verify_task.py             # PointNav task, goals, reset, and collision verification script
└── nav_arena_ros/                     # Colocated ROS 2 wrapper package (ament_python)
    ├── package.xml                    # ROS 2 package manifest
    ├── setup.py                       # ament_python build specification
    ├── launch/
    │   └── integration_test.launch.py # Lifecycle orchestration launch file
    └── nav_arena_ros/
        ├── ros2_policy_runner.py      # LifecycleNode wrapping PointNavTask in Isaac Lab
        └── dummy_ros2_policy.py       # Test P-controller policy node
```

---

## 3. Installation

Ensure your Isaac Lab virtual environment (`env_isaaclab`) and ROS 2 Jazzy workspace are configured:

```bash
source setup.env
# 1. Install core RL simulation package in editable mode
pip install -e nav_arena

# 2. Build colocated ROS 2 integration package
colcon build --symlink-install --base-paths nav_arena/nav_arena_ros
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
```

#### Visual Inspection (Interactive Viewport Window)
Launches the native Omniverse Kit GUI window on your active monitor (`DISPLAY`), frames the living room corridor and target green arrow goal marker with an external camera, and visually renders the complete verification sequence (goal reach, reset, and wall collision impact):

```bash
source setup.env
python -u nav_arena/nav_arena/scripts/verify_task.py --viz kit
```

### ROS 2 Integration & Testing
We provide an orchestrated ROS 2 launch file that launches the Isaac Lab `PointNavTask` environment as a managed **ROS 2 Lifecycle Node** alongside a dummy ROS 2 policy and RViz2. 
Using ROS 2 lifecycle state transitions (`OnStateTransition`), the test policy and RViz2 are only launched once Isaac Lab has fully configured the USD stage, compiled shaders, and reached the `ACTIVE` state.

```bash
source setup.env
ros2 launch nav_arena_ros integration_test.launch.py
```

---

## 5. Implementation Notes & Invariants

- **Quaternion Ordering**: Isaac Lab's `AssetBaseCfg.InitialStateCfg.rot` expects `(x, y, z, w)`. Identity rotation is `(0.0, 0.0, 0.0, 1.0)`.
- **Ground Clearance**: Mobile robots must spawn with ground clearance (e.g. `pos=(0.0, 0.0, 0.25)`) to avoid PhysX collision penetration depenetration impulses.
- **OmniGraph Extension Booting**: Scripts using ActionGraphs must inject `--enable omni.graph --enable omni.graph.action --enable isaacsim.ros2.bridge --enable isaacsim.ros2.nodes` into `sys.argv` before `AppLauncher` is instantiated.
- **Global Scene & Multi-Mesh Raycasting**: Indoor USD environments (like InteriorAgent) are loaded as a single global stage asset at `/World/Scene`. 2D planar LiDAR uses Isaac Lab's `MultiMeshRayCasterCfg` (`merge_prim_meshes=True`, `track_mesh_transforms=False`) to unify all room and obstacle sub-meshes for real-time GPU raycasting.
- **MDP Task & Metric Architecture**: `PointNavTask` inherits from Isaac Lab's `ManagerBasedRLEnv`. Goal poses are generated and tracked with `UniformPose2dCommandCfg` with visual arrow markers in the viewport. Collision metrics specifically bind a `ContactSensorCfg` to `{ENV_REGEX_NS}/Robot/chassis_link` so wheel-ground contacts do not trigger false-positive collisions while chassis impacts immediately register `illegal_contact`.
- **ConfigClass Import Convention**: Always import `configclass` as `from isaaclab.utils.configclass import configclass` to prevent namespace collisions where submodule imports overwrite the function handle.

### Known Constraints & Future Work
- **Single-Threaded ROS 2 / Isaac Lab Integration Constraint**: The main evaluation runner script interleaves `env.step()` synchronously with `rclpy.spin_once()`. This is an MVP design decision ensuring deterministic, lockstep execution and preventing Python weakref invalidation from Kit. 
  - **Future Work**: For high-throughput scaling or complex multi-agent parallel environments, this blocks the simulation loop during ROS 2 message processing (and vice-versa). Refactoring the ROS 2 spin loop into a background thread with thread-safe data queues will be required.
- **Configurable Action Adapter (Legged Extensibility)**: The runner script utilizes an abstract `ActionAdapter` pattern to translate ROS 2 commands into the `env.step()` action tensor. Currently implemented as a `TwistActionAdapter` for wheeled robots, it is designed to be easily extensible for `JointCommandAdapter` deployments on quadruped systems.
