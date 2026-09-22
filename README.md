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
└── nav_arena/
    ├── embodiments/                   # Robot kinematics, sensors, and ROS 2 bridges
    │   ├── __init__.py                # Embodiment package exports
    │   ├── actions.py                 # DifferentialDriveAction ActionTerm ([v, w] -> wheel speeds)
    │   ├── nova_carter.py             # NVIDIA Nova Carter articulation and actuator config
    │   ├── sensors.py                 # 2D planar LiDAR sensor factory (RayCasterCfg)
    │   └── ros2_bridge.py             # ROS 2 OmniGraphs (Clock, Odometry, TF) and Twist receiver
    ├── scenes/                        # Photorealistic scene loaders (e.g. InteriorAgent OpenUSD)
    │   ├── __init__.py                # Scene package exports
    │   └── interior_agent.py          # Dynamic InteriorAgent USD scene loader & configs
    ├── tasks/                         # Navigation tasks, MDP terms, and evaluation metrics
    └── scripts/                       # Verification tools and evaluation runners
        ├── verify_embodiment.py       # Standalone Nova Carter embodiment verification script
        └── verify_scene.py            # InteriorAgent scene & robot embodiment verification script
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

### Interactive GUI Visualization (Embodiment)
Launches the native Omniverse Kit viewport on your active monitor (`DISPLAY`), tracks Nova Carter with an external camera, and continuously runs test maneuvers:

```bash
source setup.env
python -u nav_arena/nav_arena/scripts/verify_embodiment.py --viz kit --loop
```

*(Note: On the first launch on GB10 / aarch64, allow ~60–90 seconds for Vulkan shader compilation before the viewport window renders).*

---

## 5. Implementation Notes & Invariants

- **Quaternion Ordering**: Isaac Lab's `AssetBaseCfg.InitialStateCfg.rot` expects `(x, y, z, w)`. Identity rotation is `(0.0, 0.0, 0.0, 1.0)`.
- **Ground Clearance**: Mobile robots must spawn with ground clearance (e.g. `pos=(0.0, 0.0, 0.25)`) to avoid PhysX collision penetration depenetration impulses.
- **OmniGraph Extension Booting**: Scripts using ActionGraphs must inject `--enable omni.graph --enable omni.graph.action --enable isaacsim.ros2.bridge --enable isaacsim.ros2.nodes` into `sys.argv` before `AppLauncher` is instantiated.
- **Global Scene & Multi-Mesh Raycasting**: Indoor USD environments (like InteriorAgent) are loaded as a single global stage asset at `/World/Scene`. 2D planar LiDAR uses Isaac Lab's `MultiMeshRayCasterCfg` (`merge_prim_meshes=True`, `track_mesh_transforms=False`) to unify all room and obstacle sub-meshes for real-time GPU raycasting.
