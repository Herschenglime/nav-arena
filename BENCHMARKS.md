# Navigation Arena Performance Benchmarks

## System Environment
- **Platform**: NVIDIA DGX Spark (Grace-Blackwell GB10 / aarch64)
- **Memory**: 128 GB Unified LPDDR5X / Coherent Memory (NVLink-C2C)
- **OS**: Linux 6.8 (Ubuntu 24.04 / aarch64)
- **Isaac Sim / Isaac Lab**: Isaac Lab 3.0 / Isaac Sim 5.0 (Jazzy ROS 2)
- **Environment**: InteriorAgent Scene `kujiale_0003` with Nova Carter mobile embodiment

---

## Benchmark Log

### Benchmark Protocol:
- **Test Command**:
  ```bash
  source setup.env
  python -u nav_arena/nav_arena/scripts/run_ros2_nav.py --num-steps 50
  ```
- **Simulated Physics Time**: $50 \text{ steps} \times 0.02\text{s/step} = 1.0\text{s}$

| Run ID | Date & Time | Configuration & Optimizations Applied | Wall-Clock Time (50 steps) | Step Rate (steps/s) | Real-Time Factor (RTF) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Run 0 (Baseline)** | 2026-09-29 16:13 | Unoptimized baseline (`render_interval=2`, duplicate `simulation_app.update()`, 100 Hz Warp LiDAR, `.cpu().tolist()`) | **11.20 s** | **4.5 steps/s** | **0.089×** (~11.2× slower than real-time) | Completed |
| **Run 1a (Milestone 1 - 20 FPS)** | 2026-09-29 16:15 | Decouple `render_interval=5` (20 Hz rendering), eliminate redundant `simulation_app.update()` | **2.67 s** | **18.8 steps/s** | **0.375×** (**4.2× speedup**) | Completed |
| **Run 1b (Milestone 1 - 33 FPS)** | 2026-09-29 16:24 | Decouple `render_interval=3` (~33.3 Hz rendering), eliminate redundant `simulation_app.update()` | **2.45 s** | **20.4 steps/s** | **0.408×** (**4.6× speedup**) | Completed |
| **Run 2 (Milestone 2)** | 2026-09-29 16:28 | Warp 2D LiDAR 20 Hz rate-limiting (`update_period=0.05`) | **2.44 s** | **20.5 steps/s** | **0.410×** | Completed |
| **Run 3 (Milestone 3)** | 2026-09-29 16:30 | Unified memory zero-copy transfer & non-blocking action buffer | **2.42 s** | **20.7 steps/s** | **0.414×** | Completed |

---

## Detailed Run Notes

### Run 0: Baseline
- **Observed Characteristics**:
  - Total process execution: 28 seconds (17.3s warm-up/stage parsing + 11.2s for 50 simulation steps).
  - Both physics and rendering were constrained to 50 Hz control rate, with 2 full Kit frame updates per 20ms sim-step.
  - Warp LiDAR ran at 100 Hz against all meshes in `/World/Scene`.

### Run 1b: Milestone 1 (Stepping & Viewport Decoupling at ~33.3 FPS)
- **Observed Characteristics**:
  - `render_interval = 3` (~33.3 FPS) achieved smooth interactive camera navigation in the viewport.
  - 50 simulation steps completed in **2.45 seconds** (down from 11.20 seconds).
  - Stepping rate increased from **4.5 steps/s to 20.4 steps/s** (**4.6× speedup**).
  - Real-Time Factor improved from **0.089× to 0.41×**.
  - Elimination of the duplicate `simulation_app.update()` pass prevented redundant RTX frame renders per step.

### Run 2: Milestone 2 (Warp 2D LiDAR 20 Hz Throttling)
- **Observed Characteristics**:
  - `update_period = 0.05` applied to `MultiMeshRayCasterCfg`.
  - Warp CUDA raycasting runs at 20 Hz instead of 100 Hz, freeing GPU resources during intermediate physics steps.
  - 50 simulation steps completed in **2.44 seconds** (20.5 steps/s, 0.41× RTF).
  - Greater impact will be observed in closed-loop navigation where sensor publishing and consumers are active.

### Run 3: Milestone 3 (Unified Memory Zero-Copy & Action Buffer)
- **Observed Characteristics**:
  - Replaced CUDA tensor slicing in `cmd_vel` ROS subscription callback with CPU float buffer, populating preallocated GPU tensor without clones.
  - 50 simulation steps completed in **2.42 seconds** (**20.7 steps/s**, **0.414× RTF**).
  - Profiling reveals that the remaining step time is dominated by physics solver iterations and remaining rendering cycles.

