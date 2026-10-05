# Guide: Running Experiments with `nav_arena`

The `nav_arena` CLI provides a unified interface for running single navigation episodes, orchestrating matrix batch sweeps across baselines and scenes, inspecting historical run artifacts, and validating simulation environment health.

All commands run from `/home/robopi/simulation`:

```bash
source setup.env
nav_arena <subcommand> [args...]
```

> [!NOTE]
> Isaac Lab runs headless by default. Pass `--gui` to open the interactive Omniverse Kit viewport window on your display.

---

## 1. Quick Command Reference

| Subcommand | Description | Example |
|---|---|---|
| `nav_arena run` | Run a single navigation episode | `nav_arena run --method iplanner --robot dingo --scene kujiale_0003 --route hall_straight` |
| `nav_arena sweep` | Execute a Cartesian matrix batch sweep | `nav_arena sweep sweeps/baselines_kujiale_0003.yaml` |
| `nav_arena runs list` | List batches or runs in a batch | `nav_arena runs list` or `nav_arena runs list --batch <batch_id>` |
| `nav_arena runs show` | Show metrics and diagnostics for a run/batch | `nav_arena runs show <run_id>` |
| `nav_arena runs compare` | Aggregate and compare metrics in a batch | `nav_arena runs compare <batch_id> --by method` |
| `nav_arena runs rerun` | Re-execute a historical run with same spec | `nav_arena runs rerun <run_id> [--gui]` |
| `nav_arena doctor` | Run diagnostic pre-flight environment checks | `nav_arena doctor` |
| `nav_arena routes` | List or inspect registered scene routes | `nav_arena routes list --scene kujiale_0003` |
| `nav_arena map` | Generate or inspect 2D occupancy grid maps | `nav_arena map show --scene kujiale_0003` |

For developer-level in-process testing or direct debugging without subprocess spawning, the developer shim `python -u nav_arena/nav_arena/scripts/verify_baseline.py` remains available.

---

## 2. Running a Single Episode (`nav_arena run`)

Execute a closed-loop navigation policy in an isolated worker subprocess:

```bash
# Run headless (default)
nav_arena run --method iplanner --robot dingo --scene kujiale_0003 --route hall_straight

# Run with interactive GUI viewport
nav_arena run --method navdp --robot dingo --scene kujiale_0003 --route around_table --gui

# Custom spawn/goal coordinates with policy parameter override
nav_arena run --method x_navdp --robot nova_carter --scene kujiale_0003 \
    --spawn -6.4 0.5 --goal -0.4 0.5 --goal-dist 0.5 --policy-arg fear_threshold=0.5
```

### Key Flags
- `--method`: Baseline method name (`iplanner`, `vint`, `navdp`, `viplanner`, `x_navdp`, or `nav2`).
- `--robot`: Robot embodiment (`dingo`, `nova_carter`).
- `--scene`: Scene ID (e.g. `kujiale_0003`).
- `--route`: Named route registered in the scene.
- `--spawn` & `--goal`: Manual `(x, y)` coordinates in meters (mutually exclusive with `--route`).
- `--spawn-yaw`: Initial robot yaw in radians.
- `--seed`: Random seed (default: 0).
- `--gui` / `--no-gui`: Enable or disable the interactive Omniverse Kit viewport window.
- `--follow-camera` / `--no-follow-camera`: Third-person camera that tracks the robot (default: on with `--gui`, off headless).
- `--goal-overlay` / `--no-goal-overlay`: Viewport-only goal pin, tolerance ring and the policy's current path (default: on with `--gui`, off headless). It is a UI layer, so the policy's cameras never see it.
- `--max-steps`: Episode step budget at 50 Hz (default: 1500 = 30 s).
- `--goal-dist`: Goal tolerance radius in meters (default: 0.4 m).
- `--max-speed`: Maximum linear velocity limit (default: 0.3 m/s).
- `--stall-timeout`: Seconds without forward movement before marking episode stalled (default: 10.0 s).
- `--policy-arg key=val`: Pass policy-specific hyperparameters.
- `--spec <path>`: Load base configuration from a YAML or JSON `RunSpec` file, allowing CLI flags to override fields.
- `--output <dir>`: Custom destination directory for run artifacts.

---

## 3. Batch Sweeps (`nav_arena sweep`)

Batch sweeps orchestrate multi-run evaluations across a Cartesian product of methods, robots, routes, and seeds. Each episode runs in an isolated subprocess under a strict watchdog timeout, with live atomic updates to `manifest.json` and `results.csv`.

### Defining a Sweep Spec (`sweeps/baselines_kujiale_0003.yaml`)

```yaml
name: baselines_kujiale_0003
scene: kujiale_0003
robots: [dingo]
methods: [iplanner, vint, navdp, viplanner, x_navdp]
routes: [hall_straight, around_table]
seeds: [0]

options:
  max_steps: 1500
  goal_dist: 0.4
  max_speed: 0.3
  stall_timeout: 10.0
  gui: false

timeout_s: 300
```

### Running and Resuming Sweeps

```bash
# Launch a full sweep
nav_arena sweep sweeps/baselines_kujiale_0003.yaml

# Filter sweep execution via CLI flags
nav_arena sweep sweeps/baselines_kujiale_0003.yaml --methods iplanner navdp --seeds 0 1

# Resume an interrupted or partially completed sweep
nav_arena sweep sweeps/baselines_kujiale_0003.yaml --resume
```

When `--resume` is passed, `nav_arena` reads the existing `manifest.json`, identifies runs marked `done`, and skips them, resuming execution at the first uncompleted or failed run.

---

## 4. Run Tracking & Metric Comparisons (`nav_arena runs`)

All runs record structured metadata, metrics, step logs, and sensor snapshots under `cache/runs/`.

### Directory Structure

```text
cache/runs/
├── 20261004_180000_baselines_kujiale_0003/    # Batch sweep directory
│   ├── batch.yaml                             # Frozen sweep configuration & environment metadata
│   ├── manifest.json                          # Atomic state of all runs (queued, running, done, failed)
│   ├── results.csv                            # Master tabular results across all runs
│   └── 000_iplanner_dingo_hall_straight_s0/   # Individual run artifacts
│       ├── run_spec.json                      # Exact RunSpec evaluated
│       ├── worker.log                         # Captured stdout/stderr of simulator subprocess
│       ├── summary.json                       # Final episode metrics & terminal cause
│       ├── steps.jsonl                        # Step-by-step trajectory, telemetry, & diagnostics
│       └── rgb_0000.png, depth_m_0000.npy     # Sensor captures
```

### Inspecting Runs

```bash
# List all batches and single runs
nav_arena runs list

# List runs within a specific batch
nav_arena runs list --batch 20261004_180000_baselines_kujiale_0003

# Show detailed metrics and diagnostics for a specific run
nav_arena runs show 000_iplanner_dingo_hall_straight_s0

# Compare results across methods, routes, or robots in a batch
nav_arena runs compare 20261004_180000_baselines_kujiale_0003 --by method
nav_arena runs compare 20261004_180000_baselines_kujiale_0003 --by route

# Re-run a specific episode using its historical configuration
nav_arena runs rerun 000_iplanner_dingo_hall_straight_s0 --gui
```

### Comparison Output Example

`nav_arena runs compare` generates formatted comparison tables directly from `results.csv`:

```text
========================================================================================================================
Batch: 20261004_180000_baselines_kujiale_0003 (Grouped by method)
========================================================================================================================
Method          Runs  Success  Rate (%)   TTG (s) [mean±std]   Path (m) [mean±std]  Collisions  Mean Inf (ms)
------------------------------------------------------------------------------------------------------------------------
iplanner           2        2    100.0%       18.91 ± 0.00         4.43 ± 0.00               0           4.82
navdp              2        2    100.0%       21.50 ± 1.20         4.65 ± 0.15               0          12.30
vint               2        1     50.0%       24.10 ± 0.00         5.10 ± 0.00               1          18.45
viplanner          2        0      0.0%                N/A                 N/A               0           6.10
x_navdp            2        2    100.0%       19.40 ± 0.50         4.50 ± 0.10               0          22.10
========================================================================================================================
```

---

## 5. Exit Codes & Terminal Causes

### Exit Codes

| Exit Code | Meaning |
|---|---|
| `0` | Goal reached successfully |
| `2` | Episode completed without reaching goal (`collision`, `stalled`, `max_steps`, `tipped`) |
| `1` | Infrastructure failure, crash, bad arguments, or process timeout |

### Terminal Causes

| Terminal Cause | Meaning |
|---|---|
| `goal_reached` | Robot arrived within `--goal-dist` of the target |
| `collision` | Lateral contact force exceeded threshold (Dingo) or any contact (Carter) |
| `tipped` | Roll or pitch exceeded 0.6 rad |
| `stalled` | Robot moved less than 5 cm for `--stall-timeout` simulated seconds |
| `max_steps` | Step budget exhausted while robot was still moving |
| `time_out` | Task or process watchdog timeout expired |

---

## 6. Pre-flight Diagnostics (`nav_arena doctor`)

Before starting extensive sweeps, verify your local environment:

```bash
nav_arena doctor
```

The diagnostic tool checks:
- Python interpreter & virtual environment (`env_isaaclab`).
- GPU device availability and conflicting running simulator processes.
- NavDP baseline repository checkout status.
- Baseline model checkpoint weights and SHA-256 integrity.
- Scene occupancy map cache presence.

---

## 7. Reproducibility & Evaluation Guidelines

1. **Keep Seeds Constant for Comparisons**: Learned planners (NavDP, X-NavDP) have stochastic sampling components; compare identical seeds across configurations.
2. **Standardize Limits**: When benchmarking methods against one another, ensure identical `--goal-dist` (default 0.4 m) and `--max-speed` (default 0.3 m/s).
3. **No Camera Scene Pollution**: The simulation overlay is rendered into an `omni.ui.scene` layer rather than physical USD scene geometry to ensure sensors do not view visual markers as physical obstacles.
4. **Historical Results**: See [`docs/baseline_integration_results.md`](../baseline_integration_results.md) for benchmark baselines, qualitative findings, and performance analysis.
