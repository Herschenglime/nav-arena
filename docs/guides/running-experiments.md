# Guide: running experiments today (and where this is heading)

There is no single `nav_arena` command yet. Experiments are run piecewise, one process per episode, with the scripts
below. A unified command, batch runs, run tracking, and in-place resets are designed in
[`docs/design/unified-entry-point.md`](../design/unified-entry-point.md) but not built.

All commands run from `/home/robopi/simulation`:

```bash
source setup.env
python -u nav_arena/nav_arena/scripts/<script.py> [args...]
```

Isaac Lab runs headless unless you pass `--viz kit`.

## Which script for which job

| Job | Command |
|---|---|
| Run one learned baseline on a route | `verify_baseline.py --method {iplanner,vint,navdp,viplanner,x_navdp} --robot dingo --route hall_straight` |
| Run Nav2 end to end | `verify_nav2.py` (separate ROS 2 stack, separate output) |
| Check a robot embodiment | `verify_embodiment.py --robot <name> [--camera]` |
| Check the PointNav task | `verify_task.py --robot <name> [--camera]` |
| Check scene loading / occupancy map / TF tree | `verify_scene.py`, `verify_occupancy_map.py`, `verify_tf_tree.py` |
| Generate a map | `python -u -m nav_arena.tools.map_generator --scene <id>` |
| Draw routes or a run on the map | `python -u -m nav_arena.tools.route_map ...` |

The `verify_*` scripts other than `verify_baseline.py` are engineering checks of one layer each; they are run by hand
and are not part of any planned CLI.

## Running a baseline

```bash
python -u nav_arena/nav_arena/scripts/verify_baseline.py --method iplanner --robot dingo --route hall_straight
python -u nav_arena/nav_arena/scripts/verify_baseline.py --method navdp --route around_table --viz kit
python -u nav_arena/nav_arena/scripts/verify_baseline.py --method x_navdp --robot nova_carter \
    --spawn -6.4 0.5 --goal -0.4 0.5 --goal-dist 0.5 --policy-arg fear_threshold=0.5
```

Useful flags: `--max-steps` (50 Hz; 1500 = 30 s), `--goal-dist` (tolerance, default 0.4 m), `--max-speed` (follower
limit, default 0.3 m/s), `--seed`, `--stall-timeout`, `--output`. One baseline per process: loop over methods in
your shell, one invocation each. Do not run two at once, and do not run a batch in the background while someone is
using the GUI on the same machine.

### Exit codes and terminal causes

| Exit code | Meaning |
|---|---|
| 0 | goal reached |
| 2 | episode ended without reaching the goal |
| 1 | error (bad arguments, crash, missing checkpoint, ...) |

| Terminal cause | Meaning |
|---|---|
| `goal_reached` | within `--goal-dist` of the goal |
| `collision` | lateral contact force above threshold (Dingo) or any contact (Carter) |
| `tipped` | roll or pitch above 0.6 rad |
| `stalled` | no movement beyond 5 cm for `--stall-timeout` simulated seconds (default 10), e.g. a planner that stopped permanently |
| `max_steps` | step budget exhausted while still moving |
| `time_out` | the task's own timeout |

## Run outputs

Each run writes `cache/runs/<timestamp>_<method>_<robot>_<route>/` (override with `--output` or `NAV_ARENA_RUNS_DIR`;
a directory that already exists is refused, so two runs in the same second need distinct `--output`s):

| File | Contents |
|---|---|
| `settings.json` | policy and follower settings, goal, step timing |
| `steps.jsonl` | one row per plan: sim time, position, yaw, goal distance, stop flag, diagnostics (e.g. fear), inference time, planned path |
| `summary.json` | the episode result: terminal cause, success, steps, sim time, time to goal, initial/final goal distance, path length, plans, stop requests, mean inference ms, wall time |
| `rgb_*.png`, `depth_m_*.npy` | periodic sensor snapshots (what the policy actually saw) |

Reading a result: `summary.json` first (`terminal_cause`, `final_goal_distance_m`); then
`route_map --run <dir>` to see where it happened; then `steps.jsonl` (`diagnostics`, `stop`) and the snapshots to see
why. Typical reasons: iPlanner/VIPlanner stopping permanently on a fear spike (`stalled`), NavDP finishing 0.4-0.5 m
short of a 0.4 m tolerance (`max_steps`, goal within reach at `--goal-dist 0.5`), a real contact with a table edge or
door frame (`collision`).

## Reproducibility notes

- Policies are seeded (`--seed`, default 0); stochastic planners (NavDP, X-NavDP) still vary with simulation timing,
  so compare several seeds before trusting a difference.
- Compare methods on the **same** scene, route, robot, tolerance and speed limit.
- The goal arrow is hidden from the cameras on purpose; do not enable `--show-goal-marker` for evaluation.
- Results so far and what they mean: [`docs/baseline_integration_results.md`](../baseline_integration_results.md).

## A small sweep by hand

```bash
for method in iplanner navdp x_navdp; do
  for route in hall_straight around_table; do
    python -u nav_arena/nav_arena/scripts/verify_baseline.py --method "$method" --route "$route"
  done
done
```

Each invocation boots the simulator and loads the policy from scratch, which dominates the runtime of short episodes;
that cost, the manual bookkeeping, and the lack of a combined results table are what the unified entry point is meant
to remove.
