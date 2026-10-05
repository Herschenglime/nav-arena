# Embodiment expansion: pre-refactor baseline

Recorded 2026-10-04 on `feat/embodiment-expansion` at `7604fcd` (only docs added on top of `main`, so this is the
behaviour of `main`). Phase 0 must reproduce these results for `nova_carter` and `dingo`.

All sim runs were headless via `./agy_python.sh`.

## Unit tests

`pytest -c nav_arena/pyproject.toml nav_arena/tests/unit -q`: **441 passed**.

## `verify_embodiment --camera`

| Check | nova_carter | dingo |
|---|---|---|
| DifferentialDriveAction moves wheels | PASS, displacement 0.1586 m | PASS, displacement 0.1391 m |
| 2D LiDAR hits | PASS, shape (1, 360) | PASS, shape (1, 360) |
| Camera 640x360 | PASS | PASS |
| RGB uint8, non-black | PASS, max 39 | PASS, max 240 |
| Depth valid fraction | 1.00 (max valid 0.12 m) | 0.43 (max valid 18.69 m) |
| Nearest ground return plausible | **FAIL**: 0.061 m | PASS, 1.12 m |

**Known pre-existing failure, not caused by this work:** nova_carter's camera check fails identically on every run
(reproduced twice). Every depth pixel is valid but the maximum is 0.12 m and the RGB is dark, so the camera looks
at the robot's own chassis rather than the room. The nova_carter camera mount needs its own fix (separate from this
work). Phase 0 must not make it better or worse; it only has to fail the same way.

## `verify_task --camera`

Both robots: all checks PASS (goal termination, reset, collision termination).

| | nova_carter | dingo |
|---|---|---|
| Goal reached at step | 108 | 113 |
| Collision detected at step | 197 | 116 |

## `verify_baseline --method iplanner --route hall_straight` (scene kujiale_0003, seed 0)

The Dingo run was repeated and every non-timing field matched exactly, so the simulation and planner are
**deterministic** and the post-refactor comparison can demand an exact match.

| Field | dingo | nova_carter |
|---|---|---|
| terminal_cause | goal_reached | goal_reached |
| steps | 945 | 952 |
| sim_time_s | 18.9 | 19.04 |
| initial_goal_distance_m | 5.999942779541016 | 6.0 |
| final_goal_distance_m | 0.40079620480537415 | 0.4041581451892853 |
| path_length_m | 5.613246479114334 | 5.615011942538842 |
| plans | 95 | 96 |

Full summaries are in `embodiment-expansion-baseline/summary_<robot>.json`. The per-plan logs (`steps.jsonl`) are
in `cache/baselines/pre-embodiment-expansion/steps_<robot>.jsonl` (git-ignored; regenerate from `7604fcd` if lost).
Only `mean_inference_ms` and `wall_time_s` vary between runs and should be ignored when comparing.

## How to compare after the refactor

```bash
./agy_python.sh nav_arena/nav_arena/scripts/verify_baseline.py --method iplanner --robot dingo \
    --route hall_straight --output <dir>
```

Compare `<dir>/summary.json` against `summary_dingo.json` (all fields except the two timing ones), and
`<dir>/steps.jsonl` against the cached log. Repeat for `nova_carter`, then `verify_embodiment --camera` and
`verify_task --camera` for both.

## Phase 1: Kaya baseline comparison (`verify_baseline --method iplanner --route hall_straight`)

Kaya closed-loop navigation on `hall_straight` (scene `kujiale_0003`, seed 0) with learned iPlanner policy:

| Field | kaya | dingo | nova_carter |
|---|---|---|---|
| terminal_cause | goal_reached | goal_reached | goal_reached |
| steps | 977 | 945 | 952 |
| sim_time_s | 19.54 | 18.90 | 19.04 |
| initial_goal_distance_m | 6.00003 | 5.99994 | 6.00000 |
| final_goal_distance_m | 0.40067 | 0.40080 | 0.40416 |
| path_length_m | 5.60386 | 5.61325 | 5.61501 |
| plans | 98 | 95 | 96 |
| stop_requests | 0 | 0 | 0 |

Kaya tracks the hallway trajectory cleanly to the goal within 0.5 s and 0.01 m path length of Dingo and Nova Carter.

