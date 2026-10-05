# Learned visual-navigation baselines in nav_arena: results and future work

Branch `feat/navdp-baselines`. This document records what the integration delivered, what the baselines did in
closed loop, and what is left. This file is the durable summary; the per-decision working log that preceded it was
removed (it was a temporary review log) and remains in the git history (`docs/navdp_integration_log.md`).

Related: how-to guides in [`docs/guides/`](guides/) (adding a robot, choosing routes, running experiments) and the
design for a unified entry point in [`docs/design/unified-entry-point.md`](design/unified-entry-point.md).

## What was delivered

Five learned baselines (iPlanner, ViNT, NavDP, VIPlanner, X-NavDP) now run in nav_arena on a Clearpath Dingo with an
RGB-D camera in the InteriorAgent scenes, driven by one runner and one CLI. They were ported from a labmate's untracked
Isaac harness; upstream-file fixes live on the NavDP fork (`Herschenglime/NavDP`, branch `nav-arena`, pinned `8b9ee13`).

| Area | Result |
|---|---|
| Embodiments | Registry with Nova Carter and Clearpath Dingo (`body_link`, ground-contact flag). The Dingo's ground-plane and caster-friction fixes are baked into a derived USD layer (`embodiments/assets.py`) that sublayers the untouched upstream asset, so no runtime stage patch or `replicate_physics=False` is needed for it. |
| Sensing | RGB-D camera config (640x360, f=1.4, aperture 1.88, 0.05-20 m), identical to NavDP's own training config. |
| Methods layout | `methods/in_process/` (policy contract, follower, registry, runner, `navdp_adapter/`) and `methods/ros2/nav2/` (moved). |
| Policy contract | `step(PolicyObservation) -> Plan(path_body, stop, diagnostics)`; planner at 5 Hz (ViNT 3 Hz), shared lookahead follower at 50 Hz. |
| Task | `PointNavTask` generalized over robot and camera; lateral-force collision check for Dingo; `tipped` termination; goal-image capture. |
| Runner | `scripts/verify_baseline.py`: episode recorder (`settings.json`, `steps.jsonl`, `summary.json`, RGB/depth snapshots), terminal causes `goal_reached / collision / tipped / time_out / stalled / max_steps`. |
| Viewing | Follow camera, viewport-only goal pin, tolerance ring and plan overlay (`omni.ui.scene`, not visible to the robot's cameras). |
| Tools | `tools/route_map.py` draws routes or a recorded run (trajectory, stops, plans) over the occupancy map. |
| Tests | 195 unit tests; Dingo integration tests (straight drive, camera depth, iPlanner reaches goal). |
| Bug found | `launch_simulation_app` swallowed every failure (exit 0). Fixed; it had hidden a failing integration test. |

Fidelity against the labmate's recordings (real checkpoints, his frames): iPlanner, ViNT and VIPlanner reproduce his
paths exactly; NavDP and X-NavDP match at step 0 and drift afterwards, as expected for stochastic or stateful planners
replayed without history.

## Closed-loop results

Dingo, `kujiale_0003`, headless, 30 s budget, goal tolerance 0.4 m, follower 0.3 m/s, seed 0, goal marker hidden.
14 of 20 planned runs completed.

| Method | hall_straight | around_table | through_doorway | to_far_room |
|---|---|---|---|---|
| iPlanner | goal, 18.9 s | stalled (fear stop at plan 1) | stalled (fear stop at plan 1) | not run |
| NavDP | stalled 0.47 m short | collision | collision | not run |
| X-NavDP | goal, 19.0 s | collision | goal, 12.4 s | not run |
| VIPlanner | goal, 23.6 s | stalled (fear) | stalled (fear) | not run |
| ViNT (image goal) | collision | collision | not run | not run |

iPlanner also reaches the goal on the Nova Carter (`hall_straight`, 19.0 s). Inference per plan on the GB10: iPlanner
~9 ms, ViNT 28 ms, VIPlanner ~150 ms, NavDP 150-180 ms, X-NavDP 220-280 ms.

Reading the table:
- **Only `hall_straight` is a fair test for all methods.** `around_table` and `through_doorway` are hard (a table edge, a
  ~1 m door opening). Collisions were checked against the occupancy map and are real contacts (footprint 0-0.1 m from the
  obstacle), not detector artifacts.
- **iPlanner and VIPlanner stop permanently** when predicted fear reaches 0.7. Nothing recovers, because a stopped
  robot's view never changes. That is the labmate's design and was kept. Even on `hall_straight`, iPlanner's peak fear
  is 0.6996 against the 0.7 gate.
- **NavDP often ends 0.4-0.5 m short** of a 0.4 m tolerance (his own recorded run did the same).
- **The goal marker invalidated an earlier table.** Isaac Lab's goal arrow is scene geometry, so depth cameras saw it as
  an obstacle (erasing it from one frame dropped iPlanner fear from 0.83 to 0.07). It is now off by default and the
  viewport overlay replaces it.

## Known limitations

- GUI playback looks like the robot is jolting forward (see future work). Recorded data does not show it.
- One baseline per process (upstream modules share names such as `policy_agent`).
- Embodiments that use a runtime `stage_patch_fn` (none today) must disable physics replication; the Dingo does not need one.
- A one-off Kit startup SIGSEGV (exit 139) was seen once and did not reproduce.

## Future work

### Viewport playback jolting (deferred as a visual quirk)

Symptom: in the `--viz kit` GUI the robot appears to move in jerks rather than smoothly.

Established:
- **Not the control loop.** In a recorded iPlanner run the simulated speed is 0.296-0.301 m/s in every 0.2 s plan
  interval after a ~0.4 s ramp, with no stops or dips at plan boundaries. The follower runs every 20 ms and each plan
  only swaps the path it tracks. iPlanner returns a full path to the goal, not micro-goals.
- **Wall-clock step cost is very uneven** (400-step GUI run, follow camera + overlay): plain physics step ~21 ms
  (60% of steps), step with a display render ~64 ms (30%), plan step with camera render + inference ~86 ms (10%).
  Overall ~0.5x real time.

Tried, did not fix: pacing every step to one smoothed wall-clock duration, plus lowering the display render rate
from 15 to 10 Hz. Neither changed what the GUI shows, so both were reverted (commit `bc5fbbd`, reverted afterwards).

Hypotheses not yet tested:
1. **Viewport frame presentation:** the display render only happens on `task.sim.render()` calls and on
   `render_interval` steps. Kit may present frames independently of our step timing, which would explain why pacing did nothing. Test:
   render the viewport every step with the camera sensor disabled and see whether motion is smooth; then re-enable.
2. **Follow camera sampling:** the camera pose is updated every step but drawn only on some. Test: smooth the
   camera with a low-pass filter, or update it only on render steps.
3. **Physics interpolation:** poses shown at 10-15 Hz from a 50 Hz state can alias with 0.3 m/s motion. Test: set
   `EpisodeCfg.viewer_render_hz` to a divisor of 50 (10, 25) and compare; try the Isaac Lab/Kit rendering mode that
   decouples rendering from stepping.
4. **GPU contention between the sensor render and the viewport render** (same RTX context). Test: lower camera
   resolution, or render the viewport at the same steps as the sensor.
5. **Visualizer `--viz kit` rate limit:** check Isaac Lab 3.0's visualizer update rate settings
   (`isaac-sim-mcp` `search_isaac_sim_settings` for the viewport/renderer rate).
A quick diagnostic to add first: log wall-clock time per step type to the run output so any fix is measurable
(the timing script used for the numbers above was throwaway).

### Experiments and baselines
- **Repeat with several seeds** for the stochastic planners (NavDP, X-NavDP). The results table above is partial (14 of
  20 planned runs) and was judged sufficient to show the baselines work; the missing `to_far_room` and ViNT
  `through_doorway` runs can be added whenever a full matrix is wanted.
- **Stop recovery for iPlanner/VIPlanner (postponed):** rotate in place or back up when stopped, or a looser fear
  threshold. Currently a single fear spike is a permanent stop.
- **Goal tolerance default:** whether NavDP's 0.47 m finish counts (`--goal-dist 0.5` would).
- **Compare with Nav2** on the same routes and metrics now that the runner reports time-to-goal, path length and stops.
- **Image-goal baselines** (ViNT, NavDP image goal): goal images are currently rendered at the goal pose; add
  held-out goal views.

### Engineering
- Multi-process or isolated-subprocess policy loading to lift the one-baseline-per-process limit.
- Checkpoint tooling (`tools/navdp_checkpoints.py verify/convert`, planned but low priority).
