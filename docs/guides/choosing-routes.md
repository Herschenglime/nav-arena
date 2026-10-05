# Guide: choosing routes (map, valid start/goal pairs, running them)

A **route** is a named start/goal pair in a scene, in world metres (the same frame as the occupancy map). Good routes
are what make results comparable across methods and robots, so this guide covers how to get a map, how to pick points
that are actually valid, how to try them, and how to make them permanent.

All commands run from `/home/robopi/simulation` after `source setup.env`.

## 1. Get an occupancy map

Everything below needs a cached 2D occupancy map of the scene. Check whether one exists:

```bash
ls nav_arena/cache/maps/            # one directory per scene, e.g. kujiale_0003/cs0.05_z0.01-0.60/{map.png,map.yaml}
```

If your scene is missing, generate it with `nav_arena map generate` (or the underlying tool `python -u -m nav_arena.tools.map_generator`; headless is fine, it settles physics, raycasts a z-slice, and writes `map.yaml` + `map.png`):

```bash
nav_arena map generate --scene kujiale_0004
```

You can check whether a map is cached and view its dimensions with:

```bash
nav_arena map show --scene kujiale_0004
```

| Flag | Meaning |
|---|---|
| `--scene` | InteriorAgent scene ID (`kujiale_0003`, ...) or a USD path |
| `--cell-size` | metres per pixel (default 0.05) |
| `--z-min` / `--z-max` | height band of the raycast slice (default 0.01 to 0.6 m: what a low robot can hit) |
| `--bounds-prim` | prim whose bounds define the map extent, if the default is wrong |
| `--force` | regenerate even if cached |
| `--output-dir` / `--cache-root` | write somewhere else |

Maps are cached under `cache/maps/<scene>/cs<cell>_z<zmin>-<zmax>/`. The generator conditions InteriorAgent scenes with
the open-door layer by default (doorways are cleared), which is also what the tasks use. `route_map` prefers a
`<scene>_open_doors` directory when both exist (`--closed-doors` flips that, `--map-dir` pins an exact directory).

## 2. Look at the map

```bash
python -u -m nav_arena.tools.route_map --scene kujiale_0003
```

This writes `cache/routes/<scene>_routes.png`: the map with every registered route and its A* reference path. It is
CPU-only and takes seconds. Orientation matches ROS maps: **image row 0 is the maximum y**, x increases to the right
(a mirrored reading of this image once produced unsafe routes). The grid is drawn in world metres, so you can read
coordinates straight off it.

## 3. Choose valid start and goal points

Pick candidates by eye on the image, then let the tool check them. A candidate pair is checked and drawn with:

```bash
python -u -m nav_arena.tools.route_map --scene kujiale_0003 --spawn -3 0.9 --goal -6.3 -1.2
```

```
custom (-3.0, 0.9) -> (-6.3, -1.2): straight 3.9 m, clearance S=0.55 G=0.46 m; path 4.3 m (x1.11)
```

Reading that line: `straight` is the Euclidean distance; `clearance S/G` is the distance from the start/goal to the
nearest obstacle; `path` is the shortest obstacle-inflated A* path and `x1.11` its ratio to the straight line (how much
detouring the route needs). Rules of thumb:

| Check | Guidance |
|---|---|
| Start and goal clearance | at least **0.5 m** from any obstacle (the registered routes are tested for this). The example above has `G=0.46`, so move the goal a little |
| Clearance along the way | for open-field routes keep at least **0.8 m** along the whole segment; a Dingo is ~0.7 m long and its corner can clip a wall (an early `hall_straight` spawned 0.08 m from a wall face and stalled) |
| A path must exist | `route_map` prints `NO SAFE PATH` if the points are not connected once obstacles are inflated by the robot radius (`--robot-radius`, default 0.33 m); it also appends `CLEARANCE BELOW ROBOT RADIUS` when the start or goal is closer to an obstacle than that |
| Not trivially short | the goal tolerance is 0.4 m by default; stay a few metres apart |
| Interesting detour | `x1.0` is a straight run, `x1.4` forces a detour (the `around_table` route); pick a mix across a set of routes |
| Facing | spawn yaw defaults to facing the goal (all methods then start with the same view geometry), so a start whose goal direction points into a wall is a bad start |
| Doors | a doorway of about 1 m is hard for learned planners; label such routes honestly |

## 4. Try a pair without committing it

```bash
# Using the unified CLI
nav_arena run --method iplanner --scene kujiale_0003 --spawn -6.4 0.5 --goal -0.4 0.5

# Or via developer script
python -u nav_arena/nav_arena/scripts/verify_baseline.py --method iplanner \
    --spawn -6.4 0.5 --goal -0.4 0.5
```

`--spawn` and `--goal` must be given **together** (a custom start/goal is a pair); giving one alone, or combining them
with `--route`, is a usage error, not silently ignored. `--spawn-yaw RAD` overrides the heading. The run is recorded
under `cache/runs/<timestamp>_<method>_<robot>_custom_<hex4>/`.

## 5. Make it a registered route

Edit [`scenes/routes.py`](../../nav_arena/scenes/routes.py) and add a `Route` to the scene's entry in
`INTERIOR_AGENT_ROUTES`:

```python
Route(
    "to_pantry",
    (-1.2, -3.0),             # spawn_xy
    (2.4, -3.0),              # goal_xy
    "Cross the kitchen between two islands.",
    4.1,                      # reference_path_m: the "path" value route_map printed
),
```

Use the `path` length that `route_map` printed for `reference_path_m`. Then:

1. Draw it with the others: `python -u -m nav_arena.tools.route_map --scene <scene>`.
2. Inspect registered routes using the CLI:
   ```bash
   nav_arena routes list --scene <scene>
   nav_arena routes show to_pantry --scene <scene>
   ```
3. Run the route tests: `pytest -c nav_arena/pyproject.toml nav_arena/tests/unit/test_scene_routes.py -q`. They check
   that every route's endpoints have at least 0.5 m clearance and that reference paths are at least the straight-line
   distance. Note these tests read the `kujiale_0003` map and iterate that scene's routes only; for a new scene add the
   equivalent cases (and a map fixture for it) alongside.
4. Try it:
   ```bash
   nav_arena run --method iplanner --scene <scene> --route to_pantry
   ```

`get_route(name, scene)` raises a `KeyError` that lists the available routes, and `list_routes(scene)` lists names.
Routes exist only for scenes in `INTERIOR_AGENT_ROUTES` (today `kujiale_0003`); a scene without routes can still be used
with `--spawn/--goal`.

## 6. Inspect what a run actually did

```bash
python -u -m nav_arena.tools.route_map --scene kujiale_0003 --route around_table \
    --run nav_arena/cache/runs/<run_dir> --plans-every 5
```

This overlays the recorded trajectory, the start, end and goal, red crosses where the policy asked to stop, and every
5th planned path, on the same map. It is the fastest way to see why a method stopped, collided or ended short of the
goal.
