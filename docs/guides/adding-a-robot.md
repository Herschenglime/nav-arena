# Guide: adding a robot embodiment (and validating it)

An **embodiment** couples everything nav_arena needs to know about one robot: its USD asset and articulation, the
action term that drives it, wheel geometry, the frames and mounting points for sensors, and its footprint. The robot
is then selectable everywhere with `--robot <name>` (`verify_embodiment.py`, `verify_task.py`, `verify_baseline.py`)
and via `create_point_nav_env_cfg(robot_name=...)`.

Existing examples to copy from: [`embodiments/nova_carter.py`](../../nav_arena/embodiments/nova_carter.py) (simple) and
[`embodiments/dingo.py`](../../nav_arena/embodiments/dingo.py) (single rigid body, derived USD, camera).

Today only **differential drive** is supported end to end; adding another drive type is covered in
[Adding a new drive type](#adding-a-new-drive-type-ackermann-holonomic) below.

## 1. Inspect the USD first

You need these facts before writing any code. Open the asset in Isaac Sim (or inspect it with `pxr` on the CPU) and
write down:

| Fact | Used for |
|---|---|
| Default prim (e.g. `/dingo`) and the prim paths of the rigid bodies | `body_link`, derived-asset overrides |
| Which link carries the chassis colliders | `body_link` (contact and LiDAR sensors attach here) |
| Wheel joint names (e.g. `left_wheel_joint`) | the action term |
| Wheel radius and track width (centre to centre) | `wheel_radius`, `wheel_base` |
| `metersPerUnit`, `upAxis` | must be right, or the robot is mis-scaled |
| Embedded junk: ground planes, wrong friction/materials | step 3 |
| Is there a part that touches the floor besides the wheels (caster)? | `ground_contact_on_body` |

A quick CPU inspection (a normal Python shell with the environment sourced is enough, no simulator needed):

```python
from pxr import Usd
stage = Usd.Stage.Open("path/to/robot.usd")
print(stage.GetDefaultPrim().GetPath(), stage.GetMetadata("metersPerUnit"), stage.GetMetadata("upAxis"))
for prim in stage.Traverse():
    print(prim.GetPath(), prim.GetTypeName(), prim.GetMetadata("apiSchemas"))
```

The Isaac Sim MCP can also answer questions about extension APIs while you are working out joint and sensor setup.

## 2. Write the embodiment file

Create `nav_arena/nav_arena/embodiments/<robot>.py`. Locate the asset with `nav_arena.utils.paths` (never a hardcoded
absolute path); the Dingo uses an environment override, `NAV_ARENA_DINGO_USD`, with a workspace-relative default.

```python
def create_my_robot_articulation_cfg() -> ArticulationCfg:
    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=get_robot_usd_path(),
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.25),
            rot=(0.0, 0.0, 0.0, 1.0),  # XYZW
            joint_pos={...},
            joint_vel={".*": 0.0},
        ),
        actuators={
            "wheels": ImplicitActuatorCfg(
                joint_names_expr=[...],
                effort_limit=...,
                velocity_limit_sim=...,
                stiffness=0.0,
                damping=1.0,
            )
        },
    )

ROBOT_ACTION_CFG = DifferentialDriveActionCfg(
    asset_name="robot",
    left_wheel_joint_name=...,
    right_wheel_joint_name=...,
    wheel_radius=...,
    wheel_base=...,
    max_linear_speed=...,
    max_angular_speed=...,
)

@dataclass
class MyRobotEmbodimentCfg(RobotEmbodimentCfg):
    name: str = "my_robot"
    articulation_cfg: ArticulationCfg = field(default_factory=create_my_robot_articulation_cfg)
    action_cfg: DifferentialDriveActionCfg = field(default_factory=lambda: ROBOT_ACTION_CFG)
    wheel_radius: float = ...
    wheel_base: float = ...
    body_link: str = "..."            # the USD link carrying the chassis colliders
    ground_contact_on_body: bool = False
    lidar_offset: tuple = (0.0, 0.0, 0.30)
    camera_offset: tuple = (0.0, 0.0, 0.30)
    footprint: tuple = ((x, y), ...)  # polygon in base_link, metres
    robot_radius: float = ...
    robot_height: float = ...
```

(Use the real file for exact syntax; the sketch above is the shape, not paste-ready code.) Things that matter:

- **Spawn clearance.** Never spawn at `z = 0`; use about `0.25` to avoid violent depenetration on step 0.
- **Quaternions are `(x, y, z, w)`.** Identity is `(0, 0, 0, 1)`.
- **`body_link`.** The USD link (relative to the robot root) that carries the chassis colliders. It is distinct from
  `base_frame`/`chassis_frame`, which are TF frame names synthesized independently of the USD hierarchy. If the
  asset has a single rigid body (the Dingo), every sensor attaches to that one link.
- **`ground_contact_on_body`.** Set `True` when the body link also carries a part that touches the floor (a caster
  sphere). Its resting support force would otherwise look like a collision at t=0, so collision detection then uses
  lateral (horizontal) force only. (The Dingo's resting `base_link` force is about 17.7 N.)

## 3. Fix the asset if it needs fixing

Prefer a **derived USD**: a tiny layer that sublayers the untouched upstream file and overrides only what is wrong,
generated into `cache/assets/` on first use and rebuilt when the upstream file changes. See
[`embodiments/assets.py`](../../nav_arena/embodiments/assets.py) and the Dingo for the pattern (deactivate an embedded
ground plane; set a caster material's `physxMaterial:frictionCombineMode` to `min`). Rules:

- Carry `metersPerUnit` and `upAxis` into the derived layer (sublayer metadata does not propagate).
- Fail loudly when an expected prim is missing, so an upstream layout change cannot silently produce an unfixed robot.
- Author PhysX attributes as plain USD (`apiSchemas` metadata + the attribute); `pxr.PhysxSchema` only exists in Kit.

Fall back to `stage_patch_fn` only for fixes that cannot be static USD. It runs as an Isaac Lab `prestartup` event and
forces `scene.replicate_physics=False`; the README's "Robot Assets and USD Fixes" explains why.

## 4. Attach sensors

Sensor mount geometry lives on the embodiment; the factories in
[`embodiments/sensors.py`](../../nav_arena/embodiments/sensors.py) turn it into Isaac Lab configs.

- **2D LiDAR:** `create_2d_lidar_cfg(...)` from `lidar_offset` (a ray caster on `body_link`).
- **RGB-D camera:** `create_embodiment_camera_cfg(embodiment)` uses `camera_offset` and `camera_rot` with
  `create_rgbd_camera_cfg`: 640x360, focal length 1.4, aperture 1.88, clipping 0.05 to 20 m, `distance_to_image_plane`
  depth (metric z-depth). The same optics the learned baselines were trained with, so changing them changes policy
  behavior. Intrinsics come from `pinhole_intrinsics` / `task.get_camera_intrinsics()`.
- **Orientation:** cameras use `convention="world"` (+X forward, +Z up) with the XYZW quaternion `(0, 0, 0, 1)`; do not
  copy a WXYZ quaternion from older Isaac Lab code.
- **Mount height** (`camera_offset[2]`) is the main thing to tune per robot; the camera must see the floor ahead.
  `verify_embodiment --camera` checks the nearest ground return is plausible for the mount height.
- A free-standing goal camera (`create_goal_camera_cfg`) renders goal images for image-goal policies.

## 5. Register it

Add the import and registration in `embodiments/registry.py::register_default_embodiments`, and export the new names
from `embodiments/__init__.py`. (A file can also self-register with `register_embodiment("my_robot",
MyRobotEmbodimentCfg)` at module bottom; the Dingo does.) Instances are deep-copied by `get_embodiment`.

## 6. Validate, in this order

1. **Unit tests (CPU, seconds).** Add cases to `tests/unit/test_embodiment_registry.py` (fields, registration) and
   `tests/unit/test_sensor_configs.py` (sensor prim paths and offsets), and for a derived asset a test like
   `tests/unit/test_embodiment_assets.py`.
   ```bash
   source setup.env
   pytest -c nav_arena/pyproject.toml nav_arena/tests/unit -q
   ```
2. **Embodiment check (simulator).** Drives the robot, reads LiDAR, and with `--camera` checks the sensors.
   ```bash
   python -u nav_arena/nav_arena/scripts/verify_embodiment.py --robot my_robot --camera
   ```
   Passing means: the robot moved forward more than 0.05 m under `DifferentialDriveAction` (the action term, wheel
   joints and kinematics line up); the LiDAR ray caster returned hits; the RGB is `uint8`, 640x360 and non-black; the
   depth has more than 5 % valid pixels with a plausible nearest ground return (camera orientation and height are
   right).
3. **Task check.** Goal tracking, reset, and collision detection with your `body_link` and contact settings.
   ```bash
   python -u nav_arena/nav_arena/scripts/verify_task.py --robot my_robot --camera
   ```
4. **Closed loop.** A real planner on a known-good route; compare with another robot on the same route.
   ```bash
   nav_arena run --method iplanner --robot my_robot --scene kujiale_0003 --route hall_straight
   # Or via developer script:
   python -u nav_arena/nav_arena/scripts/verify_baseline.py --method iplanner --robot my_robot --route hall_straight
   ```
5. **Optional:** an integration test modeled on `tests/integration/test_integration_dingo.py` (spawns in the scene,
   drives straight with small lateral drift, produces valid depth).

### Troubleshooting

| Symptom | Likely cause |
|---|---|
| Robot flips or explodes on step 0 | Spawned at `z = 0`, or wrong quaternion order (`(w, x, y, z)` instead of `(x, y, z, w)`) |
| Terminates as `collision` at step 0 | A floor-touching part on `body_link`: set `ground_contact_on_body=True` |
| Drives in an arc / drifts sideways | Wrong `wheel_base`/`wheel_radius`, a swapped left/right joint, or a dragging caster (friction combine mode) |
| Slides or wheels slip | Caster friction averaged with the floor: fix the material (derived USD) |
| Robot is the wrong size | Missing `metersPerUnit` in a derived layer |
| Camera image black / depth nearly all invalid | Wrong `camera_rot` or mount height, or too few render frames before reading |
| Policy stops short of the goal | A visual aid in the scene (e.g. the goal arrow) is visible to the depth camera |
| `RuntimeError: ... no prim` while building a derived asset | The upstream asset's layout changed; update the override |

## Adding a new drive type (ackermann, holonomic)

The whole stack currently assumes a two-element command `[v, omega]`:

| Where | What assumes it |
|---|---|
| `embodiments/actions.py` | `DifferentialDriveAction` has `action_dim == 2` and converts through `diff_drive_ik` |
| `embodiments/base.py` | `RobotEmbodimentCfg` only carries diff-drive geometry (`wheel_radius`, `wheel_base`, `max_angular_speed`) |
| `tasks/point_nav.py` | `ActionsCfg.robot_action` is bound from `embodiment.action_cfg` |
| `methods/in_process/controller.py` | `follow_path` returns `(v, omega)` and may turn in place |
| `methods/in_process/runner.py` | builds `torch.tensor([[v, w]])` for `task.step` |

Policies are **not** affected: they output a body-frame path, and the follower turns the path into a command. That
split is what makes a new drive tractable. A new drive type needs five pieces:

1. **Kinematics** in `embodiments/kinematics.py`: pure-torch inverse (and forward) kinematics with unit tests,
   modeled on `diff_drive_ik` / `diff_drive_fk` (validate inputs, round-trip test IK then FK).
   - *Ackermann:* `(v, steering angle)` to rear-wheel speeds and front steering angles (wheelbase, track width,
     steering limit); the turning radius is `wheelbase / tan(steer)`.
   - *Holonomic* (mecanum/omni): `(vx, vy, omega)` to per-wheel speeds from the wheel layout.
2. **An action term and config** in `embodiments/actions.py` (an Isaac Lab `ActionTerm` plus `ActionTermCfg`), like
   `DifferentialDriveAction`: resolve the joints in `__init__` (raise on a wrong joint count), set `action_dim`
   (2 for ackermann, 3 for holonomic), implement `process_actions` (scale/offset and clip to limits),
   `apply_actions` (write joint velocity targets, and for ackermann *position* targets for the steering joints), and
   `reset`.
3. **Embodiment fields.** Add a `drive_type` (`"diff"`, `"ackermann"`, `"holonomic"`) and the geometry the new drive
   needs (ackermann: wheelbase, track width, max steering angle; holonomic: wheel layout), keeping the diff-drive
   fields optional. Update the registry tests.
4. **A drive-aware follower.** `follow_path` must map a body-frame path into the command space and respect the
   drive's limits: holonomic can strafe and never needs to turn in place; ackermann cannot turn in place at all and
   has a minimum turning radius, so the follower must steer along an arc (and a stopped or blocked robot needs
   reverse or a K-turn, or the planner must be curvature-aware). Policies stay unchanged.
5. **Runner and task plumbing.** The action tensor width must come from the embodiment (a `command_dim`), not a
   hardcoded `[[v, w]]`; `task.step` and `PointNavTask` bind `embodiment.action_cfg` already.

A proposed refactor makes this slot in cleanly: a `DriveModel` per `drive_type` (command dimension, limits, and
`follow(path, goal_distance) -> command`), with `RobotEmbodimentCfg.drive` replacing the ad hoc diff-drive fields
over time. It is not built; implementing it is the first step of the ackermann/holonomic work.

Drive-specific validation (extend `verify_embodiment.py`): command each axis separately and check the result
(straight line; strafe for holonomic; in-place rotation for diff/holonomic; for ackermann, the measured arc radius
versus `wheelbase / tan(steer)`), then a `verify_baseline` run once the follower exists.
