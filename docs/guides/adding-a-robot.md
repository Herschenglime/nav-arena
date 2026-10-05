# Guide: adding a robot embodiment (and validating it)

An **embodiment** couples everything nav_arena needs to know about one robot: its USD asset and articulation, the
action term that drives it, wheel geometry, the frames and mounting points for sensors, and its footprint. The robot
is then selectable everywhere with `--robot <name>` (`verify_embodiment.py`, `verify_task.py`, `verify_baseline.py`)
and via `create_point_nav_env_cfg(robot_name=...)`.

Existing examples to copy from: [`embodiments/nova_carter.py`](../../nav_arena/embodiments/nova_carter.py) (simple),
[`embodiments/dingo.py`](../../nav_arena/embodiments/dingo.py) (single rigid body, derived USD, camera), and
[`embodiments/kaya.py`](../../nav_arena/embodiments/kaya.py) (3-omni holonomic drive, dynamic USD wheel geometry, passive rollers).

Every embodiment takes the same command, the body-frame twist `[vx, vy, wz]` (forward m/s, left m/s, yaw rad/s). A
differential drive ignores `vy`; holonomic and legged robots use it. What differs per robot is how the twist becomes
joint targets (the action term), its limits, and its physics timing. Both differential and holonomic drives are implemented
end to end; adding quadruped robots is covered in [Adding a new drive type](#adding-a-new-drive-type).

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
    drive_type: str = "diff"          # "diff" | "holonomic" | "quadruped"
    articulation_cfg: ArticulationCfg = field(default_factory=create_my_robot_articulation_cfg)
    action_cfg: DifferentialDriveActionCfg = field(default_factory=lambda: ROBOT_ACTION_CFG)
    wheel_radius: float = ...
    wheel_base: float = ...
    max_lateral_speed: float = 0.0    # m/s; 0 = cannot strafe (differential drive)
    sim_dt: float = 0.01              # physics timestep; the control step is sim_dt * decimation (0.02 s)
    decimation: int = 2
    spawn_height: float = 0.25        # root height above the floor at spawn
    body_link: str = "..."            # the USD link carrying the chassis colliders
    contact_bodies: str | None = None # regex of bodies whose contacts count as collisions (default: body_link)
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

- **Timing and spawn height.** `sim_dt`, `decimation` and `spawn_height` belong to the embodiment: a learned locomotion
  policy only works at the timing it was trained with. The control step `sim_dt * decimation` is 0.02 s (50 Hz) for
  every robot so far; the follower and the benchmark session read it from the embodiment. Camera and viewport render
  rates are physical (5 Hz and about 33 Hz) and are converted to steps with `sim_dt`.
- **`contact_bodies`.** Contact sensing normally attaches to `body_link`. Legged robots list several bodies (base,
  hips, thighs, calves) and leave out the feet, which touch the floor all the time.

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

Register the embodiment in `embodiments/registry.py::register_default_embodiments`, using a lazy `"module:Class"`
string and the drive type: `register_embodiment("my_robot", "nav_arena.embodiments.my_robot:MyRobotEmbodimentCfg",
variants=(), drive_type="diff")`. The lazy string keeps robot names and variants available to the CLI without
importing torch or Isaac Lab (the CLI must stay free of those imports). Export the new names from
`embodiments/__init__.py`'s lazy export table. (A module may also register itself on import with
`register_embodiment("my_robot", MyRobotEmbodimentCfg)`; that keeps any variants already declared for the name.)
Instances are deep-copied by `get_embodiment`.

`nav_arena robots list` shows every registered robot with its drive type and variants. `--robot` accepts a robot name
(`kaya`, the default variant) or `<robot>.<variant>` (`kaya.native`); `run` and `sweep` reject unknown names before
starting anything and record the canonical name (`kaya.mast`) in results.

### Variants

When a robot has known configurations worth comparing (a camera on a virtual mast versus the real sensor pose, a
robot with and without its camera), declare them instead of adding flags:

```python
register_embodiment(
    "kaya",
    "nav_arena.embodiments.kaya:KayaEmbodimentCfg",
    variants=[
        EmbodimentVariant("mast", "RGB-D camera on a virtual mast, comparable to the other robots.",
                          {"camera_offset": (0.0, 0.0, 0.30)}),
        EmbodimentVariant("native", "The real RealSense pose.", {"camera_offset": (0.05, 0.0, 0.10)}),
    ],
    default_variant="mast",
    drive_type="holonomic",
)
```

- A variant is data only: it replaces fields of the base embodiment. Registration fails on an unknown field, so a
  typo cannot silently do nothing; `name` and `variant` cannot be overridden.
- The bare name (`kaya`) selects `default_variant`; any other variant needs the suffix (`kaya.native`).
- Sensor availability is a field too: `sensors=("lidar",)` marks a variant without a camera. Asking such a variant for
  the camera raises an error naming the variant, before the scene loads.
- A robot without variants keeps its plain name (`dingo`), so existing results are unaffected.

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
   Passing means:
   - the robot comes to rest after the spawn drop (its pose stops changing) and then holds still under a zero
     command (catches creeping wheels or jittering rollers);
   - each drive axis it supports (forward; strafe if `max_lateral_speed > 0`; in-place rotation) reaches 70 % of the
     commanded rate in the body frame with little motion on the other axes, and a rotation keeps a fixed turning
     centre (the root may circle it if its origin is off the wheel axis). The rules are in
     `nav_arena/utils/drive_check.py`;
   - for a holonomic robot, the action term's wheel speeds match Isaac Sim's `HolonomicController` built from the
     same USD;
   - the LiDAR ray caster returned hits; the RGB is `uint8`, 640x360 and non-black; the depth has more than 5 % valid
     pixels with a plausible nearest ground return (camera orientation and height are right).
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

## Adding a new drive type

The shared pieces are done: the command is the body twist `[vx, vy, wz]` everywhere (the action term, the in-process
runner, the ROS 2 `Twist` adapter and the verify scripts), `RobotEmbodimentCfg` carries `drive_type` and
`max_lateral_speed`, the embodiment owns its physics timing and spawn height, and the follower returns `(vx, vy, wz)`.
Policies are not affected: they output a body-frame path, and the follower turns the path into a command.

A new drive type needs:

1. **Kinematics** in `embodiments/kinematics.py`: pure-torch inverse (and forward) kinematics with unit tests,
   modeled on `diff_drive_ik` / `diff_drive_fk` (validate inputs, round-trip test IK then FK).
   - *Holonomic* (mecanum/omni): twist to per-wheel speeds from the wheel layout (`u_i = a_i . (v + w x r_i)`).
   - *Ackermann:* `(v, steering angle)` to rear-wheel speeds and front steering angles; not part of the current work.
2. **An action term and config** in `embodiments/actions.py` (an Isaac Lab `ActionTerm` plus `ActionTermCfg`), like
   `DifferentialDriveAction`: resolve the joints in `__init__` (raise on a wrong joint count), keep `action_dim == 3`,
   implement `process_actions` (scale/offset and clip to limits), `apply_actions` and `reset`. A legged robot instead
   uses Isaac Lab's `PreTrainedPolicyAction`, which feeds the twist to a trained locomotion policy.
3. **Embodiment fields:** `drive_type`, `max_lateral_speed`, and for a policy-driven robot the `sim_dt`, `decimation` and
   `spawn_height` the policy was trained with.
4. **Follower behaviour.** `follow_path` already strafes when `FollowerCfg.max_lateral_speed > 0` (the benchmark
   session passes `min(embodiment.max_lateral_speed, max_speed)`): it keeps turning toward the target so the forward
   camera faces the direction of travel, and uses sideways speed for the lateral offset. A drive that cannot turn in
   place (ackermann) needs its own follower.

Drive-specific validation: `verify_embodiment.py` already commands each axis separately (see step 2 of the
validation list); a new drive type adds its own axes or limits there, then a `verify_baseline` run.
