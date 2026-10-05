# Kaya USD inspection notes

Asset: `{ISAAC_NUCLEUS_DIR}/Robots/NVIDIA/Kaya/kaya.usd` (Isaac Sim 6.0 asset root). Inspected 2026-10-04 on a local mirror with
`pxr` (no simulator). Numbers are in the `/kaya/base_link` frame unless noted: +x forward, +y left, +z up.

## Structure

| Fact | Value |
|---|---|
| Default prim | `/kaya` (carries `PhysicsArticulationRootAPI`, self-collisions off) |
| `metersPerUnit`, `upAxis` | `1.0`, `Z` |
| Root / chassis body | `/kaya/base_link` (mass 0.714 kg, COM `(0.03, 0, 0)`), origin 0.091 m above the floor when resting |
| Driven joints | `/kaya/base_link/axle_{0,1,2}_joint`, revolute, joint axis `X`, `PhysicsDriveAPI:angular` (damping 174.5, stiffness 0) |
| Wheel bodies | `/kaya/axle_{0,1,2}` |
| Free roller joints | 30 revolute joints (`/kaya/axle_i/roller_i_j_k_joint`, 10 per wheel), each with a roller body `/kaya/roller_i_j_k` and a **sphere collider** (30 sphere colliders in total). No drive: the rollers must stay passive |
| Total mass | 2.21 kg |
| Camera | `/kaya/base_link/visuals/Kaya_Body_Collapsed/realsense` (visual only, no camera prim) |
| Control frame | `/kaya/base_link/control_offset`: identity, so the command site is the `base_link` origin |

## Wheel geometry (authored in the USD, read it from there, do not hand-copy)

Each axle joint authors `isaacmecanumwheel:angle = 90.0` (omni wheel) and `isaacmecanumwheel:radius = 0.04` m, the same
attributes Isaac Sim's `HolonomicRobotUsdSetup` reads.

| Wheel | Position (x, y, z) | Axle direction in the base frame | Rolling direction (axle x up) |
|---|---|---|---|
| 0 | (-0.0980, 0.0006, -0.0505) | (-1, 0, 0) | (0, +1) |
| 1 | (0.0493, -0.0845, -0.0505) | (0.5, -0.866, 0) | (-0.866, -0.5) |
| 2 | (0.0495, 0.0857, -0.0505) | (0.5, 0.866, 0) | (0.866, -0.5) |

Three omni wheels 120 degrees apart, wheel centres about 0.098 m from the centre, 0.0405 m above the floor (wheel radius 0.04).
Positive wheel rotation about the joint axis drives the robot along the rolling direction.

## Sensors and size

- RealSense visual: lens around x = 0.03..0.07, z = 0.14..0.19 m above the floor, pitched down about 20 degrees (its local +Z is
  (0.94, 0, -0.34)). So the **native camera is about 0.16 m high**, not 0.10 m as the URDF suggested.
- Visual bounding box (world, robot at default pose): x in [-0.139, 0.127], y in [-0.143, 0.145], z up to 0.24 m. Footprint is about
  0.27 x 0.29 m, radius about 0.15 m, height about 0.24 m.
- The collision prims are `guide` purpose with odd unit scaling, so their world bounds computed with `UsdGeom.BBoxCache` are
  unreliable (they come out metres wide). Check collision behaviour in the simulator instead.

## Consequences for the embodiment

- `/groundPlane/CollisionPlane` exists in the file but **outside** the default prim, so referencing the robot does not bring it in
  (confirmed: Kaya rests at its expected height in the simulator; no derived USD needed).
- ~~The contact sensor must stay on `base_link`, because wheels would read traction as a collision.~~ **Superseded (2026-10-05):**
  Isaac Lab's contact sensor reports normal forces only, not friction. Measured over 120 steps of driving and strafing, all 34 bodies
  (chassis, 3 wheels, 30 rollers) read exactly 0 N horizontally, while the rollers carry the robot's weight vertically. So Kaya
  senses all bodies (`contact_bodies=".*"`) with `ground_contact_on_body=True` (horizontal forces only), which also catches a
  wheel or roller hitting a wall. Caveat: a sloped floor or a door sill tilts the floor normal and could read as a collision.
- 33 DOF (3 driven + 30 passive): only the three axle joints get an actuator; roller joints keep the USD's free state.
- Mount height for `kaya.mast`: 0.30 m above the floor (same viewpoint as the other robots) = `camera_offset` z of 0.30 - 0.091 = 0.209 m
  above `base_link`. For `kaya.native`: about (0.06, 0, 0.07) above base_link, pitched 20 degrees down.

## Behaviour measured in the simulator (2026-10-05)

- **Settling after spawn:** after the drop onto its wheels, Kaya shifts about 2 cm and turns about 2.4 degrees over the next
  ~3.5 s. Observed in the GUI: the wheels turn one after another (front, front-left, then slowly the rear). Unconfirmed
  hypothesis: the weight settles unevenly onto the rollers and the velocity drive (damping, no stiffness) resists speed but
  does not hold position, so loaded wheels turn slowly until the load evens out. Spawning lower (0.10 m instead of 0.15 m) did not shorten it. Afterwards it holds
  still to 0.1 mm. Instantaneous root velocity keeps chattering while the pose is still, so judge stillness on pose, not velocity.
- **Drive accuracy** (`verify_embodiment`, body frame): forward 96% of command, strafe 97% (3 mm forward, 0.012 rad yaw per 0.29 m),
  rotation 78% of command with a fixed turning centre (2 mm).
- **Kinematics:** wheel speeds match Isaac Sim's `HolonomicController` to 4.8e-7 rad/s. Isaac's controller commands the centre of
  mass (3 cm ahead of `base_link`); ours commands the `base_link` origin, the pose navigation tracks. The cross-check therefore
  builds Isaac's controller with its command site at `base_link/control_offset` (identity, so the `base_link` origin).
- **LiDAR** sits at the mast height, 0.30 m above the floor (`lidar_offset` z = 0.209 above `base_link`).
