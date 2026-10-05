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

- `/groundPlane/CollisionPlane` exists in the file but **outside** the default prim, so referencing the robot should not bring it in.
  Verify in the simulator (a stray plane would hold the robot up or flip it); add a derived USD only if it shows up.
- Wheels and rollers are separate rigid bodies that carry the floor's traction forces. The contact sensor must therefore stay on
  `base_link` only (chassis colliders), with `ground_contact_on_body=False`; a sensor on the wheel bodies would read the traction
  force as a collision.
- 33 DOF (3 driven + 30 passive): only the three axle joints get an actuator; roller joints keep the USD's free state.
- Mount height for `kaya.mast`: 0.30 m above the floor (same viewpoint as the other robots) = `camera_offset` z of 0.30 - 0.091 = 0.209 m
  above `base_link`. For `kaya.native`: about (0.06, 0, 0.07) above base_link, pitched 20 degrees down.
