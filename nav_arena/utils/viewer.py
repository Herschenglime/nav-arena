# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Viewport-only visual aids for watching a robot in the Kit GUI: a follow camera and a goal/plan overlay.

:class:`ThirdPersonView` is ported from the NavDP Isaac Sim integration (``isaac_viewer.py``). The camera is independent
of every sensor camera: it only drives the GUI viewport, so it never changes what a policy sees.

:class:`DebugOverlay` draws the goal and the planner's path as an ``omni.ui.scene`` layer composited over the viewport
widget. That is the point of using it: anything that exists in the scene, **or is drawn by the renderer**, is visible to
the robot's cameras. Isaac Lab's goal arrow is scene geometry (a depth planner treats it as an obstacle sitting on its
goal), and ``isaacsim.util.debug_draw`` was measured to leak into camera RGB (a ring and path drawn in front of the robot
were baked into its images). A UI layer is not part of any render product.

Omniverse modules are imported lazily, so this module can be imported before the app boots; construct the classes only
after it has.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np


class ThirdPersonView:
    """Chase camera behind and above the robot, pulled in when walls or furniture block the view.

    Requires PhysX scene queries (``SimulationCfg.enable_scene_query_support``, or
    ``create_point_nav_env_cfg(scene_queries=True)``) for the occlusion rays.
    """

    CAMERA_PATH = "/World/ThirdPersonCamera"

    def __init__(
        self,
        *,
        distance: float = 1.6,
        height: float = 1.2,
        ignore_prefix: str = "/World/envs/env_0/Robot/",
        smoothing_s: float = 0.15,
    ) -> None:
        """
        Args:
            distance: Camera distance behind the robot in meters.
            height: Camera height above the robot origin in meters.
            ignore_prefix: Prim path prefix of the robot; rays hitting it are ignored so the camera is never placed
                inside the robot's own chassis.
            smoothing_s: Time constant for smoothing the camera heading, in seconds.
        """
        import omni.physx
        from isaaclab.sim.utils.stage import get_current_stage
        from pxr import Gf, PhysicsSchemaTools, UsdGeom

        self._gf = Gf
        self._physics_schema_tools = PhysicsSchemaTools
        self.distance, self.height = distance, height
        self.ignore_prefix = ignore_prefix
        self.smoothing_s = smoothing_s
        self.yaw: float | None = None
        self._query = omni.physx.get_physx_scene_query_interface()
        camera = UsdGeom.Camera.Define(get_current_stage(), self.CAMERA_PATH)
        camera.CreateFocalLengthAttr(24.0)
        camera.CreateHorizontalApertureAttr(36.0)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.03, 1000.0))
        self._transform = camera.MakeMatrixXform()
        self.last_eye: np.ndarray | None = None
        self.last_target: np.ndarray | None = None

    def _visible_eye(self, target: np.ndarray, eye: np.ndarray) -> np.ndarray:
        """Pull the eye toward the target until nothing but the robot lies between them."""
        delta = eye - target
        distance = float(np.linalg.norm(delta))
        nearest = distance

        def hit_callback(hit) -> bool:
            nonlocal nearest
            collision = hit.collision
            if isinstance(collision, int):
                collision = self._physics_schema_tools.intToSdfPath(collision)
            if not str(collision).startswith(self.ignore_prefix):
                nearest = min(nearest, float(hit.distance))
            return True

        self._query.raycast_all(tuple(target), tuple(delta / distance), distance, hit_callback)
        return target + delta * max(0.15, nearest - 0.12) / distance if nearest < distance else eye

    def update(self, position: Sequence[float], yaw: float, dt: float = 0.02) -> None:
        """Move the camera behind the robot and make it the active viewport camera.

        Args:
            position: Robot world position (x, y, z).
            yaw: Robot heading in radians.
            dt: Time since the previous update, for heading smoothing.
        """
        from omni.kit.viewport.utility import get_active_viewport

        if self.yaw is None:
            self.yaw = yaw
        else:
            turn = math.atan2(math.sin(yaw - self.yaw), math.cos(yaw - self.yaw))
            self.yaw += (1 - math.exp(-dt / self.smoothing_s)) * turn
        forward = np.array([math.cos(self.yaw), math.sin(self.yaw), 0.0])
        position = np.asarray(position, dtype=float)
        target = position + forward * 0.15 + np.array([0.0, 0.0, 0.18])
        eye = position - forward * self.distance + np.array([0.0, 0.0, self.height])
        eye = self._visible_eye(target, eye)
        gf = self._gf
        pose = gf.Matrix4d().SetLookAt(gf.Vec3d(*eye), gf.Vec3d(*target), gf.Vec3d(0, 0, 1)).GetInverse()
        self._transform.Set(pose)
        self.last_eye, self.last_target = eye, target
        viewport = get_active_viewport()
        if viewport is not None and str(viewport.get_active_camera()) != self.CAMERA_PATH:
            viewport.set_active_camera(self.CAMERA_PATH)


Line = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float, float], float]
Point = tuple[tuple[float, float, float], tuple[float, float, float, float], float]


class SceneViewBackend:
    """Draws lines and points into an ``omni.ui.scene`` SceneView attached to the active viewport.

    Raises:
        RuntimeError: If there is no active viewport window (headless run) or the scene UI cannot be built.
    """

    FRAME_NAME = "nav_arena_overlay"

    def __init__(self) -> None:
        from omni.kit.viewport.utility import get_active_viewport_window
        from omni.ui import color as cl
        from omni.ui import scene as sc

        window = get_active_viewport_window()
        if window is None:
            raise RuntimeError("no active viewport window: the goal/plan overlay needs the Kit GUI (--viz kit)")
        self._sc, self._cl = sc, cl
        with window.get_frame(self.FRAME_NAME):
            self._scene_view = sc.SceneView()
        window.viewport_api.add_scene_view(self._scene_view)
        self._window = window

    def draw(self, lines: Sequence[Line], points: Sequence[Point]) -> None:
        """Replace everything previously drawn."""
        sc, cl = self._sc, self._cl
        self._scene_view.scene.clear()
        with self._scene_view.scene:
            for start, end, color, width in lines:
                sc.Line(list(start), list(end), color=cl(*color), thickness=width)
            if points:
                sc.Points(
                    [list(position) for position, _, _ in points],
                    sizes=[size for _, _, size in points],
                    colors=[cl(*color) for _, color, _ in points],
                )

    def clear(self) -> None:
        self._scene_view.scene.clear()


class DebugOverlay:
    """Draws the goal pin, goal-tolerance ring, and the policy's current path over the viewport.

    The path is cyan while the policy is driving and red while it is requesting a stop. The overlay fails soft: if the
    backend cannot draw (for example no GUI), it logs once and disables itself instead of aborting the run.
    """

    GOAL_COLOR = (0.1, 0.9, 0.2, 1.0)
    PATH_COLOR = (0.1, 0.8, 1.0, 1.0)
    STOP_COLOR = (1.0, 0.15, 0.1, 1.0)

    def __init__(
        self,
        goal_xy: Sequence[float],
        tolerance: float = 0.4,
        *,
        pin_height: float = 1.0,
        ring_segments: int = 32,
        backend: Any = None,
    ) -> None:
        """
        Args:
            goal_xy: Goal position in the world frame.
            tolerance: Goal tolerance in meters, drawn as a ring on the floor.
            pin_height: Height of the vertical goal pin in meters.
            ring_segments: Segments used to approximate the tolerance ring.
            backend: Object with ``draw(lines, points)`` and ``clear()`` (tests); defaults to the viewport scene layer.
        """
        self.goal_xy = (float(goal_xy[0]), float(goal_xy[1]))
        self.tolerance = float(tolerance)
        self.pin_height = float(pin_height)
        self.ring_segments = int(ring_segments)
        self.enabled = True
        try:
            self._backend = backend if backend is not None else SceneViewBackend()
        except Exception as exc:  # no GUI / UI extension missing: keep running without the overlay
            self._backend = None
            self._disable(f"could not create the viewport overlay: {exc}")

    @staticmethod
    def _log() -> Any:
        from nav_arena.utils import get_logger

        return get_logger("viewer")

    def _disable(self, reason: str) -> None:
        self.enabled = False
        self._log().warning(f"Goal/plan overlay disabled: {reason}")

    def build(self, path_xy: np.ndarray | Sequence[Sequence[float]], stop: bool, z: float) -> tuple[list[Line], list[Point]]:
        """Compute the primitives for the goal and a plan (pure geometry, no rendering)."""
        gx, gy = self.goal_xy
        lines: list[Line] = [((gx, gy, z), (gx, gy, z + self.pin_height), self.GOAL_COLOR, 4.0)]
        for i in range(self.ring_segments):
            a0 = 2 * math.pi * i / self.ring_segments
            a1 = 2 * math.pi * (i + 1) / self.ring_segments
            lines.append(
                (
                    (gx + self.tolerance * math.cos(a0), gy + self.tolerance * math.sin(a0), z),
                    (gx + self.tolerance * math.cos(a1), gy + self.tolerance * math.sin(a1), z),
                    self.GOAL_COLOR,
                    2.0,
                )
            )
        path = np.asarray(path_xy, dtype=float)
        path = path.reshape(0, 2) if path.size == 0 else path
        path_color = self.STOP_COLOR if stop else self.PATH_COLOR
        for a, b in zip(path[:-1], path[1:]):
            lines.append(((float(a[0]), float(a[1]), z), (float(b[0]), float(b[1]), z), path_color, 3.0))
        points: list[Point] = [((gx, gy, z + self.pin_height), self.GOAL_COLOR, 14.0)]
        if len(path):
            points.append(((float(path[-1][0]), float(path[-1][1]), z), path_color, 8.0))
        return lines, points

    def update(self, path_xy: np.ndarray | Sequence[Sequence[float]], stop: bool, z: float) -> None:
        """Redraw the goal and the given plan.

        Args:
            path_xy: Plan waypoints in the world frame, shape ``[T, >=2]`` (may be empty).
            stop: Whether the policy is requesting a stop (draws the path in red).
            z: Floor-ish height for the drawing, typically the robot's base height.
        """
        if not self.enabled:
            return
        try:
            self._backend.draw(*self.build(path_xy, stop, z))
        except Exception as exc:
            self._disable(f"drawing failed: {exc}")

    def clear(self) -> None:
        """Remove everything this overlay drew."""
        if self.enabled:
            try:
                self._backend.clear()
            except Exception as exc:
                self._disable(f"clearing failed: {exc}")
