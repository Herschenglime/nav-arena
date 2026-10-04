# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simulation application lifecycle management and boot-time extension injection."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import sys
from typing import Any, Generator


def _inject_boot_flags(
    enable_ros2: bool = True,
    enable_omap: bool = False,
    livestream: bool = True,
) -> None:
    """Inject required boot-time flags into sys.argv before AppLauncher boots."""
    flags: list[str] = []
    if enable_ros2:
        flags.extend([
            "--enable", "omni.graph",
            "--enable", "omni.graph.action",
            "--enable", "isaacsim.ros2.bridge",
            "--enable", "isaacsim.ros2.nodes",
        ])
    if enable_omap:
        flags.extend([
            "--enable", "isaacsim.asset.gen.omap",
            "--enable", "isaacsim.asset.gen.omap.ui",
        ])
    if livestream:
        flags.extend([
            "--/exts/omni.kit.livestream.app/primaryStream/allowDynamicResize=true",
            "--/app/livestream/allowResize=true",
        ])

    i = 0
    while i < len(flags):
        flag = flags[i]
        if flag == "--enable" and i + 1 < len(flags):
            ext_name = flags[i + 1]
            already_present = any(
                sys.argv[j] == "--enable" and j + 1 < len(sys.argv) and sys.argv[j + 1] == ext_name
                for j in range(len(sys.argv))
            )
            if not already_present:
                sys.argv.extend(["--enable", ext_name])
            i += 2
        else:
            if flag not in sys.argv:
                sys.argv.append(flag)
            i += 1


@contextmanager
def launch_simulation_app(
    args_cli: argparse.Namespace | dict | None = None,
    enable_ros2: bool = True,
    enable_omap: bool = False,
    livestream: bool = True,
    **app_launcher_kwargs: Any,
) -> Generator[Any, None, None]:
    """Context manager to configure boot flags, launch Isaac Lab SimulationApp, and manage lifecycle.

    Args:
        args_cli: Parsed command-line arguments or config dict for AppLauncher.
        enable_ros2: Whether to enable OmniGraph and ROS 2 bridge extensions at boot time.
        enable_omap: Whether to enable Isaac Sim 2D occupancy map extensions at boot time.
        livestream: Whether to enable dynamic viewport resizing flags and carb settings for livestreaming.
        **app_launcher_kwargs: Additional keyword arguments forwarded to AppLauncher.

    Yields:
        simulation_app: The running SimulationApp instance (`app_launcher.app`).
    """
    _inject_boot_flags(enable_ros2=enable_ros2, enable_omap=enable_omap, livestream=livestream)

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args_cli, **app_launcher_kwargs)
    simulation_app = app_launcher.app

    if livestream:
        try:
            import carb.settings

            carb_settings = carb.settings.get_settings()
            if carb_settings is not None:
                carb_settings.set("/app/livestream/allowResize", True)
                carb_settings.set("/exts/omni.kit.livestream.app/primaryStream/allowDynamicResize", True)
        except Exception:
            pass

    try:
        yield simulation_app
    finally:
        simulation_app.close()
