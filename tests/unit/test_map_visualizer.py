# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the map_visualizer tool."""

from pathlib import Path
import numpy as np
import pytest
from PIL import Image
import yaml

from nav_arena.tools.map_visualizer import load_occupancy_grid, plot_map_visualization, main


@pytest.fixture
def mock_map_dir(tmp_path: Path) -> Path:
    map_dir = tmp_path / "mock_map"
    map_dir.mkdir(parents=True)
    img = Image.fromarray(np.full((100, 100), 254, dtype=np.uint8))
    img.save(map_dir / "map.png")
    meta = {
        "image": "map.png",
        "resolution": 0.05,
        "origin": [-2.5, -2.5, 0.0],
        "occupied_thresh": 0.65,
        "free_thresh": 0.196,
    }
    (map_dir / "map.yaml").write_text(yaml.dump(meta))
    return map_dir


def test_load_occupancy_grid(mock_map_dir: Path):
    img_world, extent, res = load_occupancy_grid(mock_map_dir)
    assert img_world.shape == (100, 100)
    assert res == 0.05
    assert extent == [-2.5, 2.5, -2.5, 2.5]


def test_plot_map_visualization(mock_map_dir: Path, tmp_path: Path):
    out = tmp_path / "plot.png"
    result = plot_map_visualization(
        map_dir=mock_map_dir,
        title="Mock Map",
        spawn=(0.0, 0.0),
        goal=(1.0, 1.0),
        robot_radii={"TestRobot": 0.25},
        trajectories=[{"x": [0.0, 1.0], "y": [0.0, 1.0], "label": "Path"}],
        crop_bounds=(-1.0, 1.5, -1.0, 1.5),
        output_path=out,
        side_by_side=True,
    )
    assert result.is_file()
    assert result.stat().st_size > 0


def test_map_visualizer_cli(mock_map_dir: Path, tmp_path: Path):
    out = tmp_path / "cli_plot.png"
    code = main([
        "--map-dir", str(mock_map_dir),
        "--spawn", "0.0", "0.0",
        "--goal", "1.0", "1.0",
        "--radius", "0.3",
        "--trajectory", "0.0", "0.0", "1.0", "1.0",
        "--single-panel",
        "--output", str(out),
    ])
    assert code == 0
    assert out.is_file()
