# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""L1 unit test configuration and fixtures with zero Omniverse dependencies."""

import pytest


@pytest.fixture
def mock_temp_interior_agent_dir(tmp_path):
    """Create a mock InteriorAgent dataset directory structure for scene resolution testing."""
    base_dir = tmp_path / "InteriorAgent"
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir
