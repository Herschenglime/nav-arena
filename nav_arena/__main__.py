# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Top-level module execution entry point (python -m nav_arena)."""

from __future__ import annotations

import sys

from nav_arena.cli import main

if __name__ == "__main__":
    sys.exit(main())
