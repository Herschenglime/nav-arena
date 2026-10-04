# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Allow ``python -m nav_arena.cli``."""

import sys

from nav_arena.cli import main

if __name__ == "__main__":
    sys.exit(main())
