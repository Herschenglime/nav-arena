# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared helpers for CLI subcommand modules."""

from __future__ import annotations

import argparse


class CliError(Exception):
    """A usage or spec error to report to the user (exit code 1) without a traceback."""


def print_subcommand_help(name: str) -> int:
    """Print the help text of subcommand ``name`` (used when no action is given) and return exit code 1."""
    from nav_arena.cli import create_parser  # deferred: the package imports the subcommand modules

    for action in create_parser()._actions:
        if isinstance(action, argparse._SubParsersAction):
            subparser = action.choices.get(name)
            if subparser is not None:
                subparser.print_help()
    return 1
