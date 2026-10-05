# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Adapters exposing the NavDP-family learned baselines as in-process policies.

The network and agent implementations stay upstream in the NavDP checkout (see
:data:`nav_arena.utils.paths.NAVDP_ROOT`); this package adapts their inputs and outputs. Ported from the NavDP
Isaac Sim integration.
"""
