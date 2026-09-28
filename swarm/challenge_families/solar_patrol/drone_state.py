# The MIT License (MIT)
# Copyright © 2026 Swarm

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""Drone state inputs (task 14): where the drone is and how it is doing every step, and the site map at the start.

Positions count from the dock, so the dock is at 0, 0, 0 wherever the seed stands it.

Stand-in: the exact position and velocity, the time left, a battery that never drains, and a site map holding
only the fence.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .contract import HORIZON_S, MAX_FENCE_POINTS, SITE_MAP_SLICES, STATE_SLICES, put
from .episode import SolarEpisode


def reset(env: Any, ep: SolarEpisode) -> None:
    """Nothing is held between steps yet."""
    ep.drone_state = None


def observe(env: Any, ep: SolarEpisode, state: np.ndarray) -> None:
    """Position and velocity from the dock, compass heading, height above take-off, time left and battery."""
    position = np.asarray(env.pos[0], dtype=float) - ep.dock_position
    heading = (90.0 - math.degrees(float(env.rpy[0, 2])) + 180.0) % 360.0 - 180.0
    put(state, STATE_SLICES, "position_m", position)
    put(state, STATE_SLICES, "heading_deg", heading)
    put(state, STATE_SLICES, "velocity_mps", np.asarray(env.vel[0], dtype=float))
    put(state, STATE_SLICES, "height_above_takeoff_m", position[2])
    put(state, STATE_SLICES, "time_left_s", max(0.0, HORIZON_S - ep.time_s))
    put(state, STATE_SLICES, "battery_pct", 100.0)


def site_map(env: Any, ep: SolarEpisode, site: np.ndarray) -> None:
    """The survey of the site in metres from the dock: the fence, the panel tables and the buildings."""
    fence = (ep.fence - ep.dock_position[:2])[:MAX_FENCE_POINTS]
    padded = np.zeros((MAX_FENCE_POINTS, 2))
    padded[:len(fence)] = fence
    put(site, SITE_MAP_SLICES, "fence_count", len(fence))
    put(site, SITE_MAP_SLICES, "fence_xy", padded.reshape(-1))
