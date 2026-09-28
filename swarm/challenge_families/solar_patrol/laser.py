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

"""Laser point (task 12): the rangefinder in the camera, one distance at the crosshair, once a second.

Stand-in: the laser never gets a reading and says so.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .contract import LASER_STATUSES, STATE_SLICES, put
from .episode import SolarEpisode


def reset(env: Any, ep: SolarEpisode) -> None:
    """No reading taken yet."""
    ep.laser = None


def update(env: Any, ep: SolarEpisode) -> None:
    """Take a reading at the laser's own rate."""


def observe(env: Any, ep: SolarEpisode, state: np.ndarray) -> None:
    """The last reading: its distance, the ground point it hit, and its status."""
    put(state, STATE_SLICES, "laser_range_m", 0.0)
    put(state, STATE_SLICES, "laser_point_m", (0.0, 0.0, 0.0))
    put(state, STATE_SLICES, "laser_status", LASER_STATUSES.index("no_signal"))
