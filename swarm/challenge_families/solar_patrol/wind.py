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

"""Wind strength per seed (task 18): the wind the seed deals, and the two readings the dock reports of it.

Stand-in: every seed is still and both readings are zero.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .contract import STATE_SLICES, put
from .episode import SolarEpisode


def for_seed(seed: int) -> dict[str, Any]:
    """The task's wind fields for a seed, the ones the environment's seeded wind reads."""
    return {"wind_max_mps": 0.0, "wind_turbulence": 0.0, "wind_gusts": 0}


def reset(env: Any, ep: SolarEpisode) -> None:
    """Nothing to set up: the environment builds the physical wind from the task."""


def observe(env: Any, ep: SolarEpisode, state: np.ndarray) -> None:
    """The drone's rough wind estimate and the dock's gauge."""
    put(state, STATE_SLICES, "wind_estimate_mps", 0.0)
    put(state, STATE_SLICES, "wind_estimate_sector", 0.0)
    put(state, STATE_SLICES, "dock_wind_mps", 0.0)
