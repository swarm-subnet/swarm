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

"""The patrol task one seed describes: its clock, its map type and its wind.

The dock's place is not known here: the seed decides it when the world is built, and the environment writes it
back into the task as the start and the goal.
"""

from __future__ import annotations

from swarm.protocol import SCHEMA_VERSION, MapTask

from . import wind
from .contract import CHALLENGE_TYPE, FAMILY_ID, HORIZON_S


def solar_patrol_task(*, seed: int, sim_dt: float) -> MapTask:
    """One patrol of the solar park for a seed."""
    return MapTask(
        map_seed=int(seed),
        start=(0.0, 0.0, 0.0),
        goal=(0.0, 0.0, 0.0),
        sim_dt=float(sim_dt),
        horizon=HORIZON_S,
        challenge_type=CHALLENGE_TYPE,
        family_id=FAMILY_ID,
        version=SCHEMA_VERSION,
        moving_platform=False,
        **wind.for_seed(int(seed)),
    )
