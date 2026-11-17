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

"""The site map the model gets on the first observation: the field's outline, the tree crowns and the houses.

The layout is Sentinel's. The fence and the flight limit are both the field's outline; the table rows carry the tree
crowns, since the field has no panel tables (crowns module); the buildings are the houses around the field, nearest
the dock first, as far as the map holds them.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..solar_patrol.contract import MAX_BUILDINGS, SITE_MAP_SLICES, put
from . import crowns, flight_limit, site
from .episode import FieldEpisode


def write(env: Any, ep: FieldEpisode, vector: np.ndarray) -> None:
    """Fill the site map, in metres from the dock."""
    flight_limit.site_map(env, ep, vector)
    rows = site.buildings(ep).copy()
    rows[:, :2] -= ep.dock_position[:2]
    rows = rows[np.argsort(np.hypot(rows[:, 0], rows[:, 1]), kind="stable")][:MAX_BUILDINGS]
    padded = np.zeros((MAX_BUILDINGS, 5))
    padded[:len(rows)] = rows
    crowns.site_map(env, ep, vector)
    put(vector, SITE_MAP_SLICES, "building_count", len(rows))
    put(vector, SITE_MAP_SLICES, "buildings", padded.reshape(-1))
