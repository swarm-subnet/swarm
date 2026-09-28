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

"""Flight limit (task 15): the invisible line the drone must stay inside, and the stop line 5 m before it.

Reaching the stop line ends the patrol: the drone flies home on its own and the landing earns nothing.

Stand-in: the limit is the fence's bounding rectangle grown by 5 m on every side.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .contract import MAX_LIMIT_POINTS, SITE_MAP_SLICES, STATE_SLICES, put
from .episode import SolarEpisode

STOP_LINE_M = 5.0
MARGIN_M = 5.0
_AIRBORNE = ("taking_off", "flying", "returning", "landing")


def _inside(polygon: np.ndarray, point: np.ndarray) -> bool:
    """True when the point lies inside the closed polygon, by counting the sides a ray from it crosses."""
    x, y = float(point[0]), float(point[1])
    inside = False
    for (x1, y1), (x2, y2) in zip(polygon, np.roll(polygon, -1, axis=0)):
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def _distance(polygon: np.ndarray, point: np.ndarray) -> float:
    """Distance from the point to the nearest side of the closed polygon."""
    a = polygon
    b = np.roll(polygon, -1, axis=0)
    ab = b - a
    t = np.clip(np.einsum("ij,ij->i", point - a, ab) / np.einsum("ij,ij->i", ab, ab), 0.0, 1.0)
    return float(np.min(np.hypot(*(a + ab * t[:, None] - point).T)))


def reset(env: Any, ep: SolarEpisode) -> None:
    """Draw this seed's limit around the fence."""
    low = ep.fence.min(axis=0) - MARGIN_M
    high = ep.fence.max(axis=0) + MARGIN_M
    ep.flight_limit = {"polygon": np.array([[low[0], low[1]], [high[0], low[1]], [high[0], high[1]],
                                            [low[0], high[1]]])}


def update(env: Any, ep: SolarEpisode) -> None:
    """End the patrol once an airborne drone reaches the stop line or leaves the limit."""
    if ep.phase not in _AIRBORNE:
        return
    point = np.asarray(env.pos[0][:2], dtype=float)
    polygon = ep.flight_limit["polygon"]
    if not _inside(polygon, point) or _distance(polygon, point) <= STOP_LINE_M:
        ep.end("flight_limit")


def observe(env: Any, ep: SolarEpisode, state: np.ndarray) -> None:
    """The distance to the limit and whether the drone is inside it, as the dock reports them."""
    point = np.asarray(env.pos[0][:2], dtype=float)
    polygon = ep.flight_limit["polygon"]
    put(state, STATE_SLICES, "flight_limit_distance_m", _distance(polygon, point))
    put(state, STATE_SLICES, "inside_flight_limit", float(_inside(polygon, point)))


def site_map(env: Any, ep: SolarEpisode, site: np.ndarray) -> None:
    """The limit's shape, in metres from the dock."""
    corners = (ep.flight_limit["polygon"] - ep.dock_position[:2])[:MAX_LIMIT_POINTS]
    padded = np.zeros((MAX_LIMIT_POINTS, 2))
    padded[:len(corners)] = corners
    put(site, SITE_MAP_SLICES, "limit_count", len(corners))
    put(site, SITE_MAP_SLICES, "limit_xy", padded.reshape(-1))
