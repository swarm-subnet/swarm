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

"""The edge rule: the field's surveyed outline is the flight limit, and the drone must keep EDGE_M inside it.

No fence stands on the real field, so the line is invisible. While the drone is out of the dock, in every phase from
take-off to landing, its true position across the ground must stay inside the outline and more than EDGE_M from it:
the moment it does not, the patrol ends there, as a flight-limit end, and the seed scores 0. The state reports the
distance to the outline itself and whether the drone is inside it, as the dock reports its own custom flight area,
and the site map gives the outline as both the fence and the limit.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..solar_patrol.contract import MAX_FENCE_POINTS, MAX_LIMIT_POINTS, SITE_MAP_SLICES, STATE_SLICES, put
from ..solar_patrol.flight_limit import _distance, _inside, _sides_of
from .episode import FieldEpisode

EDGE_M = 1.0
_AIRBORNE = ("taking_off", "flying", "returning", "landing")
_ROUNDING_M = 1e-6                     # far above the distance's float error, so a skipped check never hides a trip


def reset(env: Any, ep: FieldEpisode) -> None:
    """The field's outline is this patrol's limit; no point checked yet."""
    polygon = np.asarray(ep.fence, dtype=float)
    ep.flight_limit = {"polygon": polygon, "sides": _sides_of(polygon), "checked": (0.0, 0.0), "clear_m": 0.0}
    ep.outcome.min_edge_m = None


def edge_distance(ep: FieldEpisode, point: np.ndarray) -> float:
    """How far a point lies inside the outline, negative outside it."""
    limit = ep.flight_limit
    distance = _distance(limit["polygon"], np.asarray(point, dtype=float)[:2])
    return distance if _inside(limit["sides"], point) else -distance


def update(env: Any, ep: FieldEpisode) -> None:
    """End the patrol once a drone out of the dock comes within EDGE_M of the outline or leaves it, and keep the
    closest it came."""
    if ep.phase not in _AIRBORNE:
        return
    point = np.asarray(env.pos[0][:2], dtype=float)
    limit = ep.flight_limit
    # Moved less than the clear distance since the last check: it cannot have reached the trip line.
    if math.dist(point, limit["checked"]) < limit["clear_m"]:
        return
    inside = edge_distance(ep, point)
    closest = ep.outcome.min_edge_m
    ep.outcome.min_edge_m = inside if closest is None else min(closest, inside)
    if inside <= EDGE_M:
        ep.end("flight_limit")
        ep.outcome.edge_trips = int(ep.outcome.end_reason == "flight_limit")
        return
    limit.update(checked=(float(point[0]), float(point[1])), clear_m=inside - EDGE_M - _ROUNDING_M)


def observe(env: Any, ep: FieldEpisode, state: np.ndarray) -> None:
    """The distance to the outline and whether the drone is inside it."""
    point = np.asarray(env.pos[0][:2], dtype=float)
    limit = ep.flight_limit
    put(state, STATE_SLICES, "flight_limit_distance_m", _distance(limit["polygon"], point))
    put(state, STATE_SLICES, "inside_flight_limit", float(_inside(limit["sides"], point)))


def site_map(env: Any, ep: FieldEpisode, site: np.ndarray) -> None:
    """The outline as both the fence and the limit, in metres from the dock."""
    corners = ep.flight_limit["polygon"] - ep.dock_position[:2]
    for count, name, capacity in (("fence_count", "fence_xy", MAX_FENCE_POINTS), ("limit_count", "limit_xy", MAX_LIMIT_POINTS)):
        padded = np.zeros((capacity, 2))
        padded[:min(len(corners), capacity)] = corners[:capacity]
        put(site, SITE_MAP_SLICES, count, min(len(corners), capacity))
        put(site, SITE_MAP_SLICES, name, padded.reshape(-1))
