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

"""Per-seed score: Sentinel's formula, with the field's people and its stricter rules for a flight gone wrong.

seed = 0.70 detection + 0.20 coverage + 0.10 flight, each term as Sentinel computes it (solar's score module), with
the people on the field as its threats and the coverage measured over the field. A patrol scores 0 when it never took
off or came closer than SAFE_DISTANCE_M to a person, as in Sentinel, and also when it tripped the edge rule or the
drone hit anything, the dock included, or tipped past its tilt limit: on the real field each of those ends the flight.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..solar_patrol import score as sentinel_score
from ..solar_patrol.score import PERSON_HEIGHT_M, SAFE_DISTANCE_M
from . import people
from .episode import FieldEpisode

ZEROING_ENDS = ("flight_limit", "collision")
_IN_DOCK = ("docked", "landed")

reset = sentinel_score.reset


def update(env: Any, ep: FieldEpisode) -> None:
    """Keep Sentinel's peak height above the ground and the closest the drone came to a person out of the dock."""
    sentinel_score.update(env, ep)
    if ep.phase in _IN_DOCK:
        return
    pos = np.asarray(env.pos[0], dtype=float)
    for person in people.people(ep):
        x, y = person.xy
        across = math.hypot(float(pos[0]) - x, float(pos[1]) - y)
        if across >= SAFE_DISTANCE_M:
            continue
        gap = max(float(pos[2]) - person.ground_z - PERSON_HEIGHT_M, person.ground_z - float(pos[2]), 0.0)
        distance = math.hypot(across, gap)
        closest = ep.outcome.min_threat_distance_m
        ep.outcome.min_threat_distance_m = distance if closest is None else min(closest, distance)


def score(task: Any, metrics: dict[str, Any]) -> dict[str, float]:
    """The normalised terms of one patrol, final_score among them."""
    terms = sentinel_score.score(task, metrics)
    if metrics.get("end_reason") in ZEROING_ENDS:
        terms["final_score"] = 0.0
    return terms
