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

"""Per-seed score: Sentinel's terms, with the field's people, its animals and its stricter rules for a flight gone
wrong.

seed = 0.50 people + 0.20 animals + 0.20 coverage + 0.10 flight. The people term is Sentinel's detection (solar's
score module) with the people on the field as its threats, and the coverage and flight terms are Sentinel's, the
coverage measured over the field. The animals term is the valid reports that named an animal over the animals to
report plus the reports that named an animal and failed. On a field with no animal to report it is the people term
over one plus those failed reports, so such a seed scores as 0.70 people + 0.20 coverage + 0.10 flight while no
report names an animal wrongly. A report that names an animal never counts as a false alarm.

A patrol scores 0 when it never took off or came closer than SAFE_DISTANCE_M to a person, as in Sentinel, and also
when it tripped the edge rule or the drone hit anything, the dock included, or tipped past its tilt limit: on the real
field each of those ends the flight. The distance to a person is measured to where he is at that step, as tall as he
stands, for everyone on the field or passing outside it; animals keep no distance.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..solar_patrol import score as sentinel_score
from ..solar_patrol.score import COVERAGE_WEIGHT, FLIGHT_WEIGHT, SAFE_DISTANCE_M
from . import people
from .episode import FieldEpisode

PEOPLE_WEIGHT = 0.50
ANIMALS_WEIGHT = 0.20
ZEROING_ENDS = ("flight_limit", "collision")
_IN_DOCK = ("docked", "landed")

reset = sentinel_score.reset


def update(env: Any, ep: FieldEpisode) -> None:
    """Keep Sentinel's peak height above the ground and the closest the drone came to a person out of the dock,
    measured to each person at the current step."""
    sentinel_score.update(env, ep)
    if ep.phase in _IN_DOCK:
        return
    pos = np.asarray(env.pos[0], dtype=float)
    for person in people.people(ep):
        x, y, ground = (float(v) for v in people.position(ep, person.index))
        across = math.hypot(float(pos[0]) - x, float(pos[1]) - y)
        if across >= SAFE_DISTANCE_M:
            continue
        gap = max(float(pos[2]) - ground - float(person.head_m), ground - float(pos[2]), 0.0)
        distance = math.hypot(across, gap)
        closest = ep.outcome.min_threat_distance_m
        ep.outcome.min_threat_distance_m = distance if closest is None else min(closest, distance)


def animal_term(metrics: dict[str, Any], people_term: float) -> float:
    """The animals' term of one patrol: valid animal reports over the animals to report and the failed animal
    reports; with no animal to report, the people's term over one and the failed animal reports."""
    asked, wrong = int(metrics.get("animals", 0)), int(metrics.get("wrong_animal_reports", 0))
    if asked:
        return min(int(metrics.get("valid_animal_reports", 0)), asked) / (asked + wrong)
    return people_term / (1 + wrong)


def score(task: Any, metrics: dict[str, Any]) -> dict[str, float]:
    """The normalised terms of one patrol, final_score among them."""
    terms = sentinel_score.score(task, metrics)
    terms["animal_term"] = animal_term(metrics, terms["detection_term"])
    closest = metrics["min_threat_distance_m"]
    flown = metrics["took_off"] and not (closest is not None and closest < SAFE_DISTANCE_M)
    terms["final_score"] = 0.0
    if flown and metrics.get("end_reason") not in ZEROING_ENDS:
        terms["final_score"] = (PEOPLE_WEIGHT * terms["detection_term"] + ANIMALS_WEIGHT * terms["animal_term"]
                                + COVERAGE_WEIGHT * terms["coverage_term"] + FLIGHT_WEIGHT * terms["flight_term"])
    return terms
