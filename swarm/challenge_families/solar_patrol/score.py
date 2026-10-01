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

"""Per-seed score (task 23): one score per patrol, from detection, coverage and flight.

update() keeps the flight facts the score needs while the patrol runs; score() turns a finished patrol's metrics
into the terms named in SCORE_TERMS. Every height is measured above the dock, the take-off point.

seed = 0.70 detection + 0.20 coverage + 0.10 flight
  detection  valid reports / (threats + false alarms); on a seed with no threat, the share searched
             / (1 + false alarms), so silence only pays after a real search
  coverage   the share of the park searched
  flight     half for landing in the dock after the model's own return home, half for keeping the peak height
             at 22 m or lower, falling in a straight line to 0 at 30 m
A patrol that never took off, or came closer than 5 m to a threat, scores 0.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pybullet as p

from . import theft
from .episode import SolarEpisode

DETECTION_WEIGHT = 0.70
COVERAGE_WEIGHT = 0.20
FLIGHT_WEIGHT = 0.10
LANDING_SHARE = 0.5                  # of the flight term; the rest is the height
FULL_HEIGHT_M = 22.0                 # 2 m of room above the 20 m patrol line
ZERO_HEIGHT_M = 30.0
SAFE_DISTANCE_M = 5.0                # EU rule for a C2 drone near people
PERSON_HEIGHT_M = 1.8
_IN_DOCK = ("docked", "landed")
_PAST_BODIES = 16                    # panels, fences and trees a ray may pass on its way to the terrain


def reset(env: Any, ep: SolarEpisode) -> None:
    """Nothing recorded yet."""
    ep.score = None
    ep.outcome.max_height_m = 0.0
    ep.outcome.min_threat_distance_m = None


def update(env: Any, ep: SolarEpisode) -> None:
    """Keep the highest the drone has flown above its take-off point, and the closest it came to a threat."""
    pos = np.asarray(env.pos[0], dtype=float)
    ep.outcome.max_height_m = max(ep.outcome.max_height_m, float(pos[2]) - float(ep.dock_position[2]))
    if ep.phase in _IN_DOCK:
        return
    for person in theft.people(ep):
        if not person["threat"]:
            continue
        x, y = person["xy"]
        across = math.hypot(float(pos[0]) - x, float(pos[1]) - y)
        if across >= SAFE_DISTANCE_M:
            continue
        ground = _ground(env, ep, x, y)
        gap = max(float(pos[2]) - ground - PERSON_HEIGHT_M, ground - float(pos[2]), 0.0)
        distance = math.hypot(across, gap)
        closest = ep.outcome.min_threat_distance_m
        ep.outcome.min_threat_distance_m = distance if closest is None else min(closest, distance)


def score(task: Any, metrics: dict[str, Any]) -> dict[str, float]:
    """The normalised terms of one patrol, final_score among them."""
    coverage = float(np.clip(metrics["coverage"], 0.0, 1.0))
    threats, false_alarms = int(metrics["threats"]), int(metrics["false_alarms"])
    if threats:
        detection = min(int(metrics["valid_reports"]), threats) / (threats + false_alarms)
    else:
        detection = coverage / (1 + false_alarms)
    landed = bool(metrics["landed_in_dock"]) and bool(metrics["returned_by_model"])
    height = float(np.clip((ZERO_HEIGHT_M - metrics["max_height_m"]) / (ZERO_HEIGHT_M - FULL_HEIGHT_M), 0.0, 1.0))
    flight = LANDING_SHARE * landed + (1.0 - LANDING_SHARE) * height if metrics["took_off"] else 0.0
    closest = metrics["min_threat_distance_m"]
    too_close = closest is not None and closest < SAFE_DISTANCE_M
    final = 0.0
    if metrics["took_off"] and not too_close:
        final = DETECTION_WEIGHT * detection + COVERAGE_WEIGHT * coverage + FLIGHT_WEIGHT * flight
    return {"detection_term": detection, "coverage_term": coverage, "flight_term": flight, "final_score": final}


def _ground(env: Any, ep: SolarEpisode, x: float, y: float) -> float:
    """Terrain height under a point: the first terrain body a ray straight down meets, past panels, fences and
    trees; the dock's height when none is found."""
    top = 1000.0
    for _ in range(_PAST_BODIES):
        uid, _link, _fraction, hit, _normal = p.rayTest([x, y, top], [x, y, -1000.0], physicsClientId=env.CLIENT)[0]
        if uid < 0:
            break
        if uid in ep.terrain_uids:
            return float(hit[2])
        top = float(hit[2]) - 0.01
    return float(ep.dock_position[2])
