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

"""Tree crowns: solid for the drone, given to the model in the site map's free slots.

The simulator makes a tree's trunk solid and lets a drone pass through its crown. Under the field's 10 m ceiling the
drone can reach the crowns that hang over the field, and on the real field that is a crash. So while the drone is out
of the dock, its true position across the ground inside any crown circle the survey lists ends the patrol as a
collision, at every height: the circles are already drawn generously round each crown.

The field has no panel tables, so the site map's table rows carry the crowns, Sentinel's layout unchanged: centre
east, centre north, the circle's diameter twice, and a heading of 0, nearest the dock first.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..solar_patrol.contract import MAX_TABLES, SITE_MAP_SLICES, put
from . import site
from .episode import FieldEpisode

_AIRBORNE = ("taking_off", "flying", "returning", "landing")


def circles(ep: FieldEpisode) -> np.ndarray:
    """The crown circles this patrol knows, none before the reset reads them."""
    return np.asarray(ep.crowns if ep.crowns is not None else (), dtype=float).reshape(-1, 3)


def reset(env: Any, ep: FieldEpisode) -> None:
    """The survey's crown circles, world metres; no distance kept yet."""
    ep.crowns = site.passable_outlines(ep)
    ep.outcome.min_crown_m = None


def clearance(circles: np.ndarray, point: np.ndarray) -> float:
    """How far a point lies outside the nearest crown circle, negative inside one; infinite with no crowns."""
    if not len(circles):
        return float("inf")
    return float(np.min(np.hypot(circles[:, 0] - point[0], circles[:, 1] - point[1]) - circles[:, 2]))


def outside(circles: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Which points lie outside every crown circle."""
    if not len(circles):
        return np.ones(len(points), bool)
    gaps = np.hypot(points[:, None, 0] - circles[None, :, 0], points[:, None, 1] - circles[None, :, 1]) - circles[None, :, 2]
    return np.all(gaps >= 0.0, axis=1)


def update(env: Any, ep: FieldEpisode) -> None:
    """Keep the closest a drone out of the dock came to a crown, and end the patrol as a collision inside one."""
    if ep.phase not in _AIRBORNE or not len(circles(ep)):
        return
    gap = clearance(circles(ep), np.asarray(env.pos[0][:2], dtype=float))
    closest = ep.outcome.min_crown_m
    ep.outcome.min_crown_m = gap if closest is None else min(closest, gap)
    if gap < 0.0:
        ep.end("collision")


def site_map(env: Any, ep: FieldEpisode, vector: np.ndarray) -> None:
    """The crowns in the table rows, in metres from the dock, nearest first, as many as the rows hold."""
    known = circles(ep)
    rows = np.zeros((len(known), 5))
    rows[:, :2] = known[:, :2] - ep.dock_position[:2]
    rows[:, 2] = rows[:, 3] = 2.0 * known[:, 2]
    rows = rows[np.argsort(np.hypot(rows[:, 0], rows[:, 1]), kind="stable")][:MAX_TABLES]
    padded = np.zeros((MAX_TABLES, 5))
    padded[:len(rows)] = rows
    put(vector, SITE_MAP_SLICES, "table_count", len(rows))
    put(vector, SITE_MAP_SLICES, "tables", padded.reshape(-1))
