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

Each seed draws every straight side of the fence its own distance, from the limit's own stream, for the stop line to
stand outside it, so the whole park inside the fence can always be flown. The limit stands STOP_LINE_M further out,
the pushed sides joined into one outline around the fence, the shape of a DJI custom flight area. The dock reports the
distance to it and whether the drone is inside, as DJI's does. Reaching the stop line ends the patrol there, and the
landing earns nothing.

A side with a panel table near it only draws distances that keep the stop line PANEL_CLEAR_M off every table where
the seed stands it (task 40).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import shapely
from shapely.geometry import MultiPoint, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from . import park
from .contract import MAX_LIMIT_POINTS, SITE_MAP_SLICES, STATE_SLICES, put
from .episode import SolarEpisode
from .fixed_order import dot

STOP_LINE_M = 5.0                      # DJI ends the task this near a custom flight area's edge
MAX_OUTSIDE_M = 5.0                    # the stop line stands 0 to 5 m outside each side of the fence
PANEL_CLEAR_M = 1.0                    # the stop line keeps this far off every panel table
LIMIT_SEED_STREAM = 0xF15              # the limit's own stream, so its draws never move another part's
_AIRBORNE = ("taking_off", "flying", "returning", "landing")
_ROUNDING_M = 1e-6                     # far above the distance's float error, so a skipped check never hides a stop


def _sides(fence: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each side's start and end, anticlockwise, and its unit normal pointing out of the fence."""
    a = np.array(orient(Polygon(fence), 1.0).exterior.coords[:-1])
    b = np.roll(a, -1, axis=0)
    along = b - a
    return a, b, np.column_stack([along[:, 1], -along[:, 0]]) / np.hypot(*along.T)[:, None]


def lowest(fence: np.ndarray, tables: np.ndarray) -> np.ndarray:
    """The nearest each side's stop line may stand to the fence: 0, raised on a side with a table less than
    PANEL_CLEAR_M from it until the table is cleared by that much.

    A side's gap is read MAX_OUTSIDE_M past both its ends, as far as its stop line can bend round a corner.
    """
    a, b, _ = _sides(fence)
    if not len(tables):
        return np.zeros(len(a))
    along = (b - a) / np.hypot(*(b - a).T)[:, None]
    reach = shapely.linestrings(np.stack([a - MAX_OUTSIDE_M * along, b + MAX_OUTSIDE_M * along], axis=1))
    shapes = shapely.convex_hull(shapely.multipoints(np.asarray(tables, dtype=float)))
    gaps = shapely.distance(reach[:, None], shapes[None, :]).min(axis=1)
    return np.clip(PANEL_CLEAR_M - gaps, 0.0, MAX_OUTSIDE_M)


def pushes(seed: int, low: np.ndarray) -> np.ndarray:
    """How far each side of the limit stands out from its side of the fence this seed: STOP_LINE_M, plus a distance
    drawn per side for the stop line, from the side's lowest to MAX_OUTSIDE_M."""
    return STOP_LINE_M + np.random.default_rng([LIMIT_SEED_STREAM, int(seed)]).uniform(low, MAX_OUTSIDE_M)


def outline(fence: np.ndarray, push: np.ndarray, low: np.ndarray) -> np.ndarray:
    """The fence with every side pushed out by its distance, merged into one polygon, anticlockwise, in world metres.

    A strip lies along each side and a wedge closes each corner. An outward corner's wedge reaches the point
    STOP_LINE_M, plus the larger lowest draw of its two sides, out along both sides, so no stretch of the fence
    comes nearer the limit than the stop line, and a table tucked into the corner keeps its gap.
    """
    a, b, out = _sides(fence)
    strips = [Polygon([a[i], b[i], b[i] + push[i] * out[i], a[i] + push[i] * out[i]]) for i in range(len(a))]
    wedges = []
    for i in range(len(a)):
        before, after = a[i] + push[i - 1] * out[i - 1], a[i] + push[i] * out[i]
        corner = [a[i], before, after]
        if out[i - 1, 0] * out[i, 1] - out[i - 1, 1] * out[i, 0] > 0.0:
            reach = STOP_LINE_M + max(low[i - 1], low[i])
            corner.append(a[i] + reach * (out[i - 1] + out[i]) / (1.0 + dot(out[i - 1], out[i])))
        wedges.append(MultiPoint(corner).convex_hull)
    merged = unary_union([Polygon(a)] + [piece for piece in strips + wedges if piece.area > 0.0]).simplify(0.0)
    return np.array(orient(merged, 1.0).exterior.coords[:-1])


def _sides_of(polygon: np.ndarray) -> list:
    """Each side of the closed polygon as plain floats: start east, start north, end east, end north."""
    return np.hstack([polygon, np.roll(polygon, -1, axis=0)]).tolist()


def _inside(sides: list, point: np.ndarray) -> bool:
    """True when the point lies inside the closed polygon with these sides, by counting the sides a ray from it
    crosses."""
    x, y = float(point[0]), float(point[1])
    inside = False
    for x1, y1, x2, y2 in sides:
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
    low = lowest(ep.fence, park.table_footprints(ep))
    push = pushes(ep.seed, low)
    polygon = outline(ep.fence, push, low)
    ep.flight_limit = {"pushes": push, "polygon": polygon, "sides": _sides_of(polygon), "checked": (0.0, 0.0),
                       "clear_m": 0.0}


def update(env: Any, ep: SolarEpisode) -> None:
    """End the patrol once an airborne drone reaches the stop line or leaves the limit."""
    if ep.phase not in _AIRBORNE:
        return
    point = np.asarray(env.pos[0][:2], dtype=float)
    limit = ep.flight_limit
    # Closer to the last point checked than that point was to the stop line, the drone is still well inside it.
    if math.dist(point, limit["checked"]) < limit["clear_m"]:
        return
    distance = _distance(limit["polygon"], point)
    if not _inside(limit["sides"], point) or distance <= STOP_LINE_M:
        ep.end("flight_limit")
        return
    limit.update(checked=(float(point[0]), float(point[1])), clear_m=distance - STOP_LINE_M - _ROUNDING_M)


def observe(env: Any, ep: SolarEpisode, state: np.ndarray) -> None:
    """The distance to the limit and whether the drone is inside it, as the dock reports them."""
    point = np.asarray(env.pos[0][:2], dtype=float)
    limit = ep.flight_limit
    put(state, STATE_SLICES, "flight_limit_distance_m", _distance(limit["polygon"], point))
    put(state, STATE_SLICES, "inside_flight_limit", float(_inside(limit["sides"], point)))


def site_map(env: Any, ep: SolarEpisode, site: np.ndarray) -> None:
    """The limit's shape, in metres from the dock."""
    corners = (ep.flight_limit["polygon"] - ep.dock_position[:2])[:MAX_LIMIT_POINTS]
    padded = np.zeros((MAX_LIMIT_POINTS, 2))
    padded[:len(corners)] = corners
    put(site, SITE_MAP_SLICES, "limit_count", len(corners))
    put(site, SITE_MAP_SLICES, "limit_xy", padded.reshape(-1))
