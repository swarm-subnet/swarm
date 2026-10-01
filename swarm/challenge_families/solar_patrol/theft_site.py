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

"""Theft scenarios (task 19): the park as a thief sees it, where he can get in and where he can walk.

Everything here is read from the map's manifest and the seed's shifts, without the engine, so a seed's theft can be
drawn and checked on its own: the fence panels and the gate, the forest side, the panel tables as this seed stands
them, and a walking grid inside the fence that keeps a body clear of every table, building, the fence and the dock.
Routes are the shortest walks on that grid, pulled straight wherever the line between two points stays clear.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from shapely import contains_xy, prepare
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from swarm.core.maps.solar import builder

CELL_M = 0.5                            # walking grid resolution
COMFORT_M = 1.0                         # where there is room, a walk keeps this much more from anything in its way
CROWDED_COST = 3.0                      # how much dearer a step closer than that counts, so routes leave corners wide
BODY_M = 0.35                           # half a body's width, the clearance a walking thief keeps to anything
TABLE_CLEAR_M = 0.6                     # from a table's outline: a body walks this close along a row
BUILDING_CLEAR_M = 0.8
FENCE_CLEAR_M = 0.8                     # inside the fence, except where he steps through it
DOCK_CLEAR_M = 8.0                      # nobody walks this near the dock, so take-off never starts next to a thief
OLIVE_RADIUS_M = 3.3                    # the olive's crown hangs at head height
FOREST_TIER = 0                         # the forest table's tier of trees by the fence and the road
FOREST_REACH_M = 3.0                    # a panel with a tree this close outside stands against the woods
ROAD_REACH_M = 25.0                     # a panel this near the public road the truck drives faces the road
TABLE_PARTS = ("full_table_frame", "full_table_glass", "full_table_racking", "half_table_frame", "half_table_glass",
               "half_table_racking")
BUILDINGS = ("white_unit_north", "white_unit_south", "service_object", "gate", "slab")


def yaw_of(quaternion: Sequence[float]) -> float:
    """Heading about +z of an xyzw quaternion."""
    x, y, z, w = quaternion
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def inside(ring: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Which ground points lie inside a closed ring (even-odd rule)."""
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    xs, ys = ring[:, 0], ring[:, 1]
    xn, yn = np.roll(xs, -1), np.roll(ys, -1)
    x, y = pts[:, :1], pts[:, 1:2]
    crosses = ((ys > y) != (yn > y)) & (x < (xn - xs) * (y - ys) / np.where(yn != ys, yn - ys, 1e-12) + xs)
    return crosses.sum(1) % 2 == 1


@dataclass(frozen=True)
class Table:
    """One panel table as this seed stands it: its centre, the unit axis along it and across it, and its half sizes."""

    centre: Tuple[float, float]
    along: Tuple[float, float]
    across: Tuple[float, float]
    half_length: float
    half_depth: float
    row: int

    def corners(self, grow: float = 0.0) -> np.ndarray:
        """The outline from above, grown by a margin on every side."""
        c, u, v = np.array(self.centre), np.array(self.along), np.array(self.across)
        hl, hd = self.half_length + grow, self.half_depth + grow
        return np.array([c - u * hl - v * hd, c + u * hl - v * hd, c + u * hl + v * hd, c - u * hl + v * hd])


@dataclass(frozen=True)
class Opening:
    """A place in the fence a thief can get through: a panel or the gate, its centre and height, the unit axis along
    it and the unit normal into the park, its half width, and which side it faces: the public road, or the woods."""

    kind: str
    index: int
    centre: Tuple[float, float]
    z: float
    along: Tuple[float, float]
    inward: Tuple[float, float]
    half_width: float
    road: bool
    forest: bool


class Site:
    """The park of one seed as a walking thief sees it."""

    def __init__(self, seed: int, asset_dir: str, dock: Optional[Sequence[float]] = None):
        """Read the fence, gate and tables, and lay the walking grid; dock is the seed's dock, kept clear."""
        manifest = builder.solar_manifest(asset_dir)
        placements, items = manifest["placements"], manifest["items"]
        shifts = builder.solar_shifts(seed, asset_dir)
        self.ring = np.asarray(builder.solar_fence(asset_dir), dtype=float)
        xs, ys = self.ring[:, 0], self.ring[:, 1]
        yn = np.roll(ys, -1)
        self._edges = (xs, ys, yn, np.roll(xs, -1) - xs, np.where(yn != ys, yn - ys, 1e-12))
        self.dock = None if dock is None else np.asarray(dock, dtype=float)[:2]
        self.road = _road(asset_dir)
        trees = _forest(asset_dir)
        self.tables: List[Table] = []
        rows: Dict[Tuple[float, ...], List[np.ndarray]] = {}
        blocks: List[np.ndarray] = []
        self.openings: List[Opening] = []
        olive = None
        for index, place in enumerate(placements):
            name = place["item"]
            if name in TABLE_PARTS or name in BUILDINGS or name == "olive_bark":
                moved = dict(place, **shifts.get(index, {}))
                outline = _outline(items[name], moved)
                if name in TABLE_PARTS:
                    rows.setdefault((place.get("row", -1),) + tuple(np.round(place["position"][:2], 1)), []).append(outline)
                elif name == "olive_bark":
                    olive = np.array(moved["position"][:2])
                else:
                    blocks.append(outline)
                if name == "gate":
                    self.openings.append(self._opening("gate", index, moved, items[name], trees))
            elif name == "fence_panel":
                self.openings.append(self._opening("panel", index, place, items[name], trees))
        for (row, *_), outlines in sorted(rows.items()):
            self.tables.append(_table(np.vstack(outlines), row))
        self.blocks = blocks
        self.olive = olive
        self._grid()

    def holds(self, x: float, y: float) -> bool:
        """Whether one ground point lies inside the fence: inside()'s even-odd rule on edges prepared once."""
        xs, ys, yn, dx, dy = self._edges
        return bool(np.count_nonzero(((ys > y) != (yn > y)) & (x < dx * (y - ys) / dy + xs)) % 2)

    def _opening(self, kind: str, index: int, place: dict, item: dict, trees: np.ndarray) -> Opening:
        """A panel or the gate as a way in, with its inward normal and whether it faces the forest."""
        yaw = yaw_of(place["quaternion"])
        centre = np.array(place["position"][:2], dtype=float)
        low, high = np.array(item["bounds_min"][:2]), np.array(item["bounds_max"][:2])
        if kind == "gate":
            # The gate's leaves run corner to corner of its outline, not along its local x.
            span = np.array([high[0] - low[0], low[1] - high[1]])
        else:
            span = np.array([high[0] - low[0], 0.0])
        c, s = math.cos(yaw), math.sin(yaw)
        along = np.array([c * span[0] - s * span[1], s * span[0] + c * span[1]])
        half = float(np.linalg.norm(along)) / 2.0 * float(place["scale"][0])
        along /= max(float(np.linalg.norm(along)), 1e-9)
        normal = np.array([-along[1], along[0]])
        if not inside(self.ring, centre + normal * 1.0)[0]:
            normal = -normal
        outward = centre - normal * FOREST_REACH_M
        woods = bool(len(trees)) and bool((np.hypot(*(trees - outward).T) < FOREST_REACH_M).any())
        road = bool(len(self.road)) and float(np.hypot(*(self.road - centre).T).min()) < ROAD_REACH_M
        return Opening(kind, index, (float(centre[0]), float(centre[1])), float(place["position"][2]),
                       (float(along[0]), float(along[1])), (float(normal[0]), float(normal[1])), half,
                       road, woods and not road)

    def _grid(self) -> None:
        """Mark the cells inside the fence a body can stand on, and join each to its eight neighbours."""
        low = self.ring.min(0) - 2.0
        high = self.ring.max(0) + 2.0
        self.origin = low
        self.shape = tuple(int(n) for n in np.ceil((high - low) / CELL_M).astype(int) + 1)
        xs = low[0] + np.arange(self.shape[0]) * CELL_M
        ys = low[1] + np.arange(self.shape[1]) * CELL_M
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        room = Polygon(self.ring).buffer(-FENCE_CLEAR_M, join_style="mitre")
        taken = [Polygon(table.corners(TABLE_CLEAR_M)) for table in self.tables]
        taken += [Polygon(_grown(block, BUILDING_CLEAR_M)) for block in self.blocks]
        if self.olive is not None:
            taken.append(Point(*self.olive).buffer(OLIVE_RADIUS_M + BODY_M))
        if self.dock is not None:
            taken.append(Point(*self.dock).buffer(DOCK_CLEAR_M))
        free = room.difference(unary_union(taken))
        prepare(free)
        self.walkable = contains_xy(free, gx, gy)
        self.room = distance_transform_edt(self.walkable) * CELL_M
        self._graph = _graph(self.walkable, self.room)

    def cell(self, point: Sequence[float]) -> Tuple[int, int]:
        """The grid cell holding a ground point."""
        i = int(round((float(point[0]) - self.origin[0]) / CELL_M))
        j = int(round((float(point[1]) - self.origin[1]) / CELL_M))
        return min(max(i, 0), self.shape[0] - 1), min(max(j, 0), self.shape[1] - 1)

    def point(self, cell: Tuple[int, int]) -> np.ndarray:
        """The ground point at a cell's centre."""
        return self.origin + np.array(cell, dtype=float) * CELL_M

    def free(self, point: Sequence[float]) -> bool:
        """Whether a body can stand at a ground point."""
        return bool(self.walkable[self.cell(point)])

    def clear_line(self, a: Sequence[float], b: Sequence[float], room: float = 0.0) -> bool:
        """Whether every point on the straight line between two ground points is walkable, with room to spare."""
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        n = max(2, int(math.ceil(np.linalg.norm(b - a) / (CELL_M / 2.0))) + 1)
        # Every point at once, with cell()'s own arithmetic and its round half to even.
        points = a + (b - a) * np.linspace(0.0, 1.0, n)[:, None]
        i = np.clip(np.rint((points[:, 0] - self.origin[0]) / CELL_M).astype(int), 0, self.shape[0] - 1)
        j = np.clip(np.rint((points[:, 1] - self.origin[1]) / CELL_M).astype(int), 0, self.shape[1] - 1)
        return bool(self.walkable[i, j].all() and not (self.room[i, j] < room).any())

    def distances(self, source: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Walking distance from a ground point to every cell, and each cell's predecessor on the way back to it."""
        return _distances(self, tuple(self.cell(source)))

    def nearest_free(self, point: Sequence[float], reach: float = 4.0) -> Optional[np.ndarray]:
        """The free cell centre nearest a ground point, within reach, or None."""
        i, j = self.cell(point)
        r = int(math.ceil(reach / CELL_M))
        window = self.walkable[max(0, i - r):i + r + 1, max(0, j - r):j + r + 1]
        found = np.argwhere(window)
        if not len(found):
            return None
        cells = found + np.array([max(0, i - r), max(0, j - r)])
        centres = self.origin + cells * CELL_M
        k = int(np.argmin(np.hypot(*(centres - np.asarray(point, dtype=float)[:2]).T)))
        return centres[k] if np.hypot(*(centres[k] - np.asarray(point, dtype=float)[:2])) <= reach else None

    def route(self, start: Sequence[float], goal: Sequence[float]) -> Optional[List[np.ndarray]]:
        """The shortest walk from start to goal as straight legs, or None when goal cannot be reached. A start or goal
        just off the free ground, a body stepping through the fence or a mark against a table, is joined to the nearest
        free cell by a short straight leg."""
        start, goal = np.asarray(start, dtype=float)[:2], np.asarray(goal, dtype=float)[:2]
        source = goal if self.free(goal) else self.nearest_free(goal)
        first = start if self.free(start) else self.nearest_free(start)
        if source is None or first is None:
            return None
        dist, prev = self.distances(source)
        i, j = self.cell(first)
        index = i * self.shape[1] + j
        if not np.isfinite(dist[index]):
            return None
        cells = [index]
        while prev[cells[-1]] >= 0:
            cells.append(int(prev[cells[-1]]))
        points = [self.point(divmod(c, self.shape[1])) for c in cells]
        points[0], points[-1] = first, source
        if not np.allclose(first, start):
            points.insert(0, start)
        if not np.allclose(source, goal):
            points.append(goal)
        # Pull the walk straight: from each kept point, jump to the furthest later point still in clear sight. Away
        # from its two ends a straightened leg keeps COMFORT_M to spare, so a body cutting a corner never brushes it.
        legs, k = [points[0]], 0
        last = len(points) - 1
        while k < last:
            far = last
            while far > k + 1 and not self.clear_line(points[k], points[far], 0.0 if k == 0 or far == last else COMFORT_M):
                far = (k + 1 + far) // 2 if far - k > 8 else far - 1
            legs.append(points[far])
            k = far
        return legs

    def walk_length(self, legs: Sequence[np.ndarray]) -> float:
        """Length of a walk given as straight legs."""
        return float(sum(np.linalg.norm(b - a) for a, b in zip(legs, legs[1:])))


def _outline(item: dict, place: dict) -> np.ndarray:
    """A placed item's bounding box from above, world metres, four corners."""
    low, high = np.array(item["bounds_min"][:2]), np.array(item["bounds_max"][:2])
    scale = np.array(place["scale"][:2], dtype=float)
    local = np.array([[low[0], low[1]], [high[0], low[1]], [high[0], high[1]], [low[0], high[1]]]) * scale
    yaw = yaw_of(place["quaternion"])
    c, s = math.cos(yaw), math.sin(yaw)
    return local @ np.array([[c, s], [-s, c]]) + np.array(place["position"][:2])


def _table(points: np.ndarray, row: int) -> Table:
    """The smallest box around a table's parts, aligned with its long side."""
    edge = points[1] - points[0]
    if np.linalg.norm(points[3] - points[0]) > np.linalg.norm(edge):
        edge = points[3] - points[0]
    along = edge / np.linalg.norm(edge)
    across = np.array([-along[1], along[0]])
    u, v = points @ along, points @ across
    centre = along * (u.max() + u.min()) / 2.0 + across * (v.max() + v.min()) / 2.0
    return Table((float(centre[0]), float(centre[1])), (float(along[0]), float(along[1])),
                 (float(across[0]), float(across[1])), float(u.max() - u.min()) / 2.0, float(v.max() - v.min()) / 2.0,
                 int(row))


def _grown(corners: np.ndarray, margin: float) -> np.ndarray:
    """A convex outline pushed out from its centre by a margin."""
    centre = corners.mean(0)
    out = corners - centre
    return centre + out * (1.0 + margin / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9))


def _graph(walkable: np.ndarray, room: np.ndarray) -> csr_matrix:
    """The walking grid as a graph: each free cell joined to its free neighbours, diagonals only past free corners, a
    step into a cell with less than COMFORT_M to spare counting CROWDED_COST times its length."""
    nx, ny = walkable.shape
    ids = np.arange(nx * ny).reshape(nx, ny)
    rows, cols, costs = [], [], []
    for di, dj in ((1, 0), (0, 1), (1, 1), (1, -1)):
        i0, i1 = 0, nx - di
        j0, j1 = max(0, -dj), ny - max(0, dj)
        a = walkable[i0:i1, j0:j1]
        b = walkable[i0 + di:i1 + di, j0 + dj:j1 + dj]
        both = a & b
        if di and dj:
            # A diagonal may not cut the corner of a blocked cell.
            both &= walkable[i0 + di:i1 + di, j0:j1] & walkable[i0:i1, j0 + dj:j1 + dj]
        src = ids[i0:i1, j0:j1][both]
        dst = ids[i0 + di:i1 + di, j0 + dj:j1 + dj][both]
        tight = np.minimum(room[i0:i1, j0:j1][both], room[i0 + di:i1 + di, j0 + dj:j1 + dj][both]) < COMFORT_M
        step = CELL_M * math.hypot(di, dj) * np.where(tight, CROWDED_COST, 1.0)
        rows += [src, dst]
        cols += [dst, src]
        costs += [step, step]
    return csr_matrix((np.concatenate(costs), (np.concatenate(rows), np.concatenate(cols))), shape=(nx * ny, nx * ny))


def _distances(site: Site, cell: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Walking distances and predecessors from one cell, kept per site for the sources a story asks about again."""
    cache = site.__dict__.setdefault("_paths", {})
    if cell not in cache:
        dist, prev = dijkstra(site._graph, directed=False, indices=cell[0] * site.shape[1] + cell[1],
                              return_predecessors=True)
        cache[cell] = (dist, prev)
    return cache[cell]


@lru_cache(maxsize=2)
def _road(asset_dir: str) -> np.ndarray:
    """Ground points along the public road outside the fence, from the path the farmer's truck drives on it."""
    manifest = builder.solar_manifest(asset_dir)
    for place in manifest["placements"]:
        if place.get("mover") == "pickup" and "path" in place:
            folder = os.path.join(asset_dir, manifest["items"][place["item"]]["folder"])
            with open(os.path.join(folder, place["path"]), encoding="utf-8") as handle:
                return np.asarray(json.load(handle)["rows"], dtype=float)[:, :2]
    return np.zeros((0, 2))


@lru_cache(maxsize=2)
def _forest(asset_dir: str) -> np.ndarray:
    """Ground positions of the map's trees by the fence, as the survey stands them."""
    manifest = builder.solar_manifest(asset_dir)
    forest = manifest.get("forest")
    if not forest:
        return np.zeros((0, 2))
    with np.load(os.path.join(asset_dir, forest["folder"], forest["table"])) as table:
        pos = np.asarray(table["position"], dtype=float)[np.asarray(table["tier"]) == FOREST_TIER, :2]
    ring = np.asarray(builder.solar_fence(asset_dir), dtype=float)
    low, high = ring.min(0) - 2 * FOREST_REACH_M, ring.max(0) + 2 * FOREST_REACH_M
    return pos[np.all((pos > low) & (pos < high), axis=1)]
