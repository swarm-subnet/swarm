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

"""What stands on an area, as the tables a field is cut against: tree crown circles, house rectangles, the cells a
field must keep clear of, and the top of the solid things that stand on open ground, all in area metres (x east and y
north of the area's centre).

A crown circle is centred on the tree's trunk and reaches the farthest vertex of its mesh, leaf or trunk, plus
CROWN_MARGIN_M, so every part of a tree lies inside its circle. House pieces share an id in their item name and are
joined into one rectangle per house, the smallest turned rectangle round their boxes. Every other building piece
(bridges, lift towers, stations, carriers and their ropes) never becomes a house: like water, its mesh only marks
the cells a field keeps clear of. Any other solid placement (stones, bushes) marks the cells its turned box reaches
with the height of its top, so a field can be held to what the drone can fly over.

Everything here is IEEE arithmetic done one operation at a time (no matrix product, no library sum, no trigonometry
in what a decision reads), so every machine builds the same tables from the same files.
"""

from __future__ import annotations

import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Sequence, Tuple

import numpy as np
from scipy.ndimage import binary_dilation

TREE_GROUP = "trees"
BUILDING_GROUP = "buildings"
WATER_GROUP = "water"
NOT_STANDING = ("terrain", "horizon", TREE_GROUP, BUILDING_GROUP, WATER_GROUP)  # every other solid group stands on the ground
NO_TOP = -1                            # a cell nothing solid stands on
# The drone's arms reach about 0.4 m from its centre, which is the point the crown rule checks.
CROWN_MARGIN_M = 0.5
HOUSE_PIECE = re.compile(r"^buildings_(vf\d+_\d+)_[a-z]+$")
SAMPLE_STEP_M = 0.5                    # samples along a mesh edge, closer than a cell so none is skipped
THIN_M = 0.5                           # a triangle no wider than this is marked by its sides alone


class Mesh(NamedTuple):
    """An OBJ mesh: its vertices and its triangles as indices into them."""

    vertices: np.ndarray
    triangles: np.ndarray


class Tables(NamedTuple):
    """An area's tables: crown circles (x, y, radius), house rows (centre x, centre y, length, width, heading in
    degrees), the cells a field keeps clear of, and the top of the solid things standing on each cell."""

    crowns: np.ndarray
    houses: np.ndarray
    keep_out: np.ndarray
    solid_top_cm: np.ndarray


def item_path(root: Path, manifest: Dict[str, Any], item: Dict[str, Any]) -> Path:
    """The mesh an item collides with: in the area folder, or in the shared folder for items marked shared."""
    base = Path(root)
    if item.get("root") == "shared":
        base = base / manifest["andorra"]["shared_root"]
    return base / item["folder"] / item["obj"]


@lru_cache(maxsize=512)
def read_mesh(path: str) -> Mesh:
    """The vertices and triangles of an OBJ file; faces with more corners are split into a fan."""
    lines = Path(path).read_text().splitlines()
    vertices = np.array(" ".join(line[2:] for line in lines if line.startswith("v ")).split(), dtype=float)
    faces = [[int(token.split("/", 1)[0]) - 1 for token in line[2:].split()] for line in lines if line.startswith("f ")]
    triangles = [(face[0], face[k], face[k + 1]) for face in faces for k in range(1, len(face) - 1)]
    return Mesh(vertices.reshape(-1, 3), np.array(triangles, dtype=np.int64).reshape(-1, 3))


@lru_cache(maxsize=512)
def item_reach_m(path: str) -> float:
    """How far the farthest vertex of a mesh lies from its own vertical axis, metres."""
    vertices = read_mesh(path).vertices
    return float(np.sqrt(np.max(vertices[:, 0] * vertices[:, 0] + vertices[:, 1] * vertices[:, 1])))


def rotation(quaternion: Sequence[float]) -> np.ndarray:
    """The rotation matrix of an XYZW quaternion, entry by entry."""
    x, y, z, w = (float(v) for v in quaternion)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def placed(points: np.ndarray, place: Dict[str, Any]) -> np.ndarray:
    """Item-frame points scaled, turned and moved to where a placement stands, x and y in area metres."""
    scaled = np.asarray(points, dtype=float) * np.asarray(place["scale"], dtype=float)
    turn = rotation(place["quaternion"])
    x, y, z = scaled[:, 0], scaled[:, 1], scaled[:, 2]
    east = turn[0, 0] * x + turn[0, 1] * y + turn[0, 2] * z + float(place["position"][0])
    north = turn[1, 0] * x + turn[1, 1] * y + turn[1, 2] * z + float(place["position"][1])
    return np.column_stack([east, north])


def crown_circles(root: Path, manifest: Dict[str, Any]) -> np.ndarray:
    """East, north and radius, area metres, of a circle round every tree, in placement order."""
    items, rows = manifest["items"], []
    for place in manifest["placements"]:
        item = items[place["item"]]
        if item["group"] != TREE_GROUP:
            continue
        reach = item_reach_m(str(item_path(root, manifest, item)))
        # Trees only turn about the vertical, so the wider of the two horizontal scales bounds the turned crown.
        radius = reach * max(float(place["scale"][0]), float(place["scale"][1])) + CROWN_MARGIN_M
        rows.append([float(place["position"][0]), float(place["position"][1]), radius])
    return np.asarray(rows, dtype=float).reshape(-1, 3)


def hull(points: np.ndarray) -> np.ndarray:
    """The convex hull of 2D points, anticlockwise, by the monotone chain."""
    pts = sorted(set(map(tuple, np.asarray(points, dtype=float).tolist())))
    if len(pts) < 3:
        return np.asarray(pts, dtype=float).reshape(-1, 2)

    def half(sequence: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """One side of the hull, turning left all the way."""
        chain: List[Tuple[float, float]] = []
        for point in sequence:
            while len(chain) >= 2 and ((chain[-1][0] - chain[-2][0]) * (point[1] - chain[-2][1])
                                       - (chain[-1][1] - chain[-2][1]) * (point[0] - chain[-2][0])) <= 0:
                chain.pop()
            chain.append(point)
        return chain

    lower, upper = half(pts), half(pts[::-1])
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def smallest_rectangle(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """The smallest turned rectangle round 2D points: its row (centre east, centre north, length, width and compass
    heading of the long side in degrees, 0 to 180) and its four corners, anticlockwise."""
    ring = hull(points)
    best = None
    for start, end in zip(ring, np.roll(ring, -1, axis=0)):
        edge = end - start
        size = math.sqrt(edge[0] * edge[0] + edge[1] * edge[1])
        if size == 0.0:
            continue
        along = edge / size
        across = np.array([-along[1], along[0]])
        u = ring[:, 0] * along[0] + ring[:, 1] * along[1]
        v = ring[:, 0] * across[0] + ring[:, 1] * across[1]
        area = float(u.max() - u.min()) * float(v.max() - v.min())
        if best is None or area < best[0]:
            best = (area, along, across, (float(u.min()), float(u.max())), (float(v.min()), float(v.max())))
    _area, along, across, (u0, u1), (v0, v1) = best
    corners = np.array([along * u + across * v for u, v in ((u0, v0), (u1, v0), (u1, v1), (u0, v1))])
    centre = along * ((u0 + u1) / 2.0) + across * ((v0 + v1) / 2.0)
    spans = (u1 - u0, v1 - v0)
    long_axis = along if spans[0] >= spans[1] else across
    heading = math.degrees(math.atan2(long_axis[0], long_axis[1])) % 180.0
    # Rounded so a last-digit difference in a library's arctangent never reaches the site map.
    return np.round(np.array([centre[0], centre[1], max(spans), min(spans), heading]), 4), corners


def houses(root: Path, manifest: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """One row and four corners per house, round the mesh vertices of all its pieces, which share the id in their
    item names; sorted by that id."""
    items, pieces = manifest["items"], {}
    for place in manifest["placements"]:
        match = HOUSE_PIECE.match(place["item"])
        item = items[place["item"]]
        if item["group"] != BUILDING_GROUP or match is None:
            continue
        mesh = read_mesh(str(item_path(root, manifest, item)))
        if len(mesh.vertices):
            pieces.setdefault(match.group(1), []).append(placed(mesh.vertices, place))
    found = [smallest_rectangle(np.concatenate(pieces[key])) for key in sorted(pieces)]
    return (np.asarray([row for row, _ in found], dtype=float).reshape(-1, 5),
            np.asarray([corners for _, corners in found], dtype=float).reshape(-1, 4, 2))


def mark(grid: np.ndarray, points: np.ndarray) -> None:
    """Mark the cells holding area-metre points; points off the grid are ignored."""
    rows, cols = grid.shape
    row = np.floor(rows / 2.0 - points[:, 1]).astype(np.int64)
    col = np.floor(points[:, 0] + cols / 2.0).astype(np.int64)
    keep = (row >= 0) & (row < rows) & (col >= 0) & (col < cols)
    grid[row[keep], col[keep]] = True


def _runs(counts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """For runs of the given lengths laid end to end: which run each position belongs to, and its index in it."""
    owner = np.repeat(np.arange(len(counts)), counts)
    starts = np.cumsum(counts) - counts
    return owner, np.arange(int(counts.sum())) - starts[owner]


def mark_triangles(grid: np.ndarray, corners: np.ndarray) -> None:
    """Mark the cells a set of triangles (n, 3, 2) covers, by points no farther than SAMPLE_STEP_M apart: a lattice
    over each triangle, or its three sides alone when it is no wider than THIN_M. Every point of a triangle then lies
    within a metre of a marked point, so growing the marks by one cell covers every cell the triangle touches."""
    corners = np.asarray(corners, dtype=float).reshape(-1, 3, 2)
    a, b, c = corners[:, 0], corners[:, 1], corners[:, 2]
    sides = [(a, b - a), (b, c - b), (c, a - c)]
    lengths = [np.sqrt(v[:, 0] * v[:, 0] + v[:, 1] * v[:, 1]) for _, v in sides]
    longest = np.maximum(np.maximum(lengths[0], lengths[1]), lengths[2])
    ab, ac = b - a, c - a
    twice_area = np.abs(ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0])
    thin = twice_area <= THIN_M * longest
    for (start, vector), length in zip(sides, lengths):
        start, vector, steps = start[thin], vector[thin], np.ceil(length[thin] / SAMPLE_STEP_M).astype(np.int64) + 1
        owner, k = _runs(steps + 1)
        t = (k / steps[owner])[:, None]
        mark(grid, start[owner] + t * vector[owner])
    a, ab, ac = a[~thin], ab[~thin], ac[~thin]
    steps = np.ceil(longest[~thin] / SAMPLE_STEP_M).astype(np.int64) + 1
    owner, k = _runs((steps + 1) * (steps + 1))
    i, j = k // (steps[owner] + 1), k % (steps[owner] + 1)
    keep = i + j <= steps[owner]
    owner, u, v = owner[keep], (i[keep] / steps[owner[keep]])[:, None], (j[keep] / steps[owner[keep]])[:, None]
    mark(grid, a[owner] + u * ab[owner] + v * ac[owner])


def keep_out(root: Path, manifest: Dict[str, Any], shape: Tuple[int, int]) -> np.ndarray:
    """The cells a field keeps clear of, in the survey grid's layout (rows north to south): the mesh of every
    building piece and every water surface, grown by one cell so a cell the samples skip across is still marked."""
    grid = np.zeros(shape, dtype=bool)
    items = manifest["items"]
    for place in manifest["placements"]:
        item = items[place["item"]]
        if item["group"] not in (BUILDING_GROUP, WATER_GROUP):
            continue
        mesh = read_mesh(str(item_path(root, manifest, item)))
        if len(mesh.triangles):
            mark_triangles(grid, placed(mesh.vertices, place)[mesh.triangles])
    return binary_dilation(grid, structure=np.ones((3, 3), dtype=bool))


def solid_tops(manifest: Dict[str, Any], shape: Tuple[int, int], z_shift: float = 0.0) -> np.ndarray:
    """For each cell, in the survey grid's layout, the top of the highest solid thing standing on it that is not a
    tree, a building or water, in whole centimetres above the survey's base (world z minus z_shift), rounded up; NO_TOP
    where nothing stands. A thing's turned and scaled box marks every cell it reaches, grown by one cell."""
    grid = np.full(shape, NO_TOP, dtype=np.int64)
    rows, cols = shape
    items = manifest["items"]
    for place in manifest["placements"]:
        item = items[place["item"]]
        if item["group"] in NOT_STANDING or item.get("collision") == "none" or place.get("collision") == "none":
            continue
        low, high = item["bounds_min"], item["bounds_max"]
        corners = np.array([(x, y, z) for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])],
                           dtype=float)
        xy = placed(corners, place)
        scaled = corners * np.asarray(place["scale"], dtype=float)
        turn = rotation(place["quaternion"])
        z = turn[2, 0] * scaled[:, 0] + turn[2, 1] * scaled[:, 1] + turn[2, 2] * scaled[:, 2] + float(place["position"][2])
        top = int(math.ceil((float(z.max()) - z_shift) * 100.0))
        r0 = max(0, int(math.floor(rows / 2.0 - float(xy[:, 1].max()))) - 1)
        r1 = min(rows - 1, int(math.floor(rows / 2.0 - float(xy[:, 1].min()))) + 1)
        c0 = max(0, int(math.floor(float(xy[:, 0].min()) + cols / 2.0)) - 1)
        c1 = min(cols - 1, int(math.floor(float(xy[:, 0].max()) + cols / 2.0)) + 1)
        if r0 <= r1 and c0 <= c1:
            np.maximum(grid[r0:r1 + 1, c0:c1 + 1], top, out=grid[r0:r1 + 1, c0:c1 + 1])
    return grid


def tables(root: Path, manifest: Dict[str, Any], shape: Tuple[int, int], z_shift: float = 0.0) -> Tables:
    """An area's crown circles, house rows, keep-out cells and solid tops, read from its manifest and meshes; z_shift
    turns a survey height into a world z."""
    return Tables(crown_circles(root, manifest), houses(root, manifest)[0], keep_out(root, manifest, shape),
                  solid_tops(manifest, shape, z_shift))
