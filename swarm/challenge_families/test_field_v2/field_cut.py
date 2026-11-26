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

"""The field cutter: a seed's field, cut from the open ground of one real Andorra area.

The seed draws the field's size first, then one of the areas that can hold that size, then a shape: one of four
families (a rough quadrilateral, a polygon round its centre, a smooth blob, an L), turned any way and centred on
open ground.
The shape is laid on the area's 1 m grid and trimmed to its open ground, so the field's edges follow the tree lines,
walls, roads and banks that end the open ground there. The largest piece is kept, its holes filled (a clump of trees
inside stays inside, as crowns), its border traced and simplified to at most MAX_POINTS corners, and the result must
pass every rule below; a failed try draws again on the same stream, then moves to the next area that can hold the
size. A seed never gets a smaller field than it asked for: when no area yields one, the cut fails.

A field is one of two kinds, each cut by its own recipe. An open field is trimmed to the open ground grown by 3 m, as
above. A wooded field may reach 12 m past the open ground, into the trees: less of it need be open and more of it may
lie under crowns, it is centred where less of the ground round it is open, and at least WOODED.least_crowns tree
centres must lie inside its outline. Every other rule is the same. A seed draws its kind on a stream of its own,
wooded in WOODED_SHARE of seeds, and tries a wooded field only in the areas the wooded size table lists, on shape
draws of its own; when none yields one it takes the open field, the very one it would have had as an open seed.

Every decision is exact. The grid, the traced outline and its corners are whole metres; tests on them are integer
arithmetic (sums, products and comparisons of integers, Python's own big integers where a product could overflow);
the binary morphology and labelling of scipy.ndimage are exact on booleans; the shape's corners are drawn with
IEEE operations done one at a time (no library sum, no trigonometry: directions come from a table of integers), so
the same seed cuts the same outline, byte for byte, on every machine.

Areas are passed in as plain arrays (Area): the cutter does not know how they are loaded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import binary_closing, binary_dilation, binary_erosion, binary_fill_holes, binary_opening, grey_dilation, grey_erosion, label

NO_DATA_CM = 65535
TODAY_M2 = 2755                        # the real field's surveyed area
SIZE_RANGE_TENTHS = (5, 15)            # fields from half to one and a half times TODAY_M2
SIZE_BINS = 10                         # the size table's bins across that range, a tenth of TODAY_M2 each
SIZE_TOLERANCE = 0.10                  # a cut field lands this close to the size drawn
MAX_POINTS = 64                        # the site map's outline slots
MAX_ACROSS_M = 95                      # a report names a point at most 100 m from the dock
CORE_MARGIN_M = 30                     # scenery on every side; the grids carry a 10 m border of no use
OPEN_GROWTH_M = 3                      # gives back the margin the open layer keeps: today's field's patch then holds 2,517 of its 2,755 m2
MIN_PATCH_CELLS = 10                   # open patches smaller than this neither grow nor keep two others apart
KEEP_OUT_M = 2                         # a field stays this far from houses, other structures and water
SLIVER_M = 3                           # opening radius: strips narrower than twice this are cut off
NOTCH_M = 3                            # closing radius: slits and bays narrower than twice this are filled
WINDOW_PAD = NOTCH_M + 2               # cells round a shape's box, so nothing the closing grows reaches the box's side
EDGE_BAND_M = OPEN_GROWTH_M            # the open share is read inside the band the growth gave back
MIN_OPEN_PERCENT = 75                  # today's field: 77 % of it is open ground, its edge band left out
MAX_RELIEF_CM = 800                    # under the 10 m ceiling the drone must still clear the field's top
MAX_SLOPE_TAN = Fraction(0.21255656167002213)    # tan 12 degrees, the plane fitted through the whole field
MAX_CROWNS = 64                        # the site map's table rows
MAX_CROWN_PERCENT = 20                 # merged crowns may cover at most this share of the field
WOODED_SHARE = Fraction(3, 5)          # the share of seeds that ask for a field reaching into the trees
ARM_M = 6                              # an arm narrower than twice this is too thin to fly lanes along
MAX_ARM_CELLS = 60                     # what an opening at ARM_M may take off one place: a corner, never an arm
DOCK_INSIDE_M = 5                      # the dock's margin inside the outline
DOCK_CLEAR_CM = 340                    # the dock's clearance from every crown
DOCK_SLOPE_TAN = Fraction(0.17632698070846498)   # tan 10 degrees, the dock's own limit
DOCK_RISE_CM = 600                     # the field's highest ground at most this far above the dock
# The top of any stone or bush on the field at most this far above the dock: 1.5 m under the 9.5 m the drone cruises
# at and 2 m under its 10 m ceiling, so the drone can always fly over it. Without it 9 of 400 seeds' fields let the dock
# stand more than 8 m under such a top, up to 9.87 m; with it those 9 keep their outline and lose only dock cells.
DOCK_SOLID_RISE_CM = 800
MIN_DOCK_CELLS = 10
GROWTH = (1.0, 1.15, 1.32, 1.52, 1.75)  # the shape grows through these until its trimmed piece is big enough
TRIES_PER_AREA = 300
CENTRE_OPEN_PERCENT = 60               # a shape is centred where at least this share of its neighbourhood is open
TOLERANCES = ((5, 4), (8, 5), (2, 1), (5, 2), (16, 5), (4, 1))  # outline simplification, metres as fractions
FAMILIES = ("quad", "polygon", "blob", "ell")

SIZE_SEED_STREAM = 0xF1E1D             # the size's own stream
AREA_SEED_STREAM = 0xF1E1E             # the area's own stream
SHAPE_SEED_STREAM = 0xF1E1F            # the shape's own stream, every try of every area of a seed
PROBE_SEED_STREAM = 0xF1E20            # the size table's probes, never a seed's
KIND_SEED_STREAM = 0xF1E21             # the field kind's own stream

# A quarter turn of unit directions every 7.5 degrees, as integers (10,000 cos, 10,000 sin), so no shape reads a
# library's trigonometry; the other quarters follow by exact turns.
_QUARTER = ((10000, 0), (9914, 1305), (9659, 2588), (9239, 3827), (8660, 5000), (7934, 6088), (7071, 7071),
            (6088, 7934), (5000, 8660), (3827, 9239), (2588, 9659), (1305, 9914))
_TURNS = [(x, y) for x, y in _QUARTER] + [(-y, x) for x, y in _QUARTER] + [(-x, -y) for x, y in _QUARTER] + \
         [(y, -x) for x, y in _QUARTER]
DIRECTIONS = np.array(_TURNS, dtype=float) / np.sqrt(np.array([x * x + y * y for x, y in _TURNS], dtype=float))[:, None]


class NoField(RuntimeError):
    """No area yields a field of the size the seed asks for."""


@dataclass(frozen=True)
class Recipe:
    """How one kind of field is cut: how far the open ground grows into the ground round it, the least share of the
    field that must be open, the most that crowns may cover, how open the ground round a shape's centre must be, and
    how many tree centres must lie inside the outline; shares in percent. Its number tells its draws apart from
    the open recipe's on the area, shape and probe streams, so an open field is drawn as it always was."""

    kind: str
    number: int
    growth_m: int
    open_percent: int
    crown_percent: int
    centre_open_percent: int
    least_crowns: int


OPEN = Recipe("open", 0, OPEN_GROWTH_M, MIN_OPEN_PERCENT, MAX_CROWN_PERCENT, CENTRE_OPEN_PERCENT, 0)
WOODED = Recipe("wooded", 1, 12, 45, 35, 35, 4)
RECIPES = {recipe.kind: recipe for recipe in (OPEN, WOODED)}


def streams(recipe: Recipe, *words: int) -> List[int]:
    """The words of a stream for a recipe's draws: the open recipe's as they always were, another's with its number
    added."""
    return [int(word) for word in words] + ([recipe.number] if recipe.number else [])


def disk(radius: int) -> np.ndarray:
    """The cells within a whole number of metres of a centre cell, as a structuring element."""
    r = np.arange(-radius, radius + 1)
    return (r[:, None] * r[:, None] + r[None, :] * r[None, :]) <= radius * radius


def grow_apart(open_ground: np.ndarray, radius: int) -> np.ndarray:
    """Open ground grown by a radius, each patch on its own: a cell within reach of two patches stays out, so a wall,
    a hedge or a path between two meadows still parts them."""
    patches, count = label(open_ground, structure=np.ones((3, 3), dtype=bool))
    sizes = np.bincount(patches.ravel())
    sizes[0] = 0
    patches = np.where((sizes >= MIN_PATCH_CELLS)[patches], patches, 0)
    highest = grey_dilation(patches, footprint=disk(radius), mode="constant", cval=0)
    lowest = grey_erosion(np.where(patches > 0, patches, count + 1), footprint=disk(radius), mode="constant", cval=count + 1)
    return (highest > 0) & (lowest == highest)


class Area:
    """One area as the cutter reads it: the 1 m grids (rows north to south, an even number of each) and its tables.

    open: the open-ground layer; ground_cm: ground height in centimetres, NO_DATA_CM where unknown; keep_out: cells
    of houses, other structures and water; crowns: (n, 3) east, north and radius in area metres; houses: (n, 5) the
    site map's house rows; dock: the dock layer, or None for a package without one; solid_top_cm: the top of the
    solid things standing on each cell that are not trees, in the ground's centimetres, negative where none stands, or
    None for an area with none.
    """

    def __init__(self, name: str, open_ground: np.ndarray, ground_cm: np.ndarray, keep_out: np.ndarray,
                 crowns: np.ndarray, houses: np.ndarray, dock: Optional[np.ndarray] = None,
                 solid_top_cm: Optional[np.ndarray] = None):
        """Keep the arrays and work out once what every cut of this area reads."""
        self.name = str(name)
        self.open = np.asarray(open_ground, dtype=bool)
        rows, cols = self.open.shape
        if rows % 2 or cols % 2:
            raise ValueError(f"area {name}: the grid needs an even number of rows and columns, not {rows} x {cols}")
        self.ground_cm = np.asarray(ground_cm).astype(np.int64)
        self.no_data = self.ground_cm == NO_DATA_CM
        self.keep_out = np.asarray(keep_out, dtype=bool)
        self.near_keep_out = binary_dilation(self.keep_out, structure=disk(1))
        self.crowns = np.asarray(crowns, dtype=float).reshape(-1, 3)
        # Centres to the nearest centimetre, radii rounded up, so a circle never shrinks.
        self.crowns_cm = np.column_stack([np.rint(self.crowns[:, :2] * 100.0),
                                          np.ceil(self.crowns[:, 2] * 100.0 - 1e-6) + 1.0]).astype(np.int64)
        self.houses = np.asarray(houses, dtype=float).reshape(-1, 5)
        self.dock = np.asarray(dock, dtype=bool) if dock is not None else binary_erosion(self.open, structure=disk(2))
        self.solid_top_cm = (np.full((rows, cols), -1, dtype=np.int64) if solid_top_cm is None
                             else np.asarray(solid_top_cm).astype(np.int64))
        core = np.zeros_like(self.open)
        core[CORE_MARGIN_M:rows - CORE_MARGIN_M, CORE_MARGIN_M:cols - CORE_MARGIN_M] = True
        # Where a field may reach at all: inside the core, on known ground, KEEP_OUT_M from houses and water.
        self.room = ~binary_dilation(self.keep_out, structure=disk(KEEP_OUT_M)) & ~self.no_data & core
        self._free: Dict[int, np.ndarray] = {}
        self.open_sum = np.pad(np.cumsum(np.cumsum(self.open.astype(np.int64), axis=0), axis=1), ((1, 0), (1, 0)))
        self._centres: Dict[Tuple[int, int, int], np.ndarray] = {}

    @property
    def shape(self) -> Tuple[int, int]:
        """Rows and columns of the grids."""
        return self.open.shape

    @property
    def free(self) -> np.ndarray:
        """Where an open field may reach: the open ground grown by OPEN_GROWTH_M, inside the room."""
        return self.free_for(OPEN)

    def free_for(self, recipe: Recipe) -> np.ndarray:
        """Where a field of a recipe may reach: the open ground grown by the recipe's growth, inside the room; worked
        out once an area the first time a recipe asks."""
        if recipe.growth_m not in self._free:
            self._free[recipe.growth_m] = grow_apart(self.open, recipe.growth_m) & self.room
        return self._free[recipe.growth_m]

    def centres(self, size_m2: int, recipe: Recipe = OPEN) -> np.ndarray:
        """Flat indices of the open cells a recipe's field may reach whose neighbourhood, a square of side
        0.8 x sqrt(size), is at least the recipe's centre share open."""
        side = max(3, int(math.isqrt(int(size_m2) * 64) // 10))
        key = (side, recipe.growth_m, recipe.centre_open_percent)
        if key not in self._centres:
            rows, cols = self.shape
            half = side // 2
            r0 = np.clip(np.arange(rows) - half, 0, rows)
            r1 = np.clip(np.arange(rows) - half + side, 0, rows)
            c0 = np.clip(np.arange(cols) - half, 0, cols)
            c1 = np.clip(np.arange(cols) - half + side, 0, cols)
            s = self.open_sum
            count = s[r1][:, c1] - s[r0][:, c1] - s[r1][:, c0] + s[r0][:, c0]
            good = self.free_for(recipe) & self.open & (count * 100 >= recipe.centre_open_percent * side * side)
            self._centres[key] = np.flatnonzero(good)
        return self._centres[key]


@dataclass(frozen=True)
class Field:
    """A cut field: the area it lies in, its outline (area metres, anticlockwise), its area, the crown circles that
    touch it (merged when more than MAX_CROWNS), the houses near it, the cells' centres where a dock may stand, how
    it was drawn, its kind ("open", or "wooded" when it reaches into the trees) and the crown circle of every tree
    that touches it, never merged."""

    area: str
    outline: np.ndarray
    area_m2: float
    crowns: np.ndarray
    houses: np.ndarray
    dock_spots: np.ndarray
    family: str
    target_m2: int
    tries: int
    kind: str = OPEN.kind
    trees: Optional[np.ndarray] = None


class _Reject(Exception):
    """A try broke a rule; the reason names it."""


# ---- the seed's draws ----------------------------------------------------------------------------------------------

def size_limits() -> Tuple[float, float]:
    """The smallest and largest field area, square metres."""
    return SIZE_RANGE_TENTHS[0] * TODAY_M2 / 10.0, SIZE_RANGE_TENTHS[1] * TODAY_M2 / 10.0


def size_bin(size_m2: int) -> int:
    """The size table's bin a whole number of square metres falls in."""
    return min(SIZE_BINS - 1, max(0, (10 * int(size_m2) - SIZE_RANGE_TENTHS[0] * TODAY_M2) // TODAY_M2))


def table_holds(table: Mapping[str, Sequence[bool]], name: str, which: int) -> bool:
    """Whether the size table says an area holds a size bin; an area it does not list holds none."""
    row = list(table.get(name, ()))
    return which < len(row) and bool(row[which])


def draw_kind(seed: int) -> str:
    """The kind of field the seed asks for: "wooded" in WOODED_SHARE of seeds, else "open"."""
    draw = int(np.random.default_rng([KIND_SEED_STREAM, int(seed)]).integers(WOODED_SHARE.denominator))
    return WOODED.kind if draw < WOODED_SHARE.numerator else OPEN.kind


def draw_size(seed: int, min_m2: float, max_m2: float) -> int:
    """The field size the seed asks for, whole square metres, uniform between the smallest and largest allowed."""
    rng = np.random.default_rng([SIZE_SEED_STREAM, int(seed)])
    low, high = int(math.ceil(min_m2)), int(math.floor(max_m2))
    return int(rng.integers(low, high + 1))


def unit_vector(rng: np.random.Generator) -> np.ndarray:
    """A direction drawn uniformly, by a point drawn in the unit disc and scaled to length one."""
    while True:
        u, v = rng.uniform(-1.0, 1.0, size=2)
        q = u * u + v * v
        if 0.04 <= q <= 1.0:
            return np.array([u, v]) / math.sqrt(q)


def template(rng: np.random.Generator) -> Tuple[str, np.ndarray]:
    """A shape family and its corners, anticlockwise about the origin, before scaling and turning."""
    family = FAMILIES[int(rng.integers(len(FAMILIES)))]
    if family == "quad":
        aspect = float(rng.uniform(1.0, 2.5))
        half_u, half_v = math.sqrt(aspect) / 2.0, 1.0 / (2.0 * math.sqrt(aspect))
        corners = np.array([[-half_u, -half_v], [half_u, -half_v], [half_u, half_v], [-half_u, half_v]])
        corners = corners + rng.uniform(-0.12, 0.12, size=(4, 2)) * np.array([2.0 * half_u, 2.0 * half_v])
        return family, corners
    if family == "polygon":
        count = int(rng.integers(5, 9))
        # One corner in each of count equal turns, so the polygon always wraps its centre.
        starts = [len(DIRECTIONS) * k // count for k in range(count + 1)]
        picks = [int(rng.integers(starts[k], starts[k + 1])) for k in range(count)]
        corners = DIRECTIONS[picks] * rng.uniform(0.8, 1.2, size=(count, 1))
    elif family == "blob":
        amplitude = rng.uniform(0.0, 0.12, size=3)
        phases = [unit_vector(rng) for _ in range(3)]
        steps = np.arange(0, len(DIRECTIONS), 2)
        radius = np.ones(len(steps))
        for k, (amp, phase) in enumerate(zip(amplitude, phases), start=2):
            turned = DIRECTIONS[(k * steps) % len(DIRECTIONS)]
            radius = radius + amp * (turned[:, 0] * phase[0] + turned[:, 1] * phase[1])
        corners = DIRECTIONS[steps] * radius[:, None]
    else:
        aspect = float(rng.uniform(1.0, 2.0))
        hu, hv = math.sqrt(aspect) / 2.0, 1.0 / (2.0 * math.sqrt(aspect))
        cut_u, cut_v = rng.uniform(0.3, 0.5, size=2)
        notch_u, notch_v = hu - 2.0 * hu * cut_u, hv - 2.0 * hv * cut_v
        return family, np.array([[-hu, -hv], [hu, -hv], [hu, notch_v], [notch_u, notch_v], [notch_u, hv], [-hu, hv]])
    aspect = float(rng.uniform(1.0, 2.0))
    return family, corners * np.array([math.sqrt(aspect), 1.0 / math.sqrt(aspect)])


def shoelace(corners: np.ndarray) -> float:
    """Area of a polygon from its corners, summed exactly."""
    x, y = corners[:, 0].tolist(), corners[:, 1].tolist()
    n = len(x)
    return 0.5 * math.fsum(x[k] * y[(k + 1) % n] - x[(k + 1) % n] * y[k] for k in range(n))


# ---- raster helpers ------------------------------------------------------------------------------------------------

def cells_in(corners: np.ndarray, top: int, left: int, rows: int, cols: int, shape: Tuple[int, int]) -> np.ndarray:
    """Which cells of a window have their centre inside a polygon given in area metres."""
    total_rows, total_cols = shape
    r, c = np.mgrid[top:top + rows, left:left + cols]
    x, y = c + 0.5 - total_cols / 2.0, total_rows / 2.0 - r - 0.5
    inside = np.zeros((rows, cols), dtype=bool)
    for a, b in zip(corners, np.roll(corners, -1, axis=0)):
        if a[1] == b[1]:
            continue
        crosses = (a[1] > y) != (b[1] > y)
        at = a[0] + (y - a[1]) * ((b[0] - a[0]) / (b[1] - a[1]))
        inside ^= crosses & (x < at)
    return inside


def largest_piece(mask: np.ndarray) -> np.ndarray:
    """The largest 4-connected piece of a mask, the first in raster order on a tie."""
    labels, count = label(mask)
    if count == 0:
        return mask
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(np.argmax(sizes))


def close_pinches(mask: np.ndarray) -> np.ndarray:
    """Fill the two empty cells of every 2 x 2 block whose full cells touch only at a corner, so the border is one
    simple ring."""
    mask = mask.copy()
    while True:
        a, b, c, d = mask[:-1, :-1], mask[:-1, 1:], mask[1:, :-1], mask[1:, 1:]
        pinch = (a & d & ~b & ~c) | (b & c & ~a & ~d)
        if not pinch.any():
            return mask
        rows, cols = np.nonzero(pinch)
        mask[rows, cols] = mask[rows, cols + 1] = mask[rows + 1, cols] = mask[rows + 1, cols + 1] = True


# ---- outline -------------------------------------------------------------------------------------------------------

def trace(mask: np.ndarray) -> List[Tuple[int, int]]:
    """The border of a single piece without holes or pinches, as grid corners (row line, column line), anticlockwise
    on the map (x east, y north), only where the border turns."""
    m = np.pad(mask, 1)
    edges = {}
    inner = m[1:-1, 1:-1]
    for (dr, dc), (start, end) in {(1, 0): ((1, 0), (1, 1)), (0, 1): ((1, 1), (0, 1)),
                                   (-1, 0): ((0, 1), (0, 0)), (0, -1): ((0, 0), (1, 0))}.items():
        side = inner & ~m[1 + dr:m.shape[0] - 1 + dr, 1 + dc:m.shape[1] - 1 + dc]
        for r, c in zip(*np.nonzero(side)):
            edges[(int(r) + start[0], int(c) + start[1])] = (int(r) + end[0], int(c) + end[1])
    first = min(edges)
    ring, point = [first], edges[first]
    while point != first:
        ring.append(point)
        point = edges[point]
    if len(ring) != len(edges):
        raise _Reject("outline")
    turning = []
    for k, point in enumerate(ring):
        before, after = ring[k - 1], ring[(k + 1) % len(ring)]
        if (point[0] - before[0], point[1] - before[1]) != (after[0] - point[0], after[1] - point[1]):
            turning.append(point)
    return turning


def simplify(ring: List[Tuple[int, int]], tolerance: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Douglas-Peucker on a closed ring of whole-metre corners, a corner kept when it lies farther than the tolerance
    (num, den metres) from the chord; exact integer tests."""
    num, den = tolerance
    n = len(ring)
    anchor = 0
    far = max(range(n), key=lambda k: ((ring[k][0] - ring[0][0]) ** 2 + (ring[k][1] - ring[0][1]) ** 2, -k))
    keep = {anchor, far}
    stack = [(anchor, far), (far, anchor + n)]
    while stack:
        i, j = stack.pop()
        if j - i < 2:
            continue
        a, b = ring[i % n], ring[j % n]
        er, ec = b[0] - a[0], b[1] - a[1]
        length2 = er * er + ec * ec
        best, best_k = -1, -1
        for k in range(i + 1, j):
            p = ring[k % n]
            cross = er * (p[1] - a[1]) - ec * (p[0] - a[0])
            if cross * cross > best:
                best, best_k = cross * cross, k
        if best * den * den > num * num * length2:
            keep.add(best_k % n)
            stack.extend([(i, best_k), (best_k, j)])
    return [ring[k] for k in sorted(keep)]


def is_simple(ring: List[Tuple[int, int]]) -> bool:
    """Whether a closed ring of integer corners has no two sides that meet, other than neighbours at their shared
    corner, and no side that folds back on the one before it."""
    n = len(ring)
    if n < 3:
        return False
    p = np.array(ring, dtype=np.int64)
    q = np.roll(p, -1, axis=0)
    d = q - p
    before = np.roll(d, 1, axis=0)
    if np.any((before[:, 0] * d[:, 1] - before[:, 1] * d[:, 0] == 0) & ((before * d).sum(axis=1) < 0)):
        return False
    i, j = np.triu_indices(n, k=2)
    keep = ~((i == 0) & (j == n - 1))
    i, j = i[keep], j[keep]

    def orient(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        """Sign of the turn a, b, c."""
        return np.sign((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))

    def between(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        """Whether c, on the line a b, lies within the segment's box."""
        return ((np.minimum(a[:, 0], b[:, 0]) <= c[:, 0]) & (c[:, 0] <= np.maximum(a[:, 0], b[:, 0]))
                & (np.minimum(a[:, 1], b[:, 1]) <= c[:, 1]) & (c[:, 1] <= np.maximum(a[:, 1], b[:, 1])))

    a, b, c, e = p[i], q[i], p[j], q[j]
    o1, o2, o3, o4 = orient(a, b, c), orient(a, b, e), orient(c, e, a), orient(c, e, b)
    meet = ((o1 != o2) & (o3 != o4) & (o1 != 0) & (o2 != 0) & (o3 != 0) & (o4 != 0))
    meet |= (o1 == 0) & between(a, b, c)
    meet |= (o2 == 0) & between(a, b, e)
    meet |= (o3 == 0) & between(c, e, a)
    meet |= (o4 == 0) & between(c, e, b)
    return not bool(meet.any())


def outline_of(piece: np.ndarray) -> List[Tuple[int, int]]:
    """The piece's border simplified with the smallest tolerance that leaves at most MAX_POINTS simple corners."""
    ring = trace(piece)
    for tolerance in TOLERANCES:
        corners = simplify(ring, tolerance)
        if len(corners) <= MAX_POINTS:
            if not is_simple(corners):
                raise _Reject("outline")
            return corners
    raise _Reject("points")


def polygon_cells(ring: np.ndarray, shape: Tuple[int, int]) -> Tuple[np.ndarray, int, int]:
    """Which cells have their centre inside a ring of grid corners (row line, column line), as a mask over the
    ring's box with that box's top row and left column; exact, on doubled integer coordinates."""
    top, left = int(ring[:, 0].min()), int(ring[:, 1].min())
    bottom, right = int(ring[:, 0].max()), int(ring[:, 1].max())
    r, c = np.mgrid[top:bottom, left:right]
    pr, pc = 2 * r + 1, 2 * c + 1
    inside = np.zeros(r.shape, dtype=bool)
    for (r1, c1), (r2, c2) in zip(ring.tolist(), np.roll(ring, -1, axis=0).tolist()):
        if r1 == r2:
            continue
        a_r, a_c, b_r, b_c = 2 * r1, 2 * c1, 2 * r2, 2 * c2
        crosses = (a_r > pr) != (b_r > pr)
        # Column of the side at the centre's row is a_c + (pr - a_r) (b_c - a_c) / (b_r - a_r); compare unscaled.
        lhs, rhs = (pc - a_c) * (b_r - a_r), (pr - a_r) * (b_c - a_c)
        inside ^= crosses & ((lhs < rhs) if b_r > a_r else (lhs > rhs))
    return inside, top, left


def far_from_edges(ring: np.ndarray, rows: np.ndarray, cols: np.ndarray, metres: int) -> np.ndarray:
    """Which cell centres lie at least a whole number of metres from every side of a ring of grid corners; exact."""
    pr, pc = (2 * rows + 1)[:, None], (2 * cols + 1)[:, None]
    a = 2 * ring
    b = np.roll(a, -1, axis=0)
    er, ec = (b[:, 0] - a[:, 0])[None, :], (b[:, 1] - a[:, 1])[None, :]
    wr, wc = pr - a[None, :, 0], pc - a[None, :, 1]
    dot, length2 = wr * er + wc * ec, er * er + ec * ec
    limit = (2 * metres) ** 2
    to_start = wr * wr + wc * wc
    to_end = (pr - b[None, :, 0]) ** 2 + (pc - b[None, :, 1]) ** 2
    cross = wr * ec - wc * er
    clear = np.where(dot <= 0, to_start >= limit, np.where(dot >= length2, to_end >= limit, cross * cross >= limit * length2))
    return clear.all(axis=1)


# ---- crowns --------------------------------------------------------------------------------------------------------

def centres_inside(crowns_cm: np.ndarray, ring_cm: np.ndarray) -> np.ndarray:
    """Which circles (east, north, radius, centimetres) have their centre inside a polygon (east, north corners,
    centimetres); exact."""
    if not len(crowns_cm):
        return np.zeros(0, dtype=bool)
    x, y = crowns_cm[:, 0:1], crowns_cm[:, 1:2]
    a = ring_cm[None, :, :]
    b = np.roll(ring_cm, -1, axis=0)[None, :, :]
    ex, ey = b[..., 0] - a[..., 0], b[..., 1] - a[..., 1]
    crosses = (a[..., 1] > y) != (b[..., 1] > y)
    lhs, rhs = (x - a[..., 0]) * ey, (y - a[..., 1]) * ex
    return (crosses & np.where(ey > 0, lhs < rhs, lhs > rhs)).sum(axis=1) % 2 == 1


def touching(crowns_cm: np.ndarray, ring_cm: np.ndarray) -> np.ndarray:
    """Which circles (east, north, radius, centimetres) reach into a polygon (east, north corners, centimetres)."""
    if not len(crowns_cm):
        return np.zeros(0, dtype=bool)
    x, y, radius = crowns_cm[:, 0:1], crowns_cm[:, 1:2], crowns_cm[:, 2:3]
    a = ring_cm[None, :, :]
    b = np.roll(ring_cm, -1, axis=0)[None, :, :]
    ex, ey = b[..., 0] - a[..., 0], b[..., 1] - a[..., 1]
    wx, wy = x - a[..., 0], y - a[..., 1]
    inside = centres_inside(crowns_cm, ring_cm)
    dot, length2 = wx * ex + wy * ey, ex * ex + ey * ey
    r2 = radius * radius
    near = np.where(dot <= 0, wx * wx + wy * wy <= r2,
                    np.where(dot >= length2, (x - b[..., 0]) ** 2 + (y - b[..., 1]) ** 2 <= r2,
                             (wx * ey - wy * ex) ** 2 <= r2 * length2))
    return inside | near.any(axis=1)


def merge_crowns(crowns_cm: np.ndarray, limit: int = MAX_CROWNS) -> np.ndarray:
    """Join the two circles whose covering circle is smallest, again and again, until at most limit are left; every
    circle given stays inside one returned."""
    circles = [tuple(int(v) for v in row) for row in crowns_cm]
    while len(circles) > limit:
        c = np.array(circles, dtype=float)
        dx = c[None, :, 0] - c[:, None, 0]
        dy = c[None, :, 1] - c[:, None, 1]
        d = np.sqrt(dx * dx + dy * dy)
        r1, r2 = c[:, None, 2], c[None, :, 2]
        cover = np.where(d + r2 <= r1, r1, np.where(d + r1 <= r2, r2, (d + r1 + r2) / 2.0))
        cover[np.tril_indices(len(circles))] = np.inf
        i, j = np.unravel_index(int(np.argmin(cover)), cover.shape)
        (x1, y1, s1), (x2, y2, s2) = circles[i], circles[j]
        size = float(cover[i, j])
        if size == s1:
            merged = (x1, y1, s1)
        elif size == s2:
            merged = (x2, y2, s2)
        else:
            gap = float(d[i, j])
            share = (size - s1) / gap
            merged = (int(round(x1 + share * (x2 - x1))), int(round(y1 + share * (y2 - y1))), int(math.ceil(size)) + 1)
        circles = [circle for k, circle in enumerate(circles) if k not in (i, j)] + [merged]
    return np.array(circles, dtype=np.int64).reshape(-1, 3)


def covered(crowns_cm: np.ndarray, rows: np.ndarray, cols: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Which cell centres lie inside any circle (centimetres); exact."""
    total_rows, total_cols = shape
    x = (2 * cols + 1 - total_cols) * 50
    y = (total_rows - 2 * rows - 1) * 50
    hit = np.zeros(len(rows), dtype=bool)
    for cx, cy, radius in crowns_cm.tolist():
        hit |= (x - cx) ** 2 + (y - cy) ** 2 <= radius * radius
    return hit


def crown_clear(crowns_cm: np.ndarray, rows: np.ndarray, cols: np.ndarray, shape: Tuple[int, int], clear_cm: int) -> np.ndarray:
    """Which cell centres lie at least clear_cm outside every circle (centimetres); exact."""
    total_rows, total_cols = shape
    x = (2 * cols + 1 - total_cols) * 50
    y = (total_rows - 2 * rows - 1) * 50
    ok = np.ones(len(rows), dtype=bool)
    for cx, cy, radius in crowns_cm.tolist():
        ok &= (x - cx) ** 2 + (y - cy) ** 2 >= (radius + clear_cm) ** 2
    return ok


# ---- the rules -----------------------------------------------------------------------------------------------------

def corners_cm(area: Area, ring: np.ndarray) -> np.ndarray:
    """A ring of grid corners as centimetres east and north of the area's centre."""
    rows_total, cols_total = area.shape
    return np.column_stack([(ring[:, 1] - cols_total // 2) * 100, (rows_total // 2 - ring[:, 0]) * 100]).astype(np.int64)


def reaching(area: Area, ring_cm: np.ndarray) -> np.ndarray:
    """The area's crown circles (centimetres) that reach into an outline of corners in centimetres, never merged."""
    box = (ring_cm.min(axis=0), ring_cm.max(axis=0))
    crowns = area.crowns_cm
    near = crowns[(crowns[:, 0] + crowns[:, 2] >= box[0][0]) & (crowns[:, 0] - crowns[:, 2] <= box[1][0])
                  & (crowns[:, 1] + crowns[:, 2] >= box[0][1]) & (crowns[:, 1] - crowns[:, 2] <= box[1][1])]
    return near[touching(near, ring_cm)]


def plane_too_steep(rows: np.ndarray, cols: np.ndarray, ground_cm: np.ndarray) -> bool:
    """Whether the plane fitted through the cells' ground, by least squares on exact integer sums, tilts more than
    MAX_SLOPE_TAN."""
    x, y, z = cols.astype(np.int64), -rows.astype(np.int64), ground_cm.astype(np.int64)
    n = len(z)
    sx, sy, sz = int(x.sum()), int(y.sum()), int(z.sum())
    sxx = n * int((x * x).sum()) - sx * sx
    syy = n * int((y * y).sum()) - sy * sy
    sxy = n * int((x * y).sum()) - sx * sy
    sxz = n * int((x * z).sum()) - sx * sz
    syz = n * int((y * z).sum()) - sy * sz
    det = sxx * syy - sxy * sxy
    if det == 0:
        return True
    a, b = sxz * syy - syz * sxy, syz * sxx - sxz * sxy
    # The gradient is (a, b) / det centimetres per metre.
    return Fraction(a * a + b * b) > (MAX_SLOPE_TAN * 100) ** 2 * det * det


def dock_spots(area: Area, field: np.ndarray, top: int, left: int, ring: np.ndarray, crowns_cm: np.ndarray) -> np.ndarray:
    """Grid cells of the field where the dock may stand: on the dock layer, DOCK_INSIDE_M inside the outline,
    DOCK_CLEAR_CM off every crown, on ground no steeper than DOCK_SLOPE_TAN over the 3 x 3 cells round it, with the
    field's highest ground at most DOCK_RISE_CM and the top of every solid thing standing on it at most
    DOCK_SOLID_RISE_CM above it."""
    rows, cols = np.nonzero(field)
    rows, cols = rows + top, cols + left
    ground = area.ground_cm
    highest = int(ground[rows, cols].max())
    solid = int(area.solid_top_cm[rows, cols].max())
    total_rows, total_cols = area.shape
    keep = area.dock[rows, cols] & (highest - ground[rows, cols] <= DOCK_RISE_CM)
    keep &= solid - ground[rows, cols] <= DOCK_SOLID_RISE_CM
    keep &= (rows >= 1) & (rows < total_rows - 1) & (cols >= 1) & (cols < total_cols - 1)
    rows, cols = rows[keep], cols[keep]
    if len(rows):
        patch = np.stack([ground[rows + dr, cols + dc] for dr in (-1, 0, 1) for dc in (-1, 0, 1)], axis=1)
        valid = ~(patch == NO_DATA_CM).any(axis=1)
        east = patch[:, [2, 5, 8]].sum(axis=1) - patch[:, [0, 3, 6]].sum(axis=1)
        north = patch[:, [0, 1, 2]].sum(axis=1) - patch[:, [6, 7, 8]].sum(axis=1)
        # A 3 x 3 least-squares plane has gradient (east, north) / 6 centimetres per metre.
        level = (east * east + north * north) <= float(DOCK_SLOPE_TAN * DOCK_SLOPE_TAN * 360000)
        rows, cols = rows[valid & level], cols[valid & level]
    if len(rows):
        keep = crown_clear(crowns_cm, rows, cols, area.shape, DOCK_CLEAR_CM)
        rows, cols = rows[keep], cols[keep]
    if len(rows):
        keep = far_from_edges(ring, rows, cols, DOCK_INSIDE_M)
        rows, cols = rows[keep], cols[keep]
    return np.column_stack([rows, cols])


def check(area: Area, ring: np.ndarray, low_m2: float, high_m2: float,
          recipe: Recipe = OPEN) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Every rule a field of a recipe must pass, on its outline as grid corners: its cells inside, the crowns that
    touch it (merged), the dock's cells and its area doubled; a rule broken raises with its name."""
    rows_total, cols_total = area.shape
    twice_area = abs(int(sum(ring[k - 1, 1] * ring[k, 0] - ring[k, 1] * ring[k - 1, 0] for k in range(len(ring)))))
    if not low_m2 * 2 <= twice_area <= high_m2 * 2:
        raise _Reject("size")
    gaps = ring[:, None, :] - ring[None, :, :]
    if int((gaps * gaps).sum(axis=2).max()) > MAX_ACROSS_M * MAX_ACROSS_M:
        raise _Reject("across")
    if (ring.min() < CORE_MARGIN_M or ring[:, 0].max() > rows_total - CORE_MARGIN_M
            or ring[:, 1].max() > cols_total - CORE_MARGIN_M):
        raise _Reject("margin")
    field, top, left = polygon_cells(ring, area.shape)
    window = (slice(top, top + field.shape[0]), slice(left, left + field.shape[1]))
    if (field & (area.near_keep_out[window] | area.no_data[window])).any():
        raise _Reject("keep_out")
    rows, cols = np.nonzero(field)
    ground = area.ground_cm[window][field]
    ordered = np.sort(ground)
    trim = (len(ordered) - 1) // 100
    if int(ordered[len(ordered) - 1 - trim] - ordered[trim]) > MAX_RELIEF_CM:
        raise _Reject("relief")
    if plane_too_steep(rows, cols, ground):
        raise _Reject("slope")
    inner = binary_erosion(field, structure=disk(EDGE_BAND_M))
    if 100 * int((inner & area.open[window]).sum()) < recipe.open_percent * int(inner.sum()):
        raise _Reject("open")
    lost = field & ~binary_opening(field, structure=disk(ARM_M))
    if lost.any():
        labels, _count = label(lost, structure=np.ones((3, 3), dtype=bool))
        if int(np.bincount(labels.ravel())[1:].max()) > MAX_ARM_CELLS:
            raise _Reject("arm")
    ring_cm = corners_cm(area, ring)
    crowns_cm = reaching(area, ring_cm)
    if int(centres_inside(crowns_cm, ring_cm).sum()) < recipe.least_crowns:
        raise _Reject("trees")
    if len(crowns_cm) > MAX_CROWNS:
        crowns_cm = merge_crowns(crowns_cm)
    if len(crowns_cm) and 100 * int(covered(crowns_cm, rows + top, cols + left, area.shape).sum()) > recipe.crown_percent * len(rows):
        raise _Reject("crowns")
    spots = dock_spots(area, field, top, left, ring, crowns_cm)
    if len(spots) < MIN_DOCK_CELLS:
        raise _Reject("dock")
    return field, crowns_cm, spots, twice_area


# ---- one try, one area, one seed -----------------------------------------------------------------------------------

def attempt(area: Area, rng: np.random.Generator, target_m2: int, low_m2: float, high_m2: float,
            recipe: Recipe = OPEN) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray, int]:
    """One try at a field of a recipe in an area: its family, outline corners, crowns, dock cells and doubled area;
    raises with the rule it broke."""
    family, corners = template(rng)
    turn = unit_vector(rng)
    centres = area.centres(target_m2, recipe)
    if not len(centres):
        raise _Reject("no_centre")
    centre_r, centre_c = divmod(int(centres[int(rng.integers(len(centres)))]), area.shape[1])
    rows_total, cols_total = area.shape
    centre = np.array([centre_c + 0.5 - cols_total / 2.0, rows_total / 2.0 - centre_r - 0.5])
    turned = np.column_stack([corners[:, 0] * turn[0] - corners[:, 1] * turn[1], corners[:, 0] * turn[1] + corners[:, 1] * turn[0]])
    unit = shoelace(turned)
    if unit <= 0.0:
        raise _Reject("shape")
    for growth in GROWTH:
        scaled = turned * math.sqrt(target_m2 * growth / unit) + centre
        top = max(0, int(math.floor(rows_total / 2.0 - scaled[:, 1].max())) - WINDOW_PAD)
        bottom = min(rows_total, int(math.ceil(rows_total / 2.0 - scaled[:, 1].min())) + WINDOW_PAD)
        left = max(0, int(math.floor(scaled[:, 0].min() + cols_total / 2.0)) - WINDOW_PAD)
        right = min(cols_total, int(math.ceil(scaled[:, 0].max() + cols_total / 2.0)) + WINDOW_PAD)
        if bottom <= top or right <= left:
            raise _Reject("off_grid")
        window = (slice(top, bottom), slice(left, right))
        shape_cells = cells_in(scaled, top, left, bottom - top, right - left, area.shape)
        piece = largest_piece(binary_opening(shape_cells & area.free_for(recipe)[window], structure=disk(SLIVER_M)))
        piece = largest_piece(binary_closing(piece, structure=disk(NOTCH_M)) & area.room[window])
        piece = binary_fill_holes(close_pinches(piece))
        cells = int(piece.sum())
        if cells < low_m2:
            continue
        if cells > high_m2 * (1.0 + SIZE_TOLERANCE):
            raise _Reject("size")
        ring = np.array(outline_of(piece), dtype=np.int64) + np.array([top, left])
        _cells, crowns_cm, spots, twice_area = check(area, ring, low_m2, high_m2, recipe)
        return family, ring, crowns_cm, spots, twice_area
    raise _Reject("size")


def to_metres(area: Area, ring: np.ndarray) -> np.ndarray:
    """Grid corners as area metres (east, north)."""
    rows, cols = area.shape
    return np.column_stack([ring[:, 1] - cols // 2, rows // 2 - ring[:, 0]]).astype(float)


def cut_in(area: Area, rng: np.random.Generator, target_m2: int, low_m2: float, high_m2: float,
           tries: int = TRIES_PER_AREA, reasons: Optional[Dict[str, int]] = None, recipe: Recipe = OPEN) -> Optional[Field]:
    """A field of a recipe in one area from the next tries of a stream, or None when every try breaks a rule;
    reasons, when given, counts the rule each failed try broke. An area with fewer crowns in all than the recipe
    needs yields none, and no try is drawn."""
    if len(area.crowns) < recipe.least_crowns:
        return None
    for count in range(1, tries + 1):
        try:
            family, ring, crowns_cm, spots, twice_area = attempt(area, rng, target_m2, low_m2, high_m2, recipe)
        except _Reject as reject:
            if reasons is not None:
                reasons[str(reject)] = reasons.get(str(reject), 0) + 1
            continue
        rows, cols = area.shape
        outline = to_metres(area, ring)
        centre = outline.mean(axis=0)
        houses = area.houses[np.max(np.abs(area.houses[:, :2] - centre), axis=1) <= MAX_ACROSS_M] if len(area.houses) else area.houses
        spots_xy = np.column_stack([spots[:, 1] + 0.5 - cols / 2.0, rows / 2.0 - spots[:, 0] - 0.5])
        trees = reaching(area, corners_cm(area, ring)).astype(float) / 100.0
        return Field(area.name, outline, twice_area / 2.0, crowns_cm.astype(float) / 100.0, houses, spots_xy,
                     family, int(target_m2), count, recipe.kind, trees)
    return None


def bounds(target_m2: int, min_m2: float, max_m2: float) -> Tuple[float, float]:
    """The smallest and largest area a field cut for a size may have."""
    return max(float(min_m2), target_m2 * (1.0 - SIZE_TOLERANCE)), min(float(max_m2), target_m2 * (1.0 + SIZE_TOLERANCE))


def weighted_order(rng: np.random.Generator, names: Sequence[str], weights: Mapping[str, int]) -> List[str]:
    """Names in a drawn order: each next one drawn among those left, by chance in proportion to its whole-number
    weight; a name with no weight weighs 1."""
    left, order = list(names), []
    while left:
        pick = int(rng.integers(0, sum(int(weights.get(name, 1)) for name in left)))
        for k, name in enumerate(left):
            pick -= int(weights.get(name, 1))
            if pick < 0:
                order.append(left.pop(k))
                break
    return order


def choose(seed: int, min_m2: float, max_m2: float, table: Mapping[str, Sequence[bool]],
           recipe: Recipe = OPEN, weights: Optional[Mapping[str, int]] = None) -> Tuple[int, List[str]]:
    """The size the seed asks for and, in the seed's order for a recipe, every area the table lists as holding it:
    an even shuffle, or with weights each next area drawn by its weight."""
    low, high = size_limits()
    if not low <= min_m2 <= max_m2 <= high:
        raise ValueError(f"field sizes {min_m2} to {max_m2} m2 lie outside {low} to {high}")
    target = draw_size(seed, min_m2, max_m2)
    which = size_bin(target)
    eligible = sorted(name for name in table if table_holds(table, name, which))
    rng = np.random.default_rng(streams(recipe, AREA_SEED_STREAM, seed))
    if weights is not None:
        return target, weighted_order(rng, eligible, weights)
    return target, [eligible[int(k)] for k in rng.permutation(len(eligible))]


def cut_kind(seed: int, min_m2: float, max_m2: float, table: Mapping[str, Sequence[bool]],
             area_of: Callable[[str], Area], recipe: Recipe, weights: Optional[Mapping[str, int]] = None) -> Optional[Field]:
    """The seed's field of one recipe: a size drawn between min_m2 and max_m2, then one of the areas the table lists
    as holding it, evenly or by the weights given, and a field cut there; when an area yields none, the next such
    area in the seed's order; None when no area does. Areas are read through area_of only when their turn comes."""
    target, names = choose(seed, min_m2, max_m2, table, recipe, weights)
    rng = np.random.default_rng(streams(recipe, SHAPE_SEED_STREAM, seed))
    low_m2, high_m2 = bounds(target, min_m2, max_m2)
    for name in names:
        field = cut_in(area_of(name), rng, target, low_m2, high_m2, recipe=recipe)
        if field is not None:
            return field
    return None


def cut(seed: int, min_m2: float, max_m2: float, table: Mapping[str, Sequence[bool]], area_of: Callable[[str], Area],
        wooded_table: Optional[Mapping[str, Sequence[bool]]] = None, weights: Optional[Mapping[str, int]] = None) -> Field:
    """The seed's field: wooded, from the areas the wooded table lists, when the seed asks for one and some area
    yields it; else open, from the areas the table lists. With no wooded table every field is open; with weights an
    area is picked by its weight among those that hold the seed's size."""
    if wooded_table is not None and draw_kind(seed) == WOODED.kind:
        field = cut_kind(seed, min_m2, max_m2, wooded_table, area_of, WOODED, weights)
        if field is not None:
            return field
    field = cut_kind(seed, min_m2, max_m2, table, area_of, OPEN, weights)
    if field is None:
        raise NoField(f"no area yields a field of {draw_size(seed, min_m2, max_m2)} m2 for seed {seed}")
    return field


# ---- which areas hold which sizes ----------------------------------------------------------------------------------

def bin_edges() -> List[Tuple[int, int]]:
    """Each size bin's smallest and largest whole square metres."""
    first = SIZE_RANGE_TENTHS[0]
    starts = [-(-(first + k) * TODAY_M2 // 10) for k in range(SIZE_BINS)]
    return [(starts[k], starts[k + 1] - 1) for k in range(SIZE_BINS - 1)] + [(starts[-1], SIZE_RANGE_TENTHS[1] * TODAY_M2 // 10)]


def holds(area: Area, size_bin_index: int, probes: int, needed: int, reasons: Optional[Dict[str, int]] = None,
          recipe: Recipe = OPEN) -> Tuple[bool, int, int]:
    """Whether an area holds a size bin for a recipe: at least `needed` of `probes` tries at the bin's largest size
    pass, on the probe stream; also the passes and the tries spent."""
    largest = bin_edges()[size_bin_index][1]
    low_m2, high_m2 = bounds(largest, *size_limits())
    rng = np.random.default_rng(streams(recipe, PROBE_SEED_STREAM, size_bin_index, int.from_bytes(area.name.encode(), "little")))
    passed = 0
    for spent in range(1, probes + 1):
        if cut_in(area, rng, largest, low_m2, high_m2, tries=1, reasons=reasons, recipe=recipe) is not None:
            passed += 1
            if passed >= needed:
                return True, passed, spent
    return False, passed, probes


def size_table(areas: Sequence[Area], probes: int = 200, needed: int = 4, recipe: Recipe = OPEN) -> Dict[str, Tuple[bool, ...]]:
    """For each area, which size bins it holds for a recipe."""
    return {area.name: tuple(holds(area, k, probes, needed, recipe=recipe)[0] for k in range(SIZE_BINS)) for area in areas}
