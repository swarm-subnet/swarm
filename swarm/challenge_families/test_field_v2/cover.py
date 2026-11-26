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

"""Where people who stay put stand on a field that reaches into the trees, and whether a drone can see someone under
a tree.

On a wooded field (the site's field kind) each person who stays put draws a cover on the people's own stream: under a
tree in UNDER_SHARE of draws, beside one in BESIDE_SHARE, else in the open. At most MOST_UNDER people of a seed go
under trees; a later draw for a tree's shade stands him in the open. Walkers always stroll in the open, and on an open
field nobody draws a cover.

The trees are the crown circles of the trees that reach the field, as the cutter found them, never merged into the
larger circles a site map with many trees lists. Beside a tree is BESIDE_GAP_M outside its circle, by every rule of
the open, which keeps it clear of the site map's circles too; the tree is drawn by the length of its rim. Under a tree
is inside its circle, from UNDER_CORE_M off its centre (or UNDER_CORE_SHARE of its radius when that is less) out to
UNDER_REACH_SHARE of its radius, where the leaves are thick; the tree is drawn by its area. There every rule of the open holds but the crowns' own: rays
dropped round the spot meet trunks and stones, never leaves.

The fairness rule. Someone under a tree is kept there only when a drone can see him clearly from enough of the places
it may fly, and from few enough of them that the tree does hide him. The places lie on a GRID_M grid over the field, EDGE_M inside its outline, CROWN_GAP_M outside every crown
circle and PERSON_GAP_M across the ground from everyone on the field; at each, the drone looks from CRUISE_M over the
dock and from LOW_M and MID_M over the ground under it where those lie under the CEILING_M ceiling. Only places within
REACH_M across the ground of him count. From each, a square frame at SIGHT_SCALE of the wide camera's pixel scale, a
quarter of its rays (FRAME_PX across, NEAR_FRAME_PX when he is NEAR_M or nearer), is aimed at his middle and his pixels are counted on the frame's object map,
the very map a patrol frame gives: leaves cut out by their alpha, trunks, stones, the ground and the others hiding him
as they do in flight. He is kept when CLEAR_PX or more of him, a hundred of the camera's own pixels, show from at least CLEAR_PERCENT of those places and from
at least CLEAR_LEAST of them, and from at most HIDE_PERCENT of them: seen from more, he is not hidden and another place
under a tree is drawn. One frame from straight above him at cruise height is looked at first: with OVERHEAD_PX or more
of him on it the leaves do not cover him, and no other frame is drawn. Pixel counts and the comparisons on them are integers, and the places come from IEEE
operations done one at a time, so every machine decides alike.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Any, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pybullet as p

from ..solar_patrol import camera
from ..solar_patrol.contract import RGB_SHAPE
from ..solar_patrol.flight_limit import _distance, _inside, _sides_of
from . import field_cut, grass, site
from .episode import FieldEpisode

OPEN, BESIDE, UNDER = "open", "beside", "under"
UNDER_SHARE = Fraction(25, 100)
BESIDE_SHARE = Fraction(15, 100)
MOST_UNDER = 3
BESIDE_GAP_M = (0.5, 2.0)
UNDER_CORE_M = 1.2
UNDER_CORE_SHARE = 0.4
UNDER_REACH_SHARE = 0.7                # of a crown's radius: its circle is drawn round its farthest leaf
COVER_TRIES = 400                      # draws for a spot under or beside a tree before he stands in the open
UNDER_DRAWS = 3                        # places under a tree tried after the first before he stands in the open

GRID_M = 7.5
EDGE_M = 1.5
CROWN_GAP_M = 0.3
PERSON_GAP_M = 5.0
REACH_M = 45.0
CRUISE_M = 9.5                         # over the dock, as the patrol cruises
LOW_M = 3.0                            # over the ground under the place
MID_M = 6.0
CEILING_M = 10.0                       # over the dock
SIGHT_SCALE = 0.5                      # of the wide camera's pixel scale: the rule's frames cost a quarter of the rays
FRAME_PX = 48
NEAR_FRAME_PX = 96
NEAR_M = 9.0
AIM_SHARE = {"standing": 0.5, "crouching": 0.35}   # of his height: his middle as he holds himself
CLEAR_PX = 25                          # at SIGHT_SCALE: a hundred pixels of a patrol frame
CLEAR_PERCENT = 5
CLEAR_LEAST = 6
HIDE_PERCENT = 30
OVERHEAD_PX = 10                       # fewer than this from straight above: a sweep overhead does not see him
OVERHEAD_ASIDE_M = 0.3                 # the look from above stands this far east, so its frame has an upright
LEVELS = ("cruise", "low", "mid")

# The focal length in pixels at the rule's scale, and the field of view of a square frame at that scale.
FOCAL_PX = SIGHT_SCALE * RGB_SHAPE[0] / 2.0 / math.tan(math.radians(camera.vertical_fov_deg(
    camera.WIDE_DIAGONAL_FOV_DEG, RGB_SHAPE[1], RGB_SHAPE[0]) / 2.0))
FOV_DEG = {size: 2.0 * math.degrees(math.atan(size / 2.0 / FOCAL_PX)) for size in (FRAME_PX, NEAR_FRAME_PX)}
# The thermal camera's ray-cast set draws the same object map as a patrol's colour frame, leaves cut out alike, and
# skips its light and shading; without the ray caster the colour frame itself is drawn.
SIGHT_FLAGS = (p.ER_SWARM_RAYCAST | p.ER_SWARM_THERMAL | p.ER_ALPHA_CUTOUT | camera.RASTER | camera.NO_DEPTH
               if camera.THERMAL else 0)


def draw(rng: np.random.Generator, still: Sequence[bool]) -> List[str]:
    """Everyone's cover on a wooded field, from the people's own generator: one draw for each person who stays put,
    under a tree only while fewer than MOST_UNDER are; the open for a walker."""
    covers: List[str] = []
    for stays in still:
        if not stays:
            covers.append(OPEN)
            continue
        share = Fraction(int(rng.integers(100)), 100)
        if share < UNDER_SHARE:
            covers.append(UNDER if covers.count(UNDER) < MOST_UNDER else OPEN)
        elif share < UNDER_SHARE + BESIDE_SHARE:
            covers.append(BESIDE)
        else:
            covers.append(OPEN)
    return covers


def _crown(rng: np.random.Generator, crowns: np.ndarray, power: int) -> np.ndarray:
    """One crown circle drawn with a weight of its radius in whole centimetres to a power: 2 by area, 1 by rim."""
    weights = np.rint(crowns[:, 2] * 100.0).astype(np.int64) ** power
    pick = int(rng.integers(int(weights.sum())))
    return crowns[int(np.searchsorted(np.cumsum(weights), pick, side="right"))]


def candidate(rng: np.random.Generator, cover: str, crowns: np.ndarray) -> Optional[np.ndarray]:
    """One spot drawn under or beside a crown circle, east and north; None when the field has no crown."""
    if not len(crowns):
        return None
    x, y, radius = (float(v) for v in _crown(rng, crowns, 2 if cover == UNDER else 1))
    if cover == UNDER:
        nearest = min(UNDER_CORE_M, UNDER_CORE_SHARE * radius)
        gap = float(rng.uniform(nearest, max(nearest, UNDER_REACH_SHARE * radius)))
    else:
        gap = radius + float(rng.uniform(*BESIDE_GAP_M))
    turn = field_cut.unit_vector(rng)
    return np.array([x + gap * float(turn[0]), y + gap * float(turn[1])])


def _norm(vector: np.ndarray) -> float:
    """Length of a 3-vector, one IEEE operation at a time."""
    x, y, z = (float(v) for v in vector)
    return math.sqrt(x * x + y * y + z * z)


def object_map(env: Any, ep: FieldEpisode, eye: np.ndarray, target: np.ndarray, size: int) -> np.ndarray:
    """The object map of a square frame at the wide camera's pixel scale from an eye towards a target: per pixel the
    body id, or -1 where nothing was hit. Taken on bare ground, so the measure is the same at a seed's build, where
    no grass is drawn yet, and in a flight, whatever picture came before it."""
    grass.hide(env, ep)
    direction = np.asarray(target, dtype=float) - np.asarray(eye, dtype=float)
    length = _norm(direction)
    shot = camera.View("colour", tuple(float(v) for v in eye), tuple(float(v) / length for v in direction),
                       (0.0, 0.0, 1.0), size, size, FOV_DEG[size], True, 0, far_m=camera.far_plane_m(ep))
    if SIGHT_FLAGS:
        view, projection = shot.matrices()
        seg = p.getCameraImage(size, size, view, projection, renderer=p.ER_TINY_RENDERER, flags=SIGHT_FLAGS,
                               physicsClientId=env.CLIENT)[4]
        objects = np.reshape(np.asarray(seg, dtype=np.int32), (size, size))
    else:
        objects = camera.colour_frame(env, shot)[1]
    return np.where(objects >= 0, objects & ((1 << 24) - 1), -1)


def needed(places: int) -> int:
    """How many places must see someone clearly for him to stay under a tree, of a number that may."""
    return max(CLEAR_LEAST, -(-CLEAR_PERCENT * places // 100))


def most(places: int) -> int:
    """From how many places at most someone may show clearly and still count as hidden by his tree."""
    return HIDE_PERCENT * places // 100


class Sight:
    """The places a drone may look from over one field, and what someone shows from them."""

    def __init__(self, env: Any, ep: FieldEpisode, crowns: np.ndarray):
        """Lay the grid over the field once: the points EDGE_M inside the outline and clear of the crowns, with the
        ground under each."""
        self.env, self.ep = env, ep
        outline = np.asarray(ep.fence, dtype=float)
        sides = _sides_of(outline)
        low, high = outline.min(axis=0), outline.max(axis=0)
        crowns = np.asarray(crowns, dtype=float).reshape(-1, 3)
        points, grounds = [], []
        for x in np.arange(low[0] + GRID_M / 2.0, high[0], GRID_M):
            for y in np.arange(low[1] + GRID_M / 2.0, high[1], GRID_M):
                spot = np.array([float(x), float(y)])
                if not _inside(sides, spot) or _distance(outline, spot) < EDGE_M:
                    continue
                dx, dy = crowns[:, 0] - spot[0], crowns[:, 1] - spot[1]
                reach = crowns[:, 2] + CROWN_GAP_M
                if np.any(dx * dx + dy * dy < reach * reach):
                    continue
                ground = site.ground_at(env.CLIENT, ep.terrain_uids, float(x), float(y))
                if ground is not None:
                    points.append(spot)
                    grounds.append(ground)
        self.points = np.array(points, dtype=float).reshape(-1, 2)
        self.ground = np.array(grounds, dtype=float)

    def places(self, where: np.ndarray, everyone: np.ndarray) -> np.ndarray:
        """The places that may look at someone standing at a point: east, north, up and the level's index, for every
        grid point within REACH_M of him and PERSON_GAP_M from everyone, at every level under the ceiling; nearest
        first, low before mid before cruise."""
        dx, dy = self.points[:, 0] - float(where[0]), self.points[:, 1] - float(where[1])
        keep = dx * dx + dy * dy <= REACH_M * REACH_M
        for other in np.asarray(everyone, dtype=float).reshape(-1, 3):
            ox, oy = self.points[:, 0] - other[0], self.points[:, 1] - other[1]
            keep &= ox * ox + oy * oy >= PERSON_GAP_M * PERSON_GAP_M
        dock_z = float(self.ep.dock_position[2])
        rows = []
        for (x, y), ground in zip(self.points[keep], self.ground[keep]):
            rows.append((x, y, dock_z + CRUISE_M, 0))
            for level, rise in ((1, LOW_M), (2, MID_M)):
                if ground + rise <= dock_z + CEILING_M:
                    rows.append((x, y, ground + rise, level))
        places = np.array(rows, dtype=float).reshape(-1, 4)
        flat = (places[:, 0] - float(where[0])) ** 2 + (places[:, 1] - float(where[1])) ** 2
        order = np.lexsort((flat, np.where(places[:, 3] == 0, 3, places[:, 3])))
        return places[order]

    @staticmethod
    def target(where: np.ndarray, height_m: float, pose: str) -> np.ndarray:
        """His middle as he holds himself, world metres."""
        return np.array([float(where[0]), float(where[1]), float(where[2]) + AIM_SHARE.get(pose, 0.5) * height_m])

    def pixels(self, uids: Sequence[int], target: np.ndarray, places: np.ndarray) -> Iterator[int]:
        """His pixels on the frame from each place in turn, aimed at his middle."""
        for eye in places[:, :3]:
            far = _norm(target - eye) > NEAR_M
            yield int(np.isin(object_map(self.env, self.ep, eye, target, FRAME_PX if far else NEAR_FRAME_PX), uids).sum())

    def counts(self, where: np.ndarray, height_m: float, pose: str, uids: Sequence[int],
               everyone: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Every place that may look at him and his pixels from each: the whole measure, for a proof."""
        places = self.places(where, everyone)
        return places, np.array(list(self.pixels(uids, self.target(where, height_m, pose), places)), dtype=np.int64)

    def clear(self, where: np.ndarray, height_m: float, pose: str, uids: Sequence[int], everyone: np.ndarray) -> bool:
        """Whether he shows CLEAR_PX or more from enough of the places that may look at him; the frames stop once
        the answer is sure, which never changes it."""
        places = self.places(where, everyone)
        least, left, seen = needed(len(places)), len(places), 0
        if left < least:
            return False
        for px in self.pixels(uids, self.target(where, height_m, pose), places):
            left -= 1
            seen += px >= CLEAR_PX
            if seen >= least:
                return True
            if seen + left < least:
                return False
        return False

    def overhead(self, where: np.ndarray, height_m: float, pose: str, uids: Sequence[int]) -> int:
        """His pixels on a frame from straight above him at cruise height: what a sweep passing overhead sees."""
        target = self.target(where, height_m, pose)
        eye = np.array([[float(target[0]) + OVERHEAD_ASIDE_M, float(target[1]), float(self.ep.dock_position[2]) + CRUISE_M]])
        return next(self.pixels(uids, target, eye))

    def kept(self, where: np.ndarray, height_m: float, pose: str, uids: Sequence[int], everyone: np.ndarray) -> bool:
        """Whether he may stay under his tree: the leaves cover him from straight above, and he shows CLEAR_PX or more
        from enough places to be found and from few enough to be hidden; the frames stop once the answer is sure,
        which never changes it."""
        places = self.places(where, everyone)
        least, ceiling, left, seen = needed(len(places)), most(len(places)), len(places), 0
        if left < least or ceiling < least or self.overhead(where, height_m, pose, uids) >= OVERHEAD_PX:
            return False
        for px in self.pixels(uids, self.target(where, height_m, pose), places):
            left -= 1
            seen += px >= CLEAR_PX
            if seen > ceiling or seen + left < least:
                return False
            if seen >= least and seen + left <= ceiling:
                return True
        return least <= seen <= ceiling
