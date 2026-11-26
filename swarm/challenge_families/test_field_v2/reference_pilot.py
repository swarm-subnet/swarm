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

"""The reference pilot the seed checks fly every seed with, and the verdict it brings back.

It flies through the patrol's own environment, every frame it may still read drawn as the model's would be, at the
take-off's height under the 10 m ceiling, over the free ground only: inside the field by EDGE_CLEAR_M, off every crown
by CROWN_CLEAR_M, and off every person on the field as far across the ground as keeps it PERSON_CLEAR_M from the top
of his head at that height (none where he stands low enough under it), a walker by that much off his whole patch.
People passing outside the field are neither looked for nor reported, and need no room: the edge keeps the drone
further from them than any of that. Fences of keep-off discs can cut the free ground in pieces; the drone only ever
flies on the piece its dock stands on.

It takes off, flies to a viewpoint on a ring round each person, nearest first, faces him where he is now with the
camera tilted onto him, and holds there until a frame shows him. Someone under a tree, whom no ring viewpoint sees,
and anyone a ring viewpoint did not show in time, is looked at from the places the cover module's fairness rule
measures: those on the dock's piece of free ground that show him well, best first, found once and only as far as
LOOKS_KEPT good ones. A place under the cruise height is flown to at the cruise height, then come down to straight,
through a column COLUMN_M wide where nothing solid stands within COLUMN_CLEAR_M of it and nobody within PERSON_CLEAR_M
across the ground; the pilot climbs back straight up before it flies on. Then it sweeps the free ground in lanes LANE_M
apart with the camera straight down while the time allows, or, flying a seed check, only while someone the visits gave
up on is still unreported. It flies home over the free ground, presses return home once the straight flight to the
dock stays free, and lands. It reports each person the first time a frame it is shown holds him, with the tight box
around his pixels and his reference point at that frame's step, from the object map of that very frame. The animals
to report (the goats, dogs and grown boars that live on the field) it treats alike: each takes its turn among the
people, the nearest still unreported first, from a ring viewpoint round where the animal then is, and it reports each
one as an animal on the first frame that holds it, with its box and where its middle was at that frame's step. By day it
looks through the colour feed; at night through the thermal feed, as a pilot finds people in the dark, and sweeps in
NIGHT_LANE_M lanes, as far apart as the narrower thermal lens sees.

The seed check leaves out what no pilot reads and the pass below does not need: the sweep once everyone is reported,
so nearly always the whole sweep, the laser, the stream's look on the images, every picture taken while the drone waits
in the dock for its lids, and every picture once everyone on the field and every animal to report is validly
reported.

The seed passes when every person and every animal to report was validly reported, no report was a false alarm or
a failed animal report, the drone never tripped the edge
rule or hit anything, a crown included, never came within the score's safe distance of a person, and it landed back in
the dock before the patrol's time ran out.

Run as a module it judges the seeds given on the command line, or else those read from its standard input, and keeps
their verdicts.
"""

from __future__ import annotations

import contextlib
import functools
import heapq
import io
import math
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pybullet as p
from scipy import ndimage

from swarm.constants import SIM_DT
from swarm.utils.env_factory import make_env_with_initial_obs

from ..solar_patrol import camera, coverage, sensor_noise, zoom
from ..solar_patrol import dock as sentinel_dock
from ..solar_patrol import reports as sentinel_reports
from ..solar_patrol.contract import (
    MAX_CLIMB_MPS,
    MAX_DESCENT_MPS,
    MAX_HORIZONTAL_MPS,
    MAX_YAW_RATE_DEG_S,
    RGB_SHAPE,
    ZOOM_SHAPE,
)
from ..solar_patrol.score import SAFE_DISTANCE_M
from . import animals, cover, crowns, dock, field_cut, grass, people, reports, site, wind
from .contract import ACTION_DIM, ACTION_INDEX, REPORT_REACH_M, TAKE_OFF_M
from .episode import FieldEpisode
from .family import TestFieldV2ChallengeFamily
from .seed_checks import FieldVerdict, keep
from .task import test_field_task

GRID_M = 0.5                           # the free ground is read on a grid this fine
EDGE_CLEAR_M = 3.0                     # off the field's edge: 2 m over the edge rule
CROWN_CLEAR_M = 3.0                    # off a crown circle: at a lane end next to a crown the drone swings up to 0.9 m wide
PERSON_CLEAR_M = 6.5                   # from a person's reference point: the 5 m safe-distance rule, reach, slack
LANE_M = 10.0                          # the wide camera sees about 14 m across from 10 m, less over the field's high end
NIGHT_LANE_M = 5.0                     # the thermal lens sees about 6 m across from 9.5 m
STANDOFF_M = (7.0, 8.5, 10.0)          # how far across the ground a person is looked at from
STANDOFF_TURNS = 24                    # viewpoints tried round each of those rings
SIGHT_CLEAR_M = 0.3                    # the line of sight passes this far off every crown circle
CRUISE_M = TAKE_OFF_M                  # above the dock, where the dock's take-off hands the drone over
SAMPLE_M = 0.5                         # the line between two points is checked against the free ground this often
CARROT_M = 4.0                         # the pilot's look-ahead point (carrot): this far ahead of it on its path
NEAR_CARROTS_M = (2.0, 1.0)            # nearer points it steers at when the line to that one leaves the free ground
MAX_MPS = 4.0                          # short of the 5 m/s limit, so turns stay tight
CORNER_DEG = 30.0                      # a bend sharper than this is taken at CORNER_MPS
CORNER_MPS = 1.0
BRAKE_MPS2 = 1.2                       # inside the outputs' own 2 m/s2 easing
ARRIVE_M = 1.0
HOLD_MAX_S = 10.0                      # a place no frame shows him from in this long is left for his next, then the sweep
HOME_DETOUR = 1.4                      # the way home is at most this much longer than the straight line
HOME_MPS = 3.0
HOME_RESERVE_S = 45.0                  # the dock's turn, the flight home and the landing, with margin over a bad seed
HEIGHT_GAIN_PER_S = 1.0
TURN_GAIN_PER_S = 2.0
TURNING_MPS = 1.0                      # the pilot faces where it flies only once it flies faster than this
AIM_ABOVE_GROUND_M = 0.9
PATCH_RISE_M = 0.2                     # a walker's patch may lie this much higher than the ground sampled round it
PATCH_RING_M = 1.5                     # the ground round a walker is sampled on rings this far apart, out to his rim
LOOK_PX = 2 * cover.CLEAR_PX           # a place shows someone well with this many of his pixels at the rule's scale
SEEN_PX = 10                           # and is worth a try with this many: forty of the camera's, plenty for a report
LOOKS_KEPT = 3                         # good places found for someone before the pilot stops measuring
LOOK_FRAMES = 160                      # places measured for one person at most
COLUMN_M = 2.5                         # round a low place, how wide the column the drone comes down through is checked
COLUMN_RINGS = ((1.25, 8), (COLUMN_M, 16))
COLUMN_CLEAR_M = 1.5                   # everything solid in that column stays this far under the place
CLIMB_FIRST_M = 1.0                    # this far under the cruise height, the drone climbs before it flies on
SETTLE_M = 0.3                         # within this of a place's height, the drone is there
RING_TURNS = 8
MIN_REACH_SHARE = 0.5                  # a seed whose dock's piece holds less of the free ground is cut off from its field
BEAST_TRIES = 3                        # viewpoints flown to for one animal, each laid from where it then is

# A seed check's flight hands the family these in place of the images only a model looks at.
_BLANK_RGB = np.zeros(RGB_SHAPE, dtype=np.float32)
_BLANK_ZOOM = np.zeros(ZOOM_SHAPE, dtype=np.float32)
_BLANK_RGB.flags.writeable = _BLANK_ZOOM.flags.writeable = False


class Free:
    """The ground the pilot may fly over, on a grid: inside the field by EDGE_CLEAR_M, off every crown by
    CROWN_CLEAR_M and every person by his reach past the patch he may walk in; with its pieces, and the paths across it
    between two points."""

    def __init__(self, fence: np.ndarray, crowns_xyr: np.ndarray, persons: np.ndarray,
                 patches: Optional[np.ndarray] = None, reaches: Optional[np.ndarray] = None):
        """Read which cells are free and which piece each lies on; patches gives the radius each person may walk in
        round his point, none if not given, and reaches how far past it he is kept off, PERSON_CLEAR_M if not given."""
        self.low = fence.min(axis=0) - GRID_M
        size = np.ceil((fence.max(axis=0) + GRID_M - self.low) / GRID_M).astype(int) + 1
        xs, ys = np.meshgrid(self.low[0] + GRID_M * np.arange(size[0]), self.low[1] + GRID_M * np.arange(size[1]),
                             indexing="ij")
        points = np.column_stack([xs.ravel(), ys.ravel()])
        free = sentinel_dock._inside(fence, points) & (sentinel_dock._edge_distance(fence, points) >= EDGE_CLEAR_M)
        grown = crowns_xyr.copy()
        grown[:, 2] += CROWN_CLEAR_M
        free &= crowns.outside(grown, points)
        patches = np.zeros(len(persons)) if patches is None else np.asarray(patches, dtype=float)
        reaches = np.full(len(persons), PERSON_CLEAR_M) if reaches is None else np.asarray(reaches, dtype=float)
        for person, patch, reach in zip(persons, patches, reaches):
            free &= np.hypot(points[:, 0] - person[0], points[:, 1] - person[1]) >= reach + patch
        self.free = free.reshape(size)
        # A path steps to any of a cell's eight neighbours, so a piece is joined the same way.
        self.pieces = ndimage.label(self.free, structure=np.ones((3, 3), dtype=bool))[0]
        self.crowns = crowns_xyr

    def cell(self, xy: np.ndarray) -> tuple:
        """The grid cell holding a point."""
        i, j = np.round((np.asarray(xy, dtype=float) - self.low) / GRID_M).astype(int)
        return int(np.clip(i, 0, self.free.shape[0] - 1)), int(np.clip(j, 0, self.free.shape[1] - 1))

    def point(self, cell: tuple) -> np.ndarray:
        """The centre of a grid cell."""
        return self.low + GRID_M * np.asarray(cell, dtype=float)

    def holds(self, xy: np.ndarray) -> bool:
        """Whether a point is free."""
        return bool(self.free[self.cell(xy)])

    def clear(self, a: np.ndarray, b: np.ndarray) -> bool:
        """Whether every point of the straight line from a to b is free."""
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        count = max(2, int(np.hypot(*(b - a)) / SAMPLE_M) + 2)
        return all(self.holds(p) for p in np.linspace(a, b, count))

    def nearest_free(self, xy: np.ndarray) -> Optional[tuple]:
        """The free cell nearest a point, or None when none is free."""
        cells = np.argwhere(self.free)
        if not len(cells):
            return None
        here = (np.asarray(xy, dtype=float) - self.low) / GRID_M
        return tuple(int(v) for v in cells[int(np.argmin(np.hypot(*(cells - here).T)))])

    def piece(self, xy: np.ndarray) -> int:
        """The piece of free ground a point lies on, or the one a path from it starts on when it is not free; 0 when
        nothing is free."""
        cell = self.cell(xy) if self.holds(xy) else self.nearest_free(xy)
        return 0 if cell is None else int(self.pieces[cell])

    def path(self, a: np.ndarray, b: np.ndarray) -> Optional[np.ndarray]:
        """The shortest way from a to b over free cells, pulled straight wherever a straight line stays free; it starts
        from the free cell nearest a when a itself is not free. None when b cannot be reached."""
        if self.clear(a, b):
            return np.array([a, b], dtype=float)
        start, goal = self.nearest_free(a), self.cell(b)
        if start is None or not self.free[goal]:
            return None
        came = {start: None}
        cost = {start: 0.0}
        heap = [(0.0, start)]
        steps = [(di, dj, math.hypot(di, dj)) for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj]
        while heap:
            _, here = heapq.heappop(heap)
            if here == goal:
                break
            for di, dj, step in steps:
                nxt = (here[0] + di, here[1] + dj)
                if not (0 <= nxt[0] < self.free.shape[0] and 0 <= nxt[1] < self.free.shape[1]) or not self.free[nxt]:
                    continue
                new = cost[here] + step
                if new < cost.get(nxt, math.inf):
                    cost[nxt], came[nxt] = new, here
                    heapq.heappush(heap, (new + math.hypot(goal[0] - nxt[0], goal[1] - nxt[1]), nxt))
        if goal not in came:
            return None
        cells = [goal]
        while came[cells[-1]] is not None:
            cells.append(came[cells[-1]])
        points = [np.asarray(a, dtype=float)] + [self.point(c) for c in reversed(cells)] + [np.asarray(b, dtype=float)]
        return self.pulled(points)

    def pulled(self, points: List[np.ndarray]) -> np.ndarray:
        """A path with every corner a straight free line can skip taken out."""
        kept = [points[0]]
        i = 0
        while i < len(points) - 1:
            j = len(points) - 1
            while j > i + 1 and not self.clear(points[i], points[j]):
                j -= 1
            kept.append(points[j])
            i = j
        return np.array(kept, dtype=float)

    def sight(self, eye: np.ndarray, target: np.ndarray) -> bool:
        """Whether the line from a viewpoint to a person passes clear of every crown circle."""
        if not len(self.crowns):
            return True
        grown = self.crowns.copy()
        grown[:, 2] += SIGHT_CLEAR_M
        return bool(np.all(crowns.outside(grown, np.linspace(eye, target, 30))))


def lanes(free: Free, fence: np.ndarray, lane_m: float = LANE_M) -> List[np.ndarray]:
    """The sweep's runs: lines lane_m apart along the field's long axis, cut where the ground is not free, each a start
    and an end point, flown back and forth."""
    centred = fence - fence.mean(axis=0)
    _, _, axes = np.linalg.svd(centred, full_matrices=False)
    u, v = axes[0], np.array([-axes[0][1], axes[0][0]])
    along, across = fence @ u, fence @ v
    runs: List[np.ndarray] = []
    offsets = np.arange(across.min() + lane_m / 2.0, across.max(), lane_m)
    for k, offset in enumerate(offsets):
        steps = np.arange(along.min(), along.max(), 2.0)
        points = steps[:, None] * u + offset * v
        ok = np.array([free.holds(p) for p in points] + [False])
        lane_runs, start = [], None
        for n, good in enumerate(ok):
            if good and start is None:
                start = n
            elif not good and start is not None:
                if n - 1 > start:
                    lane_runs.append(np.array([points[start], points[n - 1]]))
                start = None
        if k % 2:
            lane_runs = [run[::-1] for run in lane_runs[::-1]]
        runs += lane_runs
    return runs


def path_length(points: np.ndarray) -> float:
    """Length of the path through the points."""
    return float(np.hypot(*np.diff(np.asarray(points, dtype=float), axis=0).T).sum()) if len(points) > 1 else 0.0


def reach(head_z: float, cruise_z: float) -> float:
    """How far across the ground from someone the drone at the cruise height keeps off, so it stays PERSON_CLEAR_M from
    the top of his head: none when it flies that far above it."""
    above = max(cruise_z - head_z, 0.0)
    return math.sqrt(PERSON_CLEAR_M ** 2 - above ** 2) if above < PERSON_CLEAR_M else 0.0


def herd(ep: FieldEpisode) -> Dict[int, Tuple[int, animals.Animal]]:
    """The animals to report by their place in the pilot's count, each with its body id: they follow everyone the
    patrol holds, on the field or passing outside it, in the order the patrol keeps them."""
    first = len(people.people(ep))
    return {first + k: (uid, animal) for k, (uid, animal) in enumerate(animals.to_report(ep).items())}


class Search:
    """The reference flight: a visit to each person and each animal from a free viewpoint that sees it, then the
    sweep while the time allows, then home, pressing return home once the straight flight to the dock stays free."""

    def __init__(self, env: Any, ep: FieldEpisode, reported: set, visit: bool = True, sweep: bool = True):
        """Read the free ground for this seed and find the places that show each person under a tree; reported is the
        set of people already reported, read each decision. With sweep off the sweep is flown only while someone the
        visits gave up on is still unreported."""
        self.env, self.ep, self.reported, self.visits, self.sweeps = env, ep, reported, visit, sweep
        # The people on the field come first in the list, so a person's place here is his index.
        self.everyone = [person for person in people.people(ep) if person.inside]
        # A walker is planned for at his home, the middle of the patch he never leaves.
        self.persons = np.array([person.xy for person in self.everyone], dtype=float).reshape(-1, 2)
        self.patches = np.array([person.patch_m for person in self.everyone])
        self.cruise_z = float(ep.dock_position[2]) + CRUISE_M
        self.reaches = np.array([reach(self._head(person), self.cruise_z) for person in self.everyone])
        self.free = Free(np.asarray(ep.fence, dtype=float), np.asarray(ep.crowns, dtype=float).reshape(-1, 3), self.persons,
                         self.patches, self.reaches)
        self.dock_xy = np.asarray(ep.dock_position[:2], dtype=float)
        self.home_piece = self.free.piece(self.dock_xy)
        self.night = camera.night(env)
        self.runs = lanes(self.free, np.asarray(ep.fence, dtype=float), NIGHT_LANE_M if self.night else LANE_M)
        self.length_m = sum(path_length(run) for run in self.runs)
        self.mode = "visit" if visit else "sweep"
        self.route: Optional[np.ndarray] = None
        self.index = 0
        self.target: Optional[int] = None
        self.look_z: Optional[float] = None
        self.at_place = False
        self.arrived_s: Optional[float] = None
        self.skipped: set = set()
        self.beasts = {n: animal for n, (_uid, animal) in herd(ep).items()}
        self.tried: Dict[int, int] = dict.fromkeys(self.subjects(), 0)
        self.pressed = -2
        self.sight: Optional[cover.Sight] = None
        self.looks: Dict[int, np.ndarray] = {}
        self.measured = 0
        for n, person in enumerate(self.everyone):
            if person.cover == cover.UNDER:
                self._looks(n)

    # The plan

    def _head(self, person: people.Person) -> float:
        """The highest the top of someone's head comes: over his own ground, or over the highest ground sampled round
        a walker's patch."""
        if person.walk is None:
            return float(person.ground_z) + float(person.head_m)
        x, y = (float(v) for v in person.xy)
        rings = [float(ring) for ring in np.arange(PATCH_RING_M, person.patch_m - 1e-9, PATCH_RING_M)] + [person.patch_m]
        # A wider ring is sampled at more points, so they stay about PATCH_RING_M apart along it.
        turns = {ring: max(RING_TURNS, int(math.ceil(2.0 * math.pi * ring / PATCH_RING_M))) for ring in rings}
        spots = [(x, y)] + [(x + ring * math.cos(2.0 * math.pi * k / turns[ring]), y + ring * math.sin(2.0 * math.pi * k / turns[ring]))
                            for ring in rings for k in range(turns[ring])]
        grounds = [site.ground_at(self.env.CLIENT, self.ep.terrain_uids, sx, sy) for sx, sy in spots]
        top = max((g for g in grounds if g is not None), default=float(person.ground_z))
        return top + PATCH_RISE_M + float(person.height_m)

    def subjects(self) -> List[int]:
        """Everyone and everything to report, by its place in the pilot's count: the people on the field, then the
        animals."""
        return [*range(len(self.everyone)), *self.beasts]

    def beast(self, n: int) -> Optional[animals.Animal]:
        """The animal at a place in the pilot's count, None for a person's."""
        return self.beasts.get(n)

    def planned(self, n: int) -> np.ndarray:
        """Where a subject is planned for, east and north: a person's home, an animal where it is now."""
        beast = self.beast(n)
        return self.persons[n] if beast is None else np.array(animals.where(beast, self.ep.step), dtype=float)

    def now(self, n: int) -> np.ndarray:
        """Where a subject is at this step, east and north."""
        beast = self.beast(n)
        return people.position(self.ep, n)[:2] if beast is None else np.array(animals.where(beast, self.ep.step), dtype=float)

    def _next_person(self, pos: np.ndarray) -> Optional[int]:
        """The nearest person or animal still to report and not given up on."""
        left = [n for n in self.subjects() if n not in self.reported and n not in self.skipped]
        return min(left, key=lambda n: float(np.hypot(*(self.planned(n) - pos[:2])))) if left else None

    def viewpoint(self, n: int, pos: np.ndarray) -> Optional[np.ndarray]:
        """The free point on the drone's piece of free ground, on a ring round subject n past his patch (an animal has
        none), that sees him past every crown and is quickest to reach."""
        best, best_length = None, math.inf
        piece = self.free.piece(pos[:2])
        centre = self.planned(n)
        patch = float(self.patches[n]) if self.beast(n) is None else 0.0
        for radius in STANDOFF_M:
            for k in range(STANDOFF_TURNS):
                angle = 2.0 * math.pi * k / STANDOFF_TURNS
                spot = centre + (patch + radius) * np.array([math.cos(angle), math.sin(angle)])
                if not self.free.holds(spot) or self.free.piece(spot) != piece or not self.free.sight(spot, centre):
                    continue
                length = float(np.hypot(*(spot - pos[:2])))
                if length < best_length:
                    best, best_length = spot, length
            if best is not None:
                return best
        return None

    def goal(self, n: int, pos: np.ndarray) -> Optional[tuple]:
        """The next place person n is looked at from, east and north, with the height to come down to there (None to
        stay at the cruise height); None once every place has been tried. An animal is looked at from a ring viewpoint
        round where it is now, BEAST_TRIES times at most."""
        if self.beast(n) is not None:
            spot = self.viewpoint(n, pos) if self.tried[n] < BEAST_TRIES else None
            return None if spot is None else (spot, None)
        if self.tried[n] == 0 and self.everyone[n].cover != cover.UNDER:
            spot = self.viewpoint(n, pos)
            if spot is not None:
                return spot, None
            self.tried[n] = 1
        looks = self._looks(n)
        k = self.tried[n] - (0 if self.everyone[n].cover == cover.UNDER else 1)
        if k >= len(looks):
            return None
        x, y, z = (float(v) for v in looks[k])
        return np.array([x, y]), (z if z < self.cruise_z - CLIMB_FIRST_M else None)

    def _looks(self, n: int) -> np.ndarray:
        """The places, east, north and up, that show person n, on the dock's piece of free ground and safe to come down
        to, best first, those showing SEEN_PX of him or more: measured in the fairness rule's order, low and near first,
        until LOOKS_KEPT show LOOK_PX of him or LOOK_FRAMES were measured; worked out once."""
        if n in self.looks:
            return self.looks[n]
        person = self.everyone[n]
        if self.sight is None:
            self.sight = cover.Sight(self.env, self.ep, np.asarray(self.ep.crowns, dtype=float).reshape(-1, 3))
        where = people.position(self.ep, person.index)
        standing = np.array([people.position(self.ep, other.index) for other in self.everyone]).reshape(-1, 3)
        places = self.sight.places(where, standing)
        places = places[[self.holdable(row) for row in places]] if len(places) else places
        found: List[tuple] = []
        target = cover.Sight.target(where, person.height_m, person.pose)
        for place, px in zip(places[:LOOK_FRAMES], self.sight.pixels(list(person.body.bodies), target, places[:LOOK_FRAMES])):
            self.measured += 1
            if px >= SEEN_PX:
                found.append((px, place[:3]))
            if sum(seen >= LOOK_PX for seen, _ in found) >= LOOKS_KEPT:
                break
        found.sort(key=lambda row: -row[0])
        self.looks[n] = np.array([place for _, place in found[:LOOKS_KEPT]], dtype=float).reshape(-1, 3)
        return self.looks[n]

    def holdable(self, place: np.ndarray) -> bool:
        """Whether the drone can look from a place: on the dock's piece of free ground, and under the cruise height
        also over a column nothing solid stands in and nobody stands near."""
        xy, z = np.asarray(place[:2], dtype=float), float(place[2])
        if not self.free.holds(xy) or self.free.piece(xy) != self.home_piece:
            return False
        if z >= self.cruise_z - CLIMB_FIRST_M:
            return True
        if np.any(np.hypot(*(self.persons - xy).T) < PERSON_CLEAR_M + self.patches):
            return False
        return self.column_clear(xy, z)

    def column_clear(self, xy: np.ndarray, z: float) -> bool:
        """Whether rays dropped from over the cruise height round a point, out to COLUMN_M, meet nothing solid before
        COLUMN_CLEAR_M under z: the ground, a stone, a bush, a trunk or anyone's body."""
        x, y = (float(v) for v in xy)
        spots = [(x, y)] + [(x + ring * math.cos(2.0 * math.pi * k / turns), y + ring * math.sin(2.0 * math.pi * k / turns))
                            for ring, turns in COLUMN_RINGS for k in range(turns)]
        top, bottom = self.cruise_z + 1.0, z - COLUMN_CLEAR_M
        hits = p.rayTestBatch([[sx, sy, top] for sx, sy in spots], [[sx, sy, bottom] for sx, sy in spots],
                              physicsClientId=self.env.CLIENT)
        drone = {int(uid) for uid in self.env.DRONE_IDS}
        return all(hit[0] < 0 or hit[0] in drone for hit in hits)

    def _sweep_route(self, pos: np.ndarray) -> Optional[np.ndarray]:
        """The sweep from where the drone is: every run in turn, joined over the free ground, entered from the end
        nearer the drone."""
        runs = list(self.runs)
        if not runs:
            return None
        if np.hypot(*(runs[-1][1] - pos[:2])) < np.hypot(*(runs[0][0] - pos[:2])):
            runs = [run[::-1] for run in runs[::-1]]
        points = [np.asarray(pos[:2], dtype=float)]
        for run in runs:
            leg = self.free.path(points[-1], run[0])
            if leg is None:
                continue
            points += list(leg[1:]) + [run[1]]
        return np.array(points, dtype=float) if len(points) > 1 else None

    # Flying

    def action(self, decision: int) -> np.ndarray:
        """This decision's flight as a full action vector, the thermal feed held at night, the report values left at
        zero."""
        a = np.zeros(ACTION_DIM, dtype=np.float32)
        a[ACTION_INDEX["gimbal_tilt"]] = -1.0
        a[ACTION_INDEX["thermal"]] = 1.0 if self.night else 0.0
        ep = self.ep
        if ep.phase == "docked":
            a[ACTION_INDEX["take_off"]] = 1.0 if decision == 0 else 0.0
            return a
        if ep.phase != "flying":
            return a
        pos = np.asarray(self.env.pos[0], dtype=float)
        home_s = HOME_DETOUR * float(np.hypot(*(self.dock_xy - pos[:2]))) / HOME_MPS + HOME_RESERVE_S
        if self.mode != "home" and ep.time_s + home_s >= float(self.env.EP_LEN_SEC):
            self._switch("home")
        if self.mode == "visit":
            return self._visit(a, pos)
        if self.mode == "sweep":
            return self._sweep(a, pos)
        return self._home(a, pos, decision)

    def _switch(self, mode: str) -> None:
        """Start another part of the flight with no route laid yet."""
        self.mode, self.route, self.index, self.target, self.arrived_s = mode, None, 0, None, None
        self.look_z, self.at_place = None, False

    def _visit(self, a: np.ndarray, pos: np.ndarray) -> np.ndarray:
        """Fly to the place the current person or animal is looked at from, come down to its height when it has one,
        and hold there facing it where it is now until a frame has shown it; a place that has not in HOLD_MAX_S is
        left for its next one."""
        if self.target is not None and (self.target in self.reported or self.target in self.skipped):
            self.target, self.route = None, None
        if self.target is None:
            while True:
                n = self._next_person(pos)
                if n is None and not self.sweeps and not self._unreported():
                    self._switch("home")
                    return self._hold(a, pos)
                if n is None:
                    self._switch("sweep")
                    return self._sweep(a, pos)
                goal = self.goal(n, pos)
                if goal is None:
                    self.skipped.add(n)
                    continue
                self.tried[n] += 1
                route = self.free.path(pos[:2], goal[0])
                if route is not None:
                    self.target, self.route, self.index, self.arrived_s = n, route, 0, None
                    self.look_z, self.at_place = goal[1], False
                    break
        person = self.now(self.target)
        if float(np.hypot(*(self.route[-1] - pos[:2]))) < ARRIVE_M and np.hypot(*self.env.vel[0][:2]) < 0.5:
            self.at_place = True
        settled = self.look_z is None or abs(pos[2] - self.look_z) < SETTLE_M
        if self.at_place and settled:
            self.arrived_s = self.ep.time_s if self.arrived_s is None else self.arrived_s
            if self.ep.time_s - self.arrived_s > HOLD_MAX_S:
                self.target, self.route = None, None
                return self._hold(a, pos, self.look_z)
        return self._fly(a, pos, face=person, look=person, z=self.look_z if self.at_place else None)

    def reach_share(self) -> float:
        """The share of the field's free ground that lies on the dock's own piece: what a drone can reach at all."""
        total = int(self.free.free.sum())
        return 0.0 if not total or not self.home_piece else round(float((self.free.pieces == self.home_piece).sum()) / total, 3)

    def _unreported(self) -> bool:
        """Whether someone on the field, or an animal to report, is still to be reported."""
        return any(n not in self.reported for n in self.subjects())

    def _sweep(self, a: np.ndarray, pos: np.ndarray) -> np.ndarray:
        """Fly the sweep, then head home; with the sweep off, home as soon as everyone is reported."""
        if not self.sweeps and not self._unreported():
            self._switch("home")
            return self._hold(a, pos)
        if self.route is None:
            self.route = self._sweep_route(pos)
        if self.route is None or float(np.hypot(*(self.route[-1] - pos[:2]))) < ARRIVE_M + 1.0 and \
                self.index >= len(self.route) - 2:
            self._switch("home")
            return self._hold(a, pos)
        return self._fly(a, pos)

    def _home(self, a: np.ndarray, pos: np.ndarray, decision: int) -> np.ndarray:
        """Climb back to the cruise height if it came down, fly towards the dock over the free ground and press return
        home once the straight flight there is free."""
        if self._low(pos):
            return self._hold(a, pos)
        if self.free.clear(pos[:2], self.dock_xy):
            if decision > self.pressed + 1:
                a[ACTION_INDEX["return_home"]] = 1.0
                self.pressed = decision
            return self._hold(a, pos)
        if self.route is None:
            self.route = self.free.path(pos[:2], self.dock_xy)
            if self.route is None:
                return self._hold(a, pos)
        return self._fly(a, pos)

    def _hold(self, a: np.ndarray, pos: np.ndarray, z: Optional[float] = None) -> np.ndarray:
        """Hover where the drone is, at a height when given, else at the cruise height."""
        self._height(a, pos, z)
        return a

    def _low(self, pos: np.ndarray) -> bool:
        """Whether the drone is so far under the cruise height that it must climb before it flies on."""
        return pos[2] < self.cruise_z - CLIMB_FIRST_M

    def _height(self, a: np.ndarray, pos: np.ndarray, z: Optional[float] = None) -> None:
        """Hold a height when given, else the cruise height above the dock."""
        vz = float(np.clip(HEIGHT_GAIN_PER_S * ((self.cruise_z if z is None else z) - pos[2]), -MAX_DESCENT_MPS,
                           MAX_CLIMB_MPS))
        a[ACTION_INDEX["move_up"]] = vz / (MAX_CLIMB_MPS if vz >= 0.0 else MAX_DESCENT_MPS)

    def _fly(self, a: np.ndarray, pos: np.ndarray, face: Optional[np.ndarray] = None,
             look: Optional[np.ndarray] = None, z: Optional[float] = None) -> np.ndarray:
        """Follow the route at a speed its bends and its end allow, facing where it flies, or a point when given, the
        camera straight down or tilted onto a point when given, at a height when given; with none given and the drone
        come down, it climbs straight back to the cruise height first."""
        route = self.route
        self.index = _progress(route, self.index, pos[:2])
        carrot, left, bend = _ahead(route, self.index, pos[:2], CARROT_M)
        # Round a bend the line to the carrot cuts the corner, so a nearer carrot on the route keeps it on free ground.
        for reach_m in NEAR_CARROTS_M:
            if self.free.clear(pos[:2], carrot):
                break
            carrot = _ahead(route, self.index, pos[:2], reach_m)[0]
        speed = min(MAX_MPS, math.sqrt(2.0 * BRAKE_MPS2 * left), math.sqrt(CORNER_MPS ** 2 + 2.0 * BRAKE_MPS2 * bend))
        heading = carrot - pos[:2]
        distance = float(np.hypot(*heading))
        speed = 0.0 if z is None and self._low(pos) else min(speed, distance)
        vx, vy = (heading / distance * speed) if distance > 1e-6 else (0.0, 0.0)
        yaw = float(self.env.rpy[0, 2])
        c, s = math.cos(yaw), math.sin(yaw)
        a[ACTION_INDEX["move_forward"]] = (vx * c + vy * s) / MAX_HORIZONTAL_MPS
        a[ACTION_INDEX["move_right"]] = (vx * s - vy * c) / MAX_HORIZONTAL_MPS
        self._height(a, pos, z)
        want = None
        if face is not None:
            want = math.atan2(face[1] - pos[1], face[0] - pos[0])
        elif speed > TURNING_MPS:
            want = math.atan2(vy, vx)
        if want is not None:
            error = (want - yaw + math.pi) % (2.0 * math.pi) - math.pi
            rate = float(np.clip(math.degrees(TURN_GAIN_PER_S * error), -MAX_YAW_RATE_DEG_S, MAX_YAW_RATE_DEG_S))
            # The turn stick is clockwise positive on the compass, the simulator's yaw counter-clockwise.
            a[ACTION_INDEX["turn"]] = -rate / MAX_YAW_RATE_DEG_S
        if look is not None:
            ground = site.ground_at(self.env.CLIENT, self.ep.terrain_uids, float(look[0]), float(look[1]))
            drop = pos[2] - ((ground if ground is not None else float(self.ep.dock_position[2])) + AIM_ABOVE_GROUND_M)
            tilt = -math.degrees(math.atan2(drop, float(np.hypot(*(look - pos[:2])))))
            a[ACTION_INDEX["gimbal_tilt"]] = float(np.clip(tilt / 90.0, -1.0, 1.0))
        return a


def _segment_gap(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Distance from a point to the segment from a to b."""
    ab = b - a
    t = float(np.clip(np.dot(point - a, ab) / max(float(np.dot(ab, ab)), 1e-12), 0.0, 1.0))
    return float(np.hypot(*(a + t * ab - point)))


def _progress(route: np.ndarray, index: int, point: np.ndarray) -> int:
    """The segment of a route the drone is on: the nearer of the next two, moved past every corner within ARRIVE_M."""
    near = route[index:index + 3]
    index += int(np.argmin([_segment_gap(point, p, q) for p, q in zip(near[:-1], near[1:])] or [0]))
    while index < len(route) - 2 and float(np.hypot(*(route[index + 1] - point))) < ARRIVE_M:
        index += 1
    return index


def _ahead(route: np.ndarray, index: int, point: np.ndarray, reach: float) -> tuple:
    """On a route, from the segment index the drone is on: the point reach metres further along but never past the
    next bend sharper than CORNER_DEG, how far the route still runs, and how far that bend lies."""
    a, b = route[index], route[min(index + 1, len(route) - 1)]
    ab = b - a
    t = float(np.clip(np.dot(point - a, ab) / max(float(np.dot(ab, ab)), 1e-12), 0.0, 1.0))
    here = a + t * ab
    corners = [here] + list(route[index + 1:])
    gone, carrot, bend = 0.0, corners[-1], math.inf
    for k in range(len(corners) - 1):
        step = float(np.hypot(*(corners[k + 1] - corners[k])))
        if carrot is corners[-1] and gone + step >= reach:
            carrot = corners[k] + (corners[k + 1] - corners[k]) * ((reach - gone) / max(step, 1e-9))
        gone += step
        if bend == math.inf and k + 2 < len(corners):
            d1, d2 = corners[k + 1] - corners[k], corners[k + 2] - corners[k + 1]
            norms = float(np.hypot(*d1) * np.hypot(*d2))
            if norms > 0.0 and float(np.dot(d1, d2)) < norms * math.cos(math.radians(CORNER_DEG)):
                bend = gone
                # A carrot past a sharp corner cuts it, and past a hairpin lands back on the drone and stops it.
                if bend < reach:
                    carrot = corners[k + 1]
    return np.asarray(carrot, dtype=float), gone, bend


def box_shares(pixels: tuple, view: Any) -> tuple:
    """A (left, top, right, bottom) pixel box as shares of the frame: centre x, centre y, width and height."""
    left, top, right, bottom = pixels
    return ((left + right) / 2.0 / view.width, (top + bottom) / 2.0 / view.height,
            (right - left) / view.width, (bottom - top) / view.height)


class Eyes:
    """Perfect eyes on the feed frames the model is shown: each new frame read once, as the people on the field and
    the animals to report it holds and the tight box around each one's pixels, from a height a report counts from;
    step is the control step the last frame read was taken on."""

    def __init__(self, env: Any, ep: FieldEpisode, outside: bool = False):
        """No frame read yet; outside also takes the people passing outside the field for people to report."""
        self.env, self.ep, self.outside = env, ep, outside
        self.last: Optional[Any] = None
        self.step: Optional[int] = None

    def fresh(self) -> Dict[int, tuple]:
        """Each person and each animal to report the feed frame shown now holds, by its place in the pilot's count,
        with its box as shares of the frame; empty for a frame already read, one that sees nothing, or one taken too
        high for a report."""
        view = sensor_noise.shown_view(self.ep, "feed")
        if view is None or view is self.last:
            return {}
        # A frame that shows nothing is still a new frame: it is read, so no box of an older one is pressed on it.
        self.last, self.step = view, int(view.step)
        if view.objects is None or not view.sees:
            return {}
        if not coverage.height_above_ground(self.env, self.ep, view.eye) <= reports.MAX_REPORT_HEIGHT_M:
            return {}
        everyone = people.people(self.ep)
        owners = {uid: key for uid, key in reports.owners(self.ep).items() if self.outside or everyone[key[1]].inside}
        owners.update({uid: ("animal", n) for n, (uid, _animal) in herd(self.ep).items()})
        boxes = sentinel_reports.object_boxes(view, owners)
        return {key[1]: box_shares(pixels, view) for key, pixels in boxes.items()}


class Pilot:
    """The search with perfect eyes, reporting each person and each animal to report once, on the first frame that
    holds it, with its box and where it stood when that frame was taken, a person as a person and an animal as an
    animal; the report button is pressed at most every other decision, so each press is a fresh rise."""

    def __init__(self, env: Any, visit: bool = True, sweep: bool = True):
        """The search and its eyes."""
        self.env, self.ep = env, env._solar
        self.reported: set = set()
        self.search = Search(env, self.ep, self.reported, visit, sweep)
        self.eyes = Eyes(env, self.ep)
        self.queue: Dict[int, tuple] = {}
        self.pressed = -2

    def action(self, decision: int, obs: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        """The search's flight, and a report when a person or an animal waits to be reported on the frame shown."""
        a = self.search.action(decision)
        if self.ep.phase not in ("flying", "returning", "landing"):
            return a
        last = self.eyes.last
        seen = self.eyes.fresh()
        # A frame read anew holds the only boxes a press may carry, so one that shows nobody leaves nobody to press.
        if self.eyes.last is not last:
            self.queue = {index: box for index, box in seen.items() if index not in self.reported}
        if decision > self.pressed + 1 and self.queue:
            index, box = next(iter(sorted(self.queue.items())))
            self.press(a, box, self.point(index, self.eyes.step), animal=self.search.beast(index) is not None)
            self.reported.add(index)
            self.queue.pop(index)
            self.pressed = decision
        return a

    def point(self, index: int, step: Optional[int] = None) -> np.ndarray:
        """Where the report says the person or the animal stands, metres east and north of the dock: a person's
        reference point, an animal's middle, at the step the reported frame was taken on, now when none is given."""
        beast = self.search.beast(index)
        if beast is None:
            return people.reference(self.ep, index, step)
        stood = animals.where(beast, self.ep.step if step is None else step)
        return np.array(stood, dtype=float) - np.asarray(self.ep.dock_position[:2], dtype=float)

    @staticmethod
    def press(a: np.ndarray, box: tuple, point: np.ndarray, animal: bool = False) -> None:
        """Press report on a person, or on an animal when told, in the feed frame with a box and a position."""
        a[ACTION_INDEX["report"]] = 1.0
        a[ACTION_INDEX["report_class"]] = 1.0 if animal else 0.0
        for name, value in zip(("cx", "cy", "w", "h"), box):
            a[ACTION_INDEX[f"report_{name}"]] = float(value)
        a[ACTION_INDEX["report_east"]] = float(np.clip(point[0] / REPORT_REACH_M, -1.0, 1.0))
        a[ACTION_INDEX["report_north"]] = float(np.clip(point[1] / REPORT_REACH_M, -1.0, 1.0))


def check_flight(_runtime: Any) -> TestFieldV2ChallengeFamily:
    """The field's own runtime for a seed check's flight, with what only a model reads left out: the laser, the
    stream's look on the colour and zoom images, every picture taken while the drone waits in the dock for its lids, and
    every picture once everyone on the field and every animal to report is validly reported, as no report is pressed
    or judged after that. The flight, the people and every rule run as in a patrol."""
    runtime = TestFieldV2ChallengeFamily()
    runtime.sense = check_sense
    runtime._build_view = functools.partial(blank_view, runtime)
    return runtime


def check_sense(env: Any, ep: FieldEpisode) -> None:
    """A check's pictures and readings: the feed's pictures while a person or an animal is still to report, so the
    pilot may still read one, and the zoom; no laser."""
    left = ep.outcome.valid_reports < ep.outcome.threats or ep.outcome.valid_animal_reports < ep.outcome.animals
    if left and not waiting_for_lids(ep):
        grass.show(env, ep, camera.upcoming(env, ep, ep.step))
        camera.update(env, ep)
    grass.show(env, ep, zoom.upcoming(env, ep))
    zoom.update(env, ep)


def blank_view(runtime: TestFieldV2ChallengeFamily, env: Any, ep: FieldEpisode,
               snapshot: dict[str, Any]) -> dict[str, np.ndarray]:
    """The view the family builds, its frames' views shown as they would be, from blank colour and zoom images."""
    return TestFieldV2ChallengeFamily._build_view(runtime, env, ep, dict(snapshot, rgb=_BLANK_RGB, zoom=_BLANK_ZOOM))


def waiting_for_lids(ep: FieldEpisode) -> bool:
    """Whether the drone still rests in the dock while its lids open. The dock then lifts it to the patrol height at
    most MAX_CLIMB_MPS before handing it over, seconds longer than a frame is shown, so no picture taken now is read."""
    return ep.phase == "docked" or (ep.phase == "taking_off" and ep.dock["model"].opening < 1.0)


def check(seed: int) -> FieldVerdict:
    """Fly one seed with the reference pilot, visits only, and judge it."""
    started = time.process_time()
    with contextlib.redirect_stdout(io.StringIO()):
        env, _obs = make_env_with_initial_obs(test_field_task(seed=int(seed), sim_dt=SIM_DT), wrap_runtime=check_flight)
    try:
        ep = env._solar
        pilot = Pilot(env, sweep=False)
        decision, peak = 0, 0.0
        while True:
            _obs, _reward, terminated, truncated, _info = env.step(pilot.action(decision)[None, :])
            decision += 1
            peak = max(peak, float(env.pos[0][2]) - float(ep.dock_position[2]))
            # The outcome is fixed once the patrol ends; the lids shutting after a landing change nothing judged here.
            if terminated or truncated or ep.outcome.end_reason:
                break
        outcome = ep.outcome
        landed = outcome.end_reason == "landed" and outcome.landed_in_dock
        result = FieldVerdict(seed=int(seed), passed=False, end_reason=outcome.end_reason,
                              landed_s=round(ep.time_s, 2) if landed else None, threats=int(outcome.threats),
                              seen=int(outcome.valid_reports), wind=wind.strength(seed),
                              route_m=round(pilot.search.length_m, 1), false_alarms=int(outcome.false_alarms),
                              animals=int(outcome.animals), animals_seen=int(outcome.valid_animal_reports),
                              wrong_animal_reports=int(outcome.wrong_animal_reports),
                              min_edge_m=None if outcome.min_edge_m is None else round(outcome.min_edge_m, 2),
                              min_crown_m=None if outcome.min_crown_m is None else round(outcome.min_crown_m, 2),
                              peak_above_dock_m=round(peak, 2), night=pilot.search.night,
                              # Rounded down, so a pass a hair inside the rule is never written as on it.
                              min_person_m=None if outcome.min_threat_distance_m is None
                              else math.floor(outcome.min_threat_distance_m * 100.0) / 100.0,
                              reach_share=pilot.search.reach_share())
    finally:
        env.close()
    result.reason = failure(result)
    result.passed = not result.reason
    result.cpu_s = round(time.process_time() - started, 1)
    return result


def failure(result: FieldVerdict) -> str:
    """What failed first on a flown seed, empty when nothing did: the landing, a person left unreported, a false
    alarm, an animal left unreported, a failed animal report, or the drone coming within the score's safe distance of
    a person."""
    if result.landed_s is None:
        return f"not_landed:{result.end_reason}"
    if result.seen < result.threats:
        return "person_unreported"
    if result.false_alarms:
        return "false_alarm"
    if result.animals_seen < result.animals:
        return "animal_unreported"
    if result.wrong_animal_reports:
        return "wrong_animal_report"
    if result.min_person_m is not None and result.min_person_m < SAFE_DISTANCE_M:
        return "too_close"
    if result.reach_share is not None and result.reach_share < MIN_REACH_SHARE:
        return "field_cut_off"
    return ""


def judge(seed: int) -> FieldVerdict:
    """The seed's verdict; a seed whose own world refuses to be built fails, the same everywhere. Any other error is
    the machine's, not the seed's, and is raised so no verdict is kept for it."""
    try:
        return check(seed)
    except (dock.NoDockGround, people.NoRoom, field_cut.NoField) as exc:
        return FieldVerdict(seed=int(seed), passed=False, reason=f"error:{exc}"[:200])


def main(argv: List[str]) -> None:
    """Judge every seed named on the command line and keep its verdict; with none named, every seed read from the
    standard input, one a line, as it arrives."""
    for seed in argv or sys.stdin:
        keep(judge(int(seed)))


if __name__ == "__main__":
    main(sys.argv[1:])
