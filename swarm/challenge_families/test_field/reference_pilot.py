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

It flies through the patrol's own environment, every frame drawn as the model's would be, at the take-off's height
under the 10 m ceiling, over the free ground only: inside the field by EDGE_CLEAR_M, off every crown by CROWN_CLEAR_M,
and off every person by PERSON_CLEAR_M, since over the field's high end 10 m above the dock is only a few metres above a
person's head. It takes off, flies to a viewpoint on a ring round each person, nearest first, faces him with the camera
tilted onto him, and holds there until a frame shows him; then it sweeps the free ground in lanes LANE_M apart with the
camera straight down while the time allows, flies home over the free ground, presses return home once the straight
flight to the dock stays free, and lands. It reports each person the first time a frame it is shown holds him, with
the tight box around his pixels and his reference point, from the object map of that very frame.

The seed passes when every person was validly reported, no report was a false alarm, the drone never tripped the edge
rule or hit anything, a crown included, and it landed back in the dock before the patrol's time ran out.

Run as a module it judges the seeds given on the command line, or else those read from its standard input, and keeps
their verdicts.
"""

from __future__ import annotations

import contextlib
import heapq
import io
import math
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np

from swarm.constants import SIM_DT
from swarm.utils.env_factory import make_env_with_initial_obs

from ..solar_patrol import coverage, sensor_noise
from ..solar_patrol import dock as sentinel_dock
from ..solar_patrol import reports as sentinel_reports
from ..solar_patrol.contract import MAX_CLIMB_MPS, MAX_DESCENT_MPS, MAX_HORIZONTAL_MPS, MAX_YAW_RATE_DEG_S
from . import crowns, dock, people, reports, wind
from .contract import ACTION_DIM, ACTION_INDEX, REPORT_REACH_M, TAKE_OFF_M
from .episode import FieldEpisode
from .seed_checks import FieldVerdict, keep
from .task import test_field_task

GRID_M = 0.5                           # the free ground is read on a grid this fine
EDGE_CLEAR_M = 3.0                     # off the field's edge: 2 m over the edge rule
CROWN_CLEAR_M = 3.0                    # off a crown circle: at a lane end next to a crown the drone swings up to 0.9 m wide
PERSON_CLEAR_M = 6.5                   # from a person's reference point: the 5 m rule, half a lying body and slack
LANE_M = 10.0                          # the wide camera sees about 14 m across from 10 m, less over the field's high end
STANDOFF_M = (7.0, 8.5, 10.0)          # how far across the ground a person is looked at from
STANDOFF_TURNS = 24                    # viewpoints tried round each of those rings
SIGHT_CLEAR_M = 0.3                    # the line of sight passes this far off every crown circle
CRUISE_M = TAKE_OFF_M                  # above the dock, where the dock's take-off hands the drone over
SAMPLE_M = 0.5                         # the line between two points is checked against the free ground this often
CARROT_M = 4.0                         # the pilot steers at the point this far ahead of it on its path
NEAR_CARROTS_M = (2.0, 1.0)            # nearer points it steers at when the line to that one leaves the free ground
MAX_MPS = 4.0                          # short of the 5 m/s limit, so turns stay tight
CORNER_DEG = 30.0                      # a bend sharper than this is taken at CORNER_MPS
CORNER_MPS = 1.0
BRAKE_MPS2 = 1.2                       # inside the outputs' own 2 m/s2 easing
ARRIVE_M = 1.0
HOLD_MAX_S = 10.0                      # a person no frame shows from his viewpoint in this long is left for the sweep
HOME_DETOUR = 1.4                      # the way home is at most this much longer than the straight line
HOME_MPS = 3.0
HOME_RESERVE_S = 45.0                  # the dock's turn, its flight home and the landing, with a seed that took 29 s of it
HEIGHT_GAIN_PER_S = 1.0
TURN_GAIN_PER_S = 2.0
TURNING_MPS = 1.0                      # the pilot faces where it flies only once it flies faster than this
AIM_ABOVE_GROUND_M = 0.9


class Free:
    """The ground the pilot may fly over, on a grid: inside the field by EDGE_CLEAR_M, off every crown by
    CROWN_CLEAR_M and every person by PERSON_CLEAR_M; with the paths across it between two points."""

    def __init__(self, fence: np.ndarray, crowns_xyr: np.ndarray, persons: np.ndarray):
        """Read which cells are free."""
        self.low = fence.min(axis=0) - GRID_M
        size = np.ceil((fence.max(axis=0) + GRID_M - self.low) / GRID_M).astype(int) + 1
        xs, ys = np.meshgrid(self.low[0] + GRID_M * np.arange(size[0]), self.low[1] + GRID_M * np.arange(size[1]),
                             indexing="ij")
        points = np.column_stack([xs.ravel(), ys.ravel()])
        free = sentinel_dock._inside(fence, points) & (sentinel_dock._edge_distance(fence, points) >= EDGE_CLEAR_M)
        grown = crowns_xyr.copy()
        grown[:, 2] += CROWN_CLEAR_M
        free &= crowns.outside(grown, points)
        for person in persons:
            free &= np.hypot(points[:, 0] - person[0], points[:, 1] - person[1]) >= PERSON_CLEAR_M
        self.free = free.reshape(size)
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


def lanes(free: Free, fence: np.ndarray) -> List[np.ndarray]:
    """The sweep's runs: lines LANE_M apart along the field's long axis, cut where the ground is not free, each a start
    and an end point, flown back and forth."""
    centred = fence - fence.mean(axis=0)
    _, _, axes = np.linalg.svd(centred, full_matrices=False)
    u, v = axes[0], np.array([-axes[0][1], axes[0][0]])
    along, across = fence @ u, fence @ v
    runs: List[np.ndarray] = []
    offsets = np.arange(across.min() + LANE_M / 2.0, across.max(), LANE_M)
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


class Search:
    """The reference flight: a visit to each person from a free viewpoint that sees him, nearest first, then the sweep
    over the free ground while the time allows, then home over the free ground, pressing return home once the straight
    flight to the dock stays free. It holds the take-off's height above the dock throughout."""

    def __init__(self, env: Any, ep: FieldEpisode, reported: set, visit: bool = True):
        """Read the free ground for this seed; reported is the set of people already reported, read each decision."""
        self.env, self.ep, self.reported, self.visits = env, ep, reported, visit
        self.persons = np.array([person.xy for person in people.people(ep)], dtype=float).reshape(-1, 2)
        self.free = Free(np.asarray(ep.fence, dtype=float), np.asarray(ep.crowns, dtype=float).reshape(-1, 3), self.persons)
        self.dock_xy = np.asarray(ep.dock_position[:2], dtype=float)
        self.runs = lanes(self.free, np.asarray(ep.fence, dtype=float))
        self.length_m = sum(path_length(run) for run in self.runs)
        self.mode = "visit" if visit else "sweep"
        self.route: Optional[np.ndarray] = None
        self.index = 0
        self.target: Optional[int] = None
        self.arrived_s: Optional[float] = None
        self.skipped: set = set()
        self.pressed = -2

    # ------------------------------------------------------------------ the plan

    def _next_person(self, pos: np.ndarray) -> Optional[int]:
        """The nearest person still to report and not given up on."""
        left = [n for n in range(len(self.persons)) if n not in self.reported and n not in self.skipped]
        return min(left, key=lambda n: float(np.hypot(*(self.persons[n] - pos[:2])))) if left else None

    def viewpoint(self, n: int, pos: np.ndarray) -> Optional[np.ndarray]:
        """The free point, on a ring round person n, that sees him past every crown and is quickest to reach."""
        best, best_length = None, math.inf
        for radius in STANDOFF_M:
            for k in range(STANDOFF_TURNS):
                angle = 2.0 * math.pi * k / STANDOFF_TURNS
                spot = self.persons[n] + radius * np.array([math.cos(angle), math.sin(angle)])
                if not self.free.holds(spot) or not self.free.sight(spot, self.persons[n]):
                    continue
                length = float(np.hypot(*(spot - pos[:2])))
                if length < best_length:
                    best, best_length = spot, length
            if best is not None:
                return best
        return None

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

    # ------------------------------------------------------------------ flying

    def action(self, decision: int) -> np.ndarray:
        """This decision's flight as a full action vector, the report values left at zero."""
        a = np.zeros(ACTION_DIM, dtype=np.float32)
        a[ACTION_INDEX["gimbal_tilt"]] = -1.0
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

    def _visit(self, a: np.ndarray, pos: np.ndarray) -> np.ndarray:
        """Fly to the current person's viewpoint and hold there facing him until a frame has shown him."""
        if self.target is not None and (self.target in self.reported or self.target in self.skipped):
            self.target, self.route = None, None
        if self.target is None:
            while True:
                n = self._next_person(pos)
                if n is None:
                    self._switch("sweep")
                    return self._sweep(a, pos)
                spot = self.viewpoint(n, pos)
                route = None if spot is None else self.free.path(pos[:2], spot)
                if route is not None:
                    self.target, self.route, self.index, self.arrived_s = n, route, 0, None
                    break
                self.skipped.add(n)
        person = self.persons[self.target]
        if float(np.hypot(*(self.route[-1] - pos[:2]))) < ARRIVE_M and np.hypot(*self.env.vel[0][:2]) < 0.5:
            self.arrived_s = self.ep.time_s if self.arrived_s is None else self.arrived_s
            if self.ep.time_s - self.arrived_s > HOLD_MAX_S:
                self.skipped.add(self.target)
        return self._fly(a, pos, face=person, look=person)

    def _sweep(self, a: np.ndarray, pos: np.ndarray) -> np.ndarray:
        """Fly the sweep, then head home."""
        if self.route is None:
            self.route = self._sweep_route(pos)
        if self.route is None or float(np.hypot(*(self.route[-1] - pos[:2]))) < ARRIVE_M + 1.0 and \
                self.index >= len(self.route) - 2:
            self._switch("home")
            return self._hold(a, pos)
        return self._fly(a, pos)

    def _home(self, a: np.ndarray, pos: np.ndarray, decision: int) -> np.ndarray:
        """Fly towards the dock over the free ground and press return home once the straight flight there is free."""
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

    def _hold(self, a: np.ndarray, pos: np.ndarray) -> np.ndarray:
        """Hover where the drone is, at the cruise height."""
        self._height(a, pos)
        return a

    def _height(self, a: np.ndarray, pos: np.ndarray) -> None:
        """Hold the cruise height above the dock."""
        vz = float(np.clip(HEIGHT_GAIN_PER_S * (float(self.ep.dock_position[2]) + CRUISE_M - pos[2]),
                           -MAX_DESCENT_MPS, MAX_CLIMB_MPS))
        a[ACTION_INDEX["move_up"]] = vz / (MAX_CLIMB_MPS if vz >= 0.0 else MAX_DESCENT_MPS)

    def _fly(self, a: np.ndarray, pos: np.ndarray, face: Optional[np.ndarray] = None,
             look: Optional[np.ndarray] = None) -> np.ndarray:
        """Follow the route at a speed its bends and its end allow, facing where it flies, or a point when given, the
        camera straight down or tilted onto a point when given."""
        route = self.route
        self.index = _progress(route, self.index, pos[:2])
        carrot, left, bend = _ahead(route, self.index, pos[:2], CARROT_M)
        # Round a bend the straight line to the carrot cuts the corner: a nearer carrot on the route keeps it on free
        # ground, and the nearest still moves the drone along.
        for reach in NEAR_CARROTS_M:
            if self.free.clear(pos[:2], carrot):
                break
            carrot = _ahead(route, self.index, pos[:2], reach)[0]
        speed = min(MAX_MPS, math.sqrt(2.0 * BRAKE_MPS2 * left), math.sqrt(CORNER_MPS ** 2 + 2.0 * BRAKE_MPS2 * bend))
        heading = carrot - pos[:2]
        distance = float(np.hypot(*heading))
        speed = min(speed, distance)
        vx, vy = (heading / distance * speed) if distance > 1e-6 else (0.0, 0.0)
        yaw = float(self.env.rpy[0, 2])
        c, s = math.cos(yaw), math.sin(yaw)
        a[ACTION_INDEX["move_forward"]] = (vx * c + vy * s) / MAX_HORIZONTAL_MPS
        a[ACTION_INDEX["move_right"]] = (vx * s - vy * c) / MAX_HORIZONTAL_MPS
        self._height(a, pos)
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
            ground = people.ground_at(self.env.CLIENT, self.ep.terrain_uids, float(look[0]), float(look[1]))
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
    """Perfect eyes on the feed frames the model is shown: each new frame read once, as the people it holds and the
    tight box around each one's pixels, from a height a report counts from."""

    def __init__(self, env: Any, ep: FieldEpisode):
        """No frame read yet."""
        self.env, self.ep = env, ep
        self.last: Optional[Any] = None

    def fresh(self) -> Dict[int, tuple]:
        """Each person the feed frame shown now holds, by index, with his box as shares of the frame; empty for a frame
        already read, one that sees nothing, or one taken too high for a report."""
        view = sensor_noise.shown_view(self.ep, "feed")
        if view is None or view is self.last or view.objects is None or not view.sees:
            return {}
        self.last = view
        if not coverage.height_above_ground(self.env, self.ep, view.eye) <= reports.MAX_REPORT_HEIGHT_M:
            return {}
        boxes = sentinel_reports.object_boxes(view, reports.owners(self.ep))
        return {key[1]: box_shares(pixels, view) for key, pixels in boxes.items()}


class Pilot:
    """The search with perfect eyes, reporting each person once, on the first frame that holds him, with his box and
    his reference point; the report button is pressed at most every other decision, so each press is a fresh rise."""

    def __init__(self, env: Any, visit: bool = True):
        """The search and its eyes."""
        self.env, self.ep = env, env._solar
        self.reported: set = set()
        self.search = Search(env, self.ep, self.reported, visit)
        self.eyes = Eyes(env, self.ep)
        self.queue: Dict[int, tuple] = {}
        self.pressed = -2

    def action(self, decision: int, obs: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        """The search's flight, and a report when a person waits to be reported on the frame shown."""
        a = self.search.action(decision)
        if self.ep.phase not in ("flying", "returning", "landing"):
            return a
        seen = self.eyes.fresh()
        self.queue = {index: box for index, box in seen.items() if index not in self.reported} if seen else self.queue
        if decision > self.pressed + 1 and self.queue:
            index, box = next(iter(sorted(self.queue.items())))
            self.press(a, box, self.point(index))
            self.reported.add(index)
            self.queue.pop(index)
            self.pressed = decision
        return a

    def point(self, index: int) -> np.ndarray:
        """Where the report says the person stands, metres east and north of the dock: his reference point."""
        return people.reference(self.ep, index)

    @staticmethod
    def press(a: np.ndarray, box: tuple, point: np.ndarray) -> None:
        """Press report on a person in the feed frame with a box and a position."""
        a[ACTION_INDEX["report"]] = 1.0
        for name, value in zip(("cx", "cy", "w", "h"), box):
            a[ACTION_INDEX[f"report_{name}"]] = float(value)
        a[ACTION_INDEX["report_east"]] = float(np.clip(point[0] / REPORT_REACH_M, -1.0, 1.0))
        a[ACTION_INDEX["report_north"]] = float(np.clip(point[1] / REPORT_REACH_M, -1.0, 1.0))


def check(seed: int) -> FieldVerdict:
    """Fly one seed with the reference pilot and judge it."""
    started = time.process_time()
    with contextlib.redirect_stdout(io.StringIO()):
        env, _obs = make_env_with_initial_obs(test_field_task(seed=int(seed), sim_dt=SIM_DT))
    try:
        ep = env._solar
        pilot = Pilot(env)
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
                              min_edge_m=None if outcome.min_edge_m is None else round(outcome.min_edge_m, 2),
                              min_crown_m=None if outcome.min_crown_m is None else round(outcome.min_crown_m, 2),
                              peak_above_dock_m=round(peak, 2))
    finally:
        env.close()
    if not landed:
        result.reason = f"not_landed:{result.end_reason}"
    elif result.seen < result.threats:
        result.reason = "person_unreported"
    elif result.false_alarms:
        result.reason = "false_alarm"
    result.passed = not result.reason
    result.cpu_s = round(time.process_time() - started, 1)
    return result


def judge(seed: int) -> FieldVerdict:
    """The seed's verdict; a seed whose own world refuses to be built fails, the same everywhere. Any other error is
    the machine's, not the seed's, and is raised so no verdict is kept for it."""
    try:
        return check(seed)
    except (dock.NoDockGround, people.NoRoom) as exc:
        return FieldVerdict(seed=int(seed), passed=False, reason=f"error:{exc}"[:200])


def main(argv: List[str]) -> None:
    """Judge every seed named on the command line and keep its verdict; with none named, every seed read from the
    standard input, one a line, as it arrives."""
    for seed in argv or sys.stdin:
        keep(judge(int(seed)))


if __name__ == "__main__":
    main(sys.argv[1:])
