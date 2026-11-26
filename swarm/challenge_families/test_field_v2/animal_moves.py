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

"""How the animals near the field spend a patrol: every one planned when the seed starts, as a track in time.

The animals of a seed live together on a grid of the ground they may use. Their whole patrol is played out once, at
reset, in steps of SAMPLE_S: each animal works through a queue of things to do (play a clip for a while, walk to a
spot, drift on while grazing or sniffing), its mind fills the queue when it runs dry, and a step that would take its
body off the ground it may use, or into another animal's, is not taken. Every step's place, heading and clip frame is
kept, so where an animal stands and what pose it holds is a function of time alone, read back at any step.

Wild boars (a herd's mind) and rats or mice are planned together with the goats and dogs, each keeping clear of the
others. Birds are planned after all of them, against their finished tracks, so a seed's animals on the ground move
the same with birds or without them. A bird keeps clear of them and of the other birds, takes off when a dog or goat is about
to come close, and every so often flies low to another spot: its height over the ground under it is kept with the
track, and the straight line of its flight lies over open sky the ground lists, clear of crowns, houses and people.

Two machines plan the same seed the same way. Every choice the plan makes reads only the seed's generators (PCG64
draws, scaled by one multiply and one add), floor, comparisons, and IEEE addition, subtraction, multiplication,
division and square root, one operation at a time, which give the same bits on every machine. Headings are unit
vectors turned by fixed rational rotations, and the compass is a table of rational unit vectors, so no sine, cosine,
arctangent or exponential decides anything. The ground grid is the caller's.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .animal_skins import Body, Kind, follows, route, wrapped

SAMPLE_S = 0.1                         # the plan's step, and the spacing of the kept track
ACCEL_M_S2 = 1.5                       # how fast a walking body gathers speed or slows to a stop
ARRIVE_M = 0.3                         # a walk ends this close to its spot
SLOW_IN_TURN = 0.45                    # share of its speed a body keeps while facing straight away from its spot
TURN_T = 0.048                         # tan of half the turn one step allows: about 55 degrees a second
GAP_M = 0.1                            # bodies keep at least this much air between their reach circles
AHEAD_M = 0.6                          # a walking animal stops for another this close in front of its reach
WAIT_S = 1.2                           # and stands this long before it tries again
WAITS = 3                              # tries before it gives the walk up
WALK_LIMIT_S = 60.0                    # a walk that has not arrived by then is given up
PATH_STEP_M = 0.25                     # a straight path is checked against the ground this often
DRIFT_TURNS = (-0.006, -0.003, 0.0, 0.003, 0.006)   # tan of half the step's turn while grazing or sniffing on


def _rotation(t: float) -> Tuple[float, float]:
    """Cosine and sine of the rotation whose half-angle has tangent t, from IEEE arithmetic alone."""
    return (1.0 - t * t) / (1.0 + t * t), (2.0 * t) / (1.0 + t * t)


TURN_COS, TURN_SIN = _rotation(TURN_T)
_QUARTER = ((1.0, 0.0), (12.0 / 13.0, 5.0 / 13.0), (119.0 / 169.0, 120.0 / 169.0), (5.0 / 13.0, 12.0 / 13.0))
COMPASS = tuple(v for quarter in range(4) for v in (
    [(x, y) for x, y in _QUARTER] if quarter == 0 else
    [(-y, x) for x, y in _QUARTER] if quarter == 1 else
    [(-x, -y) for x, y in _QUARTER] if quarter == 2 else
    [(y, -x) for x, y in _QUARTER]))


def uniform(rng: np.random.Generator, low: float, high: float) -> float:
    """A float drawn evenly between two bounds, from one draw and one multiply and add."""
    return low + (high - low) * float(rng.random())


def pick(rng: np.random.Generator, weights: Sequence[Tuple[Any, float]]) -> Any:
    """One choice of a list of (choice, weight), drawn in proportion, adding the weights in order."""
    total = 0.0
    for _, weight in weights:
        total += weight
    draw = float(rng.random()) * total
    for choice, weight in weights:
        draw -= weight
        if draw < 0.0:
            return choice
    return weights[-1][0]


# The ground the animals may use
class Ground:
    """The ground the animals may use on a square grid, rows running north and columns east: open cells, which of them
    lie on the field, and the ground's height for standing a body on it.

    A body of a given reach may stand where every cell its reach can touch is open: the open cells eroded, all eight
    neighbours counted, once for the cell itself and once for each whole step of its reach.
    """

    def __init__(self, origin: Tuple[float, float], step: float, open_: np.ndarray, inside: np.ndarray,
                 height: np.ndarray, sky: Optional[np.ndarray] = None):
        """A grid whose cell (0, 0) is centred on the origin; sky, the cells a bird may fly over, is the open cells
        unless given."""
        self.origin = (float(origin[0]), float(origin[1]))
        self.step = float(step)
        self.open = np.asarray(open_, dtype=bool)
        self.inside = np.asarray(inside, dtype=bool)
        self.height = np.asarray(height, dtype=float)
        self.sky = self.open if sky is None else np.asarray(sky, dtype=bool)
        self._levels: Dict[int, np.ndarray] = {0: self.open}
        self._clear_sky: Optional[np.ndarray] = None

    def level(self, k: int) -> np.ndarray:
        """The open cells eroded k times."""
        if k not in self._levels:
            self._levels[k] = _eroded(self.level(k - 1))
        return self._levels[k]

    def side(self, on_field: bool, clear_m: float) -> "Ground":
        """The same grid with only the open cells of one side of the field's edge, clear_m and more from it: the
        ground of an animal that never crosses the edge."""
        mask = self.inside if on_field else ~self.inside
        for _ in range(int(math.ceil(clear_m / self.step))):
            mask = _eroded(mask)
        return Ground(self.origin, self.step, self.open & mask, self.inside, self.height, self.sky)

    def for_birds(self) -> "Ground":
        """The same grid with only the open cells under open sky, where a bird may stand and from where it may fly."""
        return Ground(self.origin, self.step, self.open & self.sky, self.inside, self.height, self.sky)

    def over(self, x0: float, y0: float, x1: float, y1: float) -> bool:
        """Whether every point of the straight line between two points, checked every PATH_STEP_M, lies over a cell
        of sky whose eight neighbours are sky too, so no point of it comes over anything the sky keeps out."""
        if self._clear_sky is None:
            self._clear_sky = _eroded(self.sky)
        cells = self._clear_sky
        dx, dy = x1 - x0, y1 - y0
        steps = int(math.floor(math.sqrt(dx * dx + dy * dy) / PATH_STEP_M)) + 1
        for n in range(steps + 1):
            row, col = self.cell(x0 + dx * (n / steps), y0 + dy * (n / steps))
            if not (0 <= row < cells.shape[0] and 0 <= col < cells.shape[1] and bool(cells[row, col])):
                return False
        return True

    def level_for(self, reach_m: float) -> int:
        """How often the open cells are eroded for a body that reaches this far from its centre."""
        return 1 + int(math.floor(reach_m / self.step))

    def cell(self, x: float, y: float) -> Tuple[int, int]:
        """Row and column of the cell holding a point."""
        return (int(math.floor((y - self.origin[1]) / self.step + 0.5)),
                int(math.floor((x - self.origin[0]) / self.step + 0.5)))

    def centre(self, row: int, col: int) -> Tuple[float, float]:
        """East and north of a cell's centre."""
        return self.origin[0] + col * self.step, self.origin[1] + row * self.step

    def free(self, k: int, x: float, y: float) -> bool:
        """Whether a body of erosion level k may stand with its centre on a point; nothing off the grid is free."""
        row, col = self.cell(x, y)
        cells = self.level(k)
        return 0 <= row < cells.shape[0] and 0 <= col < cells.shape[1] and bool(cells[row, col])

    def path_free(self, k: int, x0: float, y0: float, x1: float, y1: float) -> bool:
        """Whether every point of the straight line between two points, checked every PATH_STEP_M, is free."""
        dx, dy = x1 - x0, y1 - y0
        steps = int(math.floor(math.sqrt(dx * dx + dy * dy) / PATH_STEP_M)) + 1
        return all(self.free(k, x0 + dx * (n / steps), y0 + dy * (n / steps)) for n in range(steps + 1))

    def room(self, k: int, x: float, y: float, d: Tuple[float, float], reach: float) -> float:
        """How far a body can go straight on from a point along a direction, up to reach, before the ground stops it."""
        gone = 0.0
        while gone + PATH_STEP_M <= reach and self.free(k, x + d[0] * (gone + PATH_STEP_M), y + d[1] * (gone + PATH_STEP_M)):
            gone += PATH_STEP_M
        return gone

    def launches(self, k: int, x: float, y: float) -> int:
        """How many flights a body of erosion level k could take from a point: along each compass direction, every
        LAUNCH_STEP_M from the shortest flight to the longest, a spot it may stand on over a line of clear sky."""
        count = 0
        steps = int(math.floor((BIRD_FLIGHT_M[1] - BIRD_FLIGHT_M[0]) / LAUNCH_STEP_M))
        for ux, uy in COMPASS:
            for n in range(steps + 1):
                length = BIRD_FLIGHT_M[0] + n * LAUNCH_STEP_M
                end_x, end_y = x + ux * length, y + uy * length
                count += self.free(k, end_x, end_y) and self.over(x, y, end_x, end_y)
        return count

    def is_inside(self, x: float, y: float) -> bool:
        """Whether a point lies on a cell of the field."""
        row, col = self.cell(x, y)
        return 0 <= row < self.inside.shape[0] and 0 <= col < self.inside.shape[1] and bool(self.inside[row, col])

    def height_at(self, xy: np.ndarray) -> np.ndarray:
        """Ground height under each point, blended from the four cells round it; used to draw, never to decide."""
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        fx = np.clip((xy[:, 0] - self.origin[0]) / self.step, 0.0, self.height.shape[1] - 1.001)
        fy = np.clip((xy[:, 1] - self.origin[1]) / self.step, 0.0, self.height.shape[0] - 1.001)
        c, r = fx.astype(int), fy.astype(int)
        sx, sy = fx - c, fy - r
        h = self.height
        return ((h[r, c] * (1 - sx) + h[r, c + 1] * sx) * (1 - sy)
                + (h[r + 1, c] * (1 - sx) + h[r + 1, c + 1] * sx) * sy)


# One animal's plan and its track
@dataclass(frozen=True)
class Segment:
    """A stretch of the track in one clip: when it starts, and the clip and frame it eases out of, if any."""

    t0: float
    clip: str
    held: Optional[Tuple[str, float]] = None


@dataclass
class Track:
    """An animal's whole planned patrol: at every sample its place, heading, clip frame and how fast that frame runs,
    and the clip stretch it is in."""

    x: np.ndarray
    y: np.ndarray
    hx: np.ndarray
    hy: np.ndarray
    frame: np.ndarray
    rate: np.ndarray
    segment: np.ndarray
    segments: List[Segment]
    up: Optional[np.ndarray] = None       # height over the ground under it at each sample; none for a body never off it

    def index(self, t: float) -> Tuple[int, float]:
        """The sample at or before a time, and how far past it the time lies, in seconds."""
        k = min(max(int(math.floor(t / SAMPLE_S)), 0), len(self.x) - 1)
        return k, max(t - k * SAMPLE_S, 0.0)

    def place(self, t: float) -> Tuple[float, float, float, float]:
        """East, north and the unit heading at a time, between the samples round it."""
        k, tau = self.index(t)
        nxt = min(k + 1, len(self.x) - 1)
        share = tau / SAMPLE_S
        hx = self.hx[k] + (self.hx[nxt] - self.hx[k]) * share
        hy = self.hy[k] + (self.hy[nxt] - self.hy[k]) * share
        length = math.sqrt(hx * hx + hy * hy) or 1.0
        return (float(self.x[k] + (self.x[nxt] - self.x[k]) * share),
                float(self.y[k] + (self.y[nxt] - self.y[k]) * share), float(hx / length), float(hy / length))

    def lift(self, t: float) -> float:
        """How high over the ground under it the body is at a time, between the samples round it."""
        if self.up is None:
            return 0.0
        k, tau = self.index(t)
        nxt = min(k + 1, len(self.up) - 1)
        return float(self.up[k] + (self.up[nxt] - self.up[k]) * (tau / SAMPLE_S))

    def pose(self, t: float) -> Tuple[Segment, float, float]:
        """The clip stretch at a time, the frame its clip shows, and how long the stretch has run."""
        k, tau = self.index(t)
        segment = self.segments[int(self.segment[k])]
        return segment, float(self.frame[k] + self.rate[k] * tau), t - segment.t0


@dataclass
class Mover:
    """One animal while its patrol is planned: who it is, where it stands and faces, what it plays and does next."""

    kind: Kind
    body: Body
    size: float
    pace: float
    x: float
    y: float
    hx: float
    hy: float
    rng: np.random.Generator
    ground: Ground
    mind: Any = None
    reach: float = 0.0
    level: int = 0
    clip: Optional[str] = None
    frame: float = 0.0
    speed: float = 0.0
    action: Optional[tuple] = None
    actions: Deque[tuple] = field(default_factory=deque)
    since: float = 0.0
    waits: int = 0
    target: Optional[Tuple[float, float]] = None
    samples: List[tuple] = field(default_factory=list)
    segments: List[Segment] = field(default_factory=list)
    up: float = 0.0
    landing: Optional[Tuple[float, float]] = None   # the spot a queued or running flight lands on, taken from the start

    def __post_init__(self):
        """The reach at this size and the ground level it stands on."""
        self.reach = self.body.reach_m * self.size
        self.level = self.ground.level_for(self.reach)

    def full_speed(self, clip: str) -> float:
        """How fast the body travels in a clip at this animal's size and pace."""
        return self.body.clips[clip].speed_m_s * self.pace * self.size

    def holds(self) -> Tuple[Tuple[float, float], ...]:
        """Every spot of ground the body takes: where it stands unless it is in the air, and from the moment a flight is
        queued until it lands the spot it lands on."""
        here = () if self.up > 0.0 else ((self.x, self.y),)
        return here if self.landing is None else here + (self.landing,)

    def clear_of(self, x: float, y: float, others: Sequence["Mover"]) -> bool:
        """Whether the body could stand at a point with every other animal's body on the ground it takes now."""
        for other in others:
            if other is self:
                continue
            gap = self.reach + other.reach + GAP_M
            for ox, oy in other.holds():
                dx, dy = x - ox, y - oy
                if dx * dx + dy * dy < gap * gap:
                    return False
        return True

    def lands_clear(self, x: float, y: float, flight: Any, t: float, others: Sequence[Any]) -> bool:
        """Whether a flight that starts now may still land on its spot: no other mover holds the spot, and no animal
        planned before comes close to it round the touch-down. One only crossing the spot now does not stop it."""
        touch = int(round(t / SAMPLE_S)) + int(math.floor(flight.touch_s / SAMPLE_S))
        for other in others:
            if isinstance(other, Shadow) and other.near(x, y, self.reach + other.reach + LAND_GAP_M,
                                                        touch + LAND_WINDOW[0], touch + LAND_WINDOW[1]):
                return False
        return self.clear_of(x, y, [other for other in others if not isinstance(other, Shadow)])

    def can_stand(self, x: float, y: float, others: Sequence["Mover"]) -> bool:
        """Whether the body may stand at a point: on its ground, clear of every other animal."""
        return self.ground.free(self.level, x, y) and self.clear_of(x, y, others)

    def queue(self, *actions: tuple) -> None:
        """Add things to do after what is queued."""
        self.actions.extend(actions)

    def _switch(self, clip: str, t: float, frame: float = 0.0) -> None:
        """Start a clip at a time, easing out of the one playing."""
        if clip == self.clip:
            return
        held = None if self.clip is None else (self.clip, wrapped(self.body.clips[self.clip], self.frame))
        self.segments.append(Segment(t0=t, clip=clip, held=held))
        self.clip, self.frame = clip, frame

    def _begin(self, t: float, others: Sequence["Mover"] = ()) -> None:
        """Take the next thing to do, reaching its clip through the clips the table lets lie between; a flight that
        can no longer land on its spot, taken since it was queued, is dropped, and the mind asked again."""
        action = self.actions.popleft()
        clip = action[1]
        if self.clip is not None and not follows(self.kind, self.clip, clip):
            self.actions.appendleft(action)
            for between in reversed(route(self.kind, self.clip, clip)[:-1]):
                self.actions.appendleft(self._between(between, action))
            action = self.actions.popleft()
            clip = action[1]
        if action[0] == "air" and not self.lands_clear(action[2], action[3], action[4], t, others):
            self.landing = None
            self.mind.landing_taken()
            action, clip = ("play", self.clip, SAMPLE_S), self.clip
        start = action[4] if action[0] == "play" and len(action) > 4 else 0.0
        self._switch(clip, t, start)
        self.action, self.since, self.target = action, 0.0, None
        if action[0] in ("go", "air"):
            self.target = (action[2], action[3])
        if action[0] in ("play", "air"):
            self.speed = 0.0

    def _between(self, clip: str, toward: tuple) -> tuple:
        """A clip played on the way to another: a clip that ends plays to its end, one that loops for one blend, and
        on the way to a walk a travelling clip already walks."""
        spec = self.body.clips[clip]
        if toward[0] == "go" and spec.speed_m_s > 0.0:
            return ("go", clip, toward[2], toward[3], self.kind.blend_s)
        return ("play", clip, self.kind.blend_s if spec.loop else None)

    def _done(self) -> bool:
        """Whether the thing being done is over."""
        kind, clip = self.action[0], self.action[1]
        spec = self.body.clips[clip]
        if kind == "play":
            seconds = self.action[2]
            if seconds is None:
                return self.frame >= (spec.frames if spec.loop else spec.frames - 1)
            return self.since >= seconds
        if kind == "drift":
            return self.since >= self.action[2]
        if kind == "air":
            return self.target is None
        return self.target is None or (len(self.action) > 4 and self.since >= self.action[4])

    def prepare(self, k: int, others: Sequence["Mover"]) -> None:
        """Decide what the body does over the step from sample k: finish what is done, take what comes next."""
        t = k * SAMPLE_S
        if self.mind.urgent(self, k, others):
            self.actions.clear()
            self.action, self.target, self.speed, self.landing = None, None, 0.0, None
        while self.action is None or self._done():
            if not self.actions:
                self.mind.decide(self, k, others)
            self._begin(t, others)

    def record(self) -> None:
        """Keep the sample: place, heading, clip frame, how fast the frame runs over the step, and the stretch."""
        spec = self.body.clips[self.clip]
        if spec.speed_m_s > 0.0 and self.action[0] != "air":
            rate = spec.fps * self.speed / (spec.speed_m_s * self.size)
        else:
            rate = spec.fps * self.pace
        self.samples.append((self.x, self.y, self.hx, self.hy, self.frame, rate, len(self.segments) - 1, self.up))

    def move(self, k: int, others: Sequence["Mover"]) -> None:
        """Play the step: turn, travel if the step is allowed, and run the clip on."""
        kind = self.action[0]
        if kind == "air":
            self.since += SAMPLE_S
            self._fly(k)
            return
        if kind == "go":
            self._walk(others)
        elif kind == "drift":
            self._drift(k, others)
        rate = self.samples[-1][5]
        self.frame += rate * SAMPLE_S
        self.since += SAMPLE_S

    def _turn_towards(self, ux: float, uy: float, turn: Tuple[float, float] = (TURN_COS, TURN_SIN)) -> float:
        """Turn the heading towards a unit direction by at most one step's turn; returns how well it now faces it."""
        cosine, sine = turn
        facing = self.hx * ux + self.hy * uy
        if facing >= cosine:
            self.hx, self.hy = ux, uy
            return 1.0
        sine = sine if self.hx * uy - self.hy * ux >= 0.0 else -sine
        hx = self.hx * cosine - self.hy * sine
        hy = self.hx * sine + self.hy * cosine
        length = math.sqrt(hx * hx + hy * hy)
        self.hx, self.hy = hx / length, hy / length
        return self.hx * ux + self.hy * uy

    def _walk(self, others: Sequence["Mover"]) -> None:
        """One step of a walk to its spot: turn towards it, slower while turning, stop for an animal in the way."""
        tx, ty = self.target
        dx, dy = tx - self.x, ty - self.y
        gone = math.sqrt(dx * dx + dy * dy)
        if gone <= ARRIVE_M or self.since >= WALK_LIMIT_S:
            self.target, self.speed, self.waits = None, 0.0, 0
            return
        facing = self._turn_towards(dx / gone, dy / gone)
        want = self.full_speed(self.action[1]) * (SLOW_IN_TURN + (1.0 - SLOW_IN_TURN) * max(facing, 0.0))
        want = min(want, math.sqrt(2.0 * ACCEL_M_S2 * gone))
        if self._blocked(others):
            self.speed, self.target = 0.0, None
            if self.waits < WAITS:
                self.waits += 1
                self.actions.appendleft(self.action)
                self.actions.appendleft(("play", "stand", WAIT_S))
            else:
                self.waits = 0
            return
        self.speed = min(want, self.speed + ACCEL_M_S2 * SAMPLE_S) if want >= self.speed else max(want, self.speed - ACCEL_M_S2 * SAMPLE_S)
        if not self._step(others):
            self.speed, self.target, self.waits = 0.0, None, 0
        elif not self.speed and not want:
            self.target = None

    def _blocked(self, others: Sequence["Mover"]) -> bool:
        """Whether another animal stands close in front of a walking one; one in the air is no obstacle."""
        for other in others:
            if other is self or other.up > 0.0:
                continue
            dx, dy = other.x - self.x, other.y - self.y
            near = self.reach + other.reach + AHEAD_M
            if dx * dx + dy * dy < near * near and dx * self.hx + dy * self.hy > 0.0:
                return True
        return False

    def _drift(self, k: int, others: Sequence["Mover"]) -> None:
        """One step of grazing or sniffing on: a gentle turn, a slow pace, ended where the ground or the mind says no."""
        c, s = _rotation(self.action[3])
        hx, hy = self.hx * c - self.hy * s, self.hx * s + self.hy * c
        length = math.sqrt(hx * hx + hy * hy)
        self.hx, self.hy = hx / length, hy / length
        want = self.full_speed(self.action[1])
        self.speed = min(want, self.speed + ACCEL_M_S2 * SAMPLE_S)
        nx, ny = self.x + self.speed * SAMPLE_S * self.hx, self.y + self.speed * SAMPLE_S * self.hy
        if not self.mind.leash(self, k, nx, ny) or not self._step(others):
            self.speed = 0.0
            self.since = self.action[2]

    def _step(self, others: Sequence["Mover"]) -> bool:
        """Travel one step at the current speed and heading if the body may stand there; whether it did."""
        nx, ny = self.x + self.speed * SAMPLE_S * self.hx, self.y + self.speed * SAMPLE_S * self.hy
        if not self.can_stand(nx, ny, others):
            return False
        self.x, self.y = nx, ny
        return True

    def _fly(self, k: int) -> None:
        """One step of a flight, to the time the action has now run: where the body is along it, how high, which way
        it faces, and the flight's clip at the frame it has reached."""
        flight = self.action[4]
        t = self.since
        if t >= flight.end_s:
            self.x, self.y, self.up, self.target, self.landing = self.target[0], self.target[1], 0.0, None, None
            return
        self._turn_towards(flight.ux, flight.uy, (BIRD_TURN_COS, BIRD_TURN_SIN))
        gone = flight.along(t)
        self.x, self.y = flight.x0 + flight.ux * gone, flight.y0 + flight.uy * gone
        self.up = flight.height(t)
        clip, began = flight.clip(t)
        self._switch(clip, (k + 1) * SAMPLE_S)
        self.frame = (t - began) * self.body.clips[clip].fps * self.pace

    def track(self) -> Track:
        """The kept samples as a track; a body that never left the ground keeps no heights."""
        cols = list(zip(*self.samples))
        up = np.array(cols[7]) if any(cols[7]) else None
        return Track(x=np.array(cols[0]), y=np.array(cols[1]), hx=np.array(cols[2]), hy=np.array(cols[3]),
                     frame=np.array(cols[4]), rate=np.array(cols[5]), segment=np.array(cols[6], dtype=np.int64),
                     segments=list(self.segments), up=up)


class Shadow:
    """An animal planned before, standing where its track puts it at each sample, for later movers to keep clear of."""

    def __init__(self, track: Track, reach: float):
        """A shadow of a track, at its first sample, knowing the most it moves in one sample."""
        self.track, self.reach = track, float(reach)
        dx, dy = np.diff(track.x), np.diff(track.y)
        # Correctly rounded operations only: it gates a bird's decision, which must come out alike on every machine.
        self.fastest = float(np.sqrt(dx * dx + dy * dy).max(initial=0.0))
        self.target: Optional[Tuple[float, float]] = None
        self.landing: Optional[Tuple[float, float]] = None
        self.up = 0.0
        self.at(0)

    def at(self, k: int) -> None:
        """Stand where the track is at a sample."""
        k = min(k, len(self.track.x) - 1)
        self.x, self.y = float(self.track.x[k]), float(self.track.y[k])

    def holds(self) -> Tuple[Tuple[float, float], ...]:
        """The ground the body takes: where it stands."""
        return ((self.x, self.y),)

    def near(self, x: float, y: float, gap: float, first: int, last: int) -> bool:
        """Whether the body comes closer than a gap to a point at any sample from first to last."""
        n = len(self.track.x)
        first, last = min(max(first, 0), n - 1), min(max(last, 0), n - 1)
        dx = self.track.x[first:last + 1] - x
        dy = self.track.y[first:last + 1] - y
        return bool((dx * dx + dy * dy < gap * gap).any())


def live(movers: Sequence[Mover], horizon_s: float, shadows: Sequence[Shadow] = ()) -> List[Track]:
    """Play every animal's patrol out together, step by step to the horizon, beside the shadows of animals planned
    before, and return each one's track."""
    steps = int(math.ceil(horizon_s / SAMPLE_S)) + 1
    crowd = list(movers) + list(shadows)
    for k in range(steps):
        for shadow in shadows:
            shadow.at(k)
        for mover in movers:
            mover.prepare(k, crowd)
        for mover in movers:
            mover.record()
        for mover in movers:
            mover.move(k, crowd)
    return [mover.track() for mover in movers]


def _eroded(cells: np.ndarray) -> np.ndarray:
    """The cells whose eight neighbours, and they themselves, are all set; nothing off the grid is."""
    padded = np.pad(cells, 1, constant_values=False)
    kept = cells.copy()
    rows, cols = cells.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            kept &= padded[1 + dr:1 + dr + rows, 1 + dc:1 + dc + cols]
    return kept


def direction_index(hx: float, hy: float) -> int:
    """The compass direction nearest a heading."""
    best, score = 0, -2.0
    for n, (x, y) in enumerate(COMPASS):
        dot = hx * x + hy * y
        if dot > score:
            best, score = n, dot
    return best


# Goats: a herd that drifts across the grass
HERD_ROOM_LEVELS = 4                   # the herd's middle keeps this many cells further from closed ground than a goat
HERD_DRIFT_M_S = (0.04, 0.12)
HERD_LEG_S = (15.0, 45.0)
HERD_PAUSE_SHARE = 0.3
HERD_PAUSE_S = (10.0, 30.0)
HERD_LOOK_M = 6.0                      # the middle drifts on only where this much room lies ahead
GOAT_DOINGS = (("graze", 0.50), ("move", 0.28), ("look", 0.13), ("lie", 0.09))
GOAT_FIRST = (("graze", 0.45), ("stand", 0.15), ("lying", 0.12), ("look", 0.10), ("graze_walk", 0.18))
RESTING_RADIUS_M = (1.5, 0.4)          # at night the herd lies closer: this much, and this much a goat
SPOT_TRIES = 40
SPOT_MIN_M = 2.0                       # a walk to another spot is at least this long


def herd_radius(goats: int, resting: bool = False) -> float:
    """How far from the herd's middle its goats spread, closer when the herd rests for the night."""
    base, per_goat = RESTING_RADIUS_M if resting else (2.5, 0.5)
    return base + per_goat * goats


class Mind:
    """What every animal's mind answers beside its first and next doings: whether it must drop what it does now."""

    def urgent(self, mover: Mover, k: int, others: Sequence[Any]) -> bool:
        """Whether the animal drops what it does at sample k and decides afresh: never, unless a mind says so."""
        return False

    def landing_taken(self) -> None:
        """Hear that a flight's landing spot was taken before it could start; only a mind that flies queues one."""


class Herd(Mind):
    """The goats of a seed: the herd's middle drifting slowly across open ground, and each goat's choices round it; a
    herd resting for the night lies still round its start the whole patrol."""

    def __init__(self, rng: np.random.Generator, ground: Ground, level: int, start: Tuple[float, float], goats: int,
                 horizon_s: float, resting: bool = False):
        """Plan the middle's drift for the whole patrol from its start; a resting herd's middle keeps to its start, its
        drift still drawn so the herd's later draws stay where they are."""
        self.radius = herd_radius(goats, resting)
        self.ground = ground
        self.room_level = level + HERD_ROOM_LEVELS
        self.resting, self.horizon_s = resting, horizon_s
        self.cx, self.cy = self._drift(rng, start, horizon_s)
        if resting:
            self.cx, self.cy = [start[0]] * len(self.cx), [start[1]] * len(self.cy)

    def _drift(self, rng: np.random.Generator, start: Tuple[float, float], horizon_s: float) -> Tuple[List[float], List[float]]:
        """The middle's place at every sample: legs straight on at a slow pace with pauses, a new heading where the
        room ahead runs out."""
        steps = int(math.ceil(horizon_s / SAMPLE_S)) + 2
        x, y = start
        d = COMPASS[int(rng.integers(len(COMPASS)))]
        xs, ys = [], []
        left, speed = 0.0, 0.0
        for _ in range(steps):
            if left <= 0.0:
                left = uniform(rng, *HERD_LEG_S)
                speed = 0.0 if rng.random() < HERD_PAUSE_SHARE else uniform(rng, *HERD_DRIFT_M_S)
                d = self._heading(rng, x, y, d)
            nx, ny = x + speed * SAMPLE_S * d[0], y + speed * SAMPLE_S * d[1]
            if speed and self.ground.room(self.room_level, nx, ny, d, HERD_LOOK_M) >= HERD_LOOK_M:
                x, y = nx, ny
            elif speed:
                left = 0.0
            xs.append(x)
            ys.append(y)
            left -= SAMPLE_S
        return xs, ys

    def _heading(self, rng: np.random.Generator, x: float, y: float, d: Tuple[float, float]) -> Tuple[float, float]:
        """A new heading for the middle among those with room ahead, the ones nearer straight on likelier."""
        now = direction_index(*d)
        options = []
        for n, v in enumerate(COMPASS):
            turn = min((n - now) % 16, (now - n) % 16)
            if self.ground.room(self.room_level, x, y, v, HERD_LOOK_M) >= HERD_LOOK_M:
                options.append((v, 1.0 / (1.0 + turn * turn)))
        return pick(rng, options) if options else d

    def middle(self, k: int) -> Tuple[float, float]:
        """The herd's middle at a sample."""
        k = min(k, len(self.cx) - 1)
        return self.cx[k], self.cy[k]

    def leash(self, goat: Mover, k: int, x: float, y: float) -> bool:
        """Whether a goat grazing on may step to a point: still near the herd."""
        cx, cy = self.middle(k)
        reach = self.radius + 1.0
        return (x - cx) * (x - cx) + (y - cy) * (y - cy) <= reach * reach

    def spot(self, goat: Mover, k: int, others: Sequence[Mover]) -> Optional[Tuple[float, float]]:
        """A spot near the herd's middle a goat may walk to: on its ground, a real walk away, along a free straight
        line, and nobody's place or goal."""
        cx, cy = self.middle(k)
        for _ in range(SPOT_TRIES):
            u, v = uniform(goat.rng, -1.0, 1.0), uniform(goat.rng, -1.0, 1.0)
            if u * u + v * v > 1.0:
                continue
            x, y = cx + self.radius * u, cy + self.radius * v
            dx, dy = x - goat.x, y - goat.y
            if dx * dx + dy * dy < SPOT_MIN_M * SPOT_MIN_M or not goat.can_stand(x, y, others):
                continue
            if not _spare(goat, x, y, others) or not self.ground.path_free(goat.level, goat.x, goat.y, x, y):
                continue
            return x, y
        return None

    def rest(self, goat: Mover) -> None:
        """Lie for the rest of the patrol, the clip met part way through."""
        goat.queue(("play", "lying", 2.0 * self.horizon_s, None, uniform(goat.rng, 0.0, goat.body.clips["lying"].frames)))

    def first(self, goat: Mover) -> None:
        """A goat's first doing, met part way through so the herd does not start in step; lying when the herd rests."""
        if self.resting:
            self.rest(goat)
            return
        doing = pick(goat.rng, GOAT_FIRST)
        spec = goat.body.clips
        if doing == "graze":
            goat.queue(("play", "graze", uniform(goat.rng, 2.0, 9.0), None, uniform(goat.rng, 0.0, spec["graze"].frames)))
            goat.queue(("play", "head_up", None), ("play", "stand", uniform(goat.rng, 0.8, 2.5)))
        elif doing == "lying":
            goat.queue(("play", "lying", uniform(goat.rng, 5.0, 45.0), None, uniform(goat.rng, 0.0, spec["lying"].frames)),
                       ("play", "get_up", None), ("play", "stand", uniform(goat.rng, 1.0, 2.5)))
        elif doing == "graze_walk":
            goat.queue(("drift", "graze_walk", uniform(goat.rng, 2.0, 7.0), DRIFT_TURNS[int(goat.rng.integers(len(DRIFT_TURNS)))]),
                       ("play", "graze", uniform(goat.rng, 2.0, 6.0)), ("play", "head_up", None))
        else:
            goat.queue(("play", doing, uniform(goat.rng, 1.0, 5.0), None, uniform(goat.rng, 0.0, spec[doing].frames)))

    def decide(self, goat: Mover, k: int, others: Sequence[Mover]) -> None:
        """What a goat does next: catch the herd up when it has drifted off, else graze, move, look about or lie; lie on
        when the herd rests."""
        if self.resting:
            self.rest(goat)
            return
        cx, cy = self.middle(k)
        off = (goat.x - cx) * (goat.x - cx) + (goat.y - cy) * (goat.y - cy)
        rng = goat.rng
        if off > (self.radius + 0.5) * (self.radius + 0.5):
            spot = self.spot(goat, k, others)
            if spot is not None:
                far = 2.0 * self.radius + 3.0
                gait = "trot" if off > far * far and "trot" in goat.body.clips else "walk"
                goat.queue(("go", gait, spot[0], spot[1]), ("play", "stand", uniform(rng, 0.5, 1.5)))
                return
        doing = pick(rng, GOAT_DOINGS)
        if doing == "graze":
            goat.queue(("play", "head_down", None), ("play", "graze", uniform(rng, 3.0, 8.0)))
            for _ in range(int(rng.integers(0, 4))):
                goat.queue(("drift", "graze_walk", uniform(rng, 3.0, 7.0), DRIFT_TURNS[int(rng.integers(len(DRIFT_TURNS)))]),
                           ("play", "graze", uniform(rng, 2.0, 6.0)))
            goat.queue(("play", "head_up", None), ("play", "stand", uniform(rng, 0.8, 2.5)))
        elif doing == "move":
            spot = self.spot(goat, k, others)
            goat.queue(("go", "walk", spot[0], spot[1]) if spot is not None else ("play", "look", uniform(rng, 2.0, 4.0)),
                       ("play", "stand", uniform(rng, 0.6, 2.0)))
        elif doing == "look":
            goat.queue(("play", "look", uniform(rng, 4.0, 7.0)), ("play", "stand", uniform(rng, 0.5, 1.5)))
        else:
            goat.queue(("play", "lie_down", None), ("play", "lying", uniform(rng, 15.0, 45.0)), ("play", "get_up", None),
                       ("play", "stand", uniform(rng, 1.0, 2.5)))


def _spare(mover: Mover, x: float, y: float, others: Sequence[Mover]) -> bool:
    """Whether no other animal is walking or flying to a spot this close to a point."""
    for other in others:
        if other is mover:
            continue
        gap = mover.reach + other.reach + GAP_M
        for spot in (other.target, other.landing):
            if spot is None:
                continue
            dx, dy = x - spot[0], y - spot[1]
            if dx * dx + dy * dy < gap * gap:
                return False
    return True


# Dogs: each of its own mood, roaming, sniffing or resting
DOG_MOODS = (("roamer", 0.4), ("sniffer", 0.3), ("rester", 0.3))
HOME_RADIUS_M = 25.0                   # a sniffer potters within this of where it started


class Dog(Mind):
    """A dog's mind: a mood that decides its next trip or pause, and where it started."""

    def __init__(self, mood: str, home: Tuple[float, float]):
        """A dog of a mood at home where it starts."""
        self.mood, self.home = mood, home

    def leash(self, dog: Mover, k: int, x: float, y: float) -> bool:
        """Whether a dog sniffing on may step to a point: a sniffer stays near home."""
        if self.mood != "sniffer":
            return True
        dx, dy = x - self.home[0], y - self.home[1]
        return dx * dx + dy * dy <= HOME_RADIUS_M * HOME_RADIUS_M

    def first(self, dog: Mover) -> None:
        """A dog's first doing, met part way through."""
        rng = dog.rng
        if self.mood == "rester" and rng.random() < 0.7:
            dog.queue(("play", "lying", uniform(rng, 40.0, 150.0), None, uniform(rng, 0.0, dog.body.clips["lying"].frames)))
        else:
            dog.queue(("play", "stand", uniform(rng, 1.0, 4.0)))

    def decide(self, dog: Mover, k: int, others: Sequence[Mover]) -> None:
        """What a dog does next, by its mood."""
        rng = dog.rng
        if self.mood == "roamer":
            r = float(rng.random())
            if r < 0.50:
                moved = self.trip(dog, others, "walk", 6.0, 30.0)
            elif r < 0.78:
                moved = self.trip(dog, others, "trot", 10.0, 40.0)
            elif r < 0.83:
                moved = self.trip(dog, others, "run", 15.0, 40.0)
            elif r < 0.93:
                moved = False
                dog.queue(("play", "sniff" if rng.random() < 0.5 else "nose_down", uniform(rng, 2.0, 6.0)))
            else:
                moved = False
                self.lie(dog, 15.0, 60.0)
            if moved and rng.random() < 0.7:
                dog.queue(("play", "stand" if rng.random() < 0.5 else "look", uniform(rng, 1.0, 4.0)))
        elif self.mood == "sniffer":
            r = float(rng.random())
            if r < 0.45:
                dog.queue(("drift", "sniff_walk", uniform(rng, 3.0, 10.0), DRIFT_TURNS[int(rng.integers(len(DRIFT_TURNS)))]))
            elif r < 0.65:
                self.trip(dog, others, "walk", 4.0, 12.0)
            elif r < 0.85:
                dog.queue(("play", "sniff" if rng.random() < 0.5 else "nose_down", uniform(rng, 3.0, 10.0)))
            else:
                dog.queue(("play", "look" if rng.random() < 0.5 else "stand", uniform(rng, 1.0, 5.0)))
        else:
            if dog.clip == "lying":
                dog.queue(("play", "get_up", None), ("play", "look", uniform(rng, 2.0, 5.0)))
            for _ in range(int(rng.integers(1, 3))):
                if rng.random() < 0.6:
                    self.trip(dog, others, "walk", 3.0, 15.0)
                else:
                    dog.queue(("drift", "sniff_walk", uniform(rng, 2.0, 6.0), DRIFT_TURNS[int(rng.integers(len(DRIFT_TURNS)))]))
                dog.queue(("play", "sniff", uniform(rng, 2.0, 8.0)))
            self.lie(dog, 40.0, 180.0)

    @staticmethod
    def lie(dog: Mover, low: float, high: float) -> None:
        """Stand a moment, lie down, lie a while and get up."""
        dog.queue(("play", "stand", uniform(dog.rng, 1.0, 2.5)), ("play", "lie_down", None),
                  ("play", "lying", uniform(dog.rng, low, high)), ("play", "get_up", None))

    def trip(self, dog: Mover, others: Sequence[Mover], gait: str, low: float, high: float) -> bool:
        """Walk, trot or run to a spot along a free straight line, the headings nearer straight on likelier; a look
        round instead when no heading has room."""
        rng = dog.rng
        now = direction_index(dog.hx, dog.hy)
        options = []
        for n, v in enumerate(COMPASS):
            room = dog.ground.room(dog.level, dog.x, dog.y, v, high)
            if room < low:
                continue
            length = min(uniform(rng, low, high), room)
            x, y = dog.x + v[0] * length, dog.y + v[1] * length
            if self.mood == "sniffer" and not self.leash(dog, 0, x, y):
                continue
            if not _spare(dog, x, y, others):
                continue
            turn = min((n - now) % 16, (now - n) % 16)
            options.append(((x, y), 1.0 / (1.0 + turn * turn)))
        if not options:
            dog.queue(("play", "look", uniform(rng, 1.5, 4.0)))
            return False
        x, y = pick(rng, options)
        dog.queue(("go", gait, x, y))
        return True


# Rats and mice: short runs near home, with sniffing, rearing up and grooming between them
RODENT_HOME_M = 15.0                   # a rat or a mouse keeps this close to where it started
RODENT_DOINGS = (("sniff", 0.28), ("potter", 0.24), ("scurry", 0.22), ("rear", 0.16), ("groom", 0.10))
RODENT_RUN_M = (1.0, 6.0)              # one scurry is this long


class Rodent(Dog):
    """A rat's or a mouse's mind: it potters like a dog that sniffs about, but nearer home and never lying down."""

    def __init__(self, home: Tuple[float, float]):
        """A rat or a mouse at home where it starts."""
        super().__init__("sniffer", home)

    def leash(self, rodent: Mover, k: int, x: float, y: float) -> bool:
        """Whether a rat or a mouse may step to a point: still near home."""
        dx, dy = x - self.home[0], y - self.home[1]
        return dx * dx + dy * dy <= RODENT_HOME_M * RODENT_HOME_M

    def first(self, rodent: Mover) -> None:
        """Its first doing: standing, met part way through."""
        rng = rodent.rng
        rodent.queue(("play", "stand", uniform(rng, 0.5, 3.0), None, uniform(rng, 0.0, rodent.body.clips["stand"].frames)))

    def decide(self, rodent: Mover, k: int, others: Sequence[Mover]) -> None:
        """What it does next: sniff the ground, potter on with its nose down, scurry to another spot, rear up to
        scent the air or sit up and wash. Rearing and washing play once through, so it always comes down again."""
        rng = rodent.rng
        doing = pick(rng, RODENT_DOINGS)
        if doing == "sniff":
            rodent.queue(("play", "nose_down", None), ("play", "sniff", uniform(rng, 2.0, 7.0)))
        elif doing == "potter":
            rodent.queue(("drift", "sniff_walk", uniform(rng, 2.0, 6.0), DRIFT_TURNS[int(rng.integers(len(DRIFT_TURNS)))]))
        elif doing == "scurry":
            if self.trip(rodent, others, "trot", *RODENT_RUN_M):
                rodent.queue(("play", "stand", uniform(rng, 0.4, 2.0)))
        else:
            rodent.queue(("play", "look" if doing == "rear" else "groom", None), ("play", "stand", uniform(rng, 0.4, 1.5)))


# Birds: on the ground a while, then a low flight to another spot
BIRD_TURN_COS, BIRD_TURN_SIN = _rotation(0.25)   # a bird turns about 28 degrees a step, five times a goat
BIRD_DOINGS = (("peck", 0.35), ("look", 0.20), ("stand", 0.15), ("walk", 0.20), ("hop", 0.10))
BIRD_FIRST = (("peck", 0.40), ("look", 0.35), ("stand", 0.25))
BIRD_STROLL_M = (0.6, 2.5)             # a walk or a hop on the ground is this long
BIRD_FLIGHT_GAP_S = (15.0, 60.0)       # time on the ground between two flights
BIRD_RETRY_S = (2.0, 5.0)              # a bird that finds no spot to fly to tries again after this
BIRD_FLIGHT_M = (10.0, 45.0)           # how far a flight goes
BIRD_PEAK_M = (2.0, 5.0)               # how high over the ground a flight rises at most
PEAK_PER_M = 0.1                       # a flight's highest peak rises this much a metre of its length, from the lowest
FLIGHT_TRIES = 30
LAUNCH_STEP_M = 5.0                    # a start spot's flight lines are counted every this far along each compass line
MIN_LAUNCHES = 8                       # a bird starts only where at least this many of them are open, where it can
HOLD_S = 2 * SAMPLE_S                  # a flight holds its peak at least this long, so a kept sample meets it
TOUCH_SHARE = 0.7                      # the landing clip meets the ground this far through it
CLIMB_M_S = 2.5                        # a flight climbs to its peak no faster than this on average
FLOOR_M = 1.2                          # and is at least this high when its take-off clip ends
LAND_GAP_M = 1.0                       # a landing spot keeps this much more air to every dog and goat than the reaches
LAND_WINDOW = (-10, 40)                # samples round the touch-down a dog or goat must keep that air
FLUSH_GAP_M = 1.0                      # a bird takes off when a dog or goat will come this close past both reaches
FLUSH_LOOK = 20                        # samples ahead it watches a dog or goat coming
FLUSH_CALM = 30                        # samples it waits before it may take fright again


@dataclass(frozen=True)
class Flight:
    """A bird's flight in time from the start of its take-off: where along the straight line it is, how high over the
    ground under it, and which clip it plays. It leaves the ground part way through the take-off, climbs to its peak,
    flies on, glides down and lands softly, its feet meeting the ground part way through the landing clip."""

    x0: float
    y0: float
    ux: float
    uy: float
    dist: float
    speed: float
    peak: float
    lift_s: float
    rise_s: float
    top_s: float
    glide_s: float
    land_s: float
    touch_s: float
    end_s: float

    @classmethod
    def plan(cls, mover: Mover, x1: float, y1: float, peak: float) -> "Flight":
        """The flight of a bird from where it stands to a spot, rising to a peak: as fast as its clip flies, slower
        when the line is too short to climb to the peak and come down from it."""
        clips = mover.body.clips
        dx, dy = x1 - mover.x, y1 - mover.y
        dist = math.sqrt(dx * dx + dy * dy)
        rise = clips["take_off"].frames / (clips["take_off"].fps * mover.pace)
        landing = clips["land"].frames / (clips["land"].fps * mover.pace)
        lift = lift_frame(mover.body) / (clips["take_off"].fps * mover.pace)
        speeding, slowing = rise - lift, TOUCH_SHARE * landing
        # Climbing slower than this would leave the bird under FLOOR_M as the take-off ends.
        climb = min(max(2.0 * speeding, peak / CLIMB_M_S), speeding / (1.0 - math.sqrt(1.0 - FLOOR_M / peak)))
        # The landing clip starts halfway down, so the bird is at half its peak, at least a metre, when it does.
        descent = 2.0 * slowing
        level = climb + descent - speeding - slowing + HOLD_S
        speed = min(clips["fly"].speed_m_s * mover.pace, dist / ((speeding + slowing) / 2.0 + level))
        land = rise + dist / speed - (speeding + slowing) / 2.0
        touch = land + slowing
        return cls(x0=mover.x, y0=mover.y, ux=dx / dist, uy=dy / dist, dist=dist, speed=speed, peak=peak, lift_s=lift,
                   rise_s=rise, top_s=lift + climb, glide_s=touch - descent, land_s=land, touch_s=touch,
                   end_s=land + landing)

    def along(self, t: float) -> float:
        """How far along the line the bird is at a time: still until it lifts, gathering speed through the take-off,
        flying on, slowing to a stop at the touch-down."""
        if t <= self.lift_s:
            return 0.0
        if t >= self.touch_s:
            return self.dist
        speeding = self.rise_s - self.lift_s
        if t < self.rise_s:
            q = t - self.lift_s
            return self.speed * q * q / (2.0 * speeding)
        if t < self.land_s:
            return self.speed * (speeding / 2.0 + (t - self.rise_s))
        q, slowing = t - self.land_s, self.touch_s - self.land_s
        return min(self.speed * (speeding / 2.0 + (self.land_s - self.rise_s) + q - q * q / (2.0 * slowing)), self.dist)

    def height(self, t: float) -> float:
        """How high over the ground under it the bird is at a time: a quick climb to the peak that eases off, level
        flight, and a smooth descent that ends with no fall left at the touch-down."""
        if t <= self.lift_s or t >= self.touch_s:
            return 0.0
        if t < self.top_s:
            u = 1.0 - (t - self.lift_s) / (self.top_s - self.lift_s)
            return self.peak * (1.0 - u * u)
        if t < self.glide_s:
            return self.peak
        w = (t - self.glide_s) / (self.touch_s - self.glide_s)
        return self.peak * (1.0 - w * w * (3.0 - 2.0 * w))

    def clip(self, t: float) -> Tuple[str, float]:
        """The clip the bird plays at a time and when it began: take-off, flight, the glide down, the landing."""
        if t < self.rise_s:
            return "take_off", 0.0
        if t < self.glide_s:
            return "fly", self.rise_s
        if t < self.land_s:
            return "glide", self.glide_s
        return "land", self.land_s


def lift_frame(body: Body) -> int:
    """The first frame of a body's take-off whose lowest point is off the ground: where its feet leave it."""
    clip = body.clips["take_off"]
    lowest = body.poses[clip.start:clip.start + clip.frames, :, 2].min(axis=1)
    off = np.flatnonzero(lowest > 0.0)
    return int(off[0]) if len(off) else clip.frames - 1


class Bird(Mind):
    """A bird's mind: on the ground it pecks, looks about, stands, walks or hops a little; every so often, or when a
    dog or goat is about to come close, it flies low to another spot."""

    def __init__(self, rng: np.random.Generator):
        """A bird whose first flight comes part way through its first time on the ground."""
        self.next_s = uniform(rng, *BIRD_FLIGHT_GAP_S) * uniform(rng, 0.1, 1.0)
        self.fright = False
        self.calm_k = 0

    def landing_taken(self) -> None:
        """Look for another flight at the next decision."""
        self.fright = True

    def urgent(self, bird: Mover, k: int, others: Sequence[Any]) -> bool:
        """Whether a bird on the ground must take off now: a dog or goat comes close within the samples it watches."""
        if bird.up > 0.0 or k < self.calm_k or (bird.action is not None and bird.action[0] == "air"):
            return False
        for other in others:
            if not isinstance(other, Shadow):
                continue
            gap = bird.reach + other.reach + FLUSH_GAP_M
            reach = gap + other.fastest * FLUSH_LOOK
            dx, dy = other.x - bird.x, other.y - bird.y
            if dx * dx + dy * dy < reach * reach and other.near(bird.x, bird.y, gap, k, k + FLUSH_LOOK):
                self.fright, self.calm_k = True, k + FLUSH_CALM
                return True
        return False

    def first(self, bird: Mover) -> None:
        """A bird's first doing, met part way through."""
        doing = pick(bird.rng, BIRD_FIRST)
        spec = bird.body.clips[doing]
        if doing == "peck":
            bird.queue(("play", "peck", None, None, uniform(bird.rng, 0.0, spec.frames - 1.0)))
        else:
            bird.queue(("play", doing, uniform(bird.rng, 1.0, 4.0), None, uniform(bird.rng, 0.0, spec.frames)))

    def decide(self, bird: Mover, k: int, others: Sequence[Any]) -> None:
        """What a bird does next: fly when its time on the ground is up or it took fright, else something on the
        ground."""
        rng = bird.rng
        t = k * SAMPLE_S
        if self.fright or t >= self.next_s:
            self.fright = False
            flight = self.flight(bird, k, others)
            if flight is not None:
                bird.landing = (flight.x0 + flight.ux * flight.dist, flight.y0 + flight.uy * flight.dist)
                bird.queue(("air", "take_off", *bird.landing, flight))
                self.next_s = t + flight.end_s + uniform(rng, *BIRD_FLIGHT_GAP_S)
                return
            self.next_s = t + uniform(rng, *BIRD_RETRY_S)
        doing = pick(rng, BIRD_DOINGS)
        if doing == "peck":
            for _ in range(int(rng.integers(1, 4))):
                bird.queue(("play", "peck", None))
        elif doing in ("walk", "hop"):
            spot = self.stroll(bird, others)
            if spot is None:
                bird.queue(("play", "look", uniform(rng, 1.5, 4.0)))
            else:
                bird.queue(("go", doing, spot[0], spot[1]), ("play", "stand", uniform(rng, 0.3, 1.2)))
        else:
            bird.queue(("play", doing, uniform(rng, 1.0, 5.0)))

    @staticmethod
    def stroll(bird: Mover, others: Sequence[Any]) -> Optional[Tuple[float, float]]:
        """A spot a short walk or hop away along a free straight line, nobody's place or goal."""
        for _ in range(SPOT_TRIES):
            u, v = uniform(bird.rng, -1.0, 1.0), uniform(bird.rng, -1.0, 1.0)
            r2 = u * u + v * v
            if r2 > 1.0:
                continue
            x, y = bird.x + BIRD_STROLL_M[1] * u, bird.y + BIRD_STROLL_M[1] * v
            dx, dy = x - bird.x, y - bird.y
            if dx * dx + dy * dy < BIRD_STROLL_M[0] * BIRD_STROLL_M[0] or not bird.can_stand(x, y, others):
                continue
            if _spare(bird, x, y, others) and bird.ground.path_free(bird.level, bird.x, bird.y, x, y):
                return x, y
        return None

    def flight(self, bird: Mover, k: int, others: Sequence[Any]) -> Optional[Flight]:
        """A flight to a spot BIRD_FLIGHT_M away: the spot on the bird's ground, nobody's place or goal and clear of
        every dog and goat round the touch-down, the straight line to it over clear sky."""
        rng = bird.rng
        for _ in range(FLIGHT_TRIES):
            u, v = uniform(rng, -1.0, 1.0), uniform(rng, -1.0, 1.0)
            r2 = u * u + v * v
            if r2 > 1.0 or r2 < 0.04:
                continue
            r = math.sqrt(r2)
            length = uniform(rng, *BIRD_FLIGHT_M)
            x, y = bird.x + length * (u / r), bird.y + length * (v / r)
            if not bird.can_stand(x, y, others) or not _spare(bird, x, y, others):
                continue
            if not bird.ground.over(bird.x, bird.y, x, y):
                continue
            flight = Flight.plan(bird, x, y, uniform(rng, BIRD_PEAK_M[0],
                                                     min(BIRD_PEAK_M[1], BIRD_PEAK_M[0] + PEAK_PER_M * length)))
            touch = k + int(math.floor(flight.touch_s / SAMPLE_S))
            if any(isinstance(other, Shadow) and other.near(x, y, bird.reach + other.reach + LAND_GAP_M,
                                                            touch + LAND_WINDOW[0], touch + LAND_WINDOW[1])
                   for other in others):
                continue
            return flight
        return None


def place_start(rng: np.random.Generator, cells: np.ndarray, ground: Ground, mover_reach: float,
                taken: Sequence[Tuple[float, float, float]], apart_m: float, tries: int) -> Optional[Tuple[float, float]]:
    """A free cell's centre drawn from a list, clear of the bodies already placed (east, north, reach) and preferably
    apart_m from them: the first draw that is, else the furthest clear draw; None when no draw is clear."""
    best, gap = None, -1.0
    if not len(cells):
        return None
    for _ in range(tries):
        row, col = cells[int(rng.integers(len(cells)))]
        x, y = ground.centre(int(row), int(col))
        near = math.inf
        clear = True
        for ox, oy, reach in taken:
            d = math.sqrt((x - ox) * (x - ox) + (y - oy) * (y - oy))
            clear = clear and d >= mover_reach + reach + GAP_M
            near = min(near, d)
        if not clear:
            continue
        if near >= apart_m:
            return x, y
        if near > gap:
            best, gap = (x, y), near
    return best


def plan_movers(movers: Sequence[Mover], horizon_s: float, shadows: Sequence[Shadow] = ()) -> List[Track]:
    """Every animal's first doing, then the patrol played out beside the shadows of animals planned before."""
    for mover in movers:
        mover.mind.first(mover)
    return live(movers, horizon_s, shadows)
