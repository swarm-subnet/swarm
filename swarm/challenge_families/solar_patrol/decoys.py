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

"""Decoys (task 20): the harmless things the model must see and leave alone, placed per seed.

Every seed carries three kinds. The farmer's truck on the public road outside the fence and the bird circling high
above the park are the map's own movers. One to three dogs live inside the fence, where the goats used to be. A
report on any of them is a false alarm, and bodies(ep) tells the report check which body is which.

Each dog's whole patrol is planned when the seed starts, on the open ground that rays from the sky find between the
rows, so where it stands and what it does is a function of time alone and place() can put it back at any step. A
dog's mesh is rewritten only on a step a frame is taken, and only when its new pose would move on that frame by at
least STILL_PIXELS: the rewrite is the one real cost a dog has, in time and in the memory the engine keeps per
rewrite, and a dog lying or sniffing in place changes by millimetres between frames.
"""

from __future__ import annotations

import bisect
import json
import math
import os
from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pybullet as p
import shapely
import swarm_worlds
from shapely.geometry import Point, Polygon

from swarm.constants import SIM_DT

from . import camera, park, zoom
from .episode import SolarEpisode
from .fixed_order import dot, norm

DOGS_DIR = os.path.join(swarm_worlds.maps_dir(), "custom", "solar", "dogs")
DOGS_SHIPPED = os.path.isfile(os.path.join(DOGS_DIR, "dogs.json"))
THERMAL = hasattr(p, "ER_SWARM_THERMAL")
SEED_STREAM = 0xD065                   # the decoys' own stream, so a dog never moves another part's draws
MOVER_KINDS = {"pickup": "truck", "bird": "bird"}

DOG_COUNT = {1: 0.4, 2: 0.4, 3: 0.2}   # dogs a seed draws; every seed has at least one
MOODS = {"roamer": 0.4, "sniffer": 0.3, "rester": 0.3}
GRID_M = 0.5                           # spacing of the rays from the sky that find open ground
FENCE_CLEAR_M = 1.0                    # a dog keeps this far inside the fence
DOCK_CLEAR_M = 6.0                     # and this far from the dock the aircraft lands on
TRUNK_CLEAR_M = 0.8                    # and this far from a tree trunk, too thin for the rays to be sure of
START_GAP_M = 8.0                      # dogs start at least this far apart
TURN_RADIUS_M = (0.6, 0.35)            # a walked turn, and the tighter one tried where that does not fit
PATH_STEP_M = 0.25                     # a path is checked against the open ground this often
TRIES = 32                             # spots drawn for a dog's start
TURN_PREFERENCE_RAD = 0.8              # how strongly a trip prefers going straight on to turning
PIVOT_RAD_S = 1.6                      # a dog stepping round on the spot, where the aisle is too narrow to walk a turn
PIVOT_WEIGHT = 0.05                    # how much less likely stepping round is than a walked turn where both fit
FILL_RINGS = 4                         # rings of height padded round the measured ground, 2 m at the grid spacing
LIVABLE_M2 = 150.0                     # an open piece smaller than this is a scrap between panels, not a place to live
HOME_RADIUS_M = 25.0                   # a sniffer potters within this of where it started
PLAN_MARGIN_S = 10.0                   # the plan runs this far past the end of the patrol
STILL_PIXELS = 0.5                     # a new pose that moves no vertex this far on any frame is not worth a rewrite


@dataclass(frozen=True)
class Leg:
    """One stretch of a dog's patrol: a clip played on from a frame while the body travels a line or an arc, or
    turns on the spot.

    The speed eases from v_in to v over the blend, and the leg's first pose eases out of the held pose of the clip
    before it, when there was another clip before it.
    """

    t0: float
    clip: str
    frame: float
    x: float
    y: float
    heading: float
    curve: float
    v_in: float
    v: float
    held: Optional[Tuple[str, float]] = None
    spin: float = 0.0                   # rad/s the body turns on the spot, for a leg that does not travel


@dataclass
class Dog:
    """One dog of this seed: its body, looks and planned patrol, and the pose its mesh last showed."""

    body: int
    breed: str
    coat: str
    size: float
    pace: float
    mood: str
    length: float                       # nose to tail at this dog's size, metres
    blend: float                        # seconds one clip takes to ease into the next
    legs: List[Leg]
    starts: List[float] = field(default_factory=list)
    shown: Optional[np.ndarray] = None  # the render vertices its mesh last received

    def __post_init__(self):
        """Index the legs by their start times."""
        self.starts = [leg.t0 for leg in self.legs]


@dataclass(frozen=True)
class Ground:
    """The open ground inside the fence on a square grid: which cells a dog may stand on, which a path may be
    planned through, and the terrain height.

    A path is checked at points closer together than half a cell, and every cell next to a planned one, corners
    included, is open, so the dog stays on open ground between the points too.
    """

    origin: Tuple[float, float]
    step: float
    open: np.ndarray                    # rows run north, columns east
    height: np.ndarray
    planned: Optional[np.ndarray] = None

    def __post_init__(self):
        """Derive the cells a path may be planned through: the open ones whose eight neighbours are open too."""
        if self.planned is None:
            object.__setattr__(self, "planned", _eroded(self.open, corners=True))

    def cells(self, xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Row and column of the cell holding each point."""
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        return (np.floor((xy[:, 1] - self.origin[1]) / self.step + 0.5).astype(int),
                np.floor((xy[:, 0] - self.origin[0]) / self.step + 0.5).astype(int))

    def open_at(self, xy: np.ndarray) -> np.ndarray:
        """Whether each point stands on an open cell; anything off the grid is closed."""
        return self._lookup(self.open, xy)

    def planned_at(self, xy: np.ndarray) -> np.ndarray:
        """Whether each point may be on a planned path."""
        return self._lookup(self.planned, xy)

    def _lookup(self, cells: np.ndarray, xy: np.ndarray) -> np.ndarray:
        """Each point's cell in a grid of flags; anything off the grid is False."""
        rows, cols = self.cells(xy)
        inside = (rows >= 0) & (rows < cells.shape[0]) & (cols >= 0) & (cols < cells.shape[1])
        result = np.zeros(len(rows), dtype=bool)
        result[inside] = cells[rows[inside], cols[inside]]
        return result

    def height_at(self, xy: np.ndarray) -> np.ndarray:
        """Terrain height under each point, blended from the four cells round it."""
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        fx = np.clip((xy[:, 0] - self.origin[0]) / self.step, 0.0, self.height.shape[1] - 1.001)
        fy = np.clip((xy[:, 1] - self.origin[1]) / self.step, 0.0, self.height.shape[0] - 1.001)
        c, r = fx.astype(int), fy.astype(int)
        sx, sy = fx - c, fy - r
        h = self.height
        return ((h[r, c] * (1 - sx) + h[r, c + 1] * sx) * (1 - sy)
                + (h[r + 1, c] * (1 - sx) + h[r + 1, c + 1] * sx) * sy)


# ---------------------------------------------------------------------- #
# the part's hooks
# ---------------------------------------------------------------------- #
def reset(env: Any, ep: SolarEpisode) -> None:
    """Find the open ground, draw and plan this seed's dogs, and stand everything where it is at step zero."""
    ep.decoys = {"dogs": [], "ground": None}
    if not DOGS_SHIPPED:
        return
    rng = np.random.default_rng([SEED_STREAM, int(ep.seed)])
    ground = open_ground(env.CLIENT, ep.fence, ep.terrain_uids, ep.dock_position[:2],
                         park.passable_outlines(ep)[:, :2])
    horizon = float(env.EP_LEN_SEC) + PLAN_MARGIN_S
    ep.decoys = {"dogs": [_spawn(env.CLIENT, draw) for draw in draw_dogs(rng, ground, horizon)], "ground": ground}
    place(env, ep, 0)


def advance(env: Any, ep: SolarEpisode) -> None:
    """Stand the dogs where they are on the step physics is about to run, when a frame will be taken on it."""
    step = ep.step + 1
    shots = [shot for shot in (camera.upcoming(env, ep, step), zoom.upcoming(env, ep)) if shot is not None]
    if shots:
        place(env, ep, step, shots)


def place(env: Any, ep: SolarEpisode, step: int, shots: Optional[Sequence[camera.View]] = None) -> None:
    """Stand every dog where it is at a step. With no shots every mesh is posed too; with shots, only a dog whose
    new pose would move on one of those frames by STILL_PIXELS or more."""
    ground = ep.decoys["ground"]
    t = step * SIM_DT
    for dog in ep.decoys["dogs"]:
        x, y, heading, leg, tau = locate(dog, t)
        ahead = np.array([math.cos(heading), math.sin(heading)]) * dog.length / 2.0
        front, back, z = ground.height_at(np.array([[x, y] + ahead, [x, y] - ahead, [x, y]]))
        pitch = math.atan2(front - back, dog.length)
        p.resetBasePositionAndOrientation(dog.body, [x, y, float(z)], p.getQuaternionFromEuler([0.0, -pitch, heading]),
                                          physicsClientId=env.CLIENT)
        vertices = shape(dog, leg, tau)
        if shots is not None and dog.shown is not None:
            per_metre = max(_pixels(shot, np.array([x, y, float(z)]), 1.0) for shot in shots)
            if float(np.abs(vertices - dog.shown).max()) * per_metre < STILL_PIXELS:
                continue
        p.resetMeshData(dog.body, vertices, physicsClientId=env.CLIENT)
        dog.shown = vertices


def bodies(ep: SolarEpisode) -> Dict[int, str]:
    """Every decoy's body id and what it is: dog, truck or bird. None of them is ever a threat."""
    kinds = {body: MOVER_KINDS[kind] for body, kind in park.mover_kinds(ep).items() if kind in MOVER_KINDS}
    kinds.update({dog.body: "dog" for dog in ep.decoys["dogs"]})
    return kinds


def positions(ep: SolarEpisode, step: int) -> np.ndarray:
    """East, north and heading of every dog at a step, without moving anything."""
    return np.array([locate(dog, step * SIM_DT)[:3] for dog in ep.decoys["dogs"]], dtype=float).reshape(-1, 3)


# ---------------------------------------------------------------------- #
# the open ground
# ---------------------------------------------------------------------- #
def open_ground(cli: int, fence: np.ndarray, terrain: frozenset, dock_xy: np.ndarray,
                trunks: np.ndarray) -> Ground:
    """The cells inside the fence where a ray from the sky meets the terrain first, clear of the fence, the dock and
    any trunk, shrunk by one cell so a dog's body stays off what stands next to it, without the scraps too small to
    live in."""
    lo, hi = fence.min(axis=0), fence.max(axis=0)
    xs = np.arange(lo[0], hi[0] + GRID_M, GRID_M)
    ys = np.arange(lo[1], hi[1] + GRID_M, GRID_M)
    grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    area = Polygon(fence).buffer(-FENCE_CLEAR_M).difference(Point(*dock_xy).buffer(DOCK_CLEAR_M))
    for trunk in np.asarray(trunks, dtype=float).reshape(-1, 2):
        area = area.difference(Point(*trunk).buffer(TRUNK_CLEAR_M))
    keep = shapely.contains_xy(area, grid[:, 0], grid[:, 1])
    height = np.full(len(grid), np.nan)
    cells = np.flatnonzero(keep)
    for chunk in np.array_split(cells, max(1, math.ceil(len(cells) / 8192))):
        tops = np.column_stack([grid[chunk], np.full(len(chunk), 1000.0)]).tolist()
        bottoms = np.column_stack([grid[chunk], np.full(len(chunk), -1000.0)]).tolist()
        for index, hit in zip(chunk.tolist(), p.rayTestBatch(tops, bottoms, numThreads=0, physicsClientId=cli)):
            if int(hit[0]) in terrain:
                height[index] = float(hit[3][2])
    height = height.reshape(len(ys), len(xs))
    return Ground(origin=(float(xs[0]), float(ys[0])), step=GRID_M,
                  open=_livable(_eroded(~np.isnan(height), corners=False), GRID_M), height=_filled(height))


def _eroded(cells: np.ndarray, corners: bool) -> np.ndarray:
    """The cells that stay open when every cell next to a closed one closes too: the four beside it, or all eight."""
    padded = np.pad(cells, 1, constant_values=False)
    kept = cells.copy()
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)) + (((-1, -1), (-1, 1), (1, -1), (1, 1)) if corners else ()):
        kept &= padded[1 + dr:1 + dr + cells.shape[0], 1 + dc:1 + dc + cells.shape[1]]
    return kept


def _livable(open_: np.ndarray, step: float) -> np.ndarray:
    """The open cells in pieces big enough for a dog to live in; the scraps between panels and posts are dropped."""
    seen = np.zeros_like(open_)
    kept = np.zeros_like(open_)
    rows, cols = open_.shape
    for start in np.flatnonzero(open_):
        if seen.flat[start]:
            continue
        seen.flat[start] = True
        piece, queue = [], deque([int(start)])
        while queue:
            cell = queue.popleft()
            piece.append(cell)
            r, c = divmod(cell, cols)
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < rows and 0 <= nc < cols and open_[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append(nr * cols + nc)
        if len(piece) * step * step >= LIVABLE_M2:
            kept.flat[piece] = True
    return kept


def _filled(height: np.ndarray) -> np.ndarray:
    """The height grid with the cells round the measured ones given the mean of their measured neighbours, ring by
    ring, as far as a dog's body reaches past the open ground; cells further out are never read."""
    height = height.copy()
    for _ in range(FILL_RINGS):
        missing = np.isnan(height)
        if not missing.any():
            break
        padded = np.pad(height, 1, constant_values=np.nan)
        near = np.stack([padded[:-2, 1:-1], padded[2:, 1:-1], padded[1:-1, :-2], padded[1:-1, 2:]])
        counts = (~np.isnan(near)).sum(axis=0)
        sums = np.nansum(near, axis=0)
        grow = missing & (counts > 0)
        height[grow] = sums[grow] / counts[grow]
    return np.nan_to_num(height, nan=0.0)


# ---------------------------------------------------------------------- #
# drawing and planning the dogs
# ---------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _catalogue() -> Dict[str, Any]:
    """The dog package's table: bodies, coats, clips, heat and the spreads a seed draws from."""
    with open(os.path.join(DOGS_DIR, "dogs.json"), encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=4)
def _poses(breed: str) -> Tuple[np.ndarray, np.ndarray]:
    """One body's poses as float32 shared vertices per frame, and the corner order the engine draws them in."""
    with np.load(os.path.join(DOGS_DIR, _catalogue()["bodies"][breed]["poses"])) as table:
        return table["poses"].astype(np.float32), table["corners"].astype(np.int64)


def _pick(rng: np.random.Generator, weights: Dict[Any, float]) -> Any:
    """One key of a weights table, drawn in proportion."""
    keys = list(weights)
    share = np.asarray([weights[k] for k in keys], dtype=float)
    return keys[int(rng.choice(len(keys), p=share / share.sum()))]


def draw_dogs(rng: np.random.Generator, ground: Ground, horizon: float) -> List[Dict[str, Any]]:
    """This seed's dogs: each a different breed, with its coat, size, pace, mood, start and planned patrol."""
    table = _catalogue()
    count = int(_pick(rng, DOG_COUNT))
    breeds = [str(b) for b in rng.permutation(sorted(table["bodies"]))[:count]]
    free = np.argwhere(ground.planned)
    if not len(free):
        return []
    starts: List[np.ndarray] = []
    dogs = []
    for breed in breeds:
        body = table["bodies"][breed]
        coat = str(rng.choice(sorted(body["coats"])))
        size = float(rng.uniform(*table["size_spread"]))
        pace = float(rng.uniform(*table["pace_spread"]))
        mood = str(_pick(rng, MOODS))
        start = _start(rng, ground, free, starts)
        starts.append(start)
        legs = Planner(rng, ground, body["clips"], size, pace, float(table["blend_s"])).plan(mood, start, horizon)
        dogs.append({"breed": breed, "coat": coat, "size": size, "pace": pace, "mood": mood, "legs": legs})
    return dogs


def _start(rng: np.random.Generator, ground: Ground, free: np.ndarray, taken: List[np.ndarray]) -> np.ndarray:
    """An open spot away from the dogs already placed, or the furthest of the draws when none is far enough."""
    best, gap = None, -1.0
    for _ in range(TRIES):
        r, c = free[int(rng.integers(len(free)))]
        spot = np.array([ground.origin[0] + c * ground.step, ground.origin[1] + r * ground.step])
        near = min((float(np.hypot(*(spot - other))) for other in taken), default=math.inf)
        if near >= START_GAP_M:
            return spot
        if near > gap:
            best, gap = spot, near
    return best


class Planner:
    """Writes one dog's patrol as legs, trip by trip and pause by pause, the mood deciding what comes next.

    A trip is a walked turn onto a new heading followed by a straight run in the trip's own gait; it is taken only
    when the whole path, and the slide into the stop after it, stays on open ground. Speed and pose both carry over
    from one leg to the next, so nothing jumps.
    """

    def __init__(self, rng: np.random.Generator, ground: Ground, clips: Dict[str, Any], size: float, pace: float,
                 blend_s: float):
        """Start with an empty patrol for a dog of this build."""
        self.rng, self.ground, self.clips = rng, ground, clips
        self.size, self.pace, self.blend = size, pace, blend_s
        self.legs: List[Leg] = []
        self.t, self.xy, self.heading, self.home = 0.0, np.zeros(2), 0.0, np.zeros(2)
        self.clip: Optional[str] = None
        self.frame, self.v = 0.0, 0.0

    def plan(self, mood: str, start: np.ndarray, horizon: float) -> List[Leg]:
        """The whole patrol for a mood, from a start to past the horizon."""
        self.xy = self.home = np.asarray(start, dtype=float)
        self.heading = self._open_heading()
        if mood == "rester" and self.rng.random() < 0.7:
            self.play("lying", float(self.rng.uniform(40.0, 150.0)),
                      frame=float(self.rng.uniform(0.0, self.clips["lying"]["frames"])))
        else:
            self.play("stand", float(self.rng.uniform(1.0, 4.0)))
        step = {"roamer": self._roam, "sniffer": self._sniff, "rester": self._rest}[mood]
        while self.t < horizon:
            step()
        return self.legs

    def _roam(self) -> None:
        """A dog that covers the park: walks and trots, the odd sprint, short stops, now and then a lie-down."""
        r = self.rng.random()
        if r < 0.93:
            if r < 0.50:
                moved = self.trip("walk", 6.0, 30.0)
            elif r < 0.78:
                moved = self.trip("trot", 10.0, 40.0)
            elif r < 0.83:
                moved = self.trip("run", 15.0, 40.0)
            else:
                moved = False
                self.play(str(self.rng.choice(["sniff", "nose_down"])), float(self.rng.uniform(2.0, 6.0)))
            if moved and self.rng.random() < 0.7:
                self.play(str(self.rng.choice(["stand", "look"])), float(self.rng.uniform(1.0, 4.0)))
        else:
            self.lie(15.0, 60.0)
            self.play("get_up")

    def _sniff(self) -> None:
        """A dog that potters round one patch, nose down, never far from where it started."""
        r = self.rng.random()
        if r < 0.45:
            self.trip("sniff_walk", 2.0, 8.0, home=True)
        elif r < 0.65:
            self.trip("walk", 4.0, 12.0, home=True)
        elif r < 0.85:
            self.play(str(self.rng.choice(["sniff", "nose_down"])), float(self.rng.uniform(3.0, 10.0)))
        else:
            self.play(str(self.rng.choice(["look", "stand"])), float(self.rng.uniform(1.0, 5.0)))

    def _rest(self) -> None:
        """A dog that lies for minutes, gets up, looks round, ambles and sniffs a little, then lies down again."""
        if self.clip == "lying":
            self.play("get_up")
            self.play("look", float(self.rng.uniform(2.0, 5.0)))
        for _ in range(int(self.rng.integers(1, 3))):
            if self.rng.random() < 0.6:
                self.trip("walk", 3.0, 15.0)
            else:
                self.trip("sniff_walk", 2.0, 6.0)
            self.play("sniff", float(self.rng.uniform(2.0, 8.0)))
        self.lie(40.0, 180.0)

    def lie(self, low: float, high: float) -> None:
        """Stand a moment, lie down, and lie for a while."""
        self.play("stand", float(self.rng.uniform(1.0, 2.5)))
        self.play("lie_down")
        self.play("lying", float(self.rng.uniform(low, high)))

    def speed(self, clip: str) -> float:
        """How fast the body travels in a clip, so its paws stay planted at this dog's size and pace."""
        return float(self.clips[clip]["speed_m_s"]) * self.pace * self.size

    def play(self, clip: str, seconds: Optional[float] = None, curve: float = 0.0, distance: Optional[float] = None,
             frame: float = 0.0, spin: float = 0.0) -> None:
        """Add one leg: a clip for some seconds, for its own length when it does not loop, over a distance, or
        stepping round on the spot by an angle."""
        spec = self.clips[clip]
        fps = float(spec["fps"]) * self.pace
        same = clip == self.clip
        leg = Leg(t0=self.t, clip=clip, frame=self.frame if same else frame, x=float(self.xy[0]),
                  y=float(self.xy[1]), heading=self.heading, curve=curve, v_in=0.0 if spin else self.v,
                  v=0.0 if spin else self.speed(clip), held=None if same or self.clip is None else (self.clip, self.frame),
                  spin=math.copysign(PIVOT_RAD_S, spin) if spin else 0.0)
        if spin:
            seconds = abs(spin) / PIVOT_RAD_S
        elif distance is not None:
            seconds = time_for(distance, leg.v_in, leg.v, self.blend)
        elif not spec["loop"]:
            seconds = (int(spec["frames"]) - 1) / fps
        self.legs.append(leg)
        x, y, self.heading = travel(leg, seconds, self.blend)
        self.xy, self.t = np.array([x, y]), self.t + seconds
        self.clip, self.frame = clip, clip_frame(spec, leg.frame + seconds * fps)
        self.v = leg.v_in + (leg.v - leg.v_in) * min(seconds / self.blend, 1.0)

    def trip(self, gait: str, low: float, high: float, home: bool = False) -> bool:
        """Turn and travel in a gait to a spot the open ground allows, or look round a while when there is none.

        Every heading with room for the trip is a choice, the ones nearer straight on likelier, so a dog keeps to
        its aisle and turns where the aisle lets it.
        """
        slide = self.speed(gait) * self.blend / 2.0
        choices, weights = [], []
        for turn in self._turns(home):
            for radius in TURN_RADIUS_M + (0.0,):
                room = self._room(turn, radius, high + slide)
                if room is None:
                    continue
                length = min(float(self.rng.uniform(low, high)), room - slide)
                end = np.array(along(*self.xy, self.heading, self._curve(turn, radius), abs(turn) * radius)[:2])
                end_heading = self.heading + turn
                end += length * np.array([math.cos(end_heading), math.sin(end_heading)])
                if length >= low and not (home and np.hypot(*(end - self.home)) > HOME_RADIUS_M):
                    choices.append((turn, radius, length))
                    # Stepping round on the spot is what a dog does only where it cannot walk the turn.
                    weights.append(math.exp(-abs(turn) / TURN_PREFERENCE_RAD) * (1.0 if radius else PIVOT_WEIGHT))
                break
        if not choices:
            self.play("look", float(self.rng.uniform(1.5, 4.0)))
            return False
        turn, radius, length = choices[int(self.rng.choice(len(choices), p=np.divide(weights, sum(weights))))]
        if not radius and abs(turn) > 1e-6:
            if self.v > 0.0:
                self.play("stand", self.blend)
            self.play("walk", spin=turn)
        elif abs(turn) > 1e-6:
            self.play("sniff_walk" if gait == "sniff_walk" else "walk", curve=self._curve(turn, radius),
                      distance=abs(turn) * radius)
        self.play(gait, distance=length)
        return True

    def _turns(self, home: bool) -> List[float]:
        """The turns a trip considers: sixteen compass headings, a few near straight on, and home when far from it."""
        turns = [_wrap(h - self.heading) for h in np.linspace(-math.pi, math.pi, 16, endpoint=False)]
        turns += [float(t) for t in self.rng.normal(0.0, 0.3, 4)]
        if home and np.hypot(*(self.xy - self.home)) > HOME_RADIUS_M / 2.0:
            turns.append(_wrap(math.atan2(self.home[1] - self.xy[1], self.home[0] - self.xy[0]) - self.heading))
        return turns

    @staticmethod
    def _curve(turn: float, radius: float) -> float:
        """The curvature of a walked turn: one over its radius, positive to the left, none for no turn or a turn
        on the spot."""
        return math.copysign(1.0 / radius, turn) if abs(turn) > 1e-6 and radius else 0.0

    def _room(self, turn: float, radius: float, reach: float) -> Optional[float]:
        """How far a dog can run straight on, up to reach, once it has walked a turn; None when the turn itself
        leaves the open ground."""
        curve = self._curve(turn, radius)
        arc = abs(turn) * radius
        x, y = float(self.xy[0]), float(self.xy[1])
        bend = np.array([along(x, y, self.heading, curve, s)[:2] for s in np.arange(0.0, arc + PATH_STEP_M, PATH_STEP_M)])
        if not self.ground.planned_at(bend).all():
            return None
        ex, ey, _ = along(x, y, self.heading, curve, arc)
        heading = self.heading + turn
        ahead = np.arange(PATH_STEP_M, reach + PATH_STEP_M, PATH_STEP_M)
        ok = self.ground.planned_at(np.stack([ex + ahead * math.cos(heading), ey + ahead * math.sin(heading)], axis=1))
        # A cell is wider than the step, so the last open sample may already sit on the edge of a closed cell.
        return float(reach if ok.all() else ahead[int(np.argmin(ok))] - self.ground.step - PATH_STEP_M)

    def _open_heading(self) -> float:
        """The heading, of sixteen, with the longest open run ahead: along the aisle a dog starts in."""
        steps = np.arange(PATH_STEP_M, 20.0, PATH_STEP_M)
        best, reach = 0.0, -1.0
        for heading in np.linspace(-math.pi, math.pi, 16, endpoint=False):
            ok = self.ground.planned_at(self.xy + np.outer(steps, [math.cos(heading), math.sin(heading)]))
            run = float(steps[-1] if ok.all() else steps[int(np.argmin(ok))])
            if run > reach:
                best, reach = float(heading), run
        return best


def _wrap(angle: float) -> float:
    """An angle folded into -pi..pi."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


# ---------------------------------------------------------------------- #
# where a dog is, and what shape it holds
# ---------------------------------------------------------------------- #
def distance(tau: float, v_in: float, v: float, blend: float) -> float:
    """How far a leg has carried the body after tau seconds, its speed eased from v_in to v over the blend."""
    if tau < blend:
        return v_in * tau + (v - v_in) * tau * tau / (2.0 * blend)
    return (v_in + v) * blend / 2.0 + v * (tau - blend)


def time_for(gone: float, v_in: float, v: float, blend: float) -> float:
    """How long a leg takes to carry the body a distance, the inverse of distance()."""
    ramp = (v_in + v) * blend / 2.0
    if gone >= ramp:
        return blend + (gone - ramp) / v
    a = (v - v_in) / (2.0 * blend)
    if abs(a) < 1e-12:
        return gone / v_in
    return (-v_in + math.sqrt(max(v_in * v_in + 4.0 * a * gone, 0.0))) / (2.0 * a)


def along(x: float, y: float, heading: float, curve: float, gone: float) -> Tuple[float, float, float]:
    """East, north and heading after travelling a distance from a point, straight on or round an arc."""
    if abs(curve) < 1e-9:
        return x + gone * math.cos(heading), y + gone * math.sin(heading), heading
    turned = heading + curve * gone
    return x + (math.sin(turned) - math.sin(heading)) / curve, y - (math.cos(turned) - math.cos(heading)) / curve, turned


def travel(leg: Leg, tau: float, blend: float) -> Tuple[float, float, float]:
    """East, north and heading of the body tau seconds into a leg."""
    if leg.spin:
        return leg.x, leg.y, leg.heading + leg.spin * tau
    return along(leg.x, leg.y, leg.heading, leg.curve, distance(tau, leg.v_in, leg.v, blend))


def clip_frame(spec: Dict[str, Any], frame: float) -> float:
    """A frame index within a clip: wrapped round a loop, held on the last frame of one that ends."""
    frames = int(spec["frames"])
    return frame % frames if spec["loop"] else min(frame, frames - 1.0)


def locate(dog: Dog, t: float) -> Tuple[float, float, float, Leg, float]:
    """East, north and heading of a dog at a time, with the leg it is on and how far into it."""
    leg = dog.legs[max(bisect.bisect_right(dog.starts, t) - 1, 0)]
    tau = max(t - leg.t0, 0.0)
    x, y, heading = travel(leg, tau, dog.blend)
    return x, y, heading, leg, tau


def _frame_shape(poses: np.ndarray, spec: Dict[str, Any], frame: float) -> np.ndarray:
    """The shared vertices a clip holds at a fractional frame, blended between the two frames round it."""
    frame = clip_frame(spec, frame)
    first = int(math.floor(frame))
    share = frame - first
    frames = int(spec["frames"])
    second = (first + 1) % frames if spec["loop"] else min(first + 1, frames - 1)
    start = int(spec["start"])
    return poses[start + first] * (1.0 - share) + poses[start + second] * share


def shape(dog: Dog, leg: Leg, tau: float) -> np.ndarray:
    """The render vertices of a dog tau seconds into a leg, eased out of the pose held before it, at its size."""
    table = _catalogue()
    clips = table["bodies"][dog.breed]["clips"]
    poses, corners = _poses(dog.breed)
    spec = clips[leg.clip]
    vertices = _frame_shape(poses, spec, leg.frame + tau * float(spec["fps"]) * dog.pace)
    if leg.held is not None and tau < dog.blend:
        held = _frame_shape(poses, clips[leg.held[0]], leg.held[1])
        share = tau / dog.blend
        vertices = held * (1.0 - share) + vertices * share
    return vertices[corners] * np.float32(dog.size)


def _pixels(shot: camera.View, point: np.ndarray, length: float) -> float:
    """How long a body this long at a point stands on a frame, in pixels; 0 when the frame does not show it, a margin
    of that length round the frame's edges included."""
    forward = np.asarray(shot.forward, dtype=float)
    up = np.asarray(shot.up, dtype=float)
    right = np.cross(forward, up)
    right /= norm(right)
    up = np.cross(right, forward)
    offset = point - np.asarray(shot.eye, dtype=float)
    depth = dot(offset, forward)
    if depth <= 0.0:
        return 0.0
    half_v = math.tan(math.radians(shot.vertical_fov_deg) / 2.0)
    half_h = half_v * shot.width / shot.height
    reach = length / depth
    if abs(dot(offset, right)) / depth > half_h + reach or abs(dot(offset, up)) / depth > half_v + reach:
        return 0.0
    return reach * shot.height / (2.0 * half_v)


def _spawn(cli: int, draw: Dict[str, Any]) -> Dog:
    """Create one drawn dog's body with its coat and heat map, visual only, so nothing collides with it."""
    table = _catalogue()
    body = table["bodies"][draw["breed"]]
    visual = p.createVisualShape(p.GEOM_MESH, fileName=os.path.join(DOGS_DIR, body["obj"]), physicsClientId=cli)
    uid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=visual, physicsClientId=cli)
    coat = p.loadTexture(os.path.join(DOGS_DIR, body["coats"][draw["coat"]]), physicsClientId=cli)
    p.changeVisualShape(uid, -1, textureUniqueId=coat, rgbaColor=[1.0, 1.0, 1.0, 1.0], physicsClientId=cli)
    if THERMAL:
        heat = p.loadTexture(os.path.join(DOGS_DIR, body["heat_map"]), physicsClientId=cli)
        p.changeVisualShape(uid, -1, thermalTextureUniqueId=heat, temperatureRange=table["heat"]["range_c"],
                            emissivity=table["heat"]["emissivity"], physicsClientId=cli)
    return Dog(body=int(uid), breed=draw["breed"], coat=draw["coat"], size=draw["size"], pace=draw["pace"],
               mood=draw["mood"], length=draw["size"] * float(body["length_m"]), blend=float(table["blend_s"]),
               legs=draw["legs"])
