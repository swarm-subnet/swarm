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

"""People passing outside the field: in the seeds the mix gives them, one to three people walk by beyond its edge.

They are not people to find. Each walks a lane of LANE_M through the band BAND_M outside the field's outline, end to
end at a normal walk. Some stop on the way for a while; at the lane's end some leave the scene and the others stay
standing there; their arrivals spread over the patrol, so some already walk at take-off and some come late. The people
module dresses them from the same wardrobe and under the same rule as the people on the field, so nothing but where
they walk tells them apart.

A lane is drawn on the passer's own stream as a run of straight steps along the band. It starts at a point of open
ground in the band heading along the outline, and each step turns as little as it can to keep its line inside the band by LINE_SLACK_M,
on open ground no steeper than MAX_SLOPE_DEG: rays dropped on it and SIDE_M either side of it meet only the ground, and
none of those points lies under a tree crown the site lists, or within COVER_CLEAR_M of one. Where the site lists path
cells and the band holds enough of them, the lane starts on one and each step keeps to them as far as it can. His whole track is written when the lane is drawn, and the lane is drawn again unless every frame of it lies
outside the outline and inside the band, within SIDE_M of the lane's line, and APART_M from every passer planned
before him whenever both are in the scene.

A passer never reacts to the drone, so his track is the seed's alone. His body is posed, as a walker's is, only for a
picture that can see it and only when his frame changed, and kept out of the scene before he arrives and once he has
left. His reference point is the ground under his hips.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pybullet as p

from swarm.constants import SIM_DT

from ..solar_patrol.dock import _edge_distance, _inside
from ..solar_patrol.theft import AWAY_Z, _in_view
from ..solar_patrol.theft_moves import GONE, Actor, Frame, Route, Step, heading_to
from . import figures, mix, site, walkers
from .contract import HORIZON_S
from .episode import FieldEpisode
from .moves import library

OUTSIDERS_SEED_STREAM = 0x0B5E55       # the passers' own stream, so no other draw of the seed moves them
COUNT_SHARES = {1: 0.6, 2: 0.3, 3: 0.1}
BAND_M = (6.0, 40.0)                   # every frame of a passer lies this far outside the outline
LINE_SLACK_M = 1.0                     # a lane's line keeps this far inside the band, so a body's sway stays in it
SIDE_M = 0.6                           # rays are dropped this far either side of the line; no frame strays further
LANE_M = (30.0, 120.0)
STEP_M = 4.0
SAMPLE_M = 0.5                         # the ground is read this often along a step, so no wall slips between
TURN_DEG = 10.0                        # a step turns by whole multiples of this
TURNS = 3                              # and by at most this many, so a lane bends but never doubles back
WANDER_DEG = 6.0                       # the heading drifts at most this much each step, so a lane is never ruler straight
HEADINGS = 16                          # headings the first step may take
START_TRIES = 200
MAX_SLOPE_DEG = 15.0
COVER_CLEAR_M = 0.5                    # a lane keeps this far out from under every crown, which no ray meets
PATH_CELL_M = 1.0                      # the survey's grid, whose path cells the site may list
PATH_CELLS = 30                        # the band must hold this many path cells for a lane to start on one
PACE = (0.9, 1.0)                      # a normal walk, as a share of the library's plain walk
STOP_SHARE = 0.4
STOP_AT = (0.25, 0.75)                 # where along his lane a passer who stops does
STOP_S = (5.0, 30.0)
LEAVE_SHARE = 0.5
ARRIVE_S = (-40.0, 120.0)              # when he sets foot on his lane, patrol seconds; before take-off he is under way
APART_M = 1.5
LANE_TRIES = 40
HIPS_M = 0.9                           # how far above the ground a body's middle stands, for the view test


@dataclass
class Pass:
    """One passer's walk past the field: his lane, his whole track in patrol frames, how long he stops, whether he
    leaves and when he arrives, and his body and how it is posed now."""

    lane: np.ndarray
    frames: List[Frame]
    stop_s: float = 0.0
    leaves: bool = False
    arrive_s: float = 0.0
    body: Any = None
    cli: int = 0
    terrain_uids: frozenset = frozenset()
    dock_z: float = 0.0
    posed: Optional[Tuple[float, float, float]] = None
    pose_key: Optional[tuple] = None
    grounds: Dict[Tuple[float, float], float] = field(default_factory=dict)

    def frame(self, k: int) -> Frame:
        """Frame k of the patrol: before his track the first, after it the last, which shows him gone or standing."""
        return self.frames[min(max(int(k), 0), len(self.frames) - 1)]


def count(seed: int) -> int:
    """How many people pass outside the field in a seed: none unless the mix has them, else 1 to 3 by COUNT_SHARES."""
    if not mix.present(seed).outsiders:
        return 0
    rng = np.random.default_rng([OUTSIDERS_SEED_STREAM, int(seed), 0])
    keys = list(COUNT_SHARES)
    share = np.array([COUNT_SHARES[k] for k in keys], dtype=float)
    return int(keys[int(rng.choice(len(keys), p=share / share.sum()))])


# The lane

def _unit(heading: float) -> np.ndarray:
    """The ground direction of a heading, radians anticlockwise from east."""
    return np.array([math.cos(heading), math.sin(heading)])


def _in_band(outline: np.ndarray, points: np.ndarray, slack: float) -> np.ndarray:
    """Which points lie outside the outline and inside the band, slack metres short of each of its sides."""
    gap = _edge_distance(outline, points)
    return ~_inside(outline, points) & (gap >= BAND_M[0] + slack) & (gap <= BAND_M[1] - slack)


def _along(outline: np.ndarray, point: np.ndarray) -> float:
    """The heading along the outline at the side nearest a point: square to the way out to it."""
    a = outline
    ab = np.roll(outline, -1, axis=0) - a
    t = np.clip(((point - a) * ab).sum(axis=1) / (ab * ab).sum(axis=1), 0.0, 1.0)
    near = a + t[:, None] * ab
    out = point - near[int(np.argmin(np.linalg.norm(point - near, axis=1)))]
    return math.atan2(float(out[1]), float(out[0])) + math.pi / 2.0


def flat(points: np.ndarray) -> np.ndarray:
    """Open flat ground everywhere: the ground under every point at height 0."""
    return np.zeros(len(points))


def ground_test(cli: int, terrain_uids: frozenset, open_at: Optional[Callable] = None) -> Callable[[np.ndarray], np.ndarray]:
    """A reading of the ground under points from rays: its height under each, NaN where a ray meets anything else
    first or no ground lies under it. A site that knows where someone may stand gives open_at, a function of east and
    north arrays; the ground is NaN wherever it says no, so a lane keeps off what no ray meets."""

    def heights(points: np.ndarray) -> np.ndarray:
        """The terrain height under each point, NaN where it is not open ground."""
        hits = p.rayTestBatch([[x, y, site.RAY_REACH_M] for x, y in points],
                              [[x, y, -site.RAY_REACH_M] for x, y in points], physicsClientId=cli)
        found = np.array([hit[3][2] if hit[0] in terrain_uids else np.nan for hit in hits], dtype=float)
        if open_at is not None and len(found):
            points = np.asarray(points, dtype=float).reshape(-1, 2)
            found[~np.asarray(open_at(points[:, 0], points[:, 1]), dtype=bool)] = np.nan
        return found

    return heights


def _open(ground: Callable, cover: Optional[np.ndarray], low: np.ndarray, high: np.ndarray) -> Callable:
    """A reading of open ground that also takes every point under a crown circle (east, north, radius), or within
    COVER_CLEAR_M of one, for not open; only the crowns that reach the box from low to high are kept."""
    if cover is None or not len(cover):
        return ground
    cover = np.asarray(cover, dtype=float).reshape(-1, 3)
    reach = cover[:, 2:] + COVER_CLEAR_M
    cover = cover[np.all((cover[:, :2] + reach >= low) & (cover[:, :2] - reach <= high), axis=1)]

    def heights(points: np.ndarray) -> np.ndarray:
        """The ground's height under each point, NaN where it is not open or lies under a crown."""
        z = ground(points)
        if not len(cover):
            return z
        gap = np.hypot(points[:, None, 0] - cover[None, :, 0], points[:, None, 1] - cover[None, :, 1])
        return np.where(np.any(gap < cover[None, :, 2] + COVER_CLEAR_M, axis=1), np.nan, z)

    return heights


def _on_path(cells: Optional[set], points: np.ndarray) -> float:
    """The share of points that lie on a path cell; 0 with no paths."""
    if not cells:
        return 0.0
    keys = np.floor(np.asarray(points, dtype=float) / PATH_CELL_M).astype(int)
    return float(np.mean([(int(i), int(j)) in cells for i, j in keys]))


def _step(outline: np.ndarray, ground: Callable, here: np.ndarray, here_z: float,
          heading: float) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """The points of a step from here along a heading and the ground's height at its end, if its line stays in the
    band and every point of it and SIDE_M either side is open ground no steeper than MAX_SLOPE_DEG; else None."""
    way = _unit(heading)
    line = np.round(here + way * np.arange(SAMPLE_M, STEP_M + 1e-9, SAMPLE_M)[:, None], 3)
    if not _in_band(outline, line, LINE_SLACK_M).all():
        return None
    side = np.array([-way[1], way[0]]) * SIDE_M
    z = ground(np.vstack([line, line + side, line - side]))
    if not np.all(np.isfinite(z)):
        return None
    centre, left, right = z.reshape(3, -1)
    rise = math.tan(math.radians(MAX_SLOPE_DEG))
    if np.max(np.abs(np.diff(np.concatenate([[here_z], centre])))) > rise * SAMPLE_M or \
            np.max(np.abs(left - right)) > rise * 2.0 * SIDE_M:
        return None
    return line, centre, float(centre[-1])


def lane(rng: np.random.Generator, outline: np.ndarray, length: float, ground: Callable = flat,
         paths: Optional[np.ndarray] = None, cover: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """One try at a lane: its points every STEP_M, drawn from the generator, from a start in the band along the band
    for length metres or as far as it goes; None when it goes less than LANE_M[0]. ground reads the height of open
    ground under points (NaN elsewhere); paths are the centres of path cells near the field and cover the crown
    circles round it, if the site lists them."""
    outline = np.asarray(outline, dtype=float)
    low, high = outline.min(axis=0) - BAND_M[1], outline.max(axis=0) + BAND_M[1]
    ground = _open(ground, cover, low, high)
    cells = None
    start = None
    if paths is not None and len(paths):
        paths = np.asarray(paths, dtype=float).reshape(-1, 2)
        banded = paths[_in_band(outline, paths, LINE_SLACK_M)]
        if len(banded) >= PATH_CELLS:
            cells = {(int(i), int(j)) for i, j in np.floor(paths / PATH_CELL_M).astype(int)}
            start = np.round(banded[int(rng.integers(len(banded)))], 3)
    if start is None:
        for _ in range(START_TRIES):
            spot = np.round(rng.uniform(low, high), 3)
            if _in_band(outline, spot[None, :], LINE_SLACK_M)[0] and math.isfinite(ground(spot[None, :])[0]):
                start = spot
                break
        else:
            return None
    start_z = ground(start[None, :])[0]
    if not math.isfinite(start_z):
        return None
    sign = 1.0 if rng.random() < 0.5 else -1.0
    heading = _along(outline, start) + (0.0 if sign > 0 else math.pi)
    # The first step may take any heading, the nearest to along the band first; later ones turn as little as they can.
    first = sorted((2.0 * math.pi * k / HEADINGS for k in range(HEADINGS)),
                   key=lambda h: abs(math.remainder(h - heading, 2.0 * math.pi)))
    points, here, here_z = [start], start, start_z
    for n in range(int(math.ceil(length / STEP_M))):
        if n:
            heading += math.radians(float(rng.uniform(-WANDER_DEG, WANDER_DEG)))
            side = 1.0 if rng.random() < 0.5 else -1.0
            options = [heading + side * s * math.radians(TURN_DEG * k) for k in range(TURNS + 1) for s in (1.0, -1.0)
                       if k or s > 0]
        else:
            options = first
        best = None
        for option in options:
            step = _step(outline, ground, here, here_z, option)
            if step is None:
                continue
            share = _on_path(cells, step[0])
            if best is None or share > best[0]:
                best = (share, option, step)
            if not cells or share >= 1.0:
                break
        if best is None:
            break
        _share, heading, (line, _z, here_z) = best
        here = line[-1]
        points.append(here)
    if (len(points) - 1) * STEP_M < LANE_M[0]:
        return None
    return np.array(points, dtype=float)


# The track

def _walk_context() -> Dict[str, str]:
    """The loop a body is in once it has set off walking."""
    lib = library()
    return dict(lib.blocks[lib.kinds({"hub": walkers.STAND})[walkers.SET_OFF][0]]["next"])


def _split(points: np.ndarray, share: float) -> List[np.ndarray]:
    """A lane cut in two at a share of its length, the cut point ending the first part and starting the second."""
    legs = np.hypot(*np.diff(points, axis=0).T)
    at = share * float(legs.sum())
    k = int(np.searchsorted(np.cumsum(legs), at))
    cut = points[k] + (points[k + 1] - points[k]) * ((at - float(legs[:k].sum())) / legs[k])
    return [np.vstack([points[:k + 1], cut]), np.vstack([cut, points[k + 1:]])]


def track(points: np.ndarray, scale: float, pace: float, mirrored: bool, stop_at: float, stop_s: float, leaves: bool,
          arrive_s: float) -> List[Frame]:
    """A passer's whole track in patrol frames: away until he arrives, in stride along the lane, a stop of stop_s at
    stop_at of its length when stop_s is not 0, and at its end gone if he leaves, else standing. Arriving before
    take-off he is that far along, but never past half of his track."""
    lib = library()
    legs = _split(points, stop_at) if stop_s > 0.0 else [points]
    yaw = heading_to(points[1] - points[0])
    actor = Actor(lib, (float(points[0][0]), float(points[0][1])), yaw, scale=scale, mirrored=mirrored, pace=pace)
    actor.context = _walk_context()
    done = []

    def script():
        """The lane leg by leg, a stop between them, then leaving or stopping at its end."""
        for n, leg in enumerate(legs):
            route = Route(leg)
            if n:
                yield Step(walkers.SET_OFF, steer=route, stage="walking")
            yield Step("loop", cycles=None, steer=route, then=walkers.STOP, stage="walking")
            if n == len(legs) - 1 and leaves:
                yield Step("gone", stage="gone")
                return
            yield Step(walkers.STOP, steer=route, stage="walking")
            if n < len(legs) - 1:
                yield Step("hold", hold=stop_s, stage="stopped")
        done.append(True)

    actor.script = script()
    most = int((HORIZON_S - ARRIVE_S[0]) * lib.fps) + 1
    while not done and not actor.gone and len(actor.frames) < most:
        actor.run_to(len(actor.frames) + 1)
    frames = actor.frames
    start = int(round(arrive_s * lib.fps))
    if start < 0:
        return frames[min(-start, len(frames) // 2):]
    away = Frame(GONE, mirrored, float(points[0][0]), float(points[0][1]), yaw, "away")
    return [away] * start + frames


def _xy(frames: Sequence[Frame], length: int) -> Tuple[np.ndarray, np.ndarray]:
    """East and north of each of a track's first length frames, the last repeated past its end, and which of them
    show him in the scene."""
    shown = [frames[min(k, len(frames) - 1)] for k in range(length)]
    return (np.array([[f.x, f.y] for f in shown], dtype=float).reshape(-1, 2),
            np.array([f.index != GONE for f in shown], dtype=bool))


def _off_line(points: np.ndarray, line: np.ndarray) -> np.ndarray:
    """How far each point lies from the nearest point of a polyline."""
    a, ab = line[:-1], np.diff(line, axis=0)
    rel = points[:, None, :] - a[None, :, :]
    t = np.clip((rel * ab).sum(axis=2) / np.maximum((ab * ab).sum(axis=1), 1e-12), 0.0, 1.0)
    return np.min(np.linalg.norm(rel - t[..., None] * ab, axis=2), axis=1)


def keeps_out(outline: np.ndarray, points: np.ndarray, frames: Sequence[Frame]) -> bool:
    """Whether every frame of a track that shows him lies outside the outline, inside the band and within SIDE_M of
    the lane's line."""
    xy, shown = _xy(frames, len(frames))
    spots = np.unique(xy[shown], axis=0)
    if not len(spots):
        return True
    gap = _edge_distance(np.asarray(outline, dtype=float), spots)
    return bool(not _inside(np.asarray(outline, dtype=float), spots).any() and gap.min() >= BAND_M[0]
                and gap.max() <= BAND_M[1] and _off_line(spots, points).max() <= SIDE_M)


def apart(a: Sequence[Frame], b: Sequence[Frame]) -> bool:
    """Whether two tracks keep APART_M from each other in every frame both show them in the scene."""
    length = max(len(a), len(b))
    xa, shown_a = _xy(a, length)
    xb, shown_b = _xy(b, length)
    both = shown_a & shown_b
    return bool(not both.any() or np.min(np.hypot(*(xa[both] - xb[both]).T)) >= APART_M)


def plan(seed: int, n: int, outline: np.ndarray, scale: float, ground: Callable = flat,
         paths: Optional[np.ndarray] = None, others: Sequence[Pass] = (), cover: Optional[np.ndarray] = None) -> Optional[Pass]:
    """Passer n's walk, drawn on his own stream: how far, how fast, whether and where he stops, whether he leaves,
    when he arrives, then lanes until one whose track keeps out of the field and apart from the passers before him;
    None when no lane in LANE_TRIES does. scale is his build's stride times his size."""
    rng = np.random.default_rng([OUTSIDERS_SEED_STREAM, int(seed), 1, int(n)])
    length = float(rng.uniform(*LANE_M))
    pace = float(rng.uniform(*PACE))
    mirrored = bool(rng.random() < 0.5)
    stop_s = float(rng.uniform(*STOP_S)) if rng.random() < STOP_SHARE else 0.0
    stop_at = float(rng.uniform(*STOP_AT))
    leaves = bool(rng.random() < LEAVE_SHARE)
    arrive_s = float(rng.uniform(*ARRIVE_S))
    outline = np.asarray(outline, dtype=float)
    for _ in range(LANE_TRIES):
        points = lane(rng, outline, length, ground, paths, cover)
        if points is None:
            continue
        frames = track(points, scale, pace, mirrored, stop_at, stop_s, leaves, arrive_s)
        if keeps_out(outline, points, frames) and all(apart(frames, other.frames) for other in others):
            return Pass(lane=points, frames=frames, stop_s=stop_s, leaves=leaves, arrive_s=arrive_s)
    return None


# In the patrol

def start(env: Any, ep: FieldEpisode, passer: Pass, dress: Dict[str, Any], size: float) -> None:
    """Give a planned passer his body, dressed and kept out of the scene until a picture can see him."""
    passer.cli, passer.terrain_uids, passer.dock_z = env.CLIENT, ep.terrain_uids, float(ep.dock_position[2])
    passer.body = figures.Figure(figures.catalogue(), dress, walkers._pose_args(passer.frames[0], AWAY_Z), env.CLIENT,
                                 size=size, textures=figures.pictures(ep))


def ground(passer: Pass, frame: Frame) -> float:
    """The ground under a passer's hips in a frame, read once per spot; the dock's height where no terrain lies."""
    spot = (frame.x, frame.y)
    if spot not in passer.grounds:
        height = site.ground_at(passer.cli, passer.terrain_uids, frame.x, frame.y)
        passer.grounds[spot] = passer.dock_z if height is None else height
    return passer.grounds[spot]


def _frame_number(t: float) -> int:
    """The 30 fps frame of a track shown at patrol time t."""
    return max(0, int(math.floor(t * library().fps + 1e-9)))


def position(ep: FieldEpisode, passer: Pass, step: Optional[int] = None) -> np.ndarray:
    """Where a passer's reference point stood at a step up to now, and the ground under it, world metres; now when no
    step is given. Before he arrives it is his lane's start, once he has left its end."""
    frame = passer.frame(_frame_number((ep.step if step is None else min(int(step), ep.step)) * SIM_DT))
    return np.array([frame.x, frame.y, ground(passer, frame)], dtype=float)


def in_scene(ep: FieldEpisode, passer: Pass, step: Optional[int] = None) -> bool:
    """Whether a passer is in the scene at a step up to now, now when none is given."""
    t = (ep.step if step is None else min(int(step), ep.step)) * SIM_DT
    return passer.frame(_frame_number(t)).index != GONE


def advance(env: Any, ep: FieldEpisode, passers: Sequence[Pass]) -> None:
    """Pose the passers a picture taken on the step physics is about to run can see."""
    if not passers:
        return
    step = ep.step + 1
    shots = walkers.shots(env, ep, step)
    if shots:
        show(passers, step * SIM_DT, shots)


def show(passers: Sequence[Pass], t: float, shots: Sequence[Any]) -> None:
    """Pose every passer as he is at time t whose new pose, or the one his body still shows, one of the pictures can
    see, and whose frame changed; take one who is away out of the scene."""
    k = _frame_number(t)
    for passer in passers:
        frame = passer.frame(k)
        if frame.index == GONE:
            if passer.posed is not None:
                passer.body.pose(*walkers._pose_args(frame, AWAY_Z))
                passer.posed, passer.pose_key = None, None
            continue
        key = (frame.index, frame.mirrored, frame.x, frame.y, frame.yaw)
        if key == passer.pose_key:
            continue
        height = ground(passer, frame)
        now = (frame.x, frame.y, height + HIPS_M)
        if not any(_in_view(shot, now) or (passer.posed is not None and _in_view(shot, passer.posed))
                   for shot in shots):
            continue
        passer.body.pose(*walkers._pose_args(frame, height))
        passer.posed, passer.pose_key = now, key
