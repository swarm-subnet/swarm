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

"""The reference pilot the seed checks (task 24) fly every seed with, and the verdict it brings back.

It flies through the patrol's own environment: the real park, the seed's own dock spot, wind and thieves, with no
pictures drawn. It takes off, searches the whole park lane by lane at the patrol height over the ground, never higher
than the patrol height over the dock, presses return home and lands. Its lanes lie as far apart as the ground the
wide camera sees, which searches by night too with night mode on. Twice a second, as the camera would, it looks at
every thief inside the fence the way a pilot who knows where he stands would, turning and tilting towards him, and
counts his pixels in the object map of that exact view, so panels, trees and the ground hide him as they would: in
the colour camera by day, in the thermal camera by night.

The seed passes when every thief who stepped inside the fence showed at least PIXELS_ACROSS pixels across his narrow
side, from the patrol height or lower above the dock and at least MIN_DISTANCE_M away (check 1), and the pilot landed
back in the dock before the patrol's time ran out (check 2).

Run as a module it judges the seeds given on the command line, or else those read from its standard input, and keeps
their verdicts.
"""

from __future__ import annotations

import contextlib
import io
import math
import sys
import time
from functools import lru_cache
from typing import Any, Dict, List, Tuple

import numpy as np
import pybullet as p
from shapely import contains_xy
from shapely.geometry import LineString, Point, Polygon

from swarm.constants import SIM_DT
from swarm.utils.env_factory import make_env_with_initial_obs

from . import airframe, camera, dock, flight_limit, theft, wind
from .contract import (
    ACTION_DIM,
    ACTION_INDEX,
    DECISION_STEPS,
    MAX_CLIMB_MPS,
    MAX_DESCENT_MPS,
    MAX_HORIZONTAL_MPS,
    MAX_YAW_RATE_DEG_S,
    PATROL_HEIGHT_M,
    RGB_SHAPE,
    THERMAL_SHAPE,
)
from .episode import SolarEpisode
from .seed_checks import Verdict, keep
from .task import solar_patrol_task

PIXELS_ACROSS = 6.0                    # thermal DRI: a person is recognised at 6 pixels across his critical dimension
MIN_DISTANCE_M = 5.0                   # decided: the pilot keeps 5 m from any threat
PERSON_NARROW_M = 0.6                  # wider than any person's narrow side, so a view too far to pass is never drawn
CROP_PX = 128                          # the object map is drawn around the thief only, at the lens's own pixel size
AIM_ABOVE_GROUND_M = 0.9               # where the camera points on a thief: his middle standing, his back lying

FENCE_INSET_M = 4.0                    # lanes stay this far inside the fence, clear of the flight limit's stop line
CONNECT_INSET_M = 2.0                  # a hop between lanes never goes nearer the fence than this
LANE_OVERLAP = 0.1                     # neighbouring lanes' views overlap by this share of their width
LANE_ANGLES_DEG = tuple(range(0, 180, 15))
CELL_STEP_M = 0.5                      # the park is read across the lanes at this spacing to cut it into strips
GAP_GRID_M = 2.0                       # the ground a route leaves out is looked for on a grid this fine
GAP_SLACK_M = 0.5                      # how much further than half a lane the furthest ground may lie
SAMPLE_M = 2.0                         # the route is followed, and the ground under it read, at this spacing
LOOKAHEAD_M = 16.0                     # the pilot climbs for ground this far ahead before it reaches it
CARROT_M = 6.0                         # the pilot steers at the point this far ahead of it on the route
SEARCH_M = 40.0                        # its place on the route is looked for this far ahead of the last one
CORNER_DEG = 45.0                      # a turn sharper than this is taken slowly
CORNER_MPS = 1.0
BRAKE_MPS2 = 1.2                       # inside the outputs' own 2 m/s2 easing
CEILING_MARGIN_M = 0.5                 # the pilot's height over the dock stays this far under the patrol height
HEIGHT_GAIN_PER_S = 1.0
TURN_GAIN_PER_S = 2.0
TURNING_MPS = 1.0                      # the pilot faces where it flies only once it flies faster than this
RETURN_CLEAR_M = 1.0                   # return home is pressed once the straight flight home stays this far inside
RAY_REACH_M = 1000.0
RAY_BATCH = 512
GROUND_RAYS = 16                       # a ray looking for the ground passes at most this many bodies standing on it

_FRAME_DECISIONS = camera.FRAME_STEPS // DECISION_STEPS
_SEG_FLAGS = getattr(p, "ER_SWARM_RAYCAST", 0) | getattr(p, "ER_ALPHA_CUTOUT", 0)


def check(seed: int) -> Verdict:
    """Fly one seed with the reference pilot and judge it."""
    started = time.process_time()
    with contextlib.redirect_stdout(io.StringIO()):
        env, _obs = make_env_with_initial_obs(solar_patrol_task(seed=int(seed), sim_dt=SIM_DT), wrap_runtime=_Unseen)
    # The verdict reads no clearance metric, observation, reward or info, and none of them steers the flight.
    env._update_min_clearance = env._computeObs = env._computeReward = env._computeInfo = _unread
    try:
        ep = env._solar
        night = camera.night(env)
        pilot = _Pilot(env, ep, lane_spacing())
        sightings = _Sightings(env, ep, night)
        decision = 0
        while True:
            _obs, _reward, terminated, truncated, _info = env.step(pilot.action(decision)[None, :])
            decision += 1
            if decision % _FRAME_DECISIONS == 0:
                sightings.look()
            if terminated or truncated:
                break
        outcome = ep.outcome
        landed = outcome.end_reason == "landed" and outcome.landed_in_dock
        best = list(sightings.best.values())
        result = Verdict(seed=int(seed), passed=False, end_reason=outcome.end_reason,
                         landed_s=round(ep.time_s, 2) if landed else None,
                         threats=max(int(outcome.threats), len(best)),
                         seen=sum(px >= PIXELS_ACROSS for px in best), best_px=[round(px, 1) for px in best],
                         night=night, wind=wind.strength(seed), route_m=round(pilot.length_m, 1))
    finally:
        env.close()
    if not landed:
        result.reason = f"not_landed:{result.end_reason}"
    elif result.seen < result.threats:
        result.reason = "thief_unseen"
    result.passed = not result.reason
    result.cpu_s = round(time.process_time() - started, 1)
    return result


def _unread() -> None:
    """Stands in for the parts of an environment step a reference flight never reads."""


class _Unseen:
    """The patrol's runtime with nothing drawn: after each step only the parts that fly the drone run, and no
    observation is built."""

    def __init__(self, runtime: Any):
        """Wrap the family's own runtime."""
        self._runtime = runtime

    def __getattr__(self, name: str) -> Any:
        """Everything else is the family's own."""
        return getattr(self._runtime, name)

    def post_step_update(self, env: Any) -> None:
        """The dock's flight phase and the flight limit, the two parts that end or steer a flight. The readings only
        a model reads (camera, zoom, laser, wind, battery) are skipped, and the aircraft's drawn body is posed only
        for the pictures the pilot takes."""
        ep = env._solar
        ep.step += 1
        dock.update(env, ep)
        flight_limit.update(env, ep)
        if env._collision:
            ep.end("collision")

    def observation_part(self, env: Any, key: str) -> np.ndarray:
        """The pilot reads the simulator, not the observation."""
        return np.zeros(1, dtype=np.float32)


# ---------------------------------------------------------------------------------------------- the route

def lane_spacing() -> float:
    """How far apart the lanes lie: the width of ground the wide camera sees straight down from the patrol height,
    less the overlap. It searches by night too, with night mode on, as the camera and coverage parts allow."""
    height, width = RGB_SHAPE[:2]
    half = math.tan(math.radians(camera.WIDE_DIAGONAL_FOV_DEG / 2.0)) * width / math.hypot(width, height)
    return 2.0 * PATROL_HEIGHT_M * half * (1.0 - LANE_OVERLAP)


def path_length(points: np.ndarray) -> float:
    """Length of the path through the points."""
    return float(np.hypot(*np.diff(np.asarray(points, dtype=float), axis=0).T).sum())


def _along_edge(area: Polygon, a: np.ndarray, b: np.ndarray) -> List[np.ndarray]:
    """From a to b inside the area: straight when that stays inside, else along the area's edge the shorter way."""
    if area.covers(LineString([a, b])):
        return [a, b]
    ring = area.exterior
    coords = np.asarray(ring.coords)[:, :2]
    corners = coords[:-1]
    at = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(coords, axis=0).T))])[:-1]
    total = ring.length
    start, end = ring.project(Point(a)), ring.project(Point(b))

    def walk(sense: float) -> List[np.ndarray]:
        """The edge from start to end one way round: its two ends and every corner passed on the way."""
        reach = (sense * (end - start)) % total
        passed = sorted((offset, i) for i, offset in enumerate((sense * (at - start)) % total) if 0.0 < offset < reach)
        return ([np.asarray(ring.interpolate(start).coords[0])[:2]] + [corners[i] for _, i in passed]
                + [np.asarray(ring.interpolate(end).coords[0])[:2]])

    return [a] + min(walk(1.0), walk(-1.0), key=path_length) + [b]


def _pieces(area: Any, u: np.ndarray, v: np.ndarray, reach: Tuple[float, float],
            offset: float) -> List[Tuple[float, float]]:
    """Where the line at an offset across the lane direction runs inside the area, as stretches along the lanes."""
    cut = area.intersection(LineString([reach[0] * u + offset * v, reach[1] * u + offset * v]))
    stretches = []
    for piece in getattr(cut, "geoms", [cut]):
        if piece.geom_type == "LineString" and piece.length > 0.0:
            along = np.asarray(piece.coords)[:, :2] @ u
            stretches.append((float(along.min()), float(along.max())))
    return sorted(stretches)


def _cells(area: Any, fence: np.ndarray, u: np.ndarray, v: np.ndarray,
           jump: float) -> List[List[Tuple[float, float, float]]]:
    """The area cut into strips no inward corner splits, read across the lane direction every CELL_STEP_M: each strip
    a run of rows (offset, start, end), ending where the next row splits it, joins it to another, or moves one of
    its ends further than jump, as the edge does at an inward corner."""
    reach = (float((fence @ u).min()) - 1.0, float((fence @ u).max()) + 1.0)
    low, high = float((fence @ v).min()), float((fence @ v).max())
    growing: List[List[Tuple[float, float, float]]] = []
    done: List[List[Tuple[float, float, float]]] = []
    for offset in np.arange(low + CELL_STEP_M / 2.0, high, CELL_STEP_M):
        pieces = _pieces(area, u, v, reach, float(offset))
        following = []
        for a, b in pieces:
            touching = [cell for cell in growing if cell[-1][1] < b and a < cell[-1][2]]
            alone = len(touching) == 1 and sum(a2 < touching[0][-1][2] and touching[0][-1][1] < b2
                                               for a2, b2 in pieces) == 1
            if alone and max(abs(a - touching[0][-1][1]), abs(b - touching[0][-1][2])) <= jump:
                touching[0].append((float(offset), a, b))
                following.append(touching[0])
            else:
                following.append([(float(offset), a, b)])
        done += [cell for cell in growing if not any(cell is kept for kept in following)]
        growing = following
    return done + growing


def _cell_lanes(cell: List[Tuple[float, float, float]], spacing: float, u: np.ndarray,
                v: np.ndarray) -> List[np.ndarray]:
    """A strip's lanes, spread evenly across it no further apart than spacing, each a start and an end point."""
    first, last = cell[0][0] - CELL_STEP_M / 2.0, cell[-1][0] + CELL_STEP_M / 2.0
    count = max(1, math.ceil((last - first) / spacing))
    rows = np.array([row[0] for row in cell])
    lanes = []
    for k in range(count):
        offset = first + (last - first) / count * (k + 0.5)
        _row, a, b = cell[int(np.argmin(np.abs(rows - offset)))]
        lanes.append(np.array([a * u + offset * v, b * u + offset * v]))
    return lanes


def _serpentine(strips: List[List[np.ndarray]]) -> List[np.ndarray]:
    """Every strip's lanes in one run: from where the last lane ended, the nearest unflown strip is entered at its
    nearest corner and its lanes are flown back and forth from there."""
    remaining = list(strips)
    flown: List[np.ndarray] = []
    here = remaining[0][0][0]
    while remaining:
        best: Tuple[float, int, bool, bool] = (math.inf, 0, False, False)
        for i, lanes in enumerate(remaining):
            for rows_reversed in (False, True):
                first = lanes[-1] if rows_reversed else lanes[0]
                for lane_reversed in (False, True):
                    distance = float(np.hypot(*(first[int(lane_reversed)] - here)))
                    if distance < best[0]:
                        best = (distance, i, rows_reversed, lane_reversed)
        _distance, i, rows_reversed, lane_reversed = best
        lanes = remaining.pop(i)
        for k, lane in enumerate(lanes[::-1] if rows_reversed else lanes):
            flown.append(lane[::-1] if lane_reversed != (k % 2 == 1) else lane)
        here = flown[-1][1]
    return flown


def search_gap(area: Any, path: np.ndarray) -> float:
    """The furthest any ground of the area, read every GAP_GRID_M, lies from the path."""
    low_x, low_y, high_x, high_y = area.bounds
    xs, ys = np.meshgrid(np.arange(low_x, high_x, GAP_GRID_M), np.arange(low_y, high_y, GAP_GRID_M))
    grid = np.column_stack([xs.ravel(), ys.ravel()])
    grid = grid[contains_xy(area, grid[:, 0], grid[:, 1])]
    nearest = np.full(len(grid), np.inf)
    for a, b in zip(path, path[1:]):
        ab = b - a
        t = np.clip((grid - a) @ ab / max(float(ab @ ab), 1e-12), 0.0, 1.0)
        nearest = np.minimum(nearest, np.hypot(*(grid - a - t[:, None] * ab).T))
    return float(nearest.max()) if len(grid) else 0.0


@lru_cache(maxsize=8)
def _lanes(fence_key: Tuple[Tuple[float, float], ...], spacing: float) -> Tuple[np.ndarray, ...]:
    """The lanes of a full search in flying order, each a start and an end point. The park is cut into strips no
    inward corner splits, a corner being an edge that jumps half a lane or more, and each strip's lanes are spread
    evenly across it, no further apart than spacing. Of the lane directions tried, the shortest that leaves no ground
    more than half a lane from a lane wins, or the one that leaves the least when none does. The fence never moves,
    so the lanes are the same every seed."""
    fence = np.asarray(fence_key, dtype=float)
    area = Polygon(fence).buffer(-FENCE_INSET_M)
    best: Tuple[Tuple[float, float], List[np.ndarray]] = ((math.inf, math.inf), [])
    for angle in LANE_ANGLES_DEG:
        u = np.array([math.cos(math.radians(angle)), math.sin(math.radians(angle))])
        v = np.array([-u[1], u[0]])
        strips = [_cell_lanes(cell, spacing, u, v) for cell in _cells(area, fence, u, v, spacing / 2.0)]
        if not strips:
            continue
        lanes = _serpentine(strips)
        path = np.concatenate(lanes)
        gap = search_gap(area, path)
        rank = (0.0 if gap <= spacing / 2.0 + GAP_SLACK_M else gap, path_length(path))
        if rank < best[0]:
            best = (rank, lanes)
    return tuple(best[1])


def route(fence: np.ndarray, dock_xy: np.ndarray, spacing: float) -> Tuple[np.ndarray, float]:
    """The pilot's path across the ground, and how far along it the search ends: from the dock to the nearer end of
    the search, every lane, then back towards the dock, never nearer the fence than CONNECT_INSET_M off the lanes."""
    dock_xy = np.asarray(dock_xy, dtype=float)[:2]
    lanes = list(_lanes(tuple(map(tuple, np.round(np.asarray(fence, dtype=float), 6))), round(float(spacing), 6)))
    if np.hypot(*(lanes[-1][1] - dock_xy)) < np.hypot(*(lanes[0][0] - dock_xy)):
        lanes = [lane[::-1] for lane in lanes[::-1]]
    area = Polygon(fence).buffer(-CONNECT_INSET_M)
    points = _along_edge(area, dock_xy, lanes[0][0])
    for lane, nxt in zip(lanes, lanes[1:]):
        points += _along_edge(area, lane[1], nxt[0])
    points.append(lanes[-1][1])
    search = [points[0]]
    for point in points[1:]:
        if np.hypot(*(point - search[-1])) > 1e-6:
            search.append(point)
    home = _along_edge(area, lanes[-1][1], dock_xy)[1:]
    return np.asarray(search + home, dtype=float), path_length(np.asarray(search))


def _resample(path: np.ndarray, step: float) -> np.ndarray:
    """Points every step metres along the path, its end included."""
    along = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(path, axis=0).T))])
    at = np.append(np.arange(0.0, along[-1], step), along[-1])
    return np.column_stack([np.interp(at, along, path[:, 0]), np.interp(at, along, path[:, 1])])


def _tops(cli: int, xy: np.ndarray, fallback: float) -> np.ndarray:
    """The height of the first thing under each point, ground, panel or roof, or the fallback where there is none."""
    tops = np.full(len(xy), float(fallback))
    for start in range(0, len(xy), RAY_BATCH):
        chunk = xy[start:start + RAY_BATCH]
        hits = p.rayTestBatch([[x, y, RAY_REACH_M] for x, y in chunk], [[x, y, -RAY_REACH_M] for x, y in chunk],
                              physicsClientId=cli)
        for i, hit in enumerate(hits):
            if hit[0] >= 0:
                tops[start + i] = float(hit[3][2])
    return tops


def _ground_z(cli: int, x: float, y: float, terrain_uids: frozenset, fallback: float) -> float:
    """The ground's height at a point, under anything standing on it."""
    top = RAY_REACH_M
    for _ in range(GROUND_RAYS):
        hit = p.rayTest([x, y, top], [x, y, -RAY_REACH_M], physicsClientId=cli)[0]
        if hit[0] < 0:
            break
        if hit[0] in terrain_uids:
            return float(hit[3][2])
        top = float(hit[3][2]) - 0.01
    return float(fallback)


def _corner_speeds(points: np.ndarray) -> np.ndarray:
    """The fastest the pilot passes each point and can still take the sharp turns ahead of it at CORNER_MPS."""
    speeds = np.full(len(points), MAX_HORIZONTAL_MPS)
    reach = max(1, int(round(4.0 / SAMPLE_M)))
    for i in range(reach, len(points) - reach):
        before, after = points[i] - points[i - reach], points[i + reach] - points[i]
        norms = float(np.hypot(*before) * np.hypot(*after))
        if norms > 0.0 and float(before @ after) < norms * math.cos(math.radians(CORNER_DEG)):
            speeds[i] = CORNER_MPS
    for i in range(len(points) - 2, -1, -1):
        speeds[i] = min(speeds[i], math.sqrt(speeds[i + 1] ** 2 + 2.0 * BRAKE_MPS2 * SAMPLE_M))
    return speeds


class _Pilot:
    """The reference pilot: presses take-off, flies the route at its heights and speeds facing where it goes, and
    presses return home once the search is done and the straight flight home stays inside the fence."""

    def __init__(self, env: Any, ep: SolarEpisode, spacing: float):
        """Lay the route for this seed's dock and read the ground under it."""
        self.env, self.ep = env, ep
        path, search_m = route(ep.fence, ep.dock_position[:2], spacing)
        self.points = _resample(path, SAMPLE_M)
        self.length_m = path_length(path)
        self.search_end = min(len(self.points) - 1, int(math.ceil(search_m / SAMPLE_M)))
        dock_z = float(ep.dock_position[2])
        tops = _tops(env.CLIENT, self.points, dock_z)
        ahead = int(round(LOOKAHEAD_M / SAMPLE_M))
        highest = np.array([tops[i:i + ahead + 1].max() for i in range(len(tops))])
        self.heights = np.minimum(dock_z + PATROL_HEIGHT_M - CEILING_MARGIN_M, highest + PATROL_HEIGHT_M)
        self.speeds = _corner_speeds(self.points)
        self.home_area = Polygon(ep.fence).buffer(-RETURN_CLEAR_M)
        self.index = 0

    def action(self, decision: int) -> np.ndarray:
        """This decision's action vector."""
        a = np.zeros(ACTION_DIM, dtype=np.float32)
        a[ACTION_INDEX["gimbal_tilt"]] = -1.0
        if self.ep.phase == "docked":
            a[ACTION_INDEX["take_off"]] = 1.0 if decision == 0 else 0.0
            return a
        if self.ep.phase != "flying":
            return a
        pos = np.asarray(self.env.pos[0], dtype=float)
        window = self.points[self.index:self.index + int(SEARCH_M / SAMPLE_M) + 1]
        self.index += int(np.argmin(np.hypot(*(window - pos[:2]).T)))
        last = len(self.points) - 1
        if self.index >= self.search_end and (
                self.index == last or self.home_area.covers(LineString([pos[:2], self.ep.dock_position[:2]]))):
            a[ACTION_INDEX["return_home"]] = 1.0
            return a
        carrot = self.points[min(last, self.index + int(CARROT_M / SAMPLE_M))]
        heading = carrot - pos[:2]
        distance = float(np.hypot(*heading))
        speed = min(float(self.speeds[self.index]), distance)
        vx, vy = (heading / distance * speed) if distance > 1e-6 else (0.0, 0.0)
        vz = float(np.clip(HEIGHT_GAIN_PER_S * (self.heights[self.index] - pos[2]), -MAX_DESCENT_MPS, MAX_CLIMB_MPS))
        yaw = float(self.env.rpy[0, 2])
        c, s = math.cos(yaw), math.sin(yaw)
        a[ACTION_INDEX["move_forward"]] = (vx * c + vy * s) / MAX_HORIZONTAL_MPS
        a[ACTION_INDEX["move_right"]] = (vx * s - vy * c) / MAX_HORIZONTAL_MPS
        a[ACTION_INDEX["move_up"]] = vz / (MAX_CLIMB_MPS if vz >= 0.0 else MAX_DESCENT_MPS)
        if speed > TURNING_MPS:
            error = (math.atan2(vy, vx) - yaw + math.pi) % (2.0 * math.pi) - math.pi
            rate = float(np.clip(math.degrees(TURN_GAIN_PER_S * error), -MAX_YAW_RATE_DEG_S, MAX_YAW_RATE_DEG_S))
            # The turn stick is clockwise positive on the compass, the simulator's yaw counter-clockwise.
            a[ACTION_INDEX["turn"]] = -rate / MAX_YAW_RATE_DEG_S
        return a


# ---------------------------------------------------------------------------------------------- the thieves

def narrow_px(mask: np.ndarray) -> float:
    """Pixels across the narrow side of the shape a mask holds: its extent across its own long axis."""
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return 0.0
    if rows.size < 3:
        return float(min(np.ptp(rows), np.ptp(cols)) + 1)
    y, x = rows - rows.mean(), cols - cols.mean()
    long_axis = 0.5 * math.atan2(2.0 * float((x * y).mean()), float((x * x).mean() - (y * y).mean()))
    across = -x * math.sin(long_axis) + y * math.cos(long_axis)
    return round(float(across.max() - across.min() + 1.0), 3)


class _Sightings:
    """Every thief's best view so far: how many pixels across he showed the pilot looking straight at him."""

    def __init__(self, env: Any, ep: SolarEpisode, night: bool):
        """Take the lens of the search: thermal at night, colour by day."""
        self.env, self.ep = env, ep
        self.feed = camera.FEEDS[int(night)]
        height, width = (THERMAL_SHAPE if night else RGB_SHAPE)[:2]
        diagonal = camera.THERMAL_DIAGONAL_FOV_DEG if night else camera.WIDE_DIAGONAL_FOV_DEG
        half = math.tan(math.radians(camera.vertical_fov_deg(diagonal, width, height) / 2.0))
        self.pixel_rad = 2.0 * half / height
        self.crop_fov_deg = 2.0 * math.degrees(math.atan(half * CROP_PX / height))
        self.best: Dict[int, float] = {}

    def look(self) -> None:
        """Look at every thief inside the fence not yet seen well enough, from where the camera is now."""
        env, ep = self.env, self.ep
        people = [person for person in theft.people(ep) if person["threat"] and person["bodies"]]
        if not people:
            return
        index = theft.bodies(ep)
        for person in people:
            self.best.setdefault(index.get(int(person["bodies"][0]), -1), 0.0)
        eye, _forward, drone_forward = airframe.camera_pose(env, -90.0)
        if eye[2] - ep.dock_position[2] > PATROL_HEIGHT_M:
            return
        for person in people:
            n = index.get(int(person["bodies"][0]), -1)
            if self.best[n] >= PIXELS_ACROSS:
                continue
            x, y = person["xy"]
            ground = _ground_z(env.CLIENT, x, y, ep.terrain_uids, float(ep.dock_position[2]))
            ray = np.array([x, y, ground + AIM_ABOVE_GROUND_M]) - eye
            distance = float(np.linalg.norm(ray))
            if distance < MIN_DISTANCE_M or PERSON_NARROW_M / (distance * self.pixel_rad) < PIXELS_ACROSS:
                continue
            view = self._view(eye, ray / distance, drone_forward)
            airframe.update(env, ep)
            theft.show(env, ep, view)
            self.best[n] = max(self.best[n], narrow_px(np.isin(self._object_map(view), person["bodies"])))

    def _view(self, eye: np.ndarray, forward: np.ndarray, drone_forward: np.ndarray) -> camera.View:
        """The camera turned and tilted to look along forward, level as the gimbal holds it."""
        up = np.array([0.0, 0.0, 1.0]) - forward[2] * forward
        if np.linalg.norm(up) < 1e-6:
            up = np.asarray(drone_forward, dtype=float)
        up = up / np.linalg.norm(up)
        return camera.View(feed=self.feed, eye=tuple(map(float, eye)), forward=tuple(map(float, forward)),
                           up=tuple(map(float, up)), width=CROP_PX, height=CROP_PX,
                           vertical_fov_deg=self.crop_fov_deg, sees=True, step=self.ep.step)

    def _object_map(self, view: camera.View) -> np.ndarray:
        """Which body each pixel of the view shows."""
        view_matrix, projection = view.matrices()
        _w, _h, _rgb, _depth, seg = p.getCameraImage(CROP_PX, CROP_PX, view_matrix, projection,
                                                     renderer=p.ER_TINY_RENDERER, flags=_SEG_FLAGS,
                                                     physicsClientId=self.env.CLIENT)
        return np.reshape(np.asarray(seg), (CROP_PX, CROP_PX))


def judge(seed: int) -> Verdict:
    """The seed's verdict; a seed whose own world refuses to be built or flown fails, the same everywhere. Any other
    error is the machine's, not the seed's, and is raised so no verdict is kept for it."""
    try:
        return check(seed)
    except RuntimeError as exc:
        return Verdict(seed=int(seed), passed=False, reason=f"error:{exc}"[:200])


def main(argv: List[str]) -> None:
    """Judge every seed named on the command line and keep its verdict; with none named, every seed read from the
    standard input, one a line, as it arrives."""
    for seed in argv or sys.stdin:
        keep(judge(int(seed)))


if __name__ == "__main__":
    main(sys.argv[1:])
