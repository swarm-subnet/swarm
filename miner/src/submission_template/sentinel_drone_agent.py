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

"""Starter agent for the SWARM SENTINEL family (cf_solar_patrol).

Copy this file to drone_agent.py in your submission zip. It flies a whole patrol from the
state vector and the site map alone: it takes off, sweeps the park in lanes, comes home and
lands. It never looks at the images and never reports, so it finds nobody: replace the
sweep with your own search and add the reporting.

Observation (dict, every array float32 and read-only, decided 10 times a second):
    - "rgb": (480, 640, 3) in [0, 1], the wide colour camera, a new frame every 0.5 s.
    - "thermal": (512, 640, 1) in [0, 1], White Hot. Only the feed you picked is drawn;
      the other one is all zeros.
    - "zoom": (480, 640, 3) in [0, 1], the last close-up you asked for, zeros before any.
    - "state": (31,), the indices this agent reads:
        0:3   position east, north, up in metres from the dock (the dock is 0, 0, 0)
        3     compass heading in degrees, 0 north, clockwise positive, in [-180, 180)
        7     height above the dock in metres
        9     time left in seconds, counting down from 390
        19    ground distance in metres, 0.5 to 16, reads 60 when nothing is in range
        23    flight phase: 0 docked, 1 taking off, 2 flying, 3 returning, 4 landing, 5 landed
    - "site_map": (660,), sent on the FIRST observation only and all zeros after it, so
      keep it. [0] is the fence point count, [1:129] the fence as (east, north) pairs in
      metres from the dock.

Action:
    numpy array (24,). [0] move_forward, [1] move_right, [2] move_up and [3] turn are in
    [-1, 1] (shares of 5 m/s across, 3 m/s up, 2 m/s down and 90 deg/s); [4] gimbal_tilt in
    [-1, 1] maps -90 deg (straight down) to +90 deg; [5] to [23] are in [0, 1]. Buttons
    (zoom, report, take_off [21], return_home [22], cancel_return [23]) count only on the
    decision their value rises through 0.5: holding one presses it once.

Mission: take off from the dock, search the park, report every intruder with a box on the
image you saw them in, press return home and be landed in the dock before 390 s.

Scoring:
    seed = 0.70 detection + 0.20 coverage + 0.10 flight
    Every miss and every false alarm then turns 10 of your best seeds to 0.

Constraints:
    - Heights count from the dock: above 22 m the height half of the flight score drops, to 0
      at 30 m, and the drone cannot climb past 30 m; a report from a frame taken above 24 m is a false alarm.
    - 5 m before the flight limit the patrol ends on the spot, without the landing score.
    - 400 ms of compute per camera frame, shared by the five decisions of that frame.
"""

import math

import numpy as np

ACTION_DIM = 24
MAX_HORIZONTAL_MPS = 5.0
MAX_CLIMB_MPS = 3.0
MAX_DESCENT_MPS = 2.0
CRUISE_HEIGHT_M = 19.5        # above the dock: under the 20 m coverage line and the 22 m score line
TOP_HEIGHT_M = 21.5           # the highest it climbs to clear raised ground
GROUND_CLEARANCE_M = 8.0      # climb when the ground comes closer than this
LANE_SPACING_M = 25.0         # the wide camera covers about 28 m across from 20 m
FENCE_INSET_M = 4.0           # lanes stay this far inside the fence
LEG_MARGIN_M = 1.5            # legs between lanes stay this far inside the fence
CORNER_STEP_M = 3.0           # a way round a bend goes over points this far inside the fence's corners
SAMPLE_M = 1.0
WAYPOINT_M = 2.0
BRAKE_MPS2 = 1.5
HOME_MARGIN_S = 30.0          # the climb, the landing and some slack after the flight home

# State and site map indices, as listed in the docstring above.
POSITION, HEADING, HEIGHT, TIME_LEFT, GROUND, PHASE = slice(0, 2), 3, 7, 9, 19, 23
DOCKED, FLYING = 0, 2
GIMBAL, NIGHT_MODE, TAKE_OFF, RETURN_HOME = 4, 6, 21, 22


def _inside(fence, points):
    """Which points lie inside the fence polygon, by counting the sides a ray from each crosses."""
    a, b = fence, np.roll(fence, -1, axis=0)
    x, y = points[:, :1], points[:, 1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        cross = ((a[:, 1] > y) != (b[:, 1] > y)) & (x < a[:, 0] + (y - a[:, 1]) * (b[:, 0] - a[:, 0]) / (b[:, 1] - a[:, 1]))
    return cross.sum(axis=1) % 2 == 1


def _edge_distance(fence, points):
    """Distance from each point to the nearest side of the fence."""
    a = fence
    ab = np.roll(fence, -1, axis=0) - a
    rel = points[:, None, :] - a[None, :, :]
    t = np.clip((rel * ab).sum(axis=2) / (ab * ab).sum(axis=1), 0.0, 1.0)
    return np.min(np.linalg.norm(rel - t[..., None] * ab, axis=2), axis=1)


def _clear(fence, points, margin):
    """Which points are inside the fence and at least margin from its sides."""
    return _inside(fence, points) & (_edge_distance(fence, points) >= margin)


def _lanes(fence, heading_deg):
    """The sweep's waypoints for lanes along one compass heading, each lane's runs flown back and forth."""
    h = math.radians(heading_deg)
    along, across = np.array([math.sin(h), math.cos(h)]), np.array([math.cos(h), -math.sin(h)])
    lo, hi = (fence @ along).min(), (fence @ along).max()
    t = np.arange(lo, hi + SAMPLE_M, SAMPLE_M)
    low, high = (fence @ across).min(), (fence @ across).max()
    count = max(1, math.ceil((high - low) / LANE_SPACING_M))
    waypoints = []
    for k, c in enumerate(low + (np.arange(count) + 0.5) * (high - low) / count):
        points = c * across + t[:, None] * along
        keep = np.concatenate([[False], _clear(fence, points, FENCE_INSET_M), [False]])
        starts, ends = np.flatnonzero(keep[1:] & ~keep[:-1]), np.flatnonzero(~keep[1:] & keep[:-1]) - 1
        runs = [(points[s], points[e]) for s, e in zip(starts, ends) if e > s]
        if k % 2:
            runs = [(e, s) for s, e in reversed(runs)]
        waypoints += [p for run in runs for p in run]
    return np.array(waypoints).reshape(-1, 2)


def _line_clear(fence, a, b):
    """Whether the straight line from a to b keeps LEG_MARGIN_M inside the fence."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    n = max(2, int(np.hypot(*(b - a)) / SAMPLE_M) + 1)
    return bool(np.all(_clear(fence, np.linspace(a, b, n), LEG_MARGIN_M)))


def _legs_clear(fence, route):
    """Whether every straight leg of the route, from the dock on, keeps LEG_MARGIN_M inside the fence."""
    path = np.vstack([[0.0, 0.0], route])
    return all(_line_clear(fence, a, b) for a, b in zip(path[:-1], path[1:]))


def _shortest(routes):
    """The route of least length from the dock, or an empty one when there is none."""
    return min(routes, key=lambda r: float(np.hypot(*np.diff(np.vstack([[0.0, 0.0], r]), axis=0).T).sum()),
               default=np.zeros((0, 2)))


def _inner_corners(fence):
    """Each fence corner stepped CORNER_STEP_M inside, towards whichever of eight directions lands furthest from the sides."""
    steps = CORNER_STEP_M * np.array([[math.cos(k * math.pi / 4), math.sin(k * math.pi / 4)] for k in range(8)])
    corners = []
    for corner in fence:
        tries = corner + steps
        tries = tries[_clear(fence, tries, LEG_MARGIN_M)]
        if len(tries):
            corners.append(tries[np.argmax(_edge_distance(fence, tries))])
    return corners


def _way_round(a, b, corners, clear):
    """The shortest chain of corners from a to b whose legs all keep inside the fence, ending at b; None if none does."""
    nodes = [a, b, *corners]
    best, came, done = {0: 0.0}, {}, set()
    while True:
        waiting = [i for i in best if i not in done]
        if not waiting:
            return None
        i = min(waiting, key=best.get)
        if i == 1:
            chain = []
            while i:
                chain.append(nodes[i])
                i = came[i]
            return chain[::-1]
        done.add(i)
        for j in range(1, len(nodes)):
            length = best[i] + float(np.hypot(*(nodes[j] - nodes[i])))
            if j not in done and length < best.get(j, math.inf) and clear(nodes[i], nodes[j]):
                best[j], came[j] = length, i


def _joined(route, corners, clear):
    """The route from the dock with every leg that would leave the fence sent round over the corners; None if one cannot."""
    path, here = [], np.zeros(2)
    for point in route:
        if not clear(here, point):
            way = _way_round(here, point, corners, clear)
            if way is None:
                return None
            path += way[:-1]
        path.append(point)
        here = point
    return np.array(path)


def plan_route(fence):
    """The shortest lane sweep over twelve headings, started from the dock.

    A sweep whose legs all stay inside the fence is taken when there is one; on a fence that bends so that none does,
    the legs that would leave it go round over its corners instead.
    """
    sweeps = [route for heading in range(0, 180, 15) for lanes in [_lanes(fence, heading)]
              for route in (lanes, lanes[::-1]) if len(route)]
    best = _shortest([route for route in sweeps if _legs_clear(fence, route)])
    if len(best):
        return best
    corners, known = _inner_corners(fence), {}

    def clear(a, b):
        """Whether the straight leg from a to b keeps inside the fence, worked out once per leg."""
        key = (*np.round(a, 3), *np.round(b, 3))
        if key not in known:
            known[key] = _line_clear(fence, a, b)
        return known[key]

    return _shortest([way for way in (_joined(route, corners, clear) for route in sweeps) if way is not None])


class DroneFlightController:
    """Patrol baseline: takes off, sweeps the park in lanes at 19.5 m above the dock, returns home and lands."""

    def __init__(self):
        """Construction hook for your policy; the baseline loads nothing."""
        # Load your trained model here (any framework).
        self.reset()

    def reset(self):
        """Forget the last patrol: its route, where it got to, and the buttons it held."""
        self.fence = None
        self.route = None
        self.waypoint = 0
        self.homing = False
        self.held = np.zeros(ACTION_DIM, dtype=np.float32)

    def act(self, observation):
        """One decision: press take-off in the dock, fly the sweep, then press return home and let the dock land."""
        state = np.asarray(observation["state"], dtype=np.float64)
        if self.route is None:
            self.route = self._read_route(observation.get("site_map"))
        action = np.zeros(ACTION_DIM, dtype=np.float32)
        action[GIMBAL] = -1.0         # straight down
        action[NIGHT_MODE] = 1.0      # auto: the colour camera brightens itself in the dark
        phase = int(round(state[PHASE]))
        if phase == DOCKED:
            self._press(action, TAKE_OFF)
        elif phase == FLYING:
            self._fly(state, action)
        self.held = action
        return action

    def _read_route(self, site_map):
        """The sweep over the fence the first observation brings, or an empty route when it brings none."""
        if site_map is None:
            return np.zeros((0, 2))
        site_map = np.asarray(site_map, dtype=np.float64)
        count = int(round(site_map[0]))
        if count < 3:
            return np.zeros((0, 2))
        self.fence = site_map[1:1 + 2 * count].reshape(count, 2)
        return plan_route(self.fence)

    def _press(self, action, index):
        """Press a button, or release it if it was held on the last decision, so every press is a fresh rise through 0.5."""
        action[index] = 0.0 if self.held[index] > 0.5 else 1.0

    def _fly(self, state, action):
        """Steer along the sweep at cruise height; once it is done or time is short, head home and press return home."""
        xy = state[POSITION]
        if not self.homing:
            while self.waypoint < len(self.route) and np.hypot(*(self.route[self.waypoint] - xy)) < WAYPOINT_M:
                self.waypoint += 1
            home_s = np.hypot(*xy) / MAX_HORIZONTAL_MPS + HOME_MARGIN_S
            if self.waypoint >= len(self.route) or state[TIME_LEFT] < home_s:
                self.homing, self.waypoint = True, min(self.waypoint, len(self.route)) - 1
        if self.homing:
            # The dock flies home in a straight line, which ends the patrol if it crosses the flight limit's stop line.
            if self.fence is None or self.waypoint < 0 or _line_clear(self.fence, xy, (0.0, 0.0)):
                self._press(action, RETURN_HOME)
                return
            while self.waypoint > 0 and np.hypot(*(self.route[self.waypoint] - xy)) < WAYPOINT_M:
                self.waypoint -= 1
        to_go = self.route[self.waypoint] - xy
        distance = float(np.hypot(*to_go))
        speed = min(MAX_HORIZONTAL_MPS, distance, math.sqrt(2.0 * BRAKE_MPS2 * distance))
        east, north = to_go / max(distance, 1e-6) * speed
        h = math.radians(state[HEADING])
        action[0] = np.clip((east * math.sin(h) + north * math.cos(h)) / MAX_HORIZONTAL_MPS, -1.0, 1.0)
        action[1] = np.clip((east * math.cos(h) - north * math.sin(h)) / MAX_HORIZONTAL_MPS, -1.0, 1.0)
        target = CRUISE_HEIGHT_M
        if state[GROUND] < GROUND_CLEARANCE_M:
            target = min(TOP_HEIGHT_M, state[HEIGHT] + GROUND_CLEARANCE_M - state[GROUND])
        rise = target - state[HEIGHT]
        action[2] = np.clip(rise / (MAX_CLIMB_MPS if rise >= 0.0 else MAX_DESCENT_MPS), -1.0, 1.0)
