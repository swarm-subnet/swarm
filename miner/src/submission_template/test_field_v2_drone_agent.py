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

"""Starter agent for the SWARM TEST FIELD V2 family (cf_test_field_v2).

Copy this file to drone_agent.py in your submission zip. It flies a whole patrol from the
state vector and the site map alone: it takes off, sweeps the field in lanes well inside its
edge and clear of the tree crowns, comes home and lands. It never looks at the images and
never reports, so it counts nobody and nothing, and it does not know where anyone stands, so
it can fly too close to a person: replace the sweep with your own search and add the reporting.

Observation (dict, every array float32 and read-only, decided 10 times a second):
    - "rgb": (480, 640, 3) in [0, 1], the wide colour camera, a new frame every 0.5 s.
    - "thermal": (512, 640, 1) in [0, 1], White Hot. Only the feed you picked is drawn;
      the other one is all zeros.
    - "zoom": (480, 640, 3) in [0, 1], the last close-up you asked for, zeros before any.
    - "state": (31,), the indices this agent reads:
        0:3   position east, north, up in metres from the dock (the dock is 0, 0, 0)
        3     compass heading in degrees, 0 north, clockwise positive, in [-180, 180)
        7     height above the dock in metres
        9     time left in seconds, counting down from 200
        23    flight phase: 0 docked, 1 taking off, 2 flying, 3 returning, 4 landing, 5 landed
    - "site_map": (660,), sent on the FIRST observation only and all zeros after it, so
      keep it. [0] is the number of corners of the field's edge, [1:129] the edge as
      (east, north) pairs in metres from the dock; [258] the number of tree crowns,
      [259:579] five numbers per crown: centre east, centre north, its diameter twice, 0.

Action:
    numpy array (26,). [0] move_forward, [1] move_right, [2] move_up and [3] turn are in
    [-1, 1] (shares of 5 m/s across, 3 m/s up, 2 m/s down and 90 deg/s); [4] gimbal_tilt in
    [-1, 1] maps -90 deg (straight down) to +90 deg; [5] to [23] are in [0, 1]; [24]
    report_east and [25] report_north are in [-1, 1], where the reported person or animal
    stands as shares of 100 m east and north of the dock. Buttons (zoom, report, take_off [21],
    return_home [22], cancel_return [23]) count only on the decision their value rises
    through 0.5: holding one presses it once.

Mission: take off from the dock, report every person on the field and every goat, dog and
grown boar that lives on it (reported as class animal), each with a box on the image you saw
it in and where it stands, press return home and be landed in the dock before 200 s.

Scoring:
    seed = 0.50 people + 0.20 animals + 0.20 coverage + 0.10 flight
    On a seed with no animal to report, the animals term follows the people term, so such a
    seed scores 0.70 people + 0.20 coverage + 0.10 flight while no report that names an animal
    fails.

Constraints:
    - The drone must never come within 1 m of the field's edge, in any phase: the patrol ends
      there and the seed scores 0. A collision of any kind also scores 0, and flying into a
      tree crown's circle is a collision at any height.
    - The drone never rises more than 10 m above the dock. Over the field's high end that is only
      a few metres above the ground, so passing within 5 m of a person there zeroes the seed.
    - 500 ms of compute per camera frame, shared by the five decisions of that frame.
"""

import math

import numpy as np

ACTION_DIM = 26
MAX_HORIZONTAL_MPS = 5.0
MAX_CLIMB_MPS = 3.0
MAX_DESCENT_MPS = 2.0
CRUISE_HEIGHT_M = 9.0         # above the dock, under the 10 m ceiling
LANE_SPACING_M = 9.0          # the wide camera covers about 14 m across from 10 m, less over the field's high end
EDGE_INSET_M = 7.0            # lanes stay this far inside the field's edge
CROWN_MARGIN_M = 3.5          # lanes and legs stay this far off every tree crown's circle
LEG_MARGIN_M = 5.0            # legs between lanes, and the flight home, stay this far inside it: a sharp turn swings 3 m wide
CORNER_STEP_M = 6.0           # a way round a bend goes over points this far inside the edge's corners
SAMPLE_M = 1.0
WAYPOINT_M = 1.0
BRAKE_MPS2 = 1.5
HOME_MARGIN_S = 30.0          # the climb, the landing and some slack after the flight home

# State and site map indices, as listed in the docstring above.
POSITION, HEADING, HEIGHT, TIME_LEFT, PHASE = slice(0, 2), 3, 7, 9, 23
CROWN_COUNT, CROWNS = 258, 259
NO_CROWNS = np.zeros((0, 3))
DOCKED, FLYING = 0, 2
GIMBAL, NIGHT_MODE, TAKE_OFF, RETURN_HOME = 4, 6, 21, 22


def _inside(edge, points):
    """Which points lie inside the field's edge, by counting the sides a ray from each crosses."""
    a, b = edge, np.roll(edge, -1, axis=0)
    x, y = points[:, :1], points[:, 1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        cross = ((a[:, 1] > y) != (b[:, 1] > y)) & (x < a[:, 0] + (y - a[:, 1]) * (b[:, 0] - a[:, 0]) / (b[:, 1] - a[:, 1]))
    return cross.sum(axis=1) % 2 == 1


def _edge_distance(edge, points):
    """Distance from each point to the nearest side of the edge."""
    a = edge
    ab = np.roll(edge, -1, axis=0) - a
    rel = points[:, None, :] - a[None, :, :]
    t = np.clip((rel * ab).sum(axis=2) / (ab * ab).sum(axis=1), 0.0, 1.0)
    return np.min(np.linalg.norm(rel - t[..., None] * ab, axis=2), axis=1)


def _clear(edge, points, margin, crowns=NO_CROWNS):
    """Which points are inside the field's edge, at least margin from its sides and CROWN_MARGIN_M off every crown."""
    ok = _inside(edge, points) & (_edge_distance(edge, points) >= margin)
    for east, north, radius in crowns:
        ok &= np.hypot(points[:, 0] - east, points[:, 1] - north) >= radius + CROWN_MARGIN_M
    return ok


def _lanes(edge, heading_deg, crowns=NO_CROWNS):
    """The sweep's waypoints for lanes along one compass heading, each lane's runs flown back and forth."""
    h = math.radians(heading_deg)
    along, across = np.array([math.sin(h), math.cos(h)]), np.array([math.cos(h), -math.sin(h)])
    lo, hi = (edge @ along).min(), (edge @ along).max()
    t = np.arange(lo, hi + SAMPLE_M, SAMPLE_M)
    low, high = (edge @ across).min(), (edge @ across).max()
    count = max(1, math.ceil((high - low) / LANE_SPACING_M))
    waypoints = []
    for k, c in enumerate(low + (np.arange(count) + 0.5) * (high - low) / count):
        points = c * across + t[:, None] * along
        keep = np.concatenate([[False], _clear(edge, points, EDGE_INSET_M, crowns), [False]])
        starts, ends = np.flatnonzero(keep[1:] & ~keep[:-1]), np.flatnonzero(~keep[1:] & keep[:-1]) - 1
        runs = [(points[s], points[e]) for s, e in zip(starts, ends) if e > s]
        if k % 2:
            runs = [(e, s) for s, e in reversed(runs)]
        waypoints += [p for run in runs for p in run]
    return np.array(waypoints).reshape(-1, 2)


def _line_clear(edge, a, b, crowns=NO_CROWNS):
    """Whether the straight line from a to b keeps LEG_MARGIN_M inside the field's edge and clear of the crowns."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    n = max(2, int(np.hypot(*(b - a)) / SAMPLE_M) + 1)
    return bool(np.all(_clear(edge, np.linspace(a, b, n), LEG_MARGIN_M, crowns)))


def _legs_clear(edge, route, crowns=NO_CROWNS):
    """Whether every straight leg of the route, from the dock on, keeps inside the field's edge and clear of the crowns."""
    path = np.vstack([[0.0, 0.0], route])
    return all(_line_clear(edge, a, b, crowns) for a, b in zip(path[:-1], path[1:]))


def _shortest(routes):
    """The route of least length from the dock, or an empty one when there is none."""
    return min(routes, key=lambda r: float(np.hypot(*np.diff(np.vstack([[0.0, 0.0], r]), axis=0).T).sum()),
               default=np.zeros((0, 2)))


def _inner_corners(edge, crowns=NO_CROWNS):
    """Each edge corner stepped CORNER_STEP_M inside, towards whichever of eight directions lands furthest from the sides,
    and eight points round each crown just outside its margin: the points a way round a bend or a crown goes over."""
    rays = np.array([[math.cos(k * math.pi / 4), math.sin(k * math.pi / 4)] for k in range(8)])
    corners = []
    for corner in edge:
        tries = corner + CORNER_STEP_M * rays
        tries = tries[_clear(edge, tries, LEG_MARGIN_M, crowns)]
        if len(tries):
            corners.append(tries[np.argmax(_edge_distance(edge, tries))])
    for east, north, radius in crowns:
        tries = np.array([east, north]) + (radius + CROWN_MARGIN_M + 1.0) * rays
        corners += list(tries[_clear(edge, tries, LEG_MARGIN_M, crowns)])
    return corners


def _way_round(a, b, corners, clear):
    """The shortest chain of corners from a to b whose legs all keep inside the field's edge, ending at b; None if none does."""
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
    """The route from the dock with every leg that would leave the edge sent round over the corners; None if one cannot."""
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


def plan_route(edge, crowns=NO_CROWNS):
    """The shortest lane sweep over twelve headings, started from the dock, clear of the edge and the crowns.

    A sweep whose legs all stay clear is taken when there is one; otherwise the legs that would not go round over the
    edge's corners and the points round the crowns instead.
    """
    sweeps = [route for heading in range(0, 180, 15) for lanes in [_lanes(edge, heading, crowns)]
              for route in (lanes, lanes[::-1]) if len(route)]
    best = _shortest([route for route in sweeps if _legs_clear(edge, route, crowns)])
    if len(best):
        return best
    corners, known = _inner_corners(edge, crowns), {}

    def clear(a, b):
        """Whether the straight leg from a to b keeps inside the field's edge and off the crowns, worked out once per leg."""
        key = (*np.round(a, 3), *np.round(b, 3))
        if key not in known:
            known[key] = _line_clear(edge, a, b, crowns)
        return known[key]

    return _shortest([way for way in (_joined(route, corners, clear) for route in sweeps) if way is not None])


class DroneFlightController:
    """Field baseline: takes off, sweeps the field in lanes 9 m above the dock clear of the crowns, returns home and
    lands."""

    def __init__(self):
        """Construction hook for your policy; the baseline loads nothing."""
        # Load your trained model here (any framework).
        self.reset()

    def reset(self):
        """Forget the last patrol: its route, where it got to, and the buttons it held."""
        self.edge = None
        self.crowns = NO_CROWNS
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
        """The sweep over the field's edge the first observation brings, or an empty route when it brings none."""
        if site_map is None:
            return np.zeros((0, 2))
        site_map = np.asarray(site_map, dtype=np.float64)
        count = int(round(site_map[0]))
        if count < 3:
            return np.zeros((0, 2))
        self.edge = site_map[1:1 + 2 * count].reshape(count, 2)
        rows = site_map[CROWNS:CROWNS + 5 * int(round(site_map[CROWN_COUNT]))].reshape(-1, 5)
        crowns = np.column_stack([rows[:, 0], rows[:, 1], rows[:, 2] / 2.0])
        # Only a crown that reaches the field, margin included, can stand in the way.
        reach = _inside(self.edge, crowns[:, :2]) | (_edge_distance(self.edge, crowns[:, :2]) < crowns[:, 2] + CROWN_MARGIN_M)
        self.crowns = crowns[reach]
        return plan_route(self.edge, self.crowns)

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
            # The drone flies home in a straight line, which ends the patrol near the edge or through a crown.
            if self.edge is None or self.waypoint < 0 or _line_clear(self.edge, xy, (0.0, 0.0), self.crowns):
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
        rise = CRUISE_HEIGHT_M - state[HEIGHT]
        action[2] = np.clip(rise / (MAX_CLIMB_MPS if rise >= 0.0 else MAX_DESCENT_MPS), -1.0, 1.0)
