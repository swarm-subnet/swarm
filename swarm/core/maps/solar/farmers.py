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

"""Farmers on the public road outside the solar park.

The road the map ships (public_road.json) has two arms, one from the north and one from the east, that meet in the
yard outside the park's south tip, and the places along them where a truck can turn round. Each farmer comes in along
one arm, mostly from the end the last one did not, and either swings onto the other arm at the corner where they meet
and leaves by its end, or turns round in three or five moves, at a place along his arm or in the yard, and leaves back
the way he came or, from the yard, by either arm. He may stop a while before he turns. One farmer is on the road at a
time. Every pose comes baked from the road file, so a trip is only a speed profile over it: a pure
function of the seed, with no engine involved, that any check can replay to say where the truck is at any moment.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

CONFIG: Dict[str, Any] = {
    "seed_offset": 0xFA43E,        # the farmers draw from their own stream, so no other mover's draw moves
    "patrol_s": 390.0,             # the patrol the farmers are spread over
    "first_done": (0.0, 0.6),      # how far through their trip the first farmer is when the patrol starts
    "most": 4,                     # farmers a patrol holds at most
    "switch_end": 2.0 / 3.0,       # chance a farmer comes from the end the one before did not
    "through_share": 0.45,         # share of farmers who swing onto the other arm at the corner and leave by its end
    "yard_share": 0.3,             # share who drive on to the yard and leave by either arm; the rest turn back
    "gap_s": (5.0, 45.0),          # the road stands empty this long between one farmer leaving and the next coming
    "cruise_m_s": (4.0, 6.0),      # each farmer's own speed on the straight, 14 to 22 km/h on a dirt track
    "reverse_m_s": 1.2,            # backing up in the yard
    "lateral_m_s2": 1.5,           # bends are taken no faster than this sideways pull allows
    "accel_m_s2": 1.0,
    "brake_m_s2": 1.5,
    "stop_share": 0.5,             # share of farmers who stop where they turn round, before turning
    "stop_s": (10.0, 60.0),
    "pause_s": 1.5,                # the shortest stop, where the truck changes gear or puts the wheels on full lock
    "steer_s": 1.0,                # the wheels turn to their next lock over the last second of a stop
    "lock_jump": 0.05,             # a change of curvature (1/m) between samples that the truck must stop to steer
}
ARMS = ("north", "east")
AHEAD = slice(3, 11)               # a line row: x, y, curvature, then pose and twist facing along, then facing back
BACK = slice(11, 19)
TURNED = slice(4, 12)              # a turn row: x, y, gear, curvature, pose, twist


@dataclass(frozen=True)
class Turn:
    """A place a truck turns round: the arms it can come in on and leave by, the samples it starts and ends on counted
    from the lines' yard end, and its rows."""

    into: Tuple[str, ...]
    out: Tuple[str, ...]
    from_end: int
    to_end: int
    rows: np.ndarray

    @property
    def yard(self) -> bool:
        """Whether this is the turn in the yard, which takes either arm in and out."""
        return len(self.into) > 1

    @property
    def corner(self) -> bool:
        """Whether this swings from one arm onto the other at the corner where they meet."""
        return not self.yard and self.into != self.out


@dataclass(frozen=True)
class Road:
    """The baked road: the two arms as rows of rear axle samples with both poses, and every place to turn round."""

    lines: Dict[str, np.ndarray]
    turns: Tuple[Turn, ...]
    wheelbase: float
    wheel_radius: float


@lru_cache(maxsize=2)
def load_road(path: str) -> Road:
    """The road file, read once per process."""
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    lines = {name: np.asarray(raw["lines"][name], dtype=float) for name in ARMS}
    turns = tuple(Turn(tuple(t["into"]), tuple(t["out"]), int(t["from_end"]), int(t["to_end"]),
                       np.asarray(t["rows"], dtype=float)) for t in raw["turns"])
    return Road(lines, turns, float(raw["wheelbase_m"]), float(raw["wheel_radius_m"]))


@dataclass
class Trip:
    """One farmer's drive: where it comes in and leaves, and, per sample of its path, the body pose with the
    suspension's twist, the curvature its wheels steer to, the distance its wheels have rolled, and when it reaches and
    leaves the sample.

    Times are seconds from the start of the patrol; a farmer already on the road then has a start below zero.
    """

    arrive_by: str
    turn: Turn
    leave_by: str
    cruise_m_s: float
    stop_s: float
    start_s: float
    pose: np.ndarray
    curve: np.ndarray
    curve_out: np.ndarray
    rolled: np.ndarray
    speed: np.ndarray
    reach: np.ndarray
    leave: np.ndarray
    wheel_radius: float
    wheelbase: float

    @property
    def end_s(self) -> float:
        """When the truck leaves the map."""
        return self.start_s + float(self.leave[-1])

    def at(self, t: float) -> Optional[Tuple[List[float], float, float, float]]:
        """The body pose, the wheels' rolled angle, the front wheels' steer angle and the suspension's twist at time t,
        or None off the road."""
        if not self.start_s <= t <= self.end_s:
            return None
        pose, rolled, curve = self.states(np.array([t - self.start_s]))
        return (pose[0, :7].tolist(), float(rolled[0]) / self.wheel_radius, math.atan(self.wheelbase * float(curve[0])),
                float(pose[0, 7]))

    def states(self, taus: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """At each time since the trip began: the body pose with the twist after it, the distance the wheels have
        rolled and the curvature they steer to.

        Only element by element arithmetic and square roots, which round the same on every machine. Standing at a
        sample, the wheels swing to their next lock over the stop's last moments; between two samples the truck keeps
        the steady acceleration that takes it from one sample's speed to the next's.
        """
        taus = np.asarray(taus, dtype=float)
        last = len(self.reach) - 1
        k = np.searchsorted(self.reach, taus, side="right") - 1
        ahead = np.minimum(k + 1, last)
        standing = (taus <= self.leave[k]) | (k == last)
        span = self.rolled[ahead] - self.rolled[k]
        v0, v1 = self.speed[k], self.speed[ahead]
        moved = taus - self.leave[k]
        whole = np.where(standing, 1.0, self.reach[ahead] - self.leave[k])
        length = np.where(standing, 1.0, np.abs(span))
        share = np.where(v0 + v1 > 0.0, (v0 * moved + 0.5 * ((v1 - v0) / whole) * moved * moved) / length, moved / whole)
        share = np.where(standing, 0.0, np.clip(share, 0.0, 1.0))
        a, b = self.pose[k], self.pose[ahead]
        dot = a[:, 3] * b[:, 3] + a[:, 4] * b[:, 4] + a[:, 5] * b[:, 5] + a[:, 6] * b[:, 6]
        q = a[:, 3:7] + share[:, None] * (np.where(dot < 0.0, -1.0, 1.0)[:, None] * b[:, 3:7] - a[:, 3:7])
        norm = np.sqrt(q[:, 0] * q[:, 0] + q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3])
        pose = np.empty((len(taus), 8))
        pose[:, :3] = a[:, :3] + share[:, None] * (b[:, :3] - a[:, :3])
        pose[:, 3:7] = q / norm[:, None]
        pose[:, 7] = a[:, 7] + share * (b[:, 7] - a[:, 7])
        ramp = 1.0 - np.clip((self.leave[k] - taus) / CONFIG["steer_s"], 0.0, 1.0)
        curve = np.where(standing, self.curve[k] + ramp * (self.curve_out[k] - self.curve[k]),
                         self.curve_out[k] + share * (self.curve[ahead] - self.curve_out[k]))
        return pose, self.rolled[k] + share * span, curve


def _path(road: Road, arrive_by: str, turn: Turn, leave_by: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The rear axle samples of a trip in driving order: in along one arm to where it turns, the turn, out along the
    arm it leaves by.

    Returns the rear axle points, the body pose with the twist after it, the curvature the front wheels steer to, and
    the gear (+1 forward, -1 back) driven to reach each sample.
    """
    come, go = road.lines[arrive_by], road.lines[leave_by]
    arrive = come[:len(come) - turn.from_end + 1]
    # The turn starts on the arm's last sample in and ends on the first sample out, so each is kept once.
    depart = go[:len(go) - turn.to_end][::-1]
    axle = np.concatenate([arrive[:, :2], turn.rows[1:, :2], depart[:, :2]])
    pose = np.concatenate([arrive[:, AHEAD], turn.rows[1:, TURNED], depart[:, BACK]])
    # Facing back along an arm, the truck bends the other way from the line's own curvature.
    curve = np.concatenate([arrive[:, 2], turn.rows[1:, 3], -depart[:, 2]])
    gear = np.concatenate([np.ones(len(arrive)), turn.rows[1:, 2], np.ones(len(depart))])
    return axle, pose, curve, gear


def _trip(road: Road, arrive_by: str, turn: Turn, leave_by: str, cruise: float, stop_s: float) -> Trip:
    """A farmer's whole drive timed: cruising in, slowing for bends, stopping wherever the wheels must swing to a new
    lock or change gear, a while longer where it turns round if it stops there, and cruising out."""
    axle, pose, curve, gear = _path(road, arrive_by, turn, leave_by)
    # The axle points, not the body origin, set the distance between samples: the body swings past them in a turn.
    gap = axle[1:] - axle[:-1]
    step = np.sqrt(gap[:, 0] * gap[:, 0] + gap[:, 1] * gap[:, 1])
    top = np.where(gear > 0, cruise, CONFIG["reverse_m_s"])
    cap = np.minimum(top, np.sqrt(CONFIG["lateral_m_s2"] / np.maximum(np.abs(curve), 1e-9)))
    halt = np.zeros(len(pose), dtype=bool)
    halt[:-1] = (gear[1:] != gear[:-1]) | (np.abs(curve[1:] - curve[:-1]) > CONFIG["lock_jump"])
    there = len(road.lines[arrive_by]) - turn.from_end
    halt[there] = True
    cap[halt] = 0.0
    speed = cap.copy()
    for i in range(1, len(speed)):
        speed[i] = min(speed[i], math.sqrt(speed[i - 1] ** 2 + 2.0 * CONFIG["accel_m_s2"] * step[i - 1]))
    for i in range(len(speed) - 2, -1, -1):
        speed[i] = min(speed[i], math.sqrt(speed[i + 1] ** 2 + 2.0 * CONFIG["brake_m_s2"] * step[i]))
    dwell = np.where(halt, CONFIG["pause_s"], 0.0)
    dwell[there] = max(stop_s, CONFIG["pause_s"])
    both = speed[:-1] + speed[1:]
    # Two stops in a row: the truck creeps the short way between them, speeding up then braking.
    travel = np.where(both > 0.0, 2.0 * step / np.maximum(both, 1e-9), 2.0 * np.sqrt(step / CONFIG["accel_m_s2"]))
    leave = np.empty(len(pose))
    reach = np.empty(len(pose))
    reach[0], leave[0] = 0.0, dwell[0]
    for i in range(1, len(pose)):
        reach[i] = leave[i - 1] + travel[i - 1]
        leave[i] = reach[i] + dwell[i]
    curve_out = np.where(halt, np.r_[curve[1:], curve[-1]], curve)
    rolled = np.r_[0.0, np.cumsum(gear[1:] * step)]
    return Trip(arrive_by, turn, leave_by, cruise, stop_s, 0.0, pose, curve, curve_out, rolled, speed, reach, leave,
                road.wheel_radius, road.wheelbase)


def plan(seed: int, road: Road) -> List[Trip]:
    """The farmers of a seed, one after another on the road, from the one already driving when the patrol starts to
    the last one to come in before it ends, at most CONFIG["most"].

    Each farmer draws the end it comes from (mostly the one the farmer before did not), its way through (onto the other
    arm at the corner, round in the yard, or round at a place along its own arm), the end it leaves by, its cruising
    speed, and whether and how long it stops before turning; the road then stands empty for a drawn while before the
    next comes.
    """
    rng = random.Random((int(seed) ^ CONFIG["seed_offset"]) & 0xFFFFFFFF)
    trips: List[Trip] = []
    clock = None
    while len(trips) < CONFIG["most"] and (clock is None or clock < CONFIG["patrol_s"]):
        if not trips:
            arrive_by = rng.choice(ARMS)
        else:
            switch = rng.random() < CONFIG["switch_end"]
            arrive_by = [arm for arm in ARMS if (arm != trips[-1].arrive_by) == switch][0]
        corners = [turn for turn in road.turns if turn.corner and arrive_by in turn.into]
        yards = [turn for turn in road.turns if turn.yard]
        back = [turn for turn in road.turns if not turn.yard and not turn.corner and arrive_by in turn.into]
        draw = rng.random()
        if corners and draw < CONFIG["through_share"]:
            turn = rng.choice(corners)
        elif not back or draw < CONFIG["through_share"] + CONFIG["yard_share"]:
            turn = rng.choice(yards)
        else:
            turn = rng.choice(back)
        leave_by = rng.choice(turn.out)
        cruise = rng.uniform(*CONFIG["cruise_m_s"])
        stop_s = rng.uniform(*CONFIG["stop_s"]) if rng.random() < CONFIG["stop_share"] else 0.0
        trip = _trip(road, arrive_by, turn, leave_by, cruise, stop_s)
        trip.start_s = -rng.uniform(*CONFIG["first_done"]) * float(trip.leave[-1]) if clock is None else clock
        trips.append(trip)
        clock = trip.end_s + rng.uniform(*CONFIG["gap_s"])
    return trips


def where(trips: List[Trip], t: float) -> Optional[Tuple[List[float], float, float, float]]:
    """The truck's body pose, rolled wheel angle, steer angle and suspension twist at time t, or None while the road
    is empty."""
    for trip in trips:
        if trip.start_s <= t <= trip.end_s:
            return trip.at(t)
    return None


def timeline(trips: List[Trip], hz: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Every simulator step from the patrol's start to the last farmer leaving, worked out once: whether a truck is on
    the road, its body pose with the twist, its wheels' rolled angle and the curvature they steer to."""
    times = np.arange(int(math.ceil(max(trip.end_s for trip in trips) * hz)) + 1) / hz
    on = np.zeros(len(times), dtype=bool)
    pose, rolled, curve = np.zeros((len(times), 8)), np.zeros(len(times)), np.zeros(len(times))
    for trip in trips:
        hit = (times >= trip.start_s) & (times <= trip.end_s) & ~on
        pose[hit], distance, curve[hit] = trip.states(times[hit] - trip.start_s)
        rolled[hit] = distance / trip.wheel_radius
        on |= hit
    return on, pose, rolled, curve
