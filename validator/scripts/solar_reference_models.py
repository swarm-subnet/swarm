#!/usr/bin/env python3
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

"""Swarm Sentinel reference models: cheap strategies and a perfect one, each flown as a whole patrol.

A strategy that is cheap to build must not score well, and a perfect one must score close to 1. Each pilot here flies
one seed through the family's own runtime, the way a validator does: the seed checks approve it, the park is built,
the pilot decides ten times a second, and the last step's outcome is scored by the family's evaluate_rollout. The
pilots that search read the simulator's own object map of every frame the model is shown, so their eyes are perfect;
what they prove is what the rules accept, not how well a real detector would do.

The five reference models:

    silent    the full search, never reports
    panic     the full search, reports every person, animal and vehicle it sees, once each
    hover     takes off, hovers over the dock, returns home with a minute left
    random    uniform random actions
    perfect   the full search, reports each thief inside the fence once, zooming on any it sees too small

and the shortcuts tried against the rules:

    repeat       perfect, then reports every thief again on every frame he shows in
    high         perfect, flown 6 m higher, between 22 and 29 m above the dock
    scan         climbs to 60 m over the dock and zooms across the whole park before the search
    night_colour silent, searching dark nights in colour with night mode on instead of thermal
    zoom_all     perfect, spending all 80 zooms
    early_end    flies out to the flight limit's stop line before the thieves have stepped in
    bad_actions  perfect, with a broken action every seventh decision: NaN, infinity, wrong length, huge values
    big_box      perfect, with every report boxed over the whole frame

The search is the seed checks' reference route: the wide camera's lanes by day, and by night lanes narrowed to the
thermal camera's view, which is what an honest night search has to fly.

    python validator/scripts/solar_reference_models.py <seed> <pilot> <out.json>
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from shapely import contains_xy
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points

from swarm.challenge_families.solar_patrol import camera, decoys, reference_pilot, sensor_noise, theft, wind
from swarm.challenge_families.solar_patrol.contract import (
    ACTION_DIM,
    ACTION_HIGH,
    ACTION_INDEX,
    ACTION_LOW,
    MAX_CLIMB_MPS,
    MAX_DESCENT_MPS,
    MAX_HORIZONTAL_MPS,
    MAX_YAW_RATE_DEG_S,
    MAX_ZOOMS,
    PATROL_HEIGHT_M,
    THERMAL_SHAPE,
)
from swarm.challenge_families.solar_patrol.family import SolarPatrolChallengeFamily
from swarm.challenge_families.solar_patrol.reports import BODY_BITS
from swarm.constants import SIM_DT
from swarm.utils.env_factory import make_env_with_initial_obs

MIN_REPORT_PX = 12          # visible pixels before a thief is boxed: about what a person 6 px wide shows
NIGHT_MODE_ON = 0.5         # the middle third of the night mode value is on
NIGHT_MODE_AUTO = 1.0
HOME_MARGIN_S = 35.0        # time kept for the climb, the flight home and the landing, on top of the distance
HOME_SPEED_MPS = 4.0
HOVER_HOME_LEFT_S = 60.0
HIGH_OFFSET_M = 6.0         # perfect's heights, at most 19.5 m above the dock, raised into 22 to 29 m
SCAN_HEIGHT_M = 60.0        # above the dock
SCAN_GRID_M = 15.0          # ground between the points the scan zooms on, inside a 7x frame's footprint from there
SCAN_ZOOM_RESERVE = 10      # zooms kept for the search after the scan
VISIT_GIVE_UP_S = 45.0      # a thief perfect cannot get onto a frame in this long is left to the search
VISIT_STANDOFF_M = 10.0     # across the ground from the thief: the park's high ground stands up to 15 m over the dock
VISIT_INSET_M = 8.0         # the watching spot stays this far inside the fence; the stop line can lie 5 m inside it
AIM_ABOVE_GROUND_M = 0.9
VISIT_SPARE_S = 30.0        # a visit starts only with this much more time than the flight home needs
BAD_EVERY = 7
BAD_ACTIONS = (
    np.full(ACTION_DIM, np.nan, dtype=np.float32),
    np.full(ACTION_DIM, np.inf, dtype=np.float32),
    np.full(ACTION_DIM, -np.inf, dtype=np.float32),
    np.zeros(ACTION_DIM - 1, dtype=np.float32),
    np.ones(ACTION_DIM + 1, dtype=np.float32),
    np.zeros(0, dtype=np.float32),
)
WHOLE_FRAME = (0.5, 0.5, 1.0, 1.0)


def thermal_lane_spacing() -> float:
    """How far apart night lanes lie: the ground the thermal camera sees straight down from the patrol height, less
    the reference route's overlap."""
    height, width = THERMAL_SHAPE[:2]
    half = math.tan(math.radians(camera.THERMAL_DIAGONAL_FOV_DEG / 2.0)) * width / math.hypot(width, height)
    return 2.0 * PATROL_HEIGHT_M * half * (1.0 - reference_pilot.LANE_OVERLAP)


def tight_box(mask: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
    """The tight box around a mask's pixels as shares of the image: centre x, centre y, width, height."""
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return None
    height, width = mask.shape
    x0, x1, y0, y1 = cols.min(), cols.max() + 1, rows.min(), rows.max() + 1
    return ((x0 + x1) / 2 / width, (y0 + y1) / 2 / height, (x1 - x0) / width, (y1 - y0) / height)


def project(view: camera.View, point: np.ndarray) -> Optional[Tuple[float, float]]:
    """Where a world point falls on a view, as shares of its width and height from the top left; None when it lies
    behind the camera or outside the frame. The inverse of the zoom part's box ray."""
    forward, up = np.asarray(view.forward, dtype=float), np.asarray(view.up, dtype=float)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    offset = np.asarray(point, dtype=float) - np.asarray(view.eye, dtype=float)
    depth = float(offset @ forward)
    if depth <= 0.0:
        return None
    half_v = math.tan(math.radians(view.vertical_fov_deg) / 2.0)
    half_h = half_v * view.width / view.height
    cx = 0.5 + float(offset @ right) / depth / (2.0 * half_h)
    cy = 0.5 - float(offset @ up) / depth / (2.0 * half_v)
    return (cx, cy) if 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 else None


class Buttons:
    """Presses buttons so each press counts: a button held on two decisions in a row counts once, so a press is
    never made on the decision straight after the last one."""

    def __init__(self) -> None:
        """No button pressed yet."""
        self.last: Dict[str, int] = {}

    def ready(self, button: str, decision: int) -> bool:
        """Whether the button can be pressed on this decision."""
        return self.last.get(button, -2) < decision - 1

    def press(self, a: np.ndarray, decision: int, button: str, box: Tuple[float, ...], **extra: float) -> None:
        """Press a button with its box and any other values it reads."""
        self.last[button] = decision
        a[ACTION_INDEX[button]] = 1.0
        for name, value in zip(("cx", "cy", "w", "h"), box):
            a[ACTION_INDEX[f"{button}_{name}"]] = float(value)
        for name, value in extra.items():
            a[ACTION_INDEX[name]] = float(value)


class Eyes:
    """Perfect eyes: the object map of each frame the model is shown, read once per frame for thieves and decoys."""

    def __init__(self, ep: Any):
        """No frame read yet."""
        self.ep = ep
        self.seen_step = {"feed": -1, "zoom": -1}

    def fresh(self, image: str) -> Optional[camera.View]:
        """The feed or zoom frame shown now, the first time it is asked for; None otherwise."""
        view = sensor_noise.shown_view(self.ep, image)
        if view is None or view.objects is None or view.step == self.seen_step[image]:
            return None
        self.seen_step[image] = view.step
        return view

    def objects(self, view: camera.View, inside_only: bool) -> Dict[tuple, np.ndarray]:
        """Each thief (inside the fence at the frame's step when inside_only) and, when not, each decoy the frame
        shows, with the mask of its pixels."""
        ids = np.where(view.objects >= 0, view.objects & BODY_BITS, -1)
        owners: Dict[int, tuple] = {uid: ("thief", n) for uid, n in theft.bodies(self.ep).items()}
        if not inside_only:
            owners.update({uid: ("decoy", uid) for uid in decoys.bodies(self.ep)})
        shown: Dict[tuple, np.ndarray] = {}
        for key in dict.fromkeys(owners.values()):
            if inside_only and not theft.inside(self.ep, key[1], view.step):
                continue
            mask = np.isin(ids, [uid for uid, owner in owners.items() if owner == key])
            if mask.any():
                shown[key] = mask
        return shown


class Patrol:
    """The full search: the reference route at its heights, the feed for the light, and return home pressed in time
    to land before the clock runs out."""

    def __init__(self, env: Any, *, thermal_at_night: bool = True, night_mode: float = NIGHT_MODE_AUTO,
                 height_offset_m: float = 0.0):
        """Lay the route for this seed: thermal lanes by night unless told to search in colour."""
        self.env, self.ep = env, env._solar
        self.thermal = camera.night(env) and thermal_at_night
        spacing = thermal_lane_spacing() if self.thermal else reference_pilot.lane_spacing()
        self.pilot = reference_pilot._Pilot(env, self.ep, spacing)
        self.pilot.heights = self.pilot.heights + height_offset_m
        self.height_offset_m = height_offset_m
        self.night_mode = night_mode

    def action(self, decision: int) -> np.ndarray:
        """This decision's flight, with the feed set and return home pressed once the time left only just covers
        the flight home."""
        a = self.pilot.action(decision)
        a[ACTION_INDEX["thermal"]] = 1.0 if self.thermal else 0.0
        a[ACTION_INDEX["night_mode"]] = self.night_mode
        if self.ep.phase == "flying" and self.time_left_s() < self.time_home_s():
            a[ACTION_INDEX["return_home"]] = 1.0
        return a

    def time_left_s(self) -> float:
        """Seconds of patrol left."""
        return float(self.env.EP_LEN_SEC) - self.ep.time_s

    def time_home_s(self) -> float:
        """Seconds the flight home and the landing need from here."""
        offset = np.asarray(self.env.pos[0], dtype=float)[:2] - self.ep.dock_position[:2]
        return float(np.hypot(*offset)) / HOME_SPEED_MPS + HOME_MARGIN_S


class Silent:
    """The full search, never a report."""

    def __init__(self, env: Any):
        """The search's route."""
        self.patrol = Patrol(env)

    def action(self, decision: int, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """The search's flight alone."""
        return self.patrol.action(decision)


class Perfect:
    """The full search with perfect eyes and perfect knowledge: it flies over each thief inside the fence as soon as
    he steps in, reports him once on the first frame he fills enough pixels of, zooming with the 7x lens when he
    shows too small to box, and then searches the park."""

    box_override: Optional[Tuple[float, float, float, float]] = None

    def __init__(self, env: Any, **patrol: Any):
        """The search's route, nothing reported, zoomed or visited yet."""
        self.env, self.ep = env, env._solar
        self.patrol = Patrol(env, **patrol)
        self.eyes = Eyes(self.ep)
        self.buttons = Buttons()
        self.reported: set = set()
        self.zoomed: set = set()
        self.visit_started: Dict[int, float] = {}
        self.watch_area = Polygon(self.ep.fence).buffer(-VISIT_INSET_M)

    def action(self, decision: int, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """The search's flight, or a visit to a thief not reported yet, and at most one report or zoom."""
        a = self.patrol.action(decision)
        if a[ACTION_INDEX["return_home"]] < 0.5:
            self.visit(a)
        self.look(a, decision)
        return a

    def visit(self, a: np.ndarray) -> None:
        """Steer to VISIT_STANDOFF_M beside the first thief inside the fence not reported yet, at the search's height,
        facing him with the camera tilted onto him, while there is time and he has not been chased longer than
        VISIT_GIVE_UP_S."""
        ep = self.ep
        if ep.phase != "flying" or self.patrol.time_left_s() < self.patrol.time_home_s() + VISIT_SPARE_S:
            return
        index = theft.bodies(ep)
        for person in theft.people(ep):
            n = index.get(int(person["bodies"][0]), -1) if person["bodies"] else -1
            if not person["threat"] or n < 0 or n in self.reported:
                continue
            started = self.visit_started.setdefault(n, ep.time_s)
            if ep.time_s - started > VISIT_GIVE_UP_S:
                continue
            thief = np.asarray(person["xy"], dtype=float)
            pos = np.asarray(self.env.pos[0], dtype=float)
            away = pos[:2] - thief
            away = away / np.hypot(*away) if np.hypot(*away) > 1e-3 else np.array([1.0, 0.0])
            stand = thief + away * VISIT_STANDOFF_M
            if not self.watch_area.covers(Point(stand)):
                stand = np.asarray(nearest_points(self.watch_area, Point(stand))[0].coords[0], dtype=float)
            dock_z = float(ep.dock_position[2])
            ground = reference_pilot._ground_z(self.env.CLIENT, *stand, ep.terrain_uids, dock_z)
            target_z = min(dock_z + PATROL_HEIGHT_M - reference_pilot.CEILING_MARGIN_M, ground + PATROL_HEIGHT_M)
            _steer(a, self.env, np.array([*stand, target_z + self.patrol.height_offset_m]))
            head = reference_pilot._ground_z(self.env.CLIENT, *thief, ep.terrain_uids, dock_z) + AIM_ABOVE_GROUND_M
            to_thief = thief - pos[:2]
            yaw = float(self.env.rpy[0, 2])
            error = (math.atan2(to_thief[1], to_thief[0]) - yaw + math.pi) % (2.0 * math.pi) - math.pi
            a[ACTION_INDEX["turn"]] = float(np.clip(-math.degrees(2.0 * error) / MAX_YAW_RATE_DEG_S, -1.0, 1.0))
            tilt = math.degrees(math.atan2(head - pos[2], max(float(np.hypot(*to_thief)), 1e-3)))
            a[ACTION_INDEX["gimbal_tilt"]] = float(np.clip(tilt / 90.0, -1.0, 1.0))
            return

    def look(self, a: np.ndarray, decision: int) -> None:
        """Report a thief on a fresh zoom or feed frame, or zoom on one too small to box."""
        for image in ("zoom", "feed"):
            view = self.eyes.fresh(image)
            if view is None:
                continue
            for (_, n), mask in self.eyes.objects(view, inside_only=True).items():
                if n in self.reported:
                    continue
                if mask.sum() >= MIN_REPORT_PX and self.buttons.ready("report", decision):
                    self.report(a, decision, n, image, mask)
                    return
                if (image == "feed" and n not in self.zoomed and self.ep.outcome.zooms_used < MAX_ZOOMS
                        and self.buttons.ready("zoom", decision)):
                    self.zoomed.add(n)
                    self.buttons.press(a, decision, "zoom", tight_box(mask), zoom_lens=1.0)
                    return

    def report(self, a: np.ndarray, decision: int, n: int, image: str, mask: np.ndarray) -> None:
        """Report thief n with the box around his pixels on the image he showed in."""
        self.reported.add(n)
        box = self.box_override or tight_box(mask)
        self.buttons.press(a, decision, "report", box, report_class=0.0, report_image=float(image == "zoom"))


class Panic:
    """The full search, reporting every person, animal and vehicle it sees, once each, inside the fence or not."""

    def __init__(self, env: Any):
        """The search's route, nothing reported yet."""
        self.ep = env._solar
        self.patrol = Patrol(env)
        self.eyes = Eyes(self.ep)
        self.buttons = Buttons()
        self.reported: set = set()

    def action(self, decision: int, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """The search's flight, and a report on anything not reported yet."""
        a = self.patrol.action(decision)
        view = self.eyes.fresh("feed")
        if view is not None and self.buttons.ready("report", decision):
            for key, mask in self.eyes.objects(view, inside_only=False).items():
                if key not in self.reported and mask.sum() >= MIN_REPORT_PX:
                    self.reported.add(key)
                    self.buttons.press(a, decision, "report", tight_box(mask), report_class=0.0, report_image=0.0)
                    break
        return a


class Hover:
    """Takes off, hovers over the dock looking down, and presses return home with a minute left."""

    def __init__(self, env: Any):
        """Nothing to lay out."""
        self.env, self.ep = env, env._solar
        self.night = camera.night(env)

    def action(self, decision: int, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Take-off first, then still sticks until it is time to go home."""
        a = np.zeros(ACTION_DIM, dtype=np.float32)
        a[ACTION_INDEX["gimbal_tilt"]] = -1.0
        a[ACTION_INDEX["thermal"]] = float(self.night)
        a[ACTION_INDEX["take_off"]] = float(decision == 0)
        left = float(self.env.EP_LEN_SEC) - self.ep.time_s
        a[ACTION_INDEX["return_home"]] = float(self.ep.phase == "flying" and left < HOVER_HOME_LEFT_S)
        return a


class RandomPilot:
    """Uniform random actions inside the contract's bounds, from a generator fixed by the seed."""

    def __init__(self, env: Any):
        """A generator the seed fixes, so a run repeats."""
        self.rng = np.random.default_rng(int(env._solar.seed))

    def action(self, decision: int, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """One random action."""
        return self.rng.uniform(ACTION_LOW, ACTION_HIGH).astype(np.float32)


class Repeat(Perfect):
    """Perfect, then each thief already reported is reported again on every fresh frame he fills enough pixels of."""

    def look(self, a: np.ndarray, decision: int) -> None:
        """Perfect's look, or else a repeat report of a thief reported before."""
        before = a.copy()
        super().look(a, decision)
        if not np.array_equal(before, a) or not self.buttons.ready("report", decision):
            return
        view = sensor_noise.shown_view(self.ep, "feed")
        if view is None or view.objects is None or view.step == getattr(self, "_repeated_step", -1):
            return
        for (_, n), mask in self.eyes.objects(view, inside_only=True).items():
            if n in self.reported and mask.sum() >= MIN_REPORT_PX:
                self._repeated_step = view.step
                self.buttons.press(a, decision, "report", tight_box(mask), report_class=0.0, report_image=0.0)
                return


class High(Perfect):
    """Perfect flown HIGH_OFFSET_M higher, between 22 and 29 m above the dock, for a wider view."""

    def __init__(self, env: Any):
        """Perfect's route, raised."""
        super().__init__(env, height_offset_m=HIGH_OFFSET_M)


class Scan(Perfect):
    """Climbs to SCAN_HEIGHT_M over the dock, faces the park and zooms the 7x lens across all of it, reporting every
    thief a zoom frame shows, then comes down and flies perfect's search."""

    def __init__(self, env: Any):
        """Perfect's route, and the ground points of the park the scan zooms on."""
        super().__init__(env)
        fence = np.asarray(self.ep.fence, dtype=float)
        low, high = fence.min(axis=0), fence.max(axis=0)
        xs, ys = np.meshgrid(np.arange(low[0], high[0], SCAN_GRID_M) + SCAN_GRID_M / 2.0,
                             np.arange(low[1], high[1], SCAN_GRID_M) + SCAN_GRID_M / 2.0)
        points = np.column_stack([xs.ravel(), ys.ravel()])
        inside = contains_xy(Polygon(fence), points[:, 0], points[:, 1])
        ground = float(self.ep.dock_position[2])
        self.targets = [np.array([x, y, reference_pilot._ground_z(env.CLIENT, x, y, self.ep.terrain_uids, ground)])
                        for x, y in points[inside]]
        self.centre = np.append(fence.mean(axis=0), ground)
        self.scanning = True
        self.scan_reports = 0
        self.scan_end_s: Optional[float] = None

    def action(self, decision: int, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Perfect's take-off, then the climb and the scan while targets are left, then perfect's search."""
        if not self.scanning or self.ep.phase == "docked":
            return super().action(decision, obs)
        if self.ep.phase != "flying":
            return self._still()
        a = self._climb_and_face()
        before = len(self.reported)
        view = self.eyes.fresh("zoom")
        if view is not None:
            for (_, n), mask in self.eyes.objects(view, inside_only=True).items():
                if n not in self.reported and mask.sum() >= MIN_REPORT_PX and self.buttons.ready("report", decision):
                    self.report(a, decision, n, "zoom", mask)
                    self.scan_reports += len(self.reported) - before
                    return a
        feed = sensor_noise.shown_view(self.ep, "feed")
        height = float(self.env.pos[0][2] - self.ep.dock_position[2])
        if feed is None or height < SCAN_HEIGHT_M - 1.0 or not self.buttons.ready("zoom", decision):
            return a
        while self.targets and self.ep.outcome.zooms_used < MAX_ZOOMS - SCAN_ZOOM_RESERVE:
            spot = project(feed, self.targets.pop(0))
            if spot is not None:
                self.buttons.press(a, decision, "zoom", (spot[0], spot[1], 0.1, 0.1), zoom_lens=1.0)
                return a
        self.scanning, self.scan_end_s = False, self.ep.time_s
        return a

    def _still(self) -> np.ndarray:
        """No stick while the dock flies the drone."""
        a = np.zeros(ACTION_DIM, dtype=np.float32)
        a[ACTION_INDEX["gimbal_tilt"]] = -1.0
        return a

    def _climb_and_face(self) -> np.ndarray:
        """Climb straight up over the dock, turn to face the park's middle and tilt the camera onto it."""
        a = self._still()
        pos = np.asarray(self.env.pos[0], dtype=float)
        rise = float(self.ep.dock_position[2]) + SCAN_HEIGHT_M - pos[2]
        vz = float(np.clip(rise, -MAX_DESCENT_MPS, MAX_CLIMB_MPS))
        a[ACTION_INDEX["move_up"]] = vz / (MAX_CLIMB_MPS if vz >= 0.0 else MAX_DESCENT_MPS)
        drift = self.ep.dock_position[:2] - pos[:2]
        yaw = float(self.env.rpy[0, 2])
        c, s = math.cos(yaw), math.sin(yaw)
        a[ACTION_INDEX["move_forward"]] = float(np.clip((drift[0] * c + drift[1] * s) / MAX_HORIZONTAL_MPS, -1, 1))
        a[ACTION_INDEX["move_right"]] = float(np.clip((drift[0] * s - drift[1] * c) / MAX_HORIZONTAL_MPS, -1, 1))
        to_centre = self.centre - pos
        bearing = math.atan2(to_centre[1], to_centre[0])
        error = (bearing - yaw + math.pi) % (2.0 * math.pi) - math.pi
        a[ACTION_INDEX["turn"]] = float(np.clip(-math.degrees(2.0 * error) / MAX_YAW_RATE_DEG_S, -1.0, 1.0))
        tilt = -math.degrees(math.atan2(-to_centre[2], float(np.hypot(*to_centre[:2]))))
        a[ACTION_INDEX["gimbal_tilt"]] = float(np.clip(tilt / 90.0, -1.0, 1.0))
        a[ACTION_INDEX["night_mode"]] = NIGHT_MODE_AUTO
        a[ACTION_INDEX["night_vision"]] = 1.0
        return a


class NightColour(Silent):
    """Silent, searching dark nights in the wide colour camera with night mode on, on the wide camera's lanes."""

    def __init__(self, env: Any):
        """The wide camera's route and feed, day and night."""
        self.patrol = Patrol(env, thermal_at_night=False, night_mode=NIGHT_MODE_ON)


class ZoomAll(Perfect):
    """Perfect, pressing the zoom on the middle of the feed whenever it has nothing else to press, until all 80 are
    spent."""

    def look(self, a: np.ndarray, decision: int) -> None:
        """Perfect's look, or else one more zoom."""
        before = a.copy()
        super().look(a, decision)
        if (np.array_equal(before, a) and self.ep.phase == "flying" and self.ep.outcome.zooms_used < MAX_ZOOMS
                and self.buttons.ready("zoom", decision)):
            self.buttons.press(a, decision, "zoom", (0.5, 0.5, 0.1, 0.1), zoom_lens=1.0)


class EarlyEnd:
    """Takes off, then flies flat out at the nearest side of the flight limit, so its stop line ends the patrol
    before every thief has stepped inside the fence."""

    def __init__(self, env: Any):
        """Pick the nearest point of the limit to run at."""
        self.env, self.ep = env, env._solar
        polygon = np.asarray(self.ep.flight_limit["polygon"], dtype=float)
        dock = self.ep.dock_position[:2]
        sides = [_closest_on_segment(dock, a, b) for a, b in zip(polygon, np.roll(polygon, -1, axis=0))]
        self.target = min(sides, key=lambda side: side[0])[1]

    def action(self, decision: int, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Take-off, then full speed at the limit."""
        a = np.zeros(ACTION_DIM, dtype=np.float32)
        a[ACTION_INDEX["take_off"]] = float(decision == 0)
        if self.ep.phase != "flying":
            return a
        heading = self.target - np.asarray(self.env.pos[0], dtype=float)[:2]
        yaw = float(self.env.rpy[0, 2])
        c, s = math.cos(yaw), math.sin(yaw)
        direction = heading / max(float(np.hypot(*heading)), 1e-6)
        a[ACTION_INDEX["move_forward"]] = float(direction[0] * c + direction[1] * s)
        a[ACTION_INDEX["move_right"]] = float(direction[0] * s - direction[1] * c)
        return a


class BadActions(Perfect):
    """Perfect, sending a broken action every BAD_EVERY decisions instead of its own, on decisions that press no
    button, so what is measured is the broken action and not a lost press."""

    def action(self, decision: int, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Perfect's action, or a broken one."""
        a = super().action(decision, obs)
        pressed = any(a[ACTION_INDEX[button]] > 0.5 for button in ("zoom", "report", "take_off", "return_home"))
        if decision and decision % BAD_EVERY == 0 and not pressed:
            return BAD_ACTIONS[(decision // BAD_EVERY) % len(BAD_ACTIONS)]
        return a


class BigBox(Perfect):
    """Perfect, with every report boxed over the whole frame."""

    box_override = WHOLE_FRAME


def _steer(a: np.ndarray, env: Any, target: np.ndarray) -> None:
    """Set the sticks to fly towards a world point at up to the top speed, facing the way it goes, camera down."""
    pos = np.asarray(env.pos[0], dtype=float)
    heading = target[:2] - pos[:2]
    distance = float(np.hypot(*heading))
    speed = min(MAX_HORIZONTAL_MPS, distance)
    vx, vy = (heading / distance * speed) if distance > 1e-6 else (0.0, 0.0)
    vz = float(np.clip(reference_pilot.HEIGHT_GAIN_PER_S * (target[2] - pos[2]), -MAX_DESCENT_MPS, MAX_CLIMB_MPS))
    yaw = float(env.rpy[0, 2])
    c, s = math.cos(yaw), math.sin(yaw)
    a[ACTION_INDEX["move_forward"]] = (vx * c + vy * s) / MAX_HORIZONTAL_MPS
    a[ACTION_INDEX["move_right"]] = (vx * s - vy * c) / MAX_HORIZONTAL_MPS
    a[ACTION_INDEX["move_up"]] = vz / (MAX_CLIMB_MPS if vz >= 0.0 else MAX_DESCENT_MPS)
    a[ACTION_INDEX["gimbal_tilt"]] = -1.0
    error = (math.atan2(vy, vx) - yaw + math.pi) % (2.0 * math.pi) - math.pi if speed > 1.0 else 0.0
    a[ACTION_INDEX["turn"]] = float(np.clip(-math.degrees(2.0 * error) / MAX_YAW_RATE_DEG_S, -1.0, 1.0))


def _closest_on_segment(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> Tuple[float, np.ndarray]:
    """The distance from a point to a segment, and the segment's nearest point to it."""
    ab = b - a
    t = float(np.clip((point - a) @ ab / max(float(ab @ ab), 1e-12), 0.0, 1.0))
    nearest = a + t * ab
    return float(np.hypot(*(point - nearest))), nearest


PILOTS = {
    "silent": Silent, "panic": Panic, "hover": Hover, "random": RandomPilot, "perfect": Perfect,
    "repeat": Repeat, "high": High, "scan": Scan, "night_colour": NightColour, "zoom_all": ZoomAll,
    "early_end": EarlyEnd, "bad_actions": BadActions, "big_box": BigBox,
}


def _digest(state: Any, obs: Dict[str, np.ndarray], last: Dict[str, np.ndarray]) -> None:
    """Fold one decision's observation into the running hash: the vectors every decision, an image only when it is
    a new one."""
    for key in sorted(obs):
        value = obs[key]
        if key in ("rgb", "thermal", "zoom") and last.get(key) is value:
            continue
        last[key] = value
        state.update(key.encode())
        state.update(np.ascontiguousarray(value).tobytes())


def fly(seed: int, pilot_name: str) -> Dict[str, Any]:
    """Approve, build, fly and score one seed with one pilot, and every number the tables need."""
    family = SolarPatrolChallengeFamily()
    started = time.perf_counter()
    task = family.build_random_task(sim_dt=SIM_DT, seed=seed)
    approve_s = time.perf_counter() - started
    flown = int(task.map_seed)
    with contextlib.redirect_stdout(io.StringIO()):
        env, obs = make_env_with_initial_obs(task)
    build_s = time.perf_counter() - started - approve_s
    digest, last = hashlib.sha256(), {}
    brightness: List[float] = []
    try:
        pilot = PILOTS[pilot_name](env)
        decision, fly_started = 0, time.perf_counter()
        while True:
            _digest(digest, obs, last)
            if camera.night(env) and env._solar.camera["thermal"] is False and decision % 5 == 0:
                brightness.append(float(np.mean(obs["rgb"])))
            action = pilot.action(decision, obs)
            obs, _reward, terminated, truncated, info = env.step(np.asarray(action)[None, :])
            decision += 1
            if terminated or truncated:
                break
        fly_s = time.perf_counter() - fly_started
        ep = env._solar
        t_sim = float(info.get("t", env._time_alive))
        evaluation = family.evaluate_rollout(task=task, success=bool(info.get("success", False)), t=t_sim,
                                             horizon=float(task.horizon), min_clearance=info.get("min_clearance"),
                                             collision=bool(info.get("collision", False)),
                                             failure_reason=str(info.get("failure_reason", "NONE")), info=info)
        verdicts: Dict[str, int] = {}
        for _step, _report, verdict in ep.reports["log"]:
            verdicts[verdict] = verdicts.get(verdict, 0) + 1
        return {
            "seed": seed, "flown_seed": flown, "pilot": pilot_name,
            "night": bool(wind.is_night(flown)), "wind": wind.strength(flown), "theft": bool(theft.has_theft(flown)),
            "score": float(evaluation.score),
            "terms": {k: float(v) for k, v in evaluation.normalized_metrics.items()},
            "outcome": {k: v for k, v in evaluation.metrics.items() if k in ep.outcome.__dataclass_fields__},
            "time_s": round(t_sim, 2), "decisions": decision, "verdicts": verdicts,
            "scan_reports": getattr(pilot, "scan_reports", None), "scan_end_s": getattr(pilot, "scan_end_s", None),
            "night_rgb_mean": round(float(np.mean(brightness)), 4) if brightness else None,
            "digest": digest.hexdigest(),
            "pilot_code": hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:12],
            "approve_s": round(approve_s, 1), "build_s": round(build_s, 1), "fly_s": round(fly_s, 1),
        }
    finally:
        env.close()


def main(argv: List[str]) -> None:
    """Fly the seed with the pilot named and write the result as JSON."""
    seed, pilot_name, out = int(argv[0]), argv[1], argv[2]
    result = fly(seed, pilot_name)
    with open(out, "w") as handle:
        json.dump(result, handle, indent=1)
    print(json.dumps({k: result[k] for k in ("seed", "flown_seed", "pilot", "score", "time_s")}))


if __name__ == "__main__":
    main(sys.argv[1:])
