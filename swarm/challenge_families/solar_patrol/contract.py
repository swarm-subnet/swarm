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

"""The Solar Patrol contract: what the model receives, what it sends, and what one seed's result carries.

Every part of the family reads these definitions by name, so a part can be rebuilt without another one moving.
The registry entry for the family mirrors the shapes and bounds declared here, and a test holds the two together.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from swarm.constants import SIM_DT

FAMILY_ID = "cf_solar_patrol"
CHALLENGE_TYPE = 8
INTERFACE_VERSION = "submission_zip.v1"

DECISION_HZ = 10
# Physics and the flight controller keep the shared 50 Hz; one decision holds for this many of their steps.
DECISION_STEPS = int(round(1.0 / (DECISION_HZ * SIM_DT)))
PATROL_HEIGHT_M = 20.0
HEIGHT_LIMIT_M = 30.0                # the dock's height limit over the take-off point: the aircraft stops rising there
TIME_BUDGET_S = {"take_off": 40.0, "sweep": 248.0, "zoom_stops": 62.0, "landing": 40.0}
HORIZON_S = sum(TIME_BUDGET_S.values())

# The model's reach on the real aircraft, applied by the outputs part.
MAX_HORIZONTAL_MPS = 5.0
MAX_CLIMB_MPS = 3.0
MAX_DESCENT_MPS = 2.0
MAX_YAW_RATE_DEG_S = 90.0
GIMBAL_TILT_RANGE_DEG = (-90.0, 90.0)
MAX_GIMBAL_RATE_DEG_S = 100.0
MAX_ZOOMS = 80

# The model sees one colour feed or one thermal feed at a time, plus the zoom view it asked for.
RGB_SHAPE = (480, 640, 3)
THERMAL_SHAPE = (512, 640, 1)
ZOOM_SHAPE = (480, 640, 3)

BUTTON_THRESHOLD = 0.5
NIGHT_MODES = ("off", "on", "auto")
ZOOM_LENSES = (3, 7)
REPORT_CLASSES = ("person", "vehicle")
REPORT_IMAGES = ("feed", "zoom")
FLIGHT_PHASES = ("docked", "taking_off", "flying", "returning", "landing", "landed")
END_REASONS = ("landed", "timeout", "collision", "flight_limit")
LASER_STATUSES = ("normal", "too_close", "too_far", "no_signal")


@dataclass(frozen=True)
class ActionField:
    """One value of the action vector: its name and the bounds the validator clips it to."""

    name: str
    low: float
    high: float


ACTION_FIELDS = (
    ActionField("move_forward", -1.0, 1.0),      # share of MAX_HORIZONTAL_MPS along the heading
    ActionField("move_right", -1.0, 1.0),        # share of MAX_HORIZONTAL_MPS to the right of it
    ActionField("move_up", -1.0, 1.0),           # up to MAX_CLIMB_MPS above zero, MAX_DESCENT_MPS below
    ActionField("turn", -1.0, 1.0),              # share of MAX_YAW_RATE_DEG_S, clockwise positive
    ActionField("gimbal_tilt", -1.0, 1.0),       # target tilt: -1 straight down, +1 straight up
    ActionField("thermal", 0.0, 1.0),            # above the threshold the model sees the thermal feed
    ActionField("night_mode", 0.0, 1.0),         # thirds of the range pick off, on and auto
    ActionField("night_vision", 0.0, 1.0),       # above the threshold: on, at 7x zoom only
    ActionField("zoom", 0.0, 1.0),               # button
    ActionField("zoom_lens", 0.0, 1.0),          # below the threshold 3x, above it 7x
    ActionField("zoom_cx", 0.0, 1.0),            # box on the current feed frame, shares of its width and height
    ActionField("zoom_cy", 0.0, 1.0),
    ActionField("zoom_w", 0.0, 1.0),
    ActionField("zoom_h", 0.0, 1.0),
    ActionField("report", 0.0, 1.0),             # button
    ActionField("report_class", 0.0, 1.0),       # below the threshold a person, above it a vehicle
    ActionField("report_image", 0.0, 1.0),       # below the threshold the feed frame, above it the zoom view
    ActionField("report_cx", 0.0, 1.0),          # box on that image, shares of its width and height
    ActionField("report_cy", 0.0, 1.0),
    ActionField("report_w", 0.0, 1.0),
    ActionField("report_h", 0.0, 1.0),
    ActionField("take_off", 0.0, 1.0),           # button
    ActionField("return_home", 0.0, 1.0),        # button
    ActionField("cancel_return", 0.0, 1.0),      # button
)
ACTION_DIM = len(ACTION_FIELDS)
ACTION_INDEX = {f.name: i for i, f in enumerate(ACTION_FIELDS)}
ACTION_LOW = tuple(f.low for f in ACTION_FIELDS)
ACTION_HIGH = tuple(f.high for f in ACTION_FIELDS)
BUTTONS = ("zoom", "report", "take_off", "return_home", "cancel_return")

# Units are the real aircraft's: metres east, north and up from the dock, degrees, seconds, percent.
STATE_FIELDS = (
    ("position_m", 3),
    ("heading_deg", 1),                 # compass heading, 0 north, clockwise positive
    ("velocity_mps", 3),
    ("height_above_takeoff_m", 1),
    ("gimbal_tilt_deg", 1),
    ("time_left_s", 1),
    ("battery_pct", 1),
    ("wind_estimate_mps", 1),
    ("wind_estimate_sector", 1),        # 0 to 7, clockwise from north
    ("dock_wind_mps", 1),
    ("laser_range_m", 1),
    ("laser_point_m", 3),
    ("laser_status", 1),                # index into LASER_STATUSES
    ("ground_distance_m", 1),
    ("downward_sensing_ok", 1),
    ("flight_limit_distance_m", 1),
    ("inside_flight_limit", 1),
    ("flight_phase", 1),                # index into FLIGHT_PHASES
    ("camera_feed", 1),                 # 0 colour, 1 thermal
    ("frame_age_s", 1),
    ("night_mode", 1),                  # index into NIGHT_MODES
    ("night_vision", 1),
    ("zoom_lens", 1),                   # 0 before any zoom, else 3 or 7
    ("zoom_age_s", 1),
    ("zooms_left", 1),
)

MAX_FENCE_POINTS = 64
MAX_LIMIT_POINTS = 64
MAX_TABLES = 64
MAX_BUILDINGS = 16
# Sent on the first observation of a patrol and zero after it: the survey, in metres from this seed's dock.
SITE_MAP_FIELDS = (
    ("fence_count", 1),
    ("fence_xy", 2 * MAX_FENCE_POINTS),
    ("limit_count", 1),
    ("limit_xy", 2 * MAX_LIMIT_POINTS),
    ("table_count", 1),
    ("tables", 5 * MAX_TABLES),         # centre east, centre north, length, width, heading in degrees
    ("building_count", 1),
    ("buildings", 5 * MAX_BUILDINGS),   # the same five numbers per building
)


def _slices(fields: tuple[tuple[str, int], ...]) -> dict[str, slice]:
    """Each field's place in the flat vector, in declaration order."""
    slices, start = {}, 0
    for name, width in fields:
        slices[name] = slice(start, start + width)
        start += width
    return slices


STATE_SLICES = _slices(STATE_FIELDS)
STATE_DIM = sum(width for _, width in STATE_FIELDS)
SITE_MAP_SLICES = _slices(SITE_MAP_FIELDS)
SITE_MAP_DIM = sum(width for _, width in SITE_MAP_FIELDS)

OBSERVATION_SHAPES = {
    "rgb": RGB_SHAPE,
    "thermal": THERMAL_SHAPE,
    "zoom": ZOOM_SHAPE,
    "state": (STATE_DIM,),
    "site_map": (SITE_MAP_DIM,),
}


def new_state() -> np.ndarray:
    """A zeroed state vector, filled field by field by the parts that own them."""
    return np.zeros(STATE_DIM, dtype=np.float32)


def new_site_map() -> np.ndarray:
    """A zeroed site map vector."""
    return np.zeros(SITE_MAP_DIM, dtype=np.float32)


def put(vector: np.ndarray, slices: dict[str, slice], name: str, value) -> None:
    """Write one named field into a state or site map vector."""
    vector[slices[name]] = value


@dataclass(frozen=True)
class Box:
    """A box on an image, as shares of its width and height: centre, width and height."""

    cx: float
    cy: float
    w: float
    h: float


@dataclass(frozen=True)
class ZoomRequest:
    """A press of the zoom button: which lens, and the box on the current feed frame it centres on."""

    lens: int
    box: Box


@dataclass(frozen=True)
class Report:
    """A press of the report button: the class, the image the box was drawn on, and the box."""

    kind: str
    image: str
    box: Box


@dataclass(frozen=True)
class Command:
    """One decoded action. Buttons are set only on the step the value rises through the threshold."""

    move_forward: float
    move_right: float
    move_up: float
    turn: float
    gimbal_tilt: float
    thermal: bool
    night_mode: str
    night_vision: bool
    zoom: Optional[ZoomRequest]
    report: Optional[Report]
    take_off: bool
    return_home: bool
    cancel_return: bool


def _pressed(action: np.ndarray, previous: Optional[np.ndarray], name: str) -> bool:
    """True on the step a button's value rises through the threshold, so holding it counts once."""
    i = ACTION_INDEX[name]
    was = previous is not None and float(previous[i]) > BUTTON_THRESHOLD
    return float(action[i]) > BUTTON_THRESHOLD and not was


def decode_action(action: np.ndarray, previous: Optional[np.ndarray]) -> Command:
    """Read a clipped action vector into a Command, with button presses judged against the previous action."""
    a = np.asarray(action, dtype=np.float32).reshape(-1)
    v = {f.name: float(a[i]) for i, f in enumerate(ACTION_FIELDS)}

    def box(prefix: str) -> Box:
        """The box whose four values start with prefix."""
        return Box(v[f"{prefix}_cx"], v[f"{prefix}_cy"], v[f"{prefix}_w"], v[f"{prefix}_h"])

    zoom = None
    if _pressed(a, previous, "zoom"):
        zoom = ZoomRequest(ZOOM_LENSES[int(v["zoom_lens"] > BUTTON_THRESHOLD)], box("zoom"))
    report = None
    if _pressed(a, previous, "report"):
        report = Report(REPORT_CLASSES[int(v["report_class"] > BUTTON_THRESHOLD)],
                        REPORT_IMAGES[int(v["report_image"] > BUTTON_THRESHOLD)], box("report"))
    return Command(
        move_forward=v["move_forward"],
        move_right=v["move_right"],
        move_up=v["move_up"],
        turn=v["turn"],
        gimbal_tilt=v["gimbal_tilt"],
        thermal=v["thermal"] > BUTTON_THRESHOLD,
        night_mode=NIGHT_MODES[min(int(v["night_mode"] * len(NIGHT_MODES)), len(NIGHT_MODES) - 1)],
        night_vision=v["night_vision"] > BUTTON_THRESHOLD,
        zoom=zoom,
        report=report,
        take_off=_pressed(a, previous, "take_off"),
        return_home=_pressed(a, previous, "return_home"),
        cancel_return=_pressed(a, previous, "cancel_return"),
    )


@dataclass
class Outcome:
    """What one patrol has earned so far, carried in the step info and read by the scorer at the end."""

    end_reason: str = ""
    took_off: bool = False
    landed_in_dock: bool = False
    returned_by_model: bool = False
    threats: int = 0
    valid_reports: int = 0
    false_alarms: int = 0
    missed_threats: int = 0
    reports_made: int = 0
    zooms_used: int = 0
    coverage: float = 0.0
    max_height_m: float = 0.0
    min_threat_distance_m: Optional[float] = None


# The raw metrics of one seed's result: the patrol outcome plus the clock.
RESULT_METRICS = tuple(Outcome.__dataclass_fields__) + ("time_sec", "horizon_sec", "success")
# The normalised terms of one seed's result; final_score is the seed's score.
SCORE_TERMS = ("detection_term", "coverage_term", "flight_term", "final_score")


@dataclass
class Frames:
    """The images the model is shown, kept between steps so a held frame costs nothing to resend."""

    rgb: np.ndarray = field(default_factory=lambda: np.zeros(RGB_SHAPE, dtype=np.float32))
    thermal: np.ndarray = field(default_factory=lambda: np.zeros(THERMAL_SHAPE, dtype=np.float32))
    zoom: np.ndarray = field(default_factory=lambda: np.zeros(ZOOM_SHAPE, dtype=np.float32))
