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

"""Wind strength per seed (task 18): the wind the seed deals, and the two readings the dock reports of it.

Each seed is still, light or strong, in the shares measured at a weather station near the site (hourly
2019 to 2025, the 10 m wind taken up to the 20 m patrol height), with night calmer than day. The strength sets the
cap of Swarm's seeded wind: its steady part is 0.4 to 0.67 of the cap, gusts reach about 1.5 times that, and nothing
ever passes the cap, so no seed blows past the 12 m/s the M4TD and the Dock 3 are rated for.

The model never sees that wind, only the two values DJI's Cloud API sends, each pushed every 2 s:
- the aircraft's estimate, which DJI works out from the aircraft's attitude "for reference only": smoothed over
  about 12 s, reading low by up to a third, a little off in direction, in 0.1 m/s steps and one of 8 sectors for
  where the wind comes from, and zero on the ground;
- the dock's cup gauge, about 0.7 m off the ground, where the air is slower than at 20 m by the site's own
  measured wind profile, by day and by night.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from swarm.constants import SIM_DT
from swarm.core.daylight import seeded_sun

from . import park
from .contract import STATE_SLICES, put
from .episode import SolarEpisode

STRENGTHS = ("none", "light", "strong")
STILL_AIR = {"wind_max_mps": 0.0, "wind_turbulence": 0.0, "wind_gusts": 0}
DAY_SHARES = (0.05, 0.70, 0.25)          # site, by day: under 1 m/s 6 %, 1 to 4 m/s 69 %, over 4 m/s 25 %
NIGHT_SHARES = (0.10, 0.80, 0.10)        # site, by night: 10 %, 79 %, 11 %
CAP_MPS = {"none": (0.0, 0.0), "light": (2.5, 6.0), "strong": (10.0, 12.0)}  # steady wind 1 to 4 and 4 to 8 m/s
TURBULENCE = 1.0                         # Dryden's standard intensity, about 0.18 of the mean sideways at 20 m
GUSTS = (1, 2)                           # per patrol: peak 3 s wind 1.5 times the mean, the station measures 1.46
SEED_STREAM = 0x3E18                     # the wind's own streams, so its draws never move another part's

PUSH_STEPS = int(round(2.0 / SIM_DT))    # DJI: both values are in the OSD pushed at 0.5 Hz
AIRBORNE_M = 0.5                         # above the pad, where the aircraft starts estimating
ESTIMATE_TAU_S = 6.0                     # DJI's estimate is smoothed over a 10 to 15 s window
ESTIMATE_SCALE = (1.0 / 1.5, 1.0)        # and reads low by a factor of 1 to 1.5 (Varentsov et al. 2021)
ESTIMATE_TURN_TAN = 0.47                 # direction off by up to about 25 degrees, fixed per seed
GAUGE_TAU_S = 1.0                        # cup anemometer response
DAY_GAUGE_SHARE = (0.55, 0.71)           # 0.7 m wind over 20 m wind, the site's middle half of daytime hours
NIGHT_GAUGE_SHARE = (0.28, 0.48)         # and of night hours, when the air near the ground is stiller

_HALF = math.sqrt(0.5)
_SECTORS = ((0.0, 1.0), (_HALF, _HALF), (1.0, 0.0), (_HALF, -_HALF),
            (0.0, -1.0), (-_HALF, -_HALF), (-1.0, 0.0), (-_HALF, _HALF))  # east, north; clockwise from north


def is_night(seed: int) -> bool:
    """True when the environment lights this seed with the moon, drawn from the same seeded sun it uses."""
    return bool(park.SEEDED_SUN) and seeded_sun(int(seed), park.NIGHT_SHARE).night


def strength(seed: int) -> str:
    """The seed's wind strength: none, light or strong, in the site's day or night shares."""
    shares = NIGHT_SHARES if is_night(seed) else DAY_SHARES
    rng = np.random.default_rng([SEED_STREAM, int(seed)])
    return STRENGTHS[int(np.searchsorted(np.cumsum(shares), rng.random(), side="right"))]


def for_seed(seed: int) -> dict[str, Any]:
    """The task's wind fields for a seed, the ones the environment's seeded wind reads."""
    kind = strength(seed)
    if kind == "none":
        return dict(STILL_AIR)
    rng = np.random.default_rng([SEED_STREAM, int(seed), 1])
    low, high = CAP_MPS[kind]
    return {"wind_max_mps": round(float(rng.uniform(low, high)), 3), "wind_turbulence": TURBULENCE,
            "wind_gusts": int(rng.integers(GUSTS[0], GUSTS[1] + 1))}


def reset(env: Any, ep: SolarEpisode) -> None:
    """Draw this seed's reading errors and push clocks; the gauge starts on the steady wind, the estimate at zero."""
    rng = np.random.default_rng([SEED_STREAM, ep.seed, 2])
    turn = float(rng.uniform(-ESTIMATE_TURN_TAN, ESTIMATE_TURN_TAN))
    norm = math.sqrt(1.0 + turn * turn)
    gauge_share = rng.uniform(*(NIGHT_GAUGE_SHARE if is_night(ep.seed) else DAY_GAUGE_SHARE))
    ep.wind = {
        "scale": float(rng.uniform(*ESTIMATE_SCALE)),
        "turn": (1.0 / norm, turn / norm),
        "gauge_share": float(gauge_share),
        "estimate_offset": int(rng.integers(PUSH_STEPS)),
        "gauge_offset": int(rng.integers(PUSH_STEPS)),
        "smooth": np.zeros(2),
        "cups": float(gauge_share) * _speed(_steady(env)),
        "estimate_mps": 0.0,
        "estimate_sector": 0,
        "dock_mps": 0.0,
    }
    ep.wind["dock_mps"] = _tenths(ep.wind["cups"])


def update(env: Any, ep: SolarEpisode) -> None:
    """After one control step: follow the wind that step blew, and push each reading when its clock comes round."""
    w = ep.wind
    now = _current(env)
    airborne = ep.phase not in ("docked", "landed") and float(env.pos[0][2]) - float(ep.dock_position[2]) > AIRBORNE_M
    if airborne:
        w["smooth"] = w["smooth"] + (now - w["smooth"]) * (SIM_DT / ESTIMATE_TAU_S)
    else:
        w["smooth"] = np.zeros(2)
    w["cups"] += (w["gauge_share"] * _speed(now) - w["cups"]) * (SIM_DT / GAUGE_TAU_S)
    if (ep.step + w["estimate_offset"]) % PUSH_STEPS == 0:
        c, s = w["turn"]
        east, north = (float(v) for v in w["smooth"])
        seen = np.array([c * east + s * north, c * north - s * east]) * w["scale"]
        w["estimate_mps"] = _tenths(_speed(seen))
        w["estimate_sector"] = _sector(seen) if w["estimate_mps"] > 0.0 else 0
    if (ep.step + w["gauge_offset"]) % PUSH_STEPS == 0:
        w["dock_mps"] = _tenths(w["cups"])


def observe(env: Any, ep: SolarEpisode, state: np.ndarray) -> None:
    """The drone's rough wind estimate and the dock's gauge, as last pushed."""
    put(state, STATE_SLICES, "wind_estimate_mps", ep.wind["estimate_mps"])
    put(state, STATE_SLICES, "wind_estimate_sector", ep.wind["estimate_sector"])
    put(state, STATE_SLICES, "dock_wind_mps", ep.wind["dock_mps"])


def _current(env: Any) -> np.ndarray:
    """The horizontal wind (east, north) the last control step blew, zero in still air."""
    model = getattr(env, "_wind", None)
    return np.zeros(2) if model is None else np.array(model.current[:2], dtype=float)


def _steady(env: Any) -> np.ndarray:
    """The seed's steady horizontal wind, zero in still air."""
    model = getattr(env, "_wind", None)
    return np.zeros(2) if model is None else np.array(model.steady[:2], dtype=float)


def _speed(v: np.ndarray) -> float:
    """Length of a horizontal vector, without a library call that could round differently on another CPU."""
    return math.sqrt(float(v[0]) * float(v[0]) + float(v[1]) * float(v[1]))


def _sector(blowing: np.ndarray) -> int:
    """Which of the 8 compass sectors the wind comes from, 0 north and clockwise, by the nearest sector centre."""
    east, north = -float(blowing[0]), -float(blowing[1])
    return max(range(8), key=lambda k: _SECTORS[k][0] * east + _SECTORS[k][1] * north)


def _tenths(value: float) -> float:
    """A speed in DJI's 0.1 m/s steps."""
    return math.floor(value * 10.0 + 0.5) / 10.0
