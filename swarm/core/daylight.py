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

"""Seeded daylight: one sun per seed, chosen from a real sun arc, never below the horizon.

A family opts in with ``seeded_sun = True``; nothing here runs for the families that do not.
The seed picks an hour between sunrise and sunset and a heading for the map. The hour sets
how high the sun is, and the height sets its colour and strength: white and strong at noon,
orange and weak near the horizon. Every value is rounded to fixed decimals so each validator
hands the renderer the same numbers.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from swarm.constants import (
    SUN_AMBIENT_RANGE,
    SUN_DECLINATION_DEG,
    SUN_DIFFUSE_MAX,
    SUN_EXTINCTION,
    SUN_LATITUDE_DEG,
    SUN_MIN_ELEVATION_DEG,
    SUN_SEED_OFFSET,
)

# Sun colour by elevation, linearly interpolated between rows: warm near the horizon,
# white once the sun is high.
SUN_COLOR_TABLE: Tuple[Tuple[float, Tuple[float, float, float]], ...] = (
    (0.0, (1.00, 0.55, 0.30)),
    (5.0, (1.00, 0.72, 0.48)),
    (15.0, (1.00, 0.88, 0.72)),
    (30.0, (1.00, 0.96, 0.90)),
    (50.0, (1.00, 1.00, 1.00)),
)


@dataclass(frozen=True)
class SunLight:
    """One seed's sun: where it is, what colour it is and how strong it is."""

    hour: float
    elevation_deg: float
    azimuth_deg: float
    direction: Tuple[float, float, float]
    color: Tuple[float, float, float]
    ambient: float
    diffuse: float


def _elevation_deg(hour: float) -> float:
    """Sun elevation in degrees at a local solar hour on the configured arc."""
    lat = math.radians(SUN_LATITUDE_DEG)
    dec = math.radians(SUN_DECLINATION_DEG)
    hour_angle = math.radians(15.0 * (hour - 12.0))
    sin_el = math.sin(lat) * math.sin(dec) + math.cos(lat) * math.cos(dec) * math.cos(hour_angle)
    return math.degrees(math.asin(max(-1.0, min(1.0, sin_el))))


def daylight_hours() -> Tuple[float, float]:
    """The first and last hour at which the sun sits at the minimum elevation."""
    lat = math.radians(SUN_LATITUDE_DEG)
    dec = math.radians(SUN_DECLINATION_DEG)
    cos_h0 = (math.sin(math.radians(SUN_MIN_ELEVATION_DEG)) - math.sin(lat) * math.sin(dec)) / (
        math.cos(lat) * math.cos(dec)
    )
    half_day = math.degrees(math.acos(max(-1.0, min(1.0, cos_h0)))) / 15.0
    return 12.0 - half_day, 12.0 + half_day


def sun_color(elevation_deg: float) -> Tuple[float, float, float]:
    """Sun colour for an elevation, read off the colour table."""
    if elevation_deg >= SUN_COLOR_TABLE[-1][0]:
        return SUN_COLOR_TABLE[-1][1]
    for (lo_el, lo_rgb), (hi_el, hi_rgb) in zip(SUN_COLOR_TABLE, SUN_COLOR_TABLE[1:]):
        if elevation_deg <= hi_el:
            t = (elevation_deg - lo_el) / (hi_el - lo_el)
            return tuple(round(lo + (hi - lo) * t, 4) for lo, hi in zip(lo_rgb, hi_rgb))
    return SUN_COLOR_TABLE[-1][1]


def seeded_sun(seed: int) -> SunLight:
    """The sun for a seed: an hour between sunrise and sunset and a heading for the map."""
    rng = random.Random((int(seed) ^ SUN_SEED_OFFSET) & 0xFFFFFFFF)
    first, last = daylight_hours()
    hour = rng.uniform(first, last)
    azimuth_deg = rng.uniform(0.0, 360.0)
    elevation_deg = max(SUN_MIN_ELEVATION_DEG, _elevation_deg(hour))

    el = math.radians(elevation_deg)
    az = math.radians(azimuth_deg)
    direction = (
        round(math.cos(el) * math.cos(az), 6),
        round(math.cos(el) * math.sin(az), 6),
        round(math.sin(el), 6),
    )
    # Air mass thins the sun near the horizon; the sky dims with it.
    air_mass = 1.0 / math.sin(el)
    diffuse = SUN_DIFFUSE_MAX * math.exp(-SUN_EXTINCTION * (air_mass - 1.0))
    ambient_lo, ambient_hi = SUN_AMBIENT_RANGE
    ambient = ambient_lo + (ambient_hi - ambient_lo) * min(1.0, elevation_deg / 20.0)
    return SunLight(
        hour=round(hour, 3),
        elevation_deg=round(elevation_deg, 3),
        azimuth_deg=round(azimuth_deg, 3),
        direction=direction,
        color=sun_color(elevation_deg),
        ambient=round(ambient, 4),
        diffuse=round(diffuse, 4),
    )


def sun_render_kwargs(sun: SunLight) -> Dict[str, Any]:
    """The renderer arguments a colour frame needs to be lit by this sun."""
    return {
        "lightColor": list(sun.color),
        "lightAmbientCoeff": sun.ambient,
        "lightDiffuseCoeff": sun.diffuse,
    }


def apply_seeded_sun(env: Any, seed: int) -> SunLight:
    """Light an environment with the seed's sun and return it."""
    sun = seeded_sun(seed)
    env._sun = sun
    env._light_direction = list(sun.direction)
    env._light_color = list(sun.color)
    return sun


__all__: List[str] = [
    "SunLight",
    "apply_seeded_sun",
    "daylight_hours",
    "seeded_sun",
    "sun_color",
    "sun_render_kwargs",
]
