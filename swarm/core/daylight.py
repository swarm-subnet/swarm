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
orange and weak near the horizon. A family that also sets ``night_share`` hands that share of
its seeds to the moon instead: a cool, weak light from high up under a dark sky. Every value
is rounded to fixed decimals so each validator hands the renderer the same numbers.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from swarm.constants import (
    MOON_AMBIENT,
    MOON_COLOR,
    MOON_DIFFUSE,
    MOON_ELEVATION_RANGE_DEG,
    SUN_AMBIENT_RANGE,
    SUN_DECLINATION_DEG,
    SUN_DIFFUSE_MAX,
    SUN_EXTINCTION,
    SUN_LATITUDE_DEG,
    SUN_MIN_ELEVATION_DEG,
    SUN_SEED_OFFSET,
)

RGB = Tuple[float, float, float]
ColorTable = Tuple[Tuple[float, RGB], ...]

# Sun colour by elevation, linearly interpolated between rows: warm near the horizon,
# white once the sun is high.
SUN_COLOR_TABLE: ColorTable = (
    (0.0, (1.00, 0.55, 0.30)),
    (5.0, (1.00, 0.72, 0.48)),
    (15.0, (1.00, 0.88, 0.72)),
    (30.0, (1.00, 0.96, 0.90)),
    (50.0, (1.00, 1.00, 1.00)),
)
# Day sky by sun elevation, same interpolation: a warm horizon and a deep zenith at sunrise,
# pale blue over blue by mid-morning.
SKY_HORIZON_TABLE: ColorTable = (
    (0.0, (1.00, 0.62, 0.40)),
    (10.0, (0.95, 0.82, 0.68)),
    (30.0, (0.80, 0.88, 0.97)),
    (60.0, (0.74, 0.86, 0.98)),
)
SKY_ZENITH_TABLE: ColorTable = (
    (0.0, (0.25, 0.32, 0.58)),
    (10.0, (0.36, 0.52, 0.82)),
    (30.0, (0.30, 0.55, 0.92)),
    (60.0, (0.24, 0.48, 0.90)),
)
NIGHT_SKY: Tuple[RGB, RGB] = ((0.06, 0.07, 0.14), (0.01, 0.02, 0.05))


@dataclass(frozen=True)
class SunLight:
    """One seed's light: where it is, what colour it is, how strong it is and the sky under it."""

    hour: float
    elevation_deg: float
    azimuth_deg: float
    direction: RGB
    color: RGB
    ambient: float
    diffuse: float
    night: bool
    sky: Tuple[RGB, RGB]


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


def _table_color(table: ColorTable, elevation_deg: float) -> RGB:
    """A colour for an elevation, linearly interpolated between the rows of a table."""
    if elevation_deg >= table[-1][0]:
        return table[-1][1]
    for (lo_el, lo_rgb), (hi_el, hi_rgb) in zip(table, table[1:]):
        if elevation_deg <= hi_el:
            t = (elevation_deg - lo_el) / (hi_el - lo_el)
            return tuple(round(lo + (hi - lo) * t, 4) for lo, hi in zip(lo_rgb, hi_rgb))
    return table[-1][1]


def sun_color(elevation_deg: float) -> RGB:
    """Sun colour for an elevation, read off the colour table."""
    return _table_color(SUN_COLOR_TABLE, elevation_deg)


def day_sky(elevation_deg: float) -> Tuple[RGB, RGB]:
    """Horizon and zenith colours of the day sky for a sun elevation."""
    return _table_color(SKY_HORIZON_TABLE, elevation_deg), _table_color(SKY_ZENITH_TABLE, elevation_deg)


def _direction(elevation_deg: float, azimuth_deg: float) -> RGB:
    """Unit vector toward a light at an elevation and heading, rounded to fixed decimals."""
    el = math.radians(elevation_deg)
    az = math.radians(azimuth_deg)
    return (
        round(math.cos(el) * math.cos(az), 6),
        round(math.cos(el) * math.sin(az), 6),
        round(math.sin(el), 6),
    )


def _moon(rng: random.Random, azimuth_deg: float, first: float, last: float) -> SunLight:
    """A night for a seed: the moon somewhere high, an hour between sunset and sunrise."""
    elevation_deg = rng.uniform(*MOON_ELEVATION_RANGE_DEG)
    hour = (last + rng.uniform(0.0, 24.0 - (last - first))) % 24.0
    return SunLight(
        hour=round(hour, 3),
        elevation_deg=round(elevation_deg, 3),
        azimuth_deg=round(azimuth_deg, 3),
        direction=_direction(elevation_deg, azimuth_deg),
        color=MOON_COLOR,
        ambient=MOON_AMBIENT,
        diffuse=MOON_DIFFUSE,
        night=True,
        sky=NIGHT_SKY,
    )


def seeded_sun(seed: int, night_share: float = 0.0) -> SunLight:
    """The light for a seed: a sun on the day arc, or with the given share a moon instead.

    The hour and heading are drawn first, so a share of zero draws exactly the sun it
    always did; the night draw only happens when a family asks for one."""
    if not 0.0 <= night_share <= 1.0:
        raise ValueError(f"night_share must be between 0 and 1, got {night_share}")
    rng = random.Random((int(seed) ^ SUN_SEED_OFFSET) & 0xFFFFFFFF)
    first, last = daylight_hours()
    hour = rng.uniform(first, last)
    azimuth_deg = rng.uniform(0.0, 360.0)
    if night_share > 0.0 and rng.random() < night_share:
        return _moon(rng, azimuth_deg, first, last)
    elevation_deg = max(SUN_MIN_ELEVATION_DEG, _elevation_deg(hour))

    # Air mass thins the sun near the horizon; the sky dims with it.
    air_mass = 1.0 / math.sin(math.radians(elevation_deg))
    diffuse = SUN_DIFFUSE_MAX * math.exp(-SUN_EXTINCTION * (air_mass - 1.0))
    ambient_lo, ambient_hi = SUN_AMBIENT_RANGE
    ambient = ambient_lo + (ambient_hi - ambient_lo) * min(1.0, elevation_deg / 20.0)
    return SunLight(
        hour=round(hour, 3),
        elevation_deg=round(elevation_deg, 3),
        azimuth_deg=round(azimuth_deg, 3),
        direction=_direction(elevation_deg, azimuth_deg),
        color=sun_color(elevation_deg),
        ambient=round(ambient, 4),
        diffuse=round(diffuse, 4),
        night=False,
        sky=day_sky(elevation_deg),
    )


def sun_render_kwargs(sun: SunLight) -> Dict[str, Any]:
    """The renderer arguments a colour frame needs to be lit by this sun."""
    return {
        "lightColor": list(sun.color),
        "lightAmbientCoeff": sun.ambient,
        "lightDiffuseCoeff": sun.diffuse,
    }


def sky_render_kwargs(sun: SunLight) -> Dict[str, Any]:
    """The renderer arguments that paint this sun's sky behind the map."""
    horizon, zenith = sun.sky
    return {"skyHorizonColor": list(horizon), "skyZenithColor": list(zenith)}


def apply_seeded_sun(env: Any, seed: int, night_share: float = 0.0) -> SunLight:
    """Light an environment with the seed's sun, or its moon, and return it."""
    sun = seeded_sun(seed, night_share)
    env._sun = sun
    env._light_direction = list(sun.direction)
    env._light_color = list(sun.color)
    return sun


__all__: List[str] = [
    "SunLight",
    "apply_seeded_sun",
    "day_sky",
    "daylight_hours",
    "seeded_sun",
    "sky_render_kwargs",
    "sun_color",
    "sun_render_kwargs",
]
