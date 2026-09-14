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
The seed picks how high the sun stands and which way it faces, and the hour that puts it
there is read back off the arc. The height sets its colour and strength: white and strong at
the noon peak, orange and weak near the horizon. A family that also sets ``night_share`` hands
that share of its seeds to the moon instead: a cool, weak light from high up under a dark sky.
Every value is rounded to fixed decimals so each validator hands the renderer the same numbers.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from swarm.constants import (
    MOON_AMBIENT_RANGE,
    MOON_COLOR,
    MOON_DIFFUSE_RANGE,
    MOON_ELEVATION_RANGE_DEG,
    SUN_AMBIENT_RANGE,
    SUN_DECLINATION_DEG,
    SUN_DIFFUSE_MAX,
    SUN_EXTINCTION,
    SUN_LATITUDE_DEG,
    SUN_MIN_ELEVATION_DEG,
    SUN_SEED_OFFSET,
)

Vec3 = Tuple[float, float, float]
RGB = Vec3
ColorTable = Tuple[Tuple[float, RGB], ...]

# Sun colour by elevation, linearly interpolated between rows: warm near the horizon, white
# only at the noon peak, so the whole arc keeps changing rather than saturating halfway up.
SUN_COLOR_TABLE: ColorTable = (
    (3.0, (1.00, 0.50, 0.24)),
    (10.0, (1.00, 0.64, 0.38)),
    (20.0, (1.00, 0.77, 0.56)),
    (35.0, (1.00, 0.89, 0.77)),
    (60.0, (1.00, 1.00, 1.00)),
)
# A moon needs a dark background or night does not read as night. Daylight skies are not set
# here: the sky card computes them in the renderer from the light it is already given.
NIGHT_SKY: Tuple[RGB, RGB] = ((0.06, 0.07, 0.14), (0.01, 0.02, 0.05))


@dataclass(frozen=True)
class SunLight:
    """One seed's light: where it is, what colour it is, how strong it is, and for a moon the
    dark sky that goes with it."""

    hour: float
    elevation_deg: float
    azimuth_deg: float
    direction: Vec3
    color: RGB
    ambient: float
    diffuse: float
    night: bool
    sky: Optional[Tuple[RGB, RGB]]


def _elevation_deg(hour: float) -> float:
    """Sun elevation in degrees at a local solar hour on the configured arc."""
    lat = math.radians(SUN_LATITUDE_DEG)
    dec = math.radians(SUN_DECLINATION_DEG)
    hour_angle = math.radians(15.0 * (hour - 12.0))
    sin_el = math.sin(lat) * math.sin(dec) + math.cos(lat) * math.cos(dec) * math.cos(hour_angle)
    return math.degrees(math.asin(max(-1.0, min(1.0, sin_el))))


def max_elevation_deg() -> float:
    """The noon peak of the arc: the highest sun a seed can draw."""
    return _elevation_deg(12.0)


def _hour_for_elevation(elevation_deg: float, morning: bool) -> float:
    """The solar hour at which the sun reaches an elevation, before noon or after it."""
    lat = math.radians(SUN_LATITUDE_DEG)
    dec = math.radians(SUN_DECLINATION_DEG)
    cos_h = (math.sin(math.radians(elevation_deg)) - math.sin(lat) * math.sin(dec)) / (
        math.cos(lat) * math.cos(dec)
    )
    half_day = math.degrees(math.acos(max(-1.0, min(1.0, cos_h)))) / 15.0
    return 12.0 - half_day if morning else 12.0 + half_day


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


def _direction(elevation_deg: float, azimuth_deg: float) -> Vec3:
    """Unit vector toward a light at an elevation and heading, rounded to fixed decimals."""
    el = math.radians(elevation_deg)
    az = math.radians(azimuth_deg)
    return (
        round(math.cos(el) * math.cos(az), 6),
        round(math.cos(el) * math.sin(az), 6),
        round(math.sin(el), 6),
    )


def _moon(rng: random.Random, azimuth_deg: float) -> SunLight:
    """A night for a seed: the moon somewhere high, as bright as its phase, after sunset."""
    elevation_deg = rng.uniform(*MOON_ELEVATION_RANGE_DEG)
    phase = rng.random()
    first, last = daylight_hours()
    hour = (last + rng.uniform(0.0, 24.0 - (last - first))) % 24.0
    diffuse_lo, diffuse_hi = MOON_DIFFUSE_RANGE
    ambient_lo, ambient_hi = MOON_AMBIENT_RANGE
    return SunLight(
        hour=round(hour, 3),
        elevation_deg=round(elevation_deg, 3),
        azimuth_deg=round(azimuth_deg, 3),
        direction=_direction(elevation_deg, azimuth_deg),
        color=MOON_COLOR,
        ambient=round(ambient_lo + (ambient_hi - ambient_lo) * phase, 4),
        diffuse=round(diffuse_lo + (diffuse_hi - diffuse_lo) * phase, 4),
        night=True,
        sky=NIGHT_SKY,
    )


def seeded_sun(seed: int, night_share: float = 0.0) -> SunLight:
    """The light for a seed: a sun on the day arc, or with the given share a moon instead.

    The sun's height is drawn directly, not its hour, so a low golden sun is as likely as
    noon; the hour is then read back off the arc. The heading is drawn before the day or
    night coin, so a seed keeps its heading either way."""
    if not 0.0 <= night_share <= 1.0:
        raise ValueError(f"night_share must be between 0 and 1, got {night_share}")
    rng = random.Random((int(seed) ^ SUN_SEED_OFFSET) & 0xFFFFFFFF)
    azimuth_deg = rng.uniform(0.0, 360.0)
    if night_share > 0.0 and rng.random() < night_share:
        return _moon(rng, azimuth_deg)
    peak = max_elevation_deg()
    elevation_deg = rng.uniform(SUN_MIN_ELEVATION_DEG, peak)
    hour = _hour_for_elevation(elevation_deg, rng.random() < 0.5)

    # Air mass thins the sun near the horizon; the sky dims with it.
    air_mass = 1.0 / math.sin(math.radians(elevation_deg))
    diffuse = SUN_DIFFUSE_MAX * math.exp(-SUN_EXTINCTION * (air_mass - 1.0))
    ambient_lo, ambient_hi = SUN_AMBIENT_RANGE
    climb = (elevation_deg - SUN_MIN_ELEVATION_DEG) / (peak - SUN_MIN_ELEVATION_DEG)
    ambient = ambient_lo + (ambient_hi - ambient_lo) * climb
    return SunLight(
        hour=round(hour, 3),
        elevation_deg=round(elevation_deg, 3),
        azimuth_deg=round(azimuth_deg, 3),
        direction=_direction(elevation_deg, azimuth_deg),
        color=sun_color(elevation_deg),
        ambient=round(ambient, 4),
        diffuse=round(diffuse, 4),
        night=False,
        sky=None,
    )


def sun_render_kwargs(sun: SunLight) -> Dict[str, Any]:
    """The renderer arguments a colour frame needs to be lit by this sun."""
    return {
        "lightColor": list(sun.color),
        "lightAmbientCoeff": sun.ambient,
        "lightDiffuseCoeff": sun.diffuse,
    }


def sky_render_kwargs(sun: Optional[SunLight]) -> Optional[Dict[str, Any]]:
    """The renderer arguments that paint this light's own sky, or None when it has none.

    Only a moon carries a sky here: a daylight sky is the renderer's to compute from the
    light it already receives, so a sun leaves the background to the family."""
    if sun is None or sun.sky is None:
        return None
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
    "daylight_hours",
    "max_elevation_deg",
    "seeded_sun",
    "sky_render_kwargs",
    "sun_color",
    "sun_render_kwargs",
]
