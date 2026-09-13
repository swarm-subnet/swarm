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

"""Tests for the seeded sun and moon: deterministic, never below the horizon, night only by share,
and off unless a family opts in."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from swarm.challenge_families.base import ChallengeFamilyRuntime
from swarm.constants import (
    MOON_AMBIENT,
    MOON_COLOR,
    MOON_DIFFUSE,
    MOON_ELEVATION_RANGE_DEG,
    SUN_AMBIENT_RANGE,
    SUN_DIFFUSE_MAX,
    SUN_MIN_ELEVATION_DEG,
)
from swarm.core.daylight import (
    NIGHT_SKY,
    apply_seeded_sun,
    day_sky,
    daylight_hours,
    seeded_sun,
    sky_render_kwargs,
    sun_color,
    sun_render_kwargs,
)
from swarm.core.moving_drone import MovingDroneAviary

# The suns two seeds drew before the moon existed; a share of zero must still draw them.
CARD_TEN_SUNS = {
    25: (5.708, 3.099, 124.065, (-0.559319, 0.827188, 0.054065), (1.0, 0.6554, 0.4116), 0.431, 0.1225),
    96: (12.006, 60.0, 116.472, (-0.22288, 0.447578, 0.866024), (1.0, 1.0, 1.0), 0.6, 0.3468),
}


def test_same_seed_same_sun():
    """One seed gives the same sun on every call."""
    assert seeded_sun(1234) == seeded_sun(1234)


def test_different_seeds_different_suns():
    """Five seeds give five different sun directions."""
    directions = {seeded_sun(seed).direction for seed in range(1, 6)}
    assert len(directions) == 5


def test_sun_never_below_the_horizon():
    """Every seed's sun sits at or above the minimum elevation, with a unit direction pointing up."""
    first, last = daylight_hours()
    for seed in range(0, 500):
        sun = seeded_sun(seed)
        assert first <= sun.hour <= last
        assert sun.elevation_deg >= SUN_MIN_ELEVATION_DEG
        assert sun.direction[2] > 0.0
        assert math.isclose(sum(c * c for c in sun.direction), 1.0, abs_tol=1e-5)


def test_low_sun_is_warm_and_weak_high_sun_is_white_and_strong():
    """Sun colour and strength follow elevation: orange and dim low, white and full at noon."""
    assert sun_color(0.0) == (1.0, 0.55, 0.3)
    assert sun_color(50.0) == (1.0, 1.0, 1.0)
    assert sun_color(80.0) == (1.0, 1.0, 1.0)
    low = min((seeded_sun(seed) for seed in range(200)), key=lambda s: s.elevation_deg)
    high = max((seeded_sun(seed) for seed in range(200)), key=lambda s: s.elevation_deg)
    assert low.color[2] < high.color[2]
    assert low.diffuse < high.diffuse <= SUN_DIFFUSE_MAX
    assert SUN_AMBIENT_RANGE[0] <= low.ambient < high.ambient <= SUN_AMBIENT_RANGE[1]


def test_render_kwargs_carry_colour_ambient_and_diffuse():
    """The renderer arguments hold exactly the sun's colour, ambient and diffuse."""
    sun = seeded_sun(7)
    kwargs = sun_render_kwargs(sun)
    assert kwargs == {
        "lightColor": list(sun.color),
        "lightAmbientCoeff": sun.ambient,
        "lightDiffuseCoeff": sun.diffuse,
    }


def test_apply_seeded_sun_lights_the_env():
    """Applying the sun writes its direction and colour onto the env and keeps the sun on it."""
    env = SimpleNamespace()
    sun = apply_seeded_sun(env, 42)
    assert env._sun == sun == seeded_sun(42)
    assert env._light_direction == list(sun.direction)
    assert env._light_color == list(sun.color)


def test_families_are_off_by_default():
    """No family opts in unless it says so, so today's families keep today's light and no night."""
    assert ChallengeFamilyRuntime.seeded_sun is False
    assert ChallengeFamilyRuntime.night_share == 0.0
    for cls in ChallengeFamilyRuntime.__subclasses__():
        assert cls.seeded_sun is False, cls.__name__
        assert cls.night_share == 0.0, cls.__name__


def test_zero_night_share_draws_the_same_suns_as_before():
    """Without a night share every seed draws the sun it drew before the moon existed."""
    for seed, (hour, el, az, direction, color, ambient, diffuse) in CARD_TEN_SUNS.items():
        sun = seeded_sun(seed)
        assert sun == seeded_sun(seed, night_share=0.0)
        assert (sun.hour, sun.elevation_deg, sun.azimuth_deg) == (hour, el, az)
        assert (sun.direction, sun.color, sun.ambient, sun.diffuse) == (direction, color, ambient, diffuse)
        assert sun.night is False


def test_night_share_is_the_share_of_seeds_the_moon_lights():
    """A share of 0.3 gives about three moons in ten seeds, 0 gives none and 1 gives all."""
    nights = sum(seeded_sun(seed, night_share=0.3).night for seed in range(2000))
    assert 500 <= nights <= 700
    assert not any(seeded_sun(seed, night_share=0.0).night for seed in range(200))
    assert all(seeded_sun(seed, night_share=1.0).night for seed in range(200))


def test_night_share_outside_zero_to_one_is_rejected():
    """A share below 0 or above 1 is a mistake and raises."""
    with pytest.raises(ValueError):
        seeded_sun(1, night_share=1.5)
    with pytest.raises(ValueError):
        seeded_sun(1, night_share=-0.1)


def test_moon_is_high_dim_cool_and_after_sunset():
    """Every moon sits inside its elevation band above the horizon, is weaker and bluer than
    any sun, keeps the seed's heading, and its hour falls between sunset and sunrise."""
    first, last = daylight_hours()
    lo, hi = MOON_ELEVATION_RANGE_DEG
    weakest_sun = min(seeded_sun(seed).diffuse for seed in range(300))
    for seed in range(300):
        moon = seeded_sun(seed, night_share=1.0)
        assert moon.night is True
        assert lo <= moon.elevation_deg <= hi
        assert moon.direction[2] > 0.0
        assert math.isclose(sum(c * c for c in moon.direction), 1.0, abs_tol=1e-5)
        assert moon.azimuth_deg == seeded_sun(seed).azimuth_deg
        assert moon.hour > last or moon.hour < first
        assert moon.color == MOON_COLOR and moon.color[2] > moon.color[0]
        assert moon.diffuse == MOON_DIFFUSE < weakest_sun
        assert moon.ambient == MOON_AMBIENT < SUN_AMBIENT_RANGE[0]
        assert moon.sky == NIGHT_SKY


def test_sky_follows_the_light():
    """The day sky warms at the horizon for a low sun, is blue for a high one, and night is dark."""
    low_horizon, low_zenith = day_sky(3.0)
    high_horizon, high_zenith = day_sky(60.0)
    assert low_horizon[0] > low_horizon[2] and high_horizon[2] > high_horizon[0]
    assert high_zenith[2] > high_zenith[0] and high_zenith[2] > low_zenith[2]
    assert seeded_sun(96).sky == day_sky(60.0)
    night_horizon, night_zenith = NIGHT_SKY
    assert max(night_horizon) < 0.2 and max(night_zenith) < 0.1
    assert sky_render_kwargs(seeded_sun(96)) == {
        "skyHorizonColor": list(high_horizon),
        "skyZenithColor": list(high_zenith),
    }


def test_env_sky_comes_from_the_sun_unless_the_family_sets_one():
    """With a seeded sun and no family sky the camera paints the sun's sky; a family sky wins."""
    env = MovingDroneAviary.__new__(MovingDroneAviary)
    env._sky_colors = None
    env._sun = seeded_sun(7, night_share=1.0)
    assert env._sky_kwargs() == sky_render_kwargs(env._sun)
    env._sky_colors = ((0.85, 0.55, 0.25), (0.15, 0.35, 0.95))
    assert env._sky_kwargs() == {
        "skyHorizonColor": [0.85, 0.55, 0.25],
        "skyZenithColor": [0.15, 0.35, 0.95],
    }
    env._sky_colors = None
    env._sun = None
    assert env._sky_kwargs() == {}


def test_apply_seeded_sun_passes_the_night_share():
    """The env wiring hands the family's share through, so a full share lights the env with the moon."""
    env = SimpleNamespace()
    moon = apply_seeded_sun(env, 42, night_share=1.0)
    assert moon.night is True and env._sun == moon
    assert env._light_direction == list(moon.direction)
    assert env._light_color == list(MOON_COLOR)
