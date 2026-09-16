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
    MOON_AMBIENT_RANGE,
    MOON_COLOR,
    MOON_DIFFUSE_RANGE,
    MOON_ELEVATION_RANGE_DEG,
    SUN_AMBIENT_RANGE,
    SUN_DIFFUSE_MAX,
    SUN_MIN_ELEVATION_DEG,
)
from swarm.core.daylight import (
    NIGHT_SKY,
    _elevation_deg,
    apply_seeded_sun,
    daylight_hours,
    max_elevation_deg,
    seeded_sun,
    sky_render_kwargs,
    sun_color,
    sun_render_kwargs,
)
from swarm.core.moving_drone import MovingDroneAviary


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
    assert sun_color(SUN_MIN_ELEVATION_DEG) == (1.0, 0.5, 0.24)
    assert sun_color(max_elevation_deg()) == (1.0, 1.0, 1.0)
    assert sun_color(80.0) == (1.0, 1.0, 1.0)
    low = min((seeded_sun(seed) for seed in range(200)), key=lambda s: s.elevation_deg)
    high = max((seeded_sun(seed) for seed in range(200)), key=lambda s: s.elevation_deg)
    assert low.color[2] < high.color[2]
    assert low.diffuse < high.diffuse <= SUN_DIFFUSE_MAX
    assert SUN_AMBIENT_RANGE[0] <= low.ambient < high.ambient <= SUN_AMBIENT_RANGE[1]


def test_every_height_of_sun_is_equally_likely():
    """The seed draws the sun's height, not its hour, so a low golden sun is as common as noon:
    each quarter of the arc takes roughly a quarter of the seeds and no setting saturates."""
    peak = max_elevation_deg()
    edges = [SUN_MIN_ELEVATION_DEG + (peak - SUN_MIN_ELEVATION_DEG) * q / 4.0 for q in range(5)]
    suns = [seeded_sun(seed) for seed in range(4000)]
    for lo, hi in zip(edges, edges[1:]):
        share = sum(lo <= sun.elevation_deg < hi for sun in suns) / len(suns)
        assert 0.2 < share < 0.3, f"{lo:.0f} to {hi:.0f} deg took {share:.0%} of seeds"
    ambients = [sun.ambient for sun in suns]
    assert sum(a >= SUN_AMBIENT_RANGE[1] - 1e-6 for a in ambients) / len(suns) < 0.01
    assert max(ambients) - min(ambients) > 0.9 * (SUN_AMBIENT_RANGE[1] - SUN_AMBIENT_RANGE[0])


def test_the_hour_matches_the_height_it_was_drawn_for():
    """The hour is read back off the arc, so putting it through the arc returns the same height."""
    for seed in range(300):
        sun = seeded_sun(seed)
        assert math.isclose(_elevation_deg(sun.hour), sun.elevation_deg, abs_tol=0.01)


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


def test_zero_night_share_is_always_day():
    """Without a night share a seed draws a sun, and the default argument is that same share."""
    for seed in range(300):
        sun = seeded_sun(seed)
        assert sun == seeded_sun(seed, night_share=0.0)
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
        assert MOON_DIFFUSE_RANGE[0] <= moon.diffuse <= MOON_DIFFUSE_RANGE[1] < weakest_sun
        assert MOON_AMBIENT_RANGE[0] <= moon.ambient <= MOON_AMBIENT_RANGE[1] < SUN_AMBIENT_RANGE[0]
        assert moon.sky == NIGHT_SKY


def test_the_moon_has_phases():
    """Nights differ from each other: the moon's strength spans its band, and a brighter moon
    carries a brighter ambient with it."""
    moons = [seeded_sun(seed, night_share=1.0) for seed in range(2000)]
    diffuses = [moon.diffuse for moon in moons]
    span = MOON_DIFFUSE_RANGE[1] - MOON_DIFFUSE_RANGE[0]
    assert max(diffuses) - min(diffuses) > 0.9 * span
    assert len(set(diffuses)) > 500
    brightest = max(moons, key=lambda m: m.diffuse)
    dimmest = min(moons, key=lambda m: m.diffuse)
    assert brightest.ambient > dimmest.ambient


def test_only_a_moon_carries_a_sky():
    """A moon brings a dark sky so night reads as night; a sun leaves the background alone,
    because the daylight sky belongs to the renderer, not to this module."""
    night_horizon, night_zenith = NIGHT_SKY
    assert max(night_horizon) < 0.2 and max(night_zenith) < 0.1
    moon = seeded_sun(7, night_share=1.0)
    assert moon.sky == NIGHT_SKY
    assert sky_render_kwargs(moon) == {
        "skyHorizonColor": list(night_horizon),
        "skyZenithColor": list(night_zenith),
    }
    for seed in range(200):
        sun = seeded_sun(seed)
        assert sun.sky is None
        assert sky_render_kwargs(sun) is None
    assert sky_render_kwargs(None) is None


def test_env_takes_the_moon_sky_only_when_the_family_sets_none():
    """A night with no family sky paints the moon's dark sky; a family sky wins over it; a day
    seed and a family with neither keep today's white background."""
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
    env._sun = seeded_sun(7)
    assert env._sky_kwargs() == {}
    env._sun = None
    assert env._sky_kwargs() == {}


def test_apply_seeded_sun_passes_the_night_share():
    """The env wiring hands the family's share through, so a full share lights the env with the moon."""
    env = SimpleNamespace()
    moon = apply_seeded_sun(env, 42, night_share=1.0)
    assert moon.night is True and env._sun == moon
    assert env._light_direction == list(moon.direction)
    assert env._light_color == list(MOON_COLOR)
