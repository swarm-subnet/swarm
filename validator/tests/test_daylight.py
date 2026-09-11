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

"""Tests for the seeded daylight sun: deterministic, daylight only, and off unless a family opts in."""

from __future__ import annotations

import math
from types import SimpleNamespace

from swarm.challenge_families.base import ChallengeFamilyRuntime
from swarm.constants import SUN_AMBIENT_RANGE, SUN_DIFFUSE_MAX, SUN_MIN_ELEVATION_DEG
from swarm.core.daylight import apply_seeded_sun, daylight_hours, seeded_sun, sun_color, sun_render_kwargs


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
    """No family opts in unless it says so, so today's families keep today's light."""
    assert ChallengeFamilyRuntime.seeded_sun is False
    for cls in ChallengeFamilyRuntime.__subclasses__():
        assert cls.seeded_sun is False, cls.__name__
