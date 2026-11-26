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

"""The area light: on for Swarm Test Field v2 only, and under a family's area_light flag the higher lowest sun, the
graded daylight and the graded sky photo; the seeded sun's floor and the sky grade itself. No engine needed."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pybullet as p
import pytest
from PIL import Image

from swarm.challenge_families import get_challenge_family, list_registered_challenge_families
from swarm.challenge_families.test_field_v2.contract import FAMILY_ID as FIELD_FAMILY_ID
from swarm.constants import SUN_MIN_ELEVATION_DEG
from swarm.core.daylight import daylight_render_kwargs, max_elevation_deg, seeded_sun
from swarm.core.maps.andorra import rendering
from swarm.core.maps.andorra.rendering import balance_sky, grade_daylight, graded_sky_path, sky_light_colour
from swarm.core.moving_drone import MovingDroneAviary
from swarm.core.sky_pack import SkyPhoto

FLAGS = {"ER_SWARM_DAYLIGHT": 8192, "ER_SWARM_SHADOW_MAP": 64, "ER_SWARM_MOVER_SHADOW": 128, "ER_EDGE_ANTIALIAS": 256,
         "ER_ALPHA_CUTOUT": 512, "ER_TEXTURE_FILTER": 16, "ER_SPECULAR_GLINT": 1024, "ER_SWARM_LINEAR_LIGHT": 4096}


def _day_seed(start=3):
    """A seed whose light is a daytime sun."""
    seed = start
    while seeded_sun(seed, 0.5).night:
        seed += 1
    return seed


def _env(monkeypatch, area_light, photo, seed):
    """A bare env under the daylight model with the engine flags faked, its family's area light on or off."""
    for name, value in FLAGS.items():
        monkeypatch.setattr(p, name, value, raising=False)
    monkeypatch.setattr("swarm.core.moving_drone.load_sky_pack", lambda: [SkyPhoto("only", str(photo), 200.0, 40.0)])
    env = MovingDroneAviary.__new__(MovingDroneAviary)
    env.family_runtime = SimpleNamespace(seeded_sun=True, night_share=0.0, daylight=True, sky_from_sun=True, area_light=area_light)
    env._raycast_enabled = True
    env._apply_seeded_sun(seed)
    env._apply_daylight(seed)
    return env


@pytest.fixture
def photo(tmp_path):
    """A small sky photo on disk."""
    path = tmp_path / "sky.png"
    Image.new("RGB", (8, 4), (100, 140, 180)).save(path)
    return path


def test_only_the_test_field_flies_under_the_area_light():
    """Swarm Test Field v2 flies real areas under their light; every other registered family keeps today's light."""
    for family_id in list_registered_challenge_families():
        assert get_challenge_family(family_id).area_light == (family_id == FIELD_FAMILY_ID), family_id


def test_the_floor_moves_only_the_suns_height():
    """With any floor a seed keeps its heading and its day or night; the default floor is today's sun."""
    for seed in range(200):
        today, area = seeded_sun(seed, 0.2), seeded_sun(seed, 0.2, rendering.SUN_MIN_ELEVATION_DEG)
        assert seeded_sun(seed, 0.2, SUN_MIN_ELEVATION_DEG) == today
        assert (area.azimuth_deg, area.night) == (today.azimuth_deg, today.night)
        if not area.night:
            assert rendering.SUN_MIN_ELEVATION_DEG <= area.elevation_deg <= max_elevation_deg()


def test_a_wide_seed_keeps_all_its_bits():
    """Seeds that differ only above the low 32 bits light differently under the area floor."""
    seed = _day_seed()
    suns = {seeded_sun(seed + (k << 64), 0.0, rendering.SUN_MIN_ELEVATION_DEG).elevation_deg for k in range(4)}
    assert len(suns) == 4


def test_a_floor_at_or_over_the_peak_is_refused():
    """No sun can stand above the noon peak, so such a floor is an error."""
    with pytest.raises(ValueError, match="min_elevation_deg"):
        seeded_sun(1, 0.0, max_elevation_deg())


def test_an_area_family_draws_its_sun_from_the_area_floor(monkeypatch, photo):
    """Under the flag the env's sun is the seed's sun on the area floor; without it, today's sun."""
    seed = next(s for s in range(1, 500) if seeded_sun(s).elevation_deg < rendering.SUN_MIN_ELEVATION_DEG)
    assert _env(monkeypatch, True, photo, seed)._sun == seeded_sun(seed, 0.0, rendering.SUN_MIN_ELEVATION_DEG)
    assert _env(monkeypatch, False, photo, seed)._sun == seeded_sun(seed)


def test_the_area_grade_applies_only_under_the_flag(monkeypatch, photo):
    """The flag scales exposure, ambient and haze and lifts the shadow floor; without it the arguments are today's."""
    seed = _day_seed()
    monkeypatch.setattr(p, "loadTexture", lambda path, physicsClientId=0: 7)
    plain = _env(monkeypatch, False, photo, seed)._daylight_kwargs(0)
    graded = _env(monkeypatch, True, photo, _day_seed())._daylight_kwargs(0)
    today = daylight_render_kwargs(seeded_sun(seed))
    assert {k: v for k, v in plain.items() if k in today} == today
    sun = seeded_sun(seed, 0.0, rendering.SUN_MIN_ELEVATION_DEG)
    assert graded["exposure"] == pytest.approx(daylight_render_kwargs(sun)["exposure"] * rendering.EXPOSURE_SCALE)
    assert graded["lightAmbientCoeff"] == pytest.approx(today["lightAmbientCoeff"] * rendering.AMBIENT_SCALE)
    assert graded["hazeDistance"] == pytest.approx(today["hazeDistance"] * rendering.HAZE_DISTANCE_SCALE)
    assert graded["shadowLightCoeff"] == rendering.SHADOW_FLOOR


def test_the_area_sky_photo_is_graded_once(monkeypatch, photo):
    """Under the flag the engine loads the graded copy, written once per photo; without it the photo itself."""
    loads = []
    monkeypatch.setattr(p, "loadTexture", lambda path, physicsClientId=0: loads.append(path) or len(loads))
    seed = _day_seed()
    misses = graded_sky_path.cache_info().misses
    for _ in range(2):
        _env(monkeypatch, True, photo, seed)._daylight_kwargs(0)
    _env(monkeypatch, False, photo, seed)._daylight_kwargs(0)
    assert loads[0] == loads[1] != str(photo) and Path(loads[0]).is_file()
    assert loads[2] == str(photo)
    assert graded_sky_path.cache_info().misses == misses + 1


def test_grade_leaves_out_what_the_caller_left_out():
    """Optional renderer arguments that are absent stay absent."""
    assert grade_daylight({}) == {}


def _blue_sky() -> Image.Image:
    """A 64 x 32 equirectangular sky: deep blue at the zenith, pale at the horizon, a brown ground below."""
    rows = np.linspace(0.0, 1.0, 16)[:, None, None]
    upper = np.repeat((np.array([60, 110, 220]) * (1 - rows) + np.array([200, 215, 235]) * rows), 64, axis=1)
    lower = np.full((16, 64, 3), (90, 80, 60))
    return Image.fromarray(np.concatenate([upper, lower]).astype(np.uint8))


def test_sky_balance_turns_the_sky_light_towards_neutral():
    """Full balance lights an up-facing surface almost white, the shipped strength cuts most of the blue, none
    changes nothing."""
    sky = _blue_sky()
    blue = sky_light_colour(sky)
    assert blue[2] / blue[0] > 1.5
    assert np.array_equal(np.asarray(balance_sky(sky, 0.0)), np.asarray(sky))
    full = sky_light_colour(balance_sky(sky, 1.0))
    assert abs(full[2] / full[0] - 1.0) < 0.08
    shipped = sky_light_colour(balance_sky(sky, rendering.SKY_BALANCE))
    assert 1.0 <= shipped[2] / shipped[0] < 1.0 + 0.5 * (blue[2] / blue[0] - 1.0)


def test_sky_balance_is_the_same_bytes_every_time():
    """The balanced photo is rebuilt byte for byte, so every validator lights shade alike."""
    sky = _blue_sky()
    assert np.array_equal(np.asarray(balance_sky(sky, rendering.SKY_BALANCE)),
                          np.asarray(balance_sky(sky.copy(), rendering.SKY_BALANCE)))
