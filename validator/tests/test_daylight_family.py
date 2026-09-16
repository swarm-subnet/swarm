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

"""The daylight option: off for every family today, the render arguments it adds, the sky photo a seed
picks and the turn that puts the photo's sun on the seeded sun's heading. No engine needed."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pybullet as p
import pytest

from swarm.challenge_families import get_challenge_family, list_registered_challenge_families
from swarm.challenge_families.base import ChallengeFamilyRuntime
from swarm.constants import (
    DAYLIGHT_AMBIENT,
    DAYLIGHT_EXPOSURE,
    DAYLIGHT_EXPOSURE_GAIN_MAX,
    DAYLIGHT_HAZE_M,
    DAYLIGHT_SHADOW_CORE_M,
    DAYLIGHT_SKY_ELEVATION_TOLERANCE_DEG,
    DAYLIGHT_SUN_DIFFUSE_MAX,
)
from swarm.core.daylight import daylight_exposure, daylight_render_kwargs, max_elevation_deg, seeded_sun
from swarm.core.moving_drone import MovingDroneAviary
from swarm.core.sky_pack import SkyPhoto, load_sky_pack, pick_sky, sky_yaw_deg

FLAGS = {"ER_SWARM_DAYLIGHT": 8192, "ER_SWARM_SHADOW_MAP": 64, "ER_SWARM_MOVER_SHADOW": 128, "ER_EDGE_ANTIALIAS": 256,
         "ER_ALPHA_CUTOUT": 512, "ER_TEXTURE_FILTER": 16, "ER_SPECULAR_GLINT": 1024, "ER_SWARM_LINEAR_LIGHT": 4096}
PACK = [
    SkyPhoto("noon", "/skies/noon.png", 216.0, 49.9),
    SkyPhoto("afternoon", "/skies/afternoon.png", 216.0, 40.9),
    SkyPhoto("late", "/skies/late.png", 216.0, 19.2),
    SkyPhoto("sunset", "/skies/sunset.png", 215.8, 2.6),
]


def _day_seed():
    """A seed whose sun is a daytime sun."""
    seed = 3
    while seeded_sun(seed, 0.5).night:
        seed += 1
    return seed


def _env(monkeypatch, daylight=True, raycast=True, sky_from_sun=True, sun=None, seed=77):
    """A bare env with the family options given, after the daylight setup ran, with the engine flags faked."""
    for name, value in FLAGS.items():
        monkeypatch.setattr(p, name, value, raising=False)
    monkeypatch.setattr("swarm.core.moving_drone.load_sky_pack", lambda: PACK)
    env = MovingDroneAviary.__new__(MovingDroneAviary)
    env.family_runtime = SimpleNamespace(daylight=daylight, sky_from_sun=sky_from_sun)
    env._raycast_enabled = raycast
    env._sun = seeded_sun(seed) if sun is None else sun
    env._apply_daylight(seed)
    return env


def test_no_family_uses_the_daylight_model_today():
    """Every registered family keeps its light and its pixels."""
    assert ChallengeFamilyRuntime.daylight is False
    for family_id in list_registered_challenge_families():
        assert not get_challenge_family(family_id).daylight, family_id


def test_render_kwargs_carry_the_sun_and_the_film_settings():
    """The daylight arguments: the sun's colour and strength, a full shadow, exposure, haze and the core grid."""
    sun = seeded_sun(_day_seed())
    kwargs = daylight_render_kwargs(sun)
    assert kwargs["lightColor"] == list(sun.color)
    assert kwargs["lightAmbientCoeff"] == DAYLIGHT_AMBIENT
    assert 0.0 < kwargs["lightDiffuseCoeff"] <= DAYLIGHT_SUN_DIFFUSE_MAX
    assert kwargs["shadowLightCoeff"] == 0.0
    assert kwargs["exposure"] == daylight_exposure(sun.elevation_deg)
    assert DAYLIGHT_EXPOSURE <= kwargs["exposure"] <= DAYLIGHT_EXPOSURE * DAYLIGHT_EXPOSURE_GAIN_MAX
    assert kwargs["hazeDistance"] == DAYLIGHT_HAZE_M
    assert kwargs["shadowCoreRadius"] == DAYLIGHT_SHADOW_CORE_M


def test_exposure_opens_up_as_the_sun_sinks():
    """Noon keeps the base exposure; a sun near the horizon gets more, never past the gain cap."""
    assert daylight_exposure(max_elevation_deg()) == DAYLIGHT_EXPOSURE
    assert daylight_exposure(30.0) > DAYLIGHT_EXPOSURE
    assert daylight_exposure(3.0) > daylight_exposure(30.0)
    assert daylight_exposure(3.0) <= round(DAYLIGHT_EXPOSURE * DAYLIGHT_EXPOSURE_GAIN_MAX, 4)


def test_a_low_sun_is_weaker_than_a_high_one():
    """The air mass thins the daylight sun as it does the seeded sun."""
    low = next(seeded_sun(s) for s in range(1, 500) if seeded_sun(s).elevation_deg < 10)
    high = next(seeded_sun(s) for s in range(1, 500) if seeded_sun(s).elevation_deg > 50)
    assert daylight_render_kwargs(low)["lightDiffuseCoeff"] < daylight_render_kwargs(high)["lightDiffuseCoeff"]


def test_sky_yaw_puts_the_photo_sun_on_the_seeded_heading():
    """Turning the photo by its sun heading minus the seed's leaves the photo's sun on the seed's heading."""
    sun = SimpleNamespace(azimuth_deg=181.8, elevation_deg=34.0)
    photo = SkyPhoto("k48", "/skies/k48.png", 214.2, 47.9)
    assert sky_yaw_deg(photo, sun) == pytest.approx(32.4)
    assert 0.0 <= sky_yaw_deg(SkyPhoto("x", "/x.png", 10.0, 40.0), SimpleNamespace(azimuth_deg=350.0, elevation_deg=40.0)) < 360.0


def test_pick_sky_is_deterministic_and_prefers_a_matching_sun_height():
    """The same seed picks the same photo and turn; the photo's sun stands within the tolerance when one can."""
    for seed in range(1, 40):
        sun = seeded_sun(seed)
        photo, yaw = pick_sky(seed, sun, PACK)
        again, yaw_again = pick_sky(seed, sun, PACK)
        assert (photo, yaw) == (again, yaw_again)
        nearest = min(abs(candidate.sun_elevation_deg - sun.elevation_deg) for candidate in PACK)
        if nearest <= DAYLIGHT_SKY_ELEVATION_TOLERANCE_DEG:
            assert abs(photo.sun_elevation_deg - sun.elevation_deg) <= DAYLIGHT_SKY_ELEVATION_TOLERANCE_DEG
        assert yaw == sky_yaw_deg(photo, sun)


def test_pick_sky_falls_back_to_the_nearest_heights():
    """With no photo near the seed's sun height, the pick comes from the three nearest."""
    sun = SimpleNamespace(azimuth_deg=0.0, elevation_deg=80.0)
    photo, _ = pick_sky(5, sun, PACK)
    assert photo.name in {"noon", "afternoon", "late"}
    with pytest.raises(ValueError):
        pick_sky(5, sun, [])


def test_load_sky_pack_reads_the_manifest(tmp_path):
    """A pack folder is its manifest: file, name and the sun's heading and height per sky."""
    (tmp_path / "skies.json").write_text(json.dumps({"skies": [
        {"file": "a.png", "name": "a", "sun_azimuth_deg": 10.5, "sun_elevation_deg": 20.0},
        {"file": "b.png", "name": "b", "sun_azimuth_deg": 200.0, "sun_elevation_deg": 5.0},
    ]}))
    pack = load_sky_pack(str(tmp_path))
    assert [photo.name for photo in pack] == ["a", "b"]
    assert pack[0].path == str(tmp_path / "a.png")
    assert pack[1].sun_azimuth_deg == 200.0


def test_daylight_is_off_unless_the_family_asks(monkeypatch):
    """Without the option nothing changes: no flags, no arguments, no photo."""
    env = _env(monkeypatch, daylight=False)
    assert env._daylight_flags == 0
    assert env._daylight_sky is None
    assert env._daylight_kwargs(0) == {}


def test_daylight_needs_the_ray_caster_and_the_sun_sky(monkeypatch):
    """The option refuses a family that is not on the ray caster with the sun sky."""
    with pytest.raises(RuntimeError):
        _env(monkeypatch, raycast=False)
    with pytest.raises(RuntimeError):
        _env(monkeypatch, sky_from_sun=False)


def test_daylight_sets_every_picture_flag_and_picks_the_seeds_sky(monkeypatch):
    """A daytime seed turns on the daylight flag with the picture flags and carries the sky photo and its turn."""
    seed = _day_seed()
    env = _env(monkeypatch, seed=seed)
    assert env._daylight_flags == sum(FLAGS.values())
    photo, yaw = env._daylight_sky
    assert photo in PACK
    assert yaw == sky_yaw_deg(photo, env._sun)
    assert env._daylight_texture is None


def test_night_keeps_the_moon(monkeypatch):
    """A night seed renders as before: the moon's light and sky, no daylight model."""
    moon = next(seeded_sun(s, 1.0) for s in range(1, 50))
    assert moon.night
    env = _env(monkeypatch, sun=moon)
    assert env._daylight_flags == 0
    assert env._daylight_sky is None


def test_daylight_kwargs_load_the_photo_once_per_world(monkeypatch):
    """The photo is loaded into the engine on the first colour frame and reused until the world resets."""
    env = _env(monkeypatch, seed=_day_seed())
    loads = []
    monkeypatch.setattr(p, "loadTexture", lambda path, physicsClientId=0: loads.append(path) or 7)
    first = env._daylight_kwargs(0)
    second = env._daylight_kwargs(0)
    assert loads == [env._daylight_sky[0].path]
    assert first["skyTextureId"] == 7 and second["skyTextureId"] == 7
    assert first["skyYaw"] == env._daylight_sky[1]
    assert first["exposure"] == daylight_exposure(env._sun.elevation_deg)
    env._daylight_texture = None
    env._daylight_kwargs(0)
    assert len(loads) == 2


def test_daylight_refuses_a_wheel_without_the_flags(monkeypatch):
    """A wheel that lacks the daylight flag is named in the error."""
    monkeypatch.delattr(p, "ER_SWARM_DAYLIGHT", raising=False)
    monkeypatch.setattr("swarm.core.moving_drone.load_sky_pack", lambda: PACK)
    env = MovingDroneAviary.__new__(MovingDroneAviary)
    env.family_runtime = SimpleNamespace(daylight=True, sky_from_sun=True)
    env._raycast_enabled = True
    env._sun = seeded_sun(_day_seed())
    with pytest.raises(RuntimeError, match="ER_SWARM_DAYLIGHT"):
        env._apply_daylight(1)
