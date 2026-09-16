"""The sky background is opt-in per family: nobody has one today, and an env only passes the
sky arguments to the camera when its family returns one or asks for the sky of its sun."""

from types import SimpleNamespace

import pybullet as p
import pytest

from swarm.challenge_families import get_challenge_family, list_registered_challenge_families
from swarm.challenge_families.base import ChallengeFamilyRuntime
from swarm.core.daylight import seeded_sun, sky_render_kwargs
from swarm.core.moving_drone import MovingDroneAviary

SKY_SUN_FLAG = 2048


def _env_with_sky(sky):
    """A bare env carrying only the sky attribute, enough for the kwargs helper."""
    env = MovingDroneAviary.__new__(MovingDroneAviary)
    env._sky_colors = sky
    return env


def _env_for_sun(monkeypatch, sun, sky_from_sun=True, sky_clouds=False, seed=77):
    """A bare env whose family opted in as given, lit by the sun, after the sun sky setup ran."""
    monkeypatch.setattr(p, "ER_SWARM_SKY_SUN", SKY_SUN_FLAG, raising=False)
    env = _env_with_sky(None)
    env.family_runtime = SimpleNamespace(sky_from_sun=sky_from_sun, sky_clouds=sky_clouds)
    env._sun = sun
    env._apply_sun_sky(seed)
    return env


def test_no_family_paints_a_sky_today():
    """Every registered family keeps the white background, so existing pixels are untouched."""
    for family_id in list_registered_challenge_families():
        family = get_challenge_family(family_id)
        assert family.sky_colors(task=None) is None
        assert not family.sky_from_sun and not family.sky_clouds, family_id


def test_base_runtime_defaults_to_no_sky():
    """A new family inherits the white background unless it overrides sky_colors."""
    assert ChallengeFamilyRuntime().sky_colors(task=None) is None
    assert ChallengeFamilyRuntime.sky_from_sun is False
    assert ChallengeFamilyRuntime.sky_clouds is False


def test_no_sky_means_no_camera_arguments():
    """With no sky the camera call gets nothing extra, exactly as before."""
    assert _env_with_sky(None)._sky_kwargs() == {}


def test_sky_becomes_the_two_camera_colours():
    """A family sky reaches getCameraImage as the horizon and zenith colours."""
    env = _env_with_sky(((0.85, 0.55, 0.25), (0.15, 0.35, 0.95)))
    assert env._sky_kwargs() == {
        "skyHorizonColor": [0.85, 0.55, 0.25],
        "skyZenithColor": [0.15, 0.35, 0.95],
    }


def test_sun_sky_needs_the_family_flag_and_a_sun(monkeypatch):
    """Without the flag, or without a seeded sun, the camera flags and arguments stay as before."""
    assert _env_for_sun(monkeypatch, seeded_sun(3), sky_from_sun=False)._sky_flags == 0
    assert _env_for_sun(monkeypatch, None)._sky_flags == 0
    assert _env_for_sun(monkeypatch, None)._sky_kwargs() == {}


def test_sun_sky_sets_the_renderer_flag(monkeypatch):
    """A family that opted in renders with the sun sky flag and no cloud seed."""
    env = _env_for_sun(monkeypatch, seeded_sun(3))
    assert env._sky_flags == SKY_SUN_FLAG
    assert env._sky_kwargs() == {}


def test_clouds_carry_the_map_seed(monkeypatch):
    """With clouds on, the map seed reaches the renderer as the cloud seed."""
    env = _env_for_sun(monkeypatch, seeded_sun(3), sky_clouds=True, seed=4242)
    assert env._sky_flags == SKY_SUN_FLAG
    assert env._sky_kwargs() == {"skyCloudSeed": 4242}


def test_night_paints_the_moon_sky_and_no_daylight_one(monkeypatch):
    """A seed whose light is a moon never gets the daylight sky or its clouds: the renderer flag
    stays off and the camera takes the moon's own dark gradient instead."""
    moon = seeded_sun(3, night_share=1.0)
    env = _env_for_sun(monkeypatch, moon, sky_clouds=True)
    assert env._sky_flags == 0
    assert env._sky_kwargs() == sky_render_kwargs(moon)
    assert "skyCloudSeed" not in env._sky_kwargs()


def test_sun_sky_refuses_a_wheel_without_the_flag(monkeypatch):
    """An old wheel cannot paint the sun sky, so the env says so instead of rendering white."""
    monkeypatch.delattr(p, "ER_SWARM_SKY_SUN", raising=False)
    env = _env_with_sky(None)
    env.family_runtime = SimpleNamespace(sky_from_sun=True, sky_clouds=False)
    env._sun = seeded_sun(3)
    with pytest.raises(RuntimeError, match="ER_SWARM_SKY_SUN"):
        env._apply_sun_sky(3)
