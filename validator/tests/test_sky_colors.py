"""The sky background is opt-in per family: nobody has one today, and an env only passes the
sky arguments to the camera when its family returns one."""

from swarm.challenge_families import get_challenge_family, list_registered_challenge_families
from swarm.challenge_families.base import ChallengeFamilyRuntime
from swarm.core.moving_drone import MovingDroneAviary


def _env_with_sky(sky):
    """A bare env carrying only the sky attribute, enough for the kwargs helper."""
    env = MovingDroneAviary.__new__(MovingDroneAviary)
    env._sky_colors = sky
    return env


def test_no_family_paints_a_sky_today():
    """Every registered family keeps the white background, so existing pixels are untouched."""
    for family_id in list_registered_challenge_families():
        assert get_challenge_family(family_id).sky_colors(task=None) is None


def test_base_runtime_defaults_to_no_sky():
    """A new family inherits the white background unless it overrides sky_colors."""
    assert ChallengeFamilyRuntime().sky_colors(task=None) is None


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
