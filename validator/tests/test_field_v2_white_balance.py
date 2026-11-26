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

"""The colour camera's white balance hook: off for Swarm Sentinel and for Test Field v2 on its areas, whose frames stay
byte for byte as drawn, and with the field's measured table, which takes the cast it was measured on out of a grey
patch in linear light."""
from __future__ import annotations

import contextlib
import io
import math

import numpy as np
import pybullet as p
import pytest

from swarm.challenge_families.solar_patrol import camera
from swarm.challenge_families.solar_patrol.episode import SolarEpisode
from swarm.challenge_families.test_field_v2 import install, site
from swarm.challenge_families.test_field_v2 import task as field_task
from swarm.challenge_families.test_field_v2.episode import FieldEpisode
from swarm.challenge_families.test_field_v2.family import TestFieldV2ChallengeFamily
from swarm.challenge_families.test_field_v2.white_balance import WHITE_BALANCE
from swarm.constants import SIM_DT
from swarm.core.daylight import seeded_sun
from swarm.utils.env_factory import make_env_with_initial_obs
from validator.tests import area_stand_in, field_v2_stand_in
from validator.tests.test_field_v2_family import _SHIPPED

_UNITY = tuple((row[0], 1.0, 1.0, 1.0) for row in WHITE_BALANCE)


def _cast(grey: int, gains: tuple) -> np.ndarray:
    """One RGBA pixel of a grey byte carrying the cast the gains take out: each channel divided by its gain in linear
    light."""
    return np.array([[[round(255.0 * math.pow(math.pow(grey / 255.0, 2.2) / g, 1.0 / 2.2)) for g in gains] + [255]]],
                    dtype=np.uint8)


def test_neither_sentinel_nor_the_field_on_its_areas_sets_a_table():
    """Neither episode carries a table, so no colour frame is touched: the field's table was measured under another
    light than the areas are matched under."""
    assert SolarEpisode(seed=1).white_balance is None
    assert FieldEpisode(seed=1).white_balance is None


def test_the_gains_follow_the_sun_and_hold_at_the_ends_of_the_table():
    """The gains are the table's at a listed elevation, interpolated between rows and held past the first and last."""
    assert camera.white_balance_gains(WHITE_BALANCE, 8.0) == pytest.approx((0.937, 1.098, 1.167))
    assert camera.white_balance_gains(WHITE_BALANCE, 20.0) == pytest.approx((0.828, 0.871, 0.867))
    assert camera.white_balance_gains(WHITE_BALANCE, 35.0) == pytest.approx(((0.773 + 0.769) / 2, (0.773 + 0.751) / 2, (0.739 + 0.708) / 2))
    assert camera.white_balance_gains(WHITE_BALANCE, 2.0) == camera.white_balance_gains(WHITE_BALANCE, 8.0)
    assert camera.white_balance_gains(WHITE_BALANCE, 80.0) == pytest.approx((0.753, 0.707, 0.642))


def test_unity_gains_change_no_byte():
    """Through gains of one, decoding and encoding again gives every byte back."""
    assert np.array_equal(camera.white_balance_bytes((1.0, 1.0, 1.0)), np.tile(np.arange(256, dtype=np.uint8), (3, 1)))


@pytest.mark.parametrize("elevation", [row[0] for row in WHITE_BALANCE])
def test_a_grey_patch_carrying_the_measured_cast_comes_out_neutral(elevation):
    """A 50 % grey carrying the cast the table was measured on comes out grey under a low or high sun, from RGBA
    bytes and from float frames alike, the alpha kept."""
    gains = camera.white_balance_gains(WHITE_BALANCE, elevation)
    warm = _cast(128, gains)
    out = camera.balanced(warm, gains)
    assert int(out[..., :3].max()) - int(out[..., :3].min()) <= 1 and out[0, 0, 3] == 255
    floats = camera.balanced(np.true_divide(warm[..., :3], np.float32(255.0), dtype=np.float32), gains)
    assert floats.dtype == np.float32 and np.array_equal(np.rint(floats * 255.0).astype(np.uint8), out[..., :3])


@pytest.fixture
def low_sun_field(tmp_path, monkeypatch):
    """A field on the stand-in area under a sun lower than 15 degrees, out of the area light that never sets one so
    low, with a 50 % grey patch beside it, and a view straight down onto the patch."""
    if not (_SHIPPED and camera.RAYCAST and site.DAYLIGHT):
        pytest.skip("the white balance acts on the engine's daylight model, which this install lacks")
    monkeypatch.setattr(install, "area_problems", lambda: [])
    field_v2_stand_in.use_area(tmp_path, monkeypatch)
    monkeypatch.setattr(TestFieldV2ChallengeFamily, "area_light", False)
    field_v2_stand_in.use_people(tmp_path / "people", monkeypatch)
    seed = next(s for s in range(500) if not site.night(s) and seeded_sun(s).elevation_deg < 15.0)
    with contextlib.redirect_stdout(io.StringIO()):
        env, _ = make_env_with_initial_obs(field_task.test_field_task(seed=seed, sim_dt=SIM_DT))
    look = p.createVisualShape(p.GEOM_BOX, halfExtents=[3.0, 3.0, 0.05], rgbaColor=[0.5, 0.5, 0.5, 1.0],
                               physicsClientId=env.CLIENT)
    # In the stand-in's border, where no field, dock or person ever reaches.
    ground = area_stand_in.ground_z(0.0)
    p.createMultiBody(0, -1, look, [0.0, -75.0, ground + 0.06], physicsClientId=env.CLIENT)
    view = camera.View("colour", (0.0, -75.0, ground + 7.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0), 64, 48, 20.0, True, 0)
    yield env, view
    env.close()


def test_with_the_hook_off_a_frame_is_byte_for_byte_as_drawn(low_sun_field):
    """Without a table the balance is never run, and the frame equals one drawn through gains of one byte for byte."""
    env, view = low_sun_field
    env._solar.white_balance = None
    # Its own patch, so undoing it leaves the field's light as the fixture set it.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(camera, "balanced", lambda *args: pytest.fail("a frame without a table was balanced"))
        drawn, _ = camera.colour_frame(env, view)
    env._solar.white_balance = _UNITY
    through_unity, _ = camera.colour_frame(env, view)
    assert drawn.tobytes() == through_unity.tobytes()


def test_with_the_hook_on_the_field_table_is_applied_to_the_drawn_frame(low_sun_field):
    """Under a low sun the field's table maps every byte of the drawn frame through its gains at the sun's elevation,
    taking red down and blue up against green on the grey patch."""
    env, view = low_sun_field
    env._solar.white_balance = None
    drawn, _ = camera.colour_frame(env, view)
    env._solar.white_balance = WHITE_BALANCE
    balanced, _ = camera.colour_frame(env, view)
    gains = camera.white_balance_gains(WHITE_BALANCE, env._sun.elevation_deg)
    assert balanced.tobytes() == camera.balanced(drawn, gains).tobytes()
    patch_before, patch_after = drawn[12:36, 16:48].reshape(-1, 3).mean(0), balanced[12:36, 16:48].reshape(-1, 3).mean(0)
    assert patch_after[0] / patch_after[1] < patch_before[0] / patch_before[1]
    assert patch_after[2] / patch_after[1] > patch_before[2] / patch_before[1]
