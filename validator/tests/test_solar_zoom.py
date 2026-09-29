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

"""Solar Patrol zoom lenses: the 3x and 7x close-up answers the box the model drew, one decision later.

The patrols here fly the M4TD over the flat stand-in park and put a red marker on the ground in view of the wide
camera; the model's box is drawn on the marker as it shows in the wide frame, and the zoom frame must centre it.
"""
from __future__ import annotations

import contextlib
import io
import math
from types import SimpleNamespace

import numpy as np
import pybullet as p
import pytest

from swarm.challenge_families import build_benchmark_tasks
from swarm.challenge_families.solar_patrol import camera, zoom
from swarm.challenge_families.solar_patrol.contract import (
    FAMILY_ID,
    MAX_ZOOMS,
    STATE_SLICES,
    ZOOM_SHAPE,
    Box,
    decode_action,
)
from swarm.challenge_families.solar_patrol.episode import SolarEpisode
from swarm.constants import SIM_DT
from swarm.utils.env_factory import make_env_with_initial_obs
from validator.tests.test_solar_patrol_family import _M4TD_SHIPPED, _action
from validator.tests.test_solar_patrol_family import flat_park as _flat_park  # noqa: F401

_TILT = -0.5                                   # gimbal 45 degrees down
_CENTRE = (ZOOM_SHAPE[1] / 2.0, ZOOM_SHAPE[0] / 2.0)
_CENTRED_PX = 8.0


def _view(vertical_fov_deg=40.0, width=640, height=480):
    """A camera looking north from 20 m up, level, with the given lens and image size."""
    return camera.View(feed="colour", eye=(0.0, 0.0, 20.0), forward=(0.0, 1.0, 0.0), up=(0.0, 0.0, 1.0),
                       width=width, height=height, vertical_fov_deg=vertical_fov_deg, sees=True, step=0)


def _red(frame):
    """Centre (x, y) in pixels and size in pixels of the red marker in a frame, or None when it is not there."""
    mask = (frame[..., 0] > 0.25) & (frame[..., 0] > 3.0 * frame[..., 1]) & (frame[..., 0] > 3.0 * frame[..., 2])
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    return float(xs.mean() + 0.5), float(ys.mean() + 0.5), int(mask.sum())


class _Flight:
    """An M4TD patrol flown decision by decision, hovering at the patrol height with the gimbal 45 degrees down."""

    def __init__(self, seed):
        """Build the patrol, take off, and wait until the dock hands the aircraft over."""
        if not _M4TD_SHIPPED:
            pytest.skip("the installed swarm-worlds has no M4TD yet")
        task = build_benchmark_tasks(sim_dt=SIM_DT, seeds=[seed], family_id=FAMILY_ID)[0]
        with contextlib.redirect_stdout(io.StringIO()):
            self.env, self.obs = make_env_with_initial_obs(task)
        self.step(_action(take_off=1.0))
        while int(self.obs["state"][STATE_SLICES["flight_phase"]][0]) != 2:
            self.step(_action())
        for _ in range(20):
            self.step(_action(gimbal_tilt=_TILT))

    def step(self, action):
        """One decision; the patrol must still be running after it."""
        self.obs, _reward, terminated, truncated, _info = self.env.step(action[None, :])
        assert not (terminated or truncated)
        return self.obs

    def place_marker(self, ahead_m, left_m, **flying):
        """Stand a red 1 m block on the ground this far ahead of the aircraft and to its left, then fly on until the
        wide camera takes its next frame."""
        cli = self.env.CLIENT
        x, y, height = np.asarray(self.env.pos[0], dtype=float)
        yaw = float(self.env.rpy[0][2])
        spot = [x + ahead_m * math.cos(yaw) - left_m * math.sin(yaw), y + ahead_m * math.sin(yaw) + left_m * math.cos(yaw), 0.5]
        shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5], physicsClientId=cli)
        visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5], rgbaColor=[0.9, 0.05, 0.05, 1.0],
                                     physicsClientId=cli)
        p.createMultiBody(0, shape, visual, spot, physicsClientId=cli)
        self.step(_action(gimbal_tilt=_TILT, **flying))
        while self.obs["state"][STATE_SLICES["frame_age_s"]][0] > 0.05:
            self.step(_action(gimbal_tilt=_TILT, **flying))

    def zoom_on_marker(self, lens, **flying):
        """Box the marker where the wide frame shows it, press zoom with a lens, and return the observation that
        brings the close-up: the second after the press, once the request and the frame have crossed the link."""
        found = _red(self.obs["rgb"])
        assert found is not None, "the marker is not in the wide frame"
        self.step(_action(gimbal_tilt=_TILT, zoom=1.0, zoom_lens=float(lens == 7), zoom_cx=found[0] / 640.0,
                          zoom_cy=found[1] / 480.0, zoom_w=0.05, zoom_h=0.05, **flying))
        return self.step(_action(gimbal_tilt=_TILT, **flying))

    def close(self):
        """Close the environment."""
        self.env.close()


def test_the_tele_lenses_see_the_decided_angles():
    """3x covers 35 degrees and 7x 15 degrees across the diagonal of the 640 x 480 zoom frame."""
    height, width = ZOOM_SHAPE[:2]
    for lens, diagonal in ((3, 35.0), (7, 15.0)):
        half_v = math.tan(math.radians(camera.vertical_fov_deg(zoom.LENS_DIAGONAL_FOV_DEG[lens], width, height)) / 2)
        assert 2 * math.degrees(math.atan(half_v * math.hypot(width, height) / height)) == pytest.approx(diagonal)
    assert camera.vertical_fov_deg(35.0, width, height) == pytest.approx(21.43, abs=0.01)
    assert camera.vertical_fov_deg(15.0, width, height) == pytest.approx(9.03, abs=0.01)


def test_a_box_turns_into_the_ray_through_its_centre():
    """The centre of the frame looks straight ahead, and each edge half the lens's angle off it, top up and left left."""
    seen = _view(vertical_fov_deg=40.0)
    assert zoom.box_ray(seen, Box(0.5, 0.5, 0.1, 0.1)) == pytest.approx([0.0, 1.0, 0.0])
    top = zoom.box_ray(seen, Box(0.5, 0.0, 0.0, 0.0))
    assert math.degrees(math.atan2(top[2], top[1])) == pytest.approx(20.0)
    left = zoom.box_ray(seen, Box(0.0, 0.5, 0.0, 0.0))
    half_h = math.degrees(math.atan(math.tan(math.radians(20.0)) * 640 / 480))
    assert math.degrees(math.atan2(-left[0], left[1])) == pytest.approx(half_h)
    assert zoom.box_ray(seen, Box(0.5, 0.5, 0.0, 0.0)) == pytest.approx(zoom.box_ray(seen, Box(0.5, 0.5, 1.0, 1.0)))


def test_the_patrol_has_80_zooms_and_each_keeps_the_frame_it_was_boxed_on():
    """Every press up to 80 is queued with the view of the frame shown; the 81st is refused and changes nothing."""
    env = SimpleNamespace()
    ep = SolarEpisode(seed=0)
    camera.reset(env, ep)
    zoom.reset(env, ep)
    ep.camera["view"] = _view()
    press = decode_action(_action(zoom=1.0, zoom_lens=1.0, zoom_cx=0.2, zoom_cy=0.7), None)
    for i in range(MAX_ZOOMS):
        ep.step = i
        zoom.request(env, ep, press)
    assert ep.outcome.zooms_used == MAX_ZOOMS
    assert ep.zoom["pending"] == press.zoom and ep.zoom["seen"] == _view()
    ep.step = MAX_ZOOMS
    zoom.request(env, ep, press)
    assert ep.outcome.zooms_used == MAX_ZOOMS and ep.zoom["pending_step"] == MAX_ZOOMS - 1


def test_night_vision_is_black_and_white_and_lit_only_inside_its_beam():
    """In the dark the 5.7 degree beam lights the middle of the 7x frame within 100 m and leaves the rest dim; by day
    the picture is only turned black and white."""
    shot = _view(vertical_fov_deg=camera.vertical_fov_deg(15.0, 640, 480))
    frame = np.full((480, 640, 3), [0.1, 0.05, 0.02], dtype=np.float32)
    grey = float(np.dot([0.1, 0.05, 0.02], zoom.LUMA))
    near = zoom._night_vision(frame, shot, 40.0, dark=True)
    assert np.array_equal(near[..., 0], near[..., 1]) and np.array_equal(near[..., 1], near[..., 2])
    assert near[240, 320, 0] == pytest.approx(grey * zoom.NIGHT_VISION_GAIN)
    assert near[0, 0, 0] == pytest.approx(grey * zoom.NIGHT_VISION_OUTSIDE)
    beam_px = math.tan(math.radians(zoom.NIGHT_VISION_BEAM_DEG / 2)) / math.tan(math.radians(shot.vertical_fov_deg / 2)) * 240
    lit_rows = np.nonzero(near[:, 320, 0] > grey)[0]
    assert (lit_rows[-1] - lit_rows[0] + 1) / 2 == pytest.approx(beam_px, abs=1.0)
    assert np.allclose(zoom._night_vision(frame, shot, 140.0, dark=True), grey * zoom.NIGHT_VISION_OUTSIDE)
    assert np.allclose(zoom._night_vision(frame, shot, 40.0, dark=False), grey)


@pytest.mark.timeout(300)
@pytest.mark.usefixtures("_flat_park")
def test_a_zoom_centres_what_was_boxed_two_decisions_later():
    """Nothing shows before the press; the second observation after it holds the close-up, with the boxed marker in
    its middle on both lenses and 7x showing it (7/3)^2 larger in area than 3x, as the lenses' angles give."""
    flight = _Flight(seed=21)
    try:
        flight.place_marker(ahead_m=20.0, left_m=5.0)
        assert not flight.obs["zoom"].any()
        assert flight.obs["state"][STATE_SLICES["zoom_lens"]][0] == 0
        sizes = {}
        for lens in (3, 7):
            pressed_step = flight.env._solar.step
            obs = flight.zoom_on_marker(lens)
            assert obs["state"][STATE_SLICES["zoom_lens"]][0] == lens
            # The request reaches the aircraft a control step after the press; the frame is drawn a decision later.
            assert obs["state"][STATE_SLICES["zoom_age_s"]][0] <= SIM_DT + 1e-6
            assert zoom.view(flight.env._solar).step == pressed_step + 6
            found = _red(obs["zoom"])
            assert found is not None
            assert math.hypot(found[0] - _CENTRE[0], found[1] - _CENTRE[1]) < _CENTRED_PX
            sizes[lens] = found[2]
            flight.step(_action(gimbal_tilt=_TILT))
        tele, medium = (math.tan(math.radians(camera.vertical_fov_deg(zoom.LENS_DIAGONAL_FOV_DEG[k], 640, 480)) / 2)
                        for k in (7, 3))
        assert sizes[7] / sizes[3] == pytest.approx((medium / tele) ** 2, rel=0.1)
        assert flight.env._solar.outcome.zooms_used == 2
    finally:
        flight.close()


@pytest.mark.timeout(300)
@pytest.mark.usefixtures("_flat_park")
def test_a_zoom_boxed_on_an_old_frame_while_flying_and_turning_still_finds_its_target():
    """The box drawn on a wide frame 0.4 s old, pressed while the aircraft flies at 3 m/s and turns at 45 deg/s,
    still brings the marker to the middle of the 7x frame."""
    flight = _Flight(seed=22)
    flying = {"move_forward": 0.6, "turn": -0.5}
    try:
        flight.place_marker(ahead_m=24.0, left_m=6.0, **flying)
        while flight.obs["state"][STATE_SLICES["frame_age_s"]][0] < 0.35:
            flight.step(_action(gimbal_tilt=_TILT, **flying))
        found = _red(flight.zoom_on_marker(7, **flying)["zoom"])
        assert found is not None
        assert math.hypot(found[0] - _CENTRE[0], found[1] - _CENTRE[1]) < _CENTRED_PX
    finally:
        flight.close()


@pytest.mark.timeout(300)
@pytest.mark.usefixtures("_flat_park")
def test_the_same_patrol_draws_the_same_zoom():
    """Two runs of one seed with the same actions give the same zoom frame, pixel for pixel."""
    frames = []
    for _ in range(2):
        flight = _Flight(seed=23)
        try:
            flight.place_marker(ahead_m=20.0, left_m=-4.0)
            frames.append(flight.zoom_on_marker(3)["zoom"].copy())
        finally:
            flight.close()
    assert frames[0].any() and np.array_equal(frames[0], frames[1])
