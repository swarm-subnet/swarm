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

"""The colour camera and gimbal (task 10): the M4TD's lenses, two frames a second, one feed at a time, the view at
every tilt with the aircraft's own body above +70 degrees, and night mode."""

from __future__ import annotations

import dataclasses
import math
import os

import numpy as np
import pybullet as p
import pytest
import swarm_worlds

from swarm.challenge_families.solar_patrol import airframe, camera, park
from swarm.challenge_families.solar_patrol.contract import (
    DECISION_STEPS,
    RGB_SHAPE,
    STATE_SLICES,
    THERMAL_SHAPE,
    decode_action,
    new_state,
)
from swarm.challenge_families.solar_patrol.episode import SolarEpisode
from swarm.constants import MOON_AMBIENT_RANGE, MOON_DIFFUSE_RANGE, SIM_DT
from swarm.core.daylight import seeded_sun, sky_render_kwargs
from validator.tests.test_solar_patrol_family import _action, _patrol, flat_park  # noqa: F401

_M4TD_SHIPPED = os.path.isfile(os.path.join(swarm_worlds.robots_dir(), airframe.URDF))
_HEIGHT_M = 20.0
_RED = [0.9, 0.1, 0.1, 1.0]


class _Env:
    """The environment fields the camera reads, over a bare world with the aircraft level at 20 m."""

    def __init__(self, cli: int, drone: int, sun=None):
        """Hold the client, the aircraft and the seed's light."""
        self.CLIENT = cli
        self.DRONE_IDS = [drone]
        self.pos = [np.array([0.0, 0.0, _HEIGHT_M])]
        self.quat = [np.array([0.0, 0.0, 0.0, 1.0])]
        self._sun = sun
        self._render_flags = p.ER_SWARM_RAYCAST if camera.RAYCAST else 0
        self._sky_flags = 0
        self._daylight_flags = 0
        self._light_direction = [0.3, 0.2, 0.9]

    def _daylight_kwargs(self, cli: int) -> dict:
        """No daylight model."""
        return {}

    def _sky_kwargs(self) -> dict:
        """The moon's dark sky at night, as the environment hands it over, and none by day."""
        return sky_render_kwargs(self._sun) or {}


def _box(cli: int, half: list, position: list, rgba: list) -> int:
    """A static coloured box."""
    visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=rgba, physicsClientId=cli)
    return p.createMultiBody(0, -1, visual, position, physicsClientId=cli)


@pytest.fixture
def scene():
    """Grey ground with a red block straight under an invisible aircraft 20 m up, and a fresh patrol."""
    cli = p.connect(p.DIRECT)
    _box(cli, [100.0, 100.0, 0.1], [0.0, 0.0, -0.1], [0.5, 0.5, 0.5, 1.0])
    _box(cli, [3.0, 3.0, 0.5], [0.0, 0.0, 0.5], _RED)
    drone = p.createMultiBody(1.0, p.createCollisionShape(p.GEOM_SPHERE, radius=0.1, physicsClientId=cli), -1,
                              [0.0, 0.0, _HEIGHT_M], physicsClientId=cli)
    env = _Env(cli, drone)
    ep = SolarEpisode(seed=11)
    ep.park = park.heat(env._sun, ep.seed, 0)
    camera.reset(env, ep)
    yield env, ep
    p.disconnect(cli)


def _ask(env, ep, tilt=0.0, thermal=False, night_mode="off"):
    """Ask the camera for a tilt in degrees, a feed and a night mode, as the outputs part hands them over."""
    a = _action(gimbal_tilt=tilt / 90.0, thermal=float(thermal),
                night_mode={"off": 0.1, "on": 0.5, "auto": 0.9}[night_mode])
    camera.request(env, ep, decode_action(a, None))


def _centre(frame: np.ndarray) -> np.ndarray:
    """The colour at the middle of a frame."""
    h, w = frame.shape[:2]
    return frame[h // 2, w // 2]


def _is_red(colour: np.ndarray) -> bool:
    """True for the red block's colour, whatever its shading."""
    return bool(colour[0] > 0.25 and colour[0] > 2.0 * colour[1] and colour[0] > 2.0 * colour[2])


def test_the_lenses_are_the_m4td_s():
    """Wide 82 degrees across the diagonal of 640 x 480 is 69.6 x 55.1; thermal 45 degrees of 640 x 512 is 35.8 x 29.0."""
    for diagonal, (height, width), across, up in ((camera.WIDE_DIAGONAL_FOV_DEG, RGB_SHAPE[:2], 69.6, 55.1),
                                                  (camera.THERMAL_DIAGONAL_FOV_DEG, THERMAL_SHAPE[:2], 35.8, 29.0)):
        vertical = camera.vertical_fov_deg(diagonal, width, height)
        horizontal = 2.0 * math.degrees(math.atan(math.tan(math.radians(vertical / 2.0)) * width / height))
        assert vertical == pytest.approx(up, abs=0.05)
        assert horizontal == pytest.approx(across, abs=0.05)
    assert (camera.WIDE_DIAGONAL_FOV_DEG, camera.THERMAL_DIAGONAL_FOV_DEG) == (82.0, 45.0)
    assert RGB_SHAPE == (480, 640, 3) and THERMAL_SHAPE == (512, 640, 1)


def test_two_frames_a_second_on_decision_boundaries():
    """A frame every 0.5 s, which is every fifth decision."""
    assert camera.FRAME_STEPS * SIM_DT == pytest.approx(0.5)
    assert camera.FRAME_STEPS % DECISION_STEPS == 0


def test_the_ray_caster_draws_whenever_the_engine_has_it():
    """The family asks for the ray caster exactly when the installed engine carries it."""
    assert camera.RENDER_BACKEND == ("raycast" if hasattr(p, "ER_SWARM_RAYCAST") else "tiny")


@pytest.mark.parametrize("name", camera.NEEDED_FLAGS)
def test_a_ray_caster_without_a_needed_flag_is_refused(monkeypatch, name):
    """A ray-cast wheel missing a flag the frames rest on fails at reset instead of drawing another picture."""
    monkeypatch.setattr(camera, "RAYCAST", True)
    for other in camera.NEEDED_FLAGS:
        monkeypatch.setattr(p, other, getattr(p, other, 1), raising=False)
    monkeypatch.delattr(p, name, raising=False)
    with pytest.raises(RuntimeError, match=name):
        camera.reset(None, SolarEpisode(seed=0))


def test_a_close_look_smooths_every_edge_and_fills_creases_when_the_engine_can():
    """A zoom's flags drop outline-only smoothing and ask for crease fill exactly when the installed engine carries it."""
    outline = getattr(p, "ER_SWARM_EDGE_OUTLINE", 0)
    fill = getattr(p, "ER_SWARM_CREASE_FILL", 0)
    assert not camera.CLOSE_PICTURE_FLAGS & outline
    assert bool(camera.CLOSE_PICTURE_FLAGS & fill) == bool(camera.RAYCAST and fill)
    assert camera.CLOSE_PICTURE_FLAGS & ~(outline | fill) == camera.PICTURE_FLAGS & ~(outline | fill)


def test_edges_are_smoothed_on_outlines_only_when_the_engine_can():
    """The colour camera asks for outline-only edge smoothing exactly when the installed engine carries it."""
    outline = getattr(p, "ER_SWARM_EDGE_OUTLINE", 0)
    assert bool(camera.PICTURE_FLAGS & outline) == bool(camera.RAYCAST and outline)


def test_the_tilt_runs_from_straight_down_to_straight_up(scene):
    """The model's -1 to +1 is -90 to +90 degrees."""
    env, ep = scene
    for tilt in (-90.0, -45.0, 0.0, 70.0, 90.0):
        _ask(env, ep, tilt=tilt)
        assert ep.camera["tilt_deg"] == pytest.approx(tilt, abs=1e-4)


def test_the_frame_follows_the_tilt(scene):
    """Straight down the block below fills the middle of the frame; level, it is out of view."""
    env, ep = scene
    _ask(env, ep, tilt=-90.0)
    camera.capture(env, ep)
    down = ep.frames.rgb
    assert down.shape == RGB_SHAPE and down.dtype == np.float32
    assert 0.0 <= float(down.min()) and float(down.max()) <= 1.0
    assert _is_red(_centre(down))
    _ask(env, ep, tilt=0.0)
    camera.capture(env, ep)
    assert not _is_red(_centre(ep.frames.rgb))
    assert np.count_nonzero(np.apply_along_axis(_is_red, 2, ep.frames.rgb[::16, ::16])) == 0


def test_the_view_is_the_wide_lens_where_the_frame_was_taken(scene):
    """The kept view is the airframe's camera pose at the tilt, through the 82 degree lens, at the frame's step."""
    env, ep = scene
    ep.step = 25
    _ask(env, ep, tilt=-30.0)
    camera.capture(env, ep)
    shot = camera.view(ep)
    eye, forward, up = airframe.camera_pose(env, -30.0)
    assert shot.feed == "colour" and shot.step == 25 and shot.sees
    assert np.allclose(shot.eye, eye) and np.allclose(shot.forward, forward) and np.allclose(shot.up, up)
    assert (shot.width, shot.height) == (640, 480)
    assert shot.vertical_fov_deg == pytest.approx(camera.vertical_fov_deg(82.0, 640, 480))
    assert forward[2] == pytest.approx(-math.sin(math.radians(30.0)), abs=1e-6)


def test_one_feed_at_a_time(scene):
    """The picked feed carries the frame and the other image is zero."""
    env, ep = scene
    _ask(env, ep, tilt=-90.0)
    camera.capture(env, ep)
    assert np.any(ep.frames.rgb) and not np.any(ep.frames.thermal)
    _ask(env, ep, tilt=-90.0, thermal=True)
    camera.capture(env, ep)
    assert ep.frames.thermal.shape == THERMAL_SHAPE and ep.frames.thermal.dtype == np.float32
    assert not np.any(ep.frames.rgb)
    assert camera.view(ep).feed == "thermal" and camera.view(ep).width == 640 and camera.view(ep).height == 512
    if camera.THERMAL:
        assert np.any(ep.frames.thermal) and float(ep.frames.thermal.max()) <= 1.0


def test_frames_arrive_twice_a_second_and_a_switch_shows_from_the_next(scene):
    """The first frame comes with the first observation, then one every 25 control steps; a feed picked between
    frames shows from the next frame, and the state says which feed and how old the frame shown is."""
    env, ep = scene
    state = new_state()
    camera.observe(env, ep, state)
    assert ep.camera["captures"] == 1
    _ask(env, ep, thermal=True)
    for step in range(1, 51):
        ep.step = step
        camera.update(env, ep)
        camera.observe(env, ep, state)
        feed = state[STATE_SLICES["camera_feed"]][0]
        age = state[STATE_SLICES["frame_age_s"]][0]
        if step < 25:
            assert feed == 0.0 and age == pytest.approx(step * SIM_DT)
        else:
            assert feed == 1.0 and age == pytest.approx((step % 25) * SIM_DT)
    assert ep.camera["captures"] == 3


def test_the_same_view_draws_the_same_pixels(scene):
    """Two frames of the same view are identical."""
    env, ep = scene
    _ask(env, ep, tilt=-60.0)
    camera.capture(env, ep)
    first = ep.frames.rgb.copy()
    camera.capture(env, ep)
    assert np.array_equal(first, ep.frames.rgb)


@pytest.mark.skipif(not hasattr(p, "ER_SWARM_FRAME_REUSE"), reason="engine without frame reuse")
def test_a_still_camera_gets_the_frames_it_would_draw(scene, monkeypatch):
    """Colour, night and thermal frames ask the engine for a still camera's last frame when it can hand one back, and
    each frame equals the one drawn without asking, grain included; the fork's own test shows the frame is handed back."""
    env, ep = scene
    reuse = getattr(p, "ER_SWARM_FRAME_REUSE", 0)
    assert camera.FRAME_REUSE == reuse
    for thermal, night_mode in ((False, "off"), (False, "on"), (True, "off")):
        if night_mode == "on":
            _moon(env, 1.0)
        _ask(env, ep, tilt=-60.0, thermal=thermal, night_mode=night_mode)
        for flag in (reuse, 0, reuse, 0, reuse, 0):
            monkeypatch.setattr(camera, "FRAME_REUSE", flag)
            ep.camera["captures"] = 5
            camera.capture(env, ep)
            frame = (ep.frames.rgb.copy(), ep.frames.thermal.copy(), camera.view(ep).objects.copy())
            if flag:
                asked = frame
            else:
                assert all(np.array_equal(a, b) for a, b in zip(asked, frame))


@pytest.mark.skipif(not hasattr(p, "ER_SWARM_RASTER"), reason="engine without the painted frame")
def test_every_frame_is_painted(scene, monkeypatch):
    """Wide, close, night and thermal frames all ask the engine to paint what each dot sees."""
    env, ep = scene
    raster = getattr(p, "ER_SWARM_RASTER", 0)
    assert camera.PICTURE_FLAGS & raster and camera.CLOSE_PICTURE_FLAGS & raster
    asked = []
    real = p.getCameraImage

    def spy(*args, **kwargs):
        """getCameraImage, noting the flags each frame asks for."""
        asked.append(kwargs.get("flags", 0))
        return real(*args, **kwargs)

    monkeypatch.setattr(p, "getCameraImage", spy)
    for thermal, night_mode in ((False, "off"), (False, "on"), (True, "off")):
        if night_mode == "on":
            _moon(env, 1.0)
        _ask(env, ep, tilt=-60.0, thermal=thermal, night_mode=night_mode)
        camera.capture(env, ep)
    assert len(asked) >= 3 and all(flags & raster for flags in asked)


def _moonlit(env, seed=11):
    """Light the scene by the seed's moon."""
    env._sun = seeded_sun(seed, 1.0)
    assert env._sun.night


def _moon(env, phase, elevation_deg=45.0):
    """Light the scene by a moon of the given phase, 0 a crescent and 1 full, at the given height."""
    low, high = MOON_DIFFUSE_RANGE
    ambient_low, ambient_high = MOON_AMBIENT_RANGE
    up = math.radians(elevation_deg)
    env._sun = dataclasses.replace(seeded_sun(11, 1.0), diffuse=low + (high - low) * phase,
                                   ambient=ambient_low + (ambient_high - ambient_low) * phase,
                                   direction=(0.0, math.cos(up), math.sin(up)))


def test_at_night_night_mode_off_is_dark_and_sees_nothing(scene):
    """With night mode off the frame stays dark and the view counts as seeing nothing; night mode brightens it."""
    env, ep = scene
    _moonlit(env)
    _ask(env, ep, tilt=-90.0, night_mode="off")
    camera.capture(env, ep)
    assert not camera.view(ep).sees
    dark = ep.frames.rgb.copy()
    _ask(env, ep, tilt=-90.0, night_mode="on")
    camera.capture(env, ep)
    assert camera.view(ep).sees
    if camera.LOW_LIGHT:
        assert float(ep.frames.rgb.mean()) > 2.0 * float(dark.mean())
    else:
        assert np.array_equal(ep.frames.rgb, dark)


def test_auto_night_mode_turns_on_in_the_dark_only(scene):
    """Auto is on under the moon, drawing the frame night mode draws, and changes nothing by day."""
    env, ep = scene
    _ask(env, ep, tilt=-90.0, night_mode="off")
    camera.capture(env, ep)
    day = ep.frames.rgb.copy()
    for mode in ("on", "auto"):
        _ask(env, ep, tilt=-90.0, night_mode=mode)
        camera.capture(env, ep)
        assert np.array_equal(ep.frames.rgb, day) and camera.view(ep).sees
    _moonlit(env)
    frames = []
    for mode in ("on", "auto"):
        ep.camera["captures"] = 7
        _ask(env, ep, tilt=-90.0, night_mode=mode)
        camera.capture(env, ep)
        assert camera.view(ep).sees
        frames.append(ep.frames.rgb.copy())
    assert np.array_equal(frames[0], frames[1])


def test_the_thermal_feed_sees_at_night(scene):
    """Thermal needs no light, so a thermal frame at night counts as seeing."""
    env, ep = scene
    _moonlit(env)
    _ask(env, ep, thermal=True, night_mode="off")
    camera.capture(env, ep)
    assert camera.view(ep).sees


def test_moonlight_follows_the_phase_and_the_height():
    """A full moon at the zenith gives 0.25 lux plus the sky's glow; a crescent and a low moon give far less."""
    env = _Env(0, 0)
    _moon(env, 1.0, 90.0)
    assert camera.moon_lux(env._sun) == pytest.approx(camera.FULL_MOON_ZENITH_LUX + camera.SKY_GLOW_LUX)
    full_high = camera.moon_lux(env._sun)
    _moon(env, 1.0, 30.0)
    assert camera.moon_lux(env._sun) == pytest.approx(0.5 * camera.FULL_MOON_ZENITH_LUX + camera.SKY_GLOW_LUX)
    _moon(env, 0.0, 90.0)
    assert camera.moon_lux(env._sun) < 0.1 * full_high


@pytest.mark.skipif(not camera.LOW_LIGHT, reason="engine without the low-light camera")
def test_the_night_camera_follows_the_lens_the_mode_and_the_moon():
    """The 7x lens collects a quarter of the wide lens's light, Night Scene doubles the exposure and lifts the ISO cap
    16 times, a full moon gives far more light than a crescent, and every frame draws its own grain."""
    env = _Env(0, 0)
    _moon(env, 1.0)
    shot = camera.View(feed="colour", eye=(0.0, 0.0, 20.0), forward=(0.0, 0.0, -1.0), up=(0.0, 1.0, 0.0), width=640,
                       height=480, vertical_fov_deg=55.1, sees=True, step=0)
    flags, off = camera.night_camera(env, 1, shot, False, 11, 0)
    assert flags == p.ER_SWARM_LOW_LIGHT
    _flags, wide = camera.night_camera(env, 1, shot, True, 11, 0)
    _flags, tele = camera.night_camera(env, 7, shot, True, 11, 0)
    assert wide["sensorPhotons"] == pytest.approx(2.0 * off["sensorPhotons"], rel=1e-5)
    assert wide["sensorGainCap"] == pytest.approx(32.0 * off["sensorGainCap"], rel=1e-5)
    assert tele["sensorPhotons"] / wide["sensorPhotons"] == pytest.approx((1.7 / 2.8) ** 2 * 48.8 / 70.3, rel=1e-4)
    assert camera.night_camera(env, 1, shot, True, 11, 1)[1]["sensorSeed"] != wide["sensorSeed"]
    _moon(env, 0.0)
    assert camera.night_camera(env, 1, shot, True, 11, 0)[1]["sensorPhotons"] < 0.1 * wide["sensorPhotons"]
    env._sun = None
    assert camera.night_camera(env, 1, shot, True, 11, 0) == (0, {})


@pytest.mark.skipif(not camera.LOW_LIGHT, reason="engine without the low-light camera")
def test_night_mode_frames_carry_grain_that_moves(scene):
    """Two night mode frames of the same view share their brightness but not their grain, and a crescent's frame is
    darker and grainier than a full moon's."""
    env, ep = scene
    _moon(env, 1.0)
    _ask(env, ep, tilt=-90.0, night_mode="on")
    camera.capture(env, ep)
    first = ep.frames.rgb.copy()
    camera.capture(env, ep)
    assert not np.array_equal(first, ep.frames.rgb)
    assert float(first.mean()) == pytest.approx(float(ep.frames.rgb.mean()), abs=0.01)
    ground = (slice(0, 60), slice(0, 60))
    _moon(env, 0.0, 15.0)
    camera.capture(env, ep)
    crescent = ep.frames.rgb
    assert float(crescent.mean()) < float(first.mean())
    assert float(crescent[ground].std() / crescent[ground].mean()) > 2.0 * float(first[ground].std() / first[ground].mean())


@pytest.mark.skipif(not _M4TD_SHIPPED, reason=f"the installed swarm-worlds has no {airframe.URDF} yet")
def test_the_aircraft_blocks_the_view_above_seventy_degrees():
    """The body is out of view up to +60 degrees, covers about a third of the frame at +70 and all of it at +90."""
    cli = p.connect(p.DIRECT)
    try:
        robots = swarm_worlds.robots_dir()
        drone = p.loadURDF(os.path.join(robots, airframe.URDF), [0.0, 0.0, _HEIGHT_M], physicsClientId=cli)
        moving = p.loadURDF(os.path.join(robots, airframe.MOVING_URDF), [0.0, 0.0, _HEIGHT_M], physicsClientId=cli)
        joints = {p.getJointInfo(moving, j, physicsClientId=cli)[1].decode(): j
                  for j in range(p.getNumJoints(moving, physicsClientId=cli))}
        env = _Env(cli, drone)
        ep = SolarEpisode(seed=11)
        ep.park = park.heat(env._sun, ep.seed, 0)
        camera.reset(env, ep)
        shares = {}
        for tilt in (-90.0, 0.0, 60.0, 70.0, 90.0):
            p.resetJointState(moving, joints["gimbal_tilt"], math.radians(tilt), physicsClientId=cli)
            _ask(env, ep, tilt=tilt)
            camera.capture(env, ep)
            shot = camera.view(ep)
            view_matrix, projection = shot.matrices()
            seg = p.getCameraImage(shot.width, shot.height, view_matrix, projection, renderer=p.ER_TINY_RENDERER,
                                   flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX | env._render_flags,
                                   physicsClientId=cli)[4]
            ids = np.asarray(seg).reshape(shot.height, shot.width)
            ids = np.where(ids < 0, -1, ids & ((1 << 24) - 1))
            shares[tilt] = float(np.isin(ids, [drone, moving]).mean())
        assert shares[-90.0] == 0.0 and shares[0.0] == 0.0
        assert shares[60.0] < 0.1
        assert 0.2 < shares[70.0] < 0.5
        assert shares[90.0] > 0.99
    finally:
        p.disconnect(cli)


def test_a_patrol_sees_through_the_camera(flat_park):  # noqa: F811
    """On a real patrol a new frame of the picked feed arrives every fifth decision and the other feed stays zero."""
    def pilot(i, obs):
        """Take off, look down, and switch to thermal after 12 s."""
        return _action(take_off=float(i == 0), gimbal_tilt=-0.5, thermal=float(i >= 120))

    log = _patrol(3, pilot, max_decisions=200)
    fresh = 0
    for obs in log["observations"]:
        state = obs["state"]
        feed = state[STATE_SLICES["camera_feed"]][0]
        assert 0.0 <= state[STATE_SLICES["frame_age_s"]][0] < 0.5 - 1e-6
        if feed == 0.0:
            assert np.any(obs["rgb"]) and not np.any(obs["thermal"])
        else:
            assert not np.any(obs["rgb"])
        # A new frame reaches the model within the decision it was taken in, the link's delay after it.
        fresh += int(state[STATE_SLICES["frame_age_s"]][0] < DECISION_STEPS * SIM_DT - 1e-6)
    assert fresh == 1 + 200 // 5
    assert log["observations"][-1]["state"][STATE_SLICES["camera_feed"]][0] == 1.0
