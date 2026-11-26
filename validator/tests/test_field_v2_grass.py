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

"""Test Field v2's grass: stood in hidden patches, and drawn only for the pictures whose view can show it."""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from swarm.challenge_families.solar_patrol import camera, zoom
from swarm.challenge_families.solar_patrol.contract import RGB_SHAPE, THERMAL_SHAPE, ZOOM_SHAPE
from swarm.challenge_families.test_field_v2 import cover, grass
from swarm.challenge_families.test_field_v2.contract import TAKE_OFF_M
from swarm.challenge_families.test_field_v2.family import TestFieldV2ChallengeFamily
from swarm.core.maps.andorra.builder import Patch
from validator.scripts import generate_video

SIDE = 5                               # the test lawn is SIDE x SIDE patches on flat ground at height 0
TOP_M = 0.4                            # the top of every patch's grass


def _lawn() -> grass.Lawn:
    """A square of patches on flat ground, body n * SIDE + m at column n and row m, centred on the origin."""
    half = SIDE * grass.CELL_M / 2.0
    patches = [Patch(n * SIDE + m, (n * grass.CELL_M - half, m * grass.CELL_M - half, -0.1),
                     ((n + 1) * grass.CELL_M - half, (m + 1) * grass.CELL_M - half, TOP_M))
               for n in range(SIDE) for m in range(SIDE)]
    return grass.Lawn(bodies=np.array([patch.body for patch in patches]), low=np.array([patch.low for patch in patches]),
                      high=np.array([patch.high for patch in patches]), shown=np.zeros(len(patches), dtype=bool))


def _view(eye, forward=(0.0, 0.0, -1.0), feed="colour", diagonal=camera.WIDE_DIAGONAL_FOV_DEG, shape=RGB_SHAPE) -> camera.View:
    """A picture's view from a place, straight down unless told, through a lens named by its diagonal field of view."""
    length = math.sqrt(sum(c * c for c in forward))
    forward = tuple(c / length for c in forward)
    up = (0.0, 1.0, 0.0) if abs(forward[2]) > 0.9 else (0.0, 0.0, 1.0)
    height, width = shape[:2]
    return camera.View(feed=feed, eye=tuple(float(c) for c in eye), forward=forward, up=up, width=width, height=height,
                       vertical_fov_deg=camera.vertical_fov_deg(diagonal, width, height), sees=True, step=0)


def _middle(lawn: grass.Lawn) -> np.ndarray:
    """The middle of each patch across the ground."""
    return (lawn.low[:, :2] + lawn.high[:, :2]) / 2.0


def test_the_wide_lens_shows_grass_only_from_low_over_it_and_near():
    """From the height a patrol takes off to, the wide lens shows no grass at all; from low over the ground it shows
    every patch within reach across the ground whichever way it looks, and none further."""
    lawn = _lawn()
    assert TAKE_OFF_M - TOP_M > grass.HEIGHT_M
    assert not grass.wanted(lawn, _view((0.0, 0.0, TAKE_OFF_M))).any()
    assert not grass.wanted(lawn, _view((0.0, 0.0, TOP_M + grass.HEIGHT_M + 0.01))).any()
    for eye in ((0.0, 0.0, TOP_M + grass.HEIGHT_M - 0.01), (-SIDE * grass.CELL_M / 2.0 - grass.REACH_M + grass.CELL_M / 2.0, 0.0, 3.0)):
        low = grass.wanted(lawn, _view(eye))
        across = np.linalg.norm(np.maximum(np.maximum(lawn.low[:, :2] - eye[:2], eye[:2] - lawn.high[:, :2]), 0.0), axis=1)
        assert 0 < low.sum() < len(low) and np.array_equal(low, across <= grass.REACH_M)
        assert np.array_equal(low, grass.wanted(lawn, _view(eye, forward=(-1.0, 0.0, 0.0))))


def test_the_thermal_camera_never_shows_grass():
    """Through the thermal camera no patch is drawn, however low and near the camera is."""
    lawn = _lawn()
    view = _view((0.0, 0.0, 1.5), feed="thermal", diagonal=camera.THERMAL_DIAGONAL_FOV_DEG, shape=THERMAL_SHAPE)
    assert not grass.wanted(lawn, view).any()


def test_a_long_lens_shows_the_grass_it_looks_at_from_the_patrols_height():
    """A zoom lens magnifies, so it shows grass from the patrol's height that the wide lens does not, but only the
    patches its narrow frame can hold: the one it looks at among them, never one behind the camera."""
    lawn = _lawn()
    eye = (-20.0, 0.0, TAKE_OFF_M)
    target = np.array([18.0, 6.0, 0.0])
    for lens, diagonal in zoom.LENS_DIAGONAL_FOV_DEG.items():
        view = _view(eye, forward=tuple(target - eye), diagonal=diagonal, shape=ZOOM_SHAPE)
        assert grass.magnification(view) == pytest.approx(lens, rel=0.12)
        shown = grass.wanted(lawn, view)
        aimed = np.argmin(np.linalg.norm(_middle(lawn) - target[:2], axis=1))
        assert shown[aimed] and 0 < shown.sum() < len(shown) / 2
        assert not shown[_middle(lawn)[:, 0] < eye[0] - grass.CELL_M].any()
    assert grass.magnification(_view(eye)) == 1.0


def test_only_the_patches_that_change_are_switched(monkeypatch):
    """Showing a view draws exactly the patches it wants and touches only those whose state changes; no view and a
    seed without grass switch nothing."""
    switched = []
    monkeypatch.setattr(grass.p, "changeVisualShape", lambda body, link, rgbaColor, physicsClientId: switched.append((body, rgbaColor[3])))
    lawn = _lawn()
    env, ep = SimpleNamespace(CLIENT=0), SimpleNamespace(park={"grass": lawn})
    low = _view((-SIDE * grass.CELL_M / 2.0 - grass.REACH_M + grass.CELL_M / 2.0, 0.0, 3.0))
    want = grass.wanted(lawn, low)
    grass.show(env, ep, low)
    assert sorted(switched) == [(int(body), 1.0) for body in lawn.bodies[want]] and np.array_equal(lawn.shown, want)
    switched.clear()
    grass.show(env, ep, low)
    grass.show(env, ep, None)
    assert switched == []
    grass.show(env, ep, _view((0.0, 0.0, TAKE_OFF_M)))
    assert sorted(switched) == [(int(body), 0.0) for body in lawn.bodies[want]] and not lawn.shown.any()
    switched.clear()
    grass.show(env, SimpleNamespace(park=None), low)
    grass.show(env, SimpleNamespace(park={}), low)
    assert switched == []


def test_a_reading_meets_the_grass_the_wide_lens_shows_from_there_whatever_was_drawn_last(monkeypatch):
    """Shown for a reading from a point, the lawn is exactly what the wide colour lens shows from there, whether the
    last picture left every patch drawn or none; hidden for a measure, no patch is drawn."""
    monkeypatch.setattr(grass.p, "changeVisualShape", lambda body, link, rgbaColor, physicsClientId: None)
    lawn = _lawn()
    env, ep = SimpleNamespace(CLIENT=0), SimpleNamespace(park={"grass": lawn})
    eye = (-SIDE * grass.CELL_M / 2.0 - grass.REACH_M + grass.CELL_M / 2.0, 0.0, 3.0)
    want = grass.wanted(lawn, _view(eye))
    assert 0 < want.sum() < len(want)
    for left in (np.zeros(len(want), dtype=bool), np.ones(len(want), dtype=bool)):
        lawn.shown = left
        grass.show_near(env, ep, eye)
        assert np.array_equal(lawn.shown, want)
    grass.show_near(env, ep, (0.0, 0.0, TAKE_OFF_M))
    assert not lawn.shown.any()
    lawn.shown = np.ones(len(want), dtype=bool)
    grass.hide(env, ep)
    assert not lawn.shown.any()
    grass.hide(env, SimpleNamespace(park=None))
    grass.show_near(env, SimpleNamespace(park={}), eye)


@pytest.mark.parametrize("family_id", ["cf_test_field_v2", "cf_solar_patrol", "cf_test_field"])
def test_a_recording_updates_test_field_v2_grass_from_its_own_eye(monkeypatch, family_id):
    """Only the v2 recording switches grass, using the recording eye before the draw."""
    calls = []
    eye = np.array([1.0, 2.0, 3.0])
    runtime = TestFieldV2ChallengeFamily()
    runtime.family_id = family_id
    cam = generate_video._CameraBase(0, 640, 480, 50)
    cam.solar_env = SimpleNamespace(family_runtime=runtime, _solar=object())
    monkeypatch.setattr(grass, "show_near", lambda env, ep, point: calls.append(("grass", tuple(point))))
    monkeypatch.setattr(generate_video, "_solar_frame", lambda env, point, *args: calls.append(("frame", tuple(point))))
    cam._shoot(eye, (0.0, 0.0, 0.0))
    if family_id == "cf_test_field_v2":
        assert calls == [("grass", tuple(eye)), ("frame", tuple(eye))]
    else:
        assert calls == [("frame", tuple(eye))]


def test_what_a_place_shows_of_someone_is_measured_on_bare_ground(monkeypatch):
    """The frame that counts someone's pixels from a place is drawn after every patch is hidden, so the count is the
    same at a seed's build and in a flight."""
    calls = []
    monkeypatch.setattr(cover, "SIGHT_FLAGS", 0)
    monkeypatch.setattr(cover.camera, "far_plane_m", lambda ep: 100.0)
    monkeypatch.setattr(cover.grass, "hide", lambda env, ep: calls.append("bare ground"))
    monkeypatch.setattr(cover.camera, "colour_frame",
                        lambda env, shot: calls.append("frame") or (None, np.full((shot.height, shot.width), -1)))
    seen = cover.object_map(None, None, np.array([0.0, 0.0, 9.5]), np.array([3.0, 4.0, 1.0]), cover.FRAME_PX)
    assert calls == ["bare ground", "frame"] and seen.shape == (cover.FRAME_PX, cover.FRAME_PX) and (seen == -1).all()


def test_laid_grass_starts_hidden(monkeypatch):
    """Every patch an area load stood is hidden when the lawn is laid, and a load with no patch gives an empty lawn."""
    switched = []
    monkeypatch.setattr(grass.p, "changeVisualShape", lambda body, link, rgbaColor, physicsClientId: switched.append((body, rgbaColor[3])))
    patches = (Patch(7, (0.0, 0.0, 0.0), (12.0, 12.0, 0.4)), Patch(9, (12.0, 0.0, 0.0), (24.0, 12.0, 0.5)))
    lawn = grass.lay(0, patches)
    assert switched == [(7, 0.0), (9, 0.0)] and not lawn.shown.any()
    assert lawn.bodies.tolist() == [7, 9] and lawn.high[1].tolist() == [24.0, 12.0, 0.5]
    empty = grass.lay(0, ())
    assert len(empty.bodies) == 0 and empty.low.shape == (0, 3)
    assert not grass.wanted(empty, _view((0.0, 0.0, 2.0))).any()
