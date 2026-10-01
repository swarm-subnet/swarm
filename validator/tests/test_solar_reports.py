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

"""Solar Patrol reports: every report judged against the object map of the exact frame the model saw.

The frames here are real renders of stand-in people and a dog on flat ground, seen straight down from 20 m through the
wide lens. Each box is worked out from the camera's geometry, never from the object map it is judged against.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pybullet as p
import pytest

from swarm.challenge_families.solar_patrol import camera, decoys, reports, sensor_noise, theft
from swarm.challenge_families.solar_patrol.contract import STATE_SLICES, Box, Report
from swarm.challenge_families.solar_patrol.episode import SolarEpisode
from validator.tests.test_solar_patrol_family import _action
from validator.tests.test_solar_patrol_family import flat_park as _flat_park  # noqa: F401
from validator.tests.test_solar_zoom import _TILT, _Flight

_PERSON = (0.25, 0.15, 0.9)                     # half extents of a standing person, metres
_DOG = (0.45, 0.15, 0.3)
_FRAME_STEP = 7


def _block(cli, centre, half, colour):
    """A static visual box standing on the ground at centre (x, y); returns its body id and world corners."""
    visual = p.createVisualShape(p.GEOM_BOX, halfExtents=list(half), rgbaColor=list(colour) + [1.0], physicsClientId=cli)
    uid = p.createMultiBody(0, -1, visual, [centre[0], centre[1], half[2]], physicsClientId=cli)
    lo = np.array([centre[0] - half[0], centre[1] - half[1], 0.0])
    return uid, (lo, lo + 2.0 * np.asarray(half))


def _box_of(view, corners, grow=1.0, shift=(0.0, 0.0)):
    """The box, as shares of the image, around a world box's corners seen through a view; grown by a factor and
    shifted by a share of its own width and height."""
    view_matrix, projection = view.matrices()
    to_clip = np.asarray(projection).reshape(4, 4).T @ np.asarray(view_matrix).reshape(4, 4).T
    lo, hi = corners
    points = np.array([[x, y, z, 1.0] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    clip = points @ to_clip.T
    u = (clip[:, 0] / clip[:, 3] + 1.0) / 2.0
    v = (1.0 - clip[:, 1] / clip[:, 3]) / 2.0
    w, h = u.max() - u.min(), v.max() - v.min()
    return Box(cx=(u.min() + u.max()) / 2.0 + shift[0] * w, cy=(v.min() + v.max()) / 2.0 + shift[1] * h, w=w * grow, h=h * grow)


def _draw(cli, view):
    """The view with the object map of one frame drawn from it."""
    view_matrix, projection = view.matrices()
    _w, _h, _rgb, _depth, seg = p.getCameraImage(view.width, view.height, view_matrix, projection,
                                                 renderer=p.ER_TINY_RENDERER, physicsClientId=cli)
    return replace(view, objects=np.reshape(np.asarray(seg, dtype=np.int32), (view.height, view.width)))


@pytest.fixture
def park(monkeypatch):
    """Two thieves (the second built from two bodies, standing outside the fence), a dog and empty ground, drawn
    once from 20 m; the thieves and decoys parts answer for them, and the patrol was shown that frame."""
    cli = p.connect(p.DIRECT)
    ground = p.createVisualShape(p.GEOM_BOX, halfExtents=[60.0, 60.0, 0.05], rgbaColor=[0.3, 0.5, 0.2, 1.0], physicsClientId=cli)
    p.createMultiBody(0, -1, ground, [0.0, 0.0, -0.05], physicsClientId=cli)
    inside_uid, inside = _block(cli, (-3.0, 2.0), _PERSON, (0.6, 0.2, 0.2))
    legs_uid, legs = _block(cli, (3.0, 2.0), (0.25, 0.15, 0.45), (0.2, 0.2, 0.6))
    torso = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.25, 0.15, 0.45], rgbaColor=[0.6, 0.6, 0.2, 1.0], physicsClientId=cli)
    torso_uid = p.createMultiBody(0, -1, torso, [3.0, 2.0, 1.35], physicsClientId=cli)
    outside = (legs[0], legs[1] + np.array([0.0, 0.0, 0.9]))
    dog_uid, dog = _block(cli, (0.0, -3.0), _DOG, (0.5, 0.35, 0.2))
    view = camera.View(feed="colour", eye=(0.0, 0.0, 20.0), forward=(0.0, 0.0, -1.0), up=(0.0, 1.0, 0.0), width=640,
                       height=480, vertical_fov_deg=camera.vertical_fov_deg(camera.WIDE_DIAGONAL_FOV_DEG, 640, 480),
                       sees=True, step=_FRAME_STEP)
    shown = {"feed": _draw(cli, view), "zoom": None}
    asked = []

    def inside_fence(ep, thief, step):
        """Thief 0 stands inside the fence at the frame's step and nowhere else; thief 1 never does."""
        asked.append(step)
        return thief == 0 and step == _FRAME_STEP

    monkeypatch.setattr(sensor_noise, "shown_view", lambda ep, image: shown[image])
    monkeypatch.setattr(theft, "bodies", lambda ep: {inside_uid: 0, legs_uid: 1, torso_uid: 1})
    monkeypatch.setattr(theft, "inside", inside_fence)
    monkeypatch.setattr(decoys, "bodies", lambda ep: {dog_uid: "dog"})
    ep = SolarEpisode(seed=0, step=50)
    reports.reset(None, ep)
    yield SimpleNamespace(ep=ep, view=shown["feed"], shown=shown, inside=inside, outside=outside, dog=dog, asked=asked)
    p.disconnect(cli)


def _report(park, corners, kind="person", image="feed", **fit):
    """Submit one report boxed on those corners and return its verdict."""
    reports.submit(None, park.ep, Report(kind, image, _box_of(park.view, corners, **fit)))
    return park.ep.reports["log"][-1][2]


def test_a_box_on_a_thief_inside_the_fence_is_a_valid_report(park):
    """A person boxed where the frame shows him, inside the fence at the frame's moment, counts as valid."""
    assert _report(park, park.inside) == "valid"
    assert park.ep.outcome.valid_reports == 1 and park.ep.outcome.false_alarms == 0


def test_a_second_report_of_the_same_thief_is_a_false_alarm(park):
    """Reporting a thief already reported is rejected and costs a false alarm, even on a better box."""
    assert _report(park, park.inside) == "valid"
    assert _report(park, park.inside, grow=1.05) == "repeat"
    assert park.ep.outcome.valid_reports == 1 and park.ep.outcome.false_alarms == 1
    assert park.ep.outcome.reports_made == 2


def test_a_box_on_a_dog_is_a_false_alarm(park):
    """A dog is a decoy: reporting it as a person fails on the class."""
    assert _report(park, park.dog) == "wrong_class"
    assert park.ep.outcome.false_alarms == 1


def test_a_box_on_empty_ground_is_a_false_alarm(park):
    """A box where nothing stands sits on no object."""
    empty = (np.array([-1.0, -1.0, 0.0]), np.array([-0.5, 0.0, 1.8]))
    assert _report(park, empty) == "no_object"


def test_a_person_outside_the_fence_is_no_threat(park):
    """A person drawn from two bodies is one object, and outside the fence he is not a threat."""
    assert _report(park, park.outside) == "not_threat"
    assert park.asked == [_FRAME_STEP]


def test_the_threat_check_reads_the_frame_moment_not_the_patrol_clock(park):
    """Whether a thief stood inside the fence is asked for the frame's step, which the patrol has long passed."""
    assert _report(park, park.inside) == "valid"
    assert park.asked == [_FRAME_STEP] and park.ep.step != _FRAME_STEP


def test_the_vehicle_class_never_passes(park):
    """V1 has no vehicle class: the right box with the vehicle class is a false alarm."""
    assert _report(park, park.inside, kind="vehicle") == "wrong_class"


@pytest.mark.parametrize("fit, verdict", [
    ({"shift": (0.1, 0.1)}, "valid"),           # overlap 0.68
    ({"grow": 1.3}, "valid"),                   # 0.59
    ({"grow": 1.6}, "no_object"),               # 0.39
    ({"shift": (0.4, 0.0)}, "no_object"),       # 0.43
])
def test_the_box_must_overlap_the_thief_by_half(park, fit, verdict):
    """A box a little off or a little large still counts; one far too large or shifted by much of the thief does not."""
    assert _report(park, park.inside, **fit) == verdict


def test_a_report_before_any_frame_is_a_false_alarm(park):
    """With no zoom frame shown yet, a report on the zoom image has nothing to be read against."""
    assert _report(park, park.inside, image="zoom") == "no_frame"
    assert park.ep.outcome.false_alarms == 1


def test_a_report_on_the_zoom_frame_reads_the_zoom_frame(park):
    """A report on the zoom image is judged on the zoom frame's own map, and one thief is reported only once
    whichever image the model boxed him on."""
    park.shown["zoom"] = park.view
    assert _report(park, park.inside, image="zoom") == "valid"
    assert _report(park, park.inside, image="feed") == "repeat"


def test_the_object_map_leaves_the_frame_untouched(monkeypatch):
    """Asking the engine for the object map changes no pixel of the colour frame."""
    cli = p.connect(p.DIRECT)
    try:
        _block(cli, (0.0, 0.0), _PERSON, (0.6, 0.2, 0.2))
        env = SimpleNamespace(CLIENT=cli, _sun=None, _render_flags=0, _sky_flags=0, _daylight_flags=0,
                              _light_direction=[0.3, 0.2, 1.0], _daylight_kwargs=lambda client: {}, _sky_kwargs=dict)
        view = camera.View(feed="colour", eye=(0.0, -8.0, 12.0), forward=(0.0, 0.55, -0.83), up=(0.0, 0.83, 0.55),
                           width=640, height=480, vertical_fov_deg=50.0, sees=True, step=0)
        frame, objects = camera.colour_frame(env, view)
        draw = p.getCameraImage

        def without_map(*args, **kwargs):
            """The same draw with the object map switched off."""
            kwargs["flags"] |= p.ER_NO_SEGMENTATION_MASK
            return draw(*args, **kwargs)

        monkeypatch.setattr(p, "getCameraImage", without_map)
        plain, _ = camera.colour_frame(env, view)
        assert np.array_equal(frame, plain)
        assert objects.shape == (480, 640) and (objects >= 0).any()
    finally:
        p.disconnect(cli)


@pytest.mark.timeout(300)
@pytest.mark.usefixtures("_flat_park")
def test_a_thief_who_moved_after_the_frame_still_counts_where_the_frame_showed_him(monkeypatch):
    """In a real patrol, through the report button and the link's delay, the report is read against the frame the
    model saw: a thief who walked off after it was taken counts where the frame showed him, and where he stands now
    that frame shows nothing."""
    flight = _Flight(seed=21)
    env, ep = flight.env, flight.env._solar
    x, y, _z = np.asarray(env.pos[0], dtype=float)
    yaw = float(env.rpy[0][2])
    ahead = np.array([np.cos(yaw), np.sin(yaw)])
    spot = np.array([x, y]) + 20.0 * ahead
    uid, before = _block(env.CLIENT, spot, _PERSON, (0.6, 0.2, 0.2))
    monkeypatch.setattr(theft, "bodies", lambda ep: {uid: 0})
    monkeypatch.setattr(theft, "inside", lambda ep, thief, step: True)
    flight.step(_action(gimbal_tilt=_TILT))
    while flight.obs["state"][STATE_SLICES["frame_age_s"]][0] > 0.05:
        flight.step(_action(gimbal_tilt=_TILT))
    seen = sensor_noise.shown_view(ep, "feed")
    step = np.append(3.0 * np.array([-ahead[1], ahead[0]]), 0.0)
    p.resetBasePositionAndOrientation(uid, [spot[0] + step[0], spot[1] + step[1], _PERSON[2]], [0, 0, 0, 1],
                                      physicsClientId=env.CLIENT)
    try:
        assert reports.boxed_object(seen, _box_of(seen, (before[0] + step, before[1] + step)), {uid: ("thief", 0)}) is None
        box = _box_of(seen, before)
        flight.step(_action(gimbal_tilt=_TILT, report=1.0, report_cx=box.cx, report_cy=box.cy, report_w=box.w,
                            report_h=box.h))
        assert [verdict for _step, _report, verdict in ep.reports["log"]] == ["valid"]
        assert ep.outcome.valid_reports == 1 and ep.outcome.false_alarms == 0
    finally:
        env.close()
