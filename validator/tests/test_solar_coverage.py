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

"""Coverage cells (task 22): the seeded grid inside the fence, the ground under each cell, which frames mark cells,
and the share searched over a whole patrol."""

from __future__ import annotations

import math
import os
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pybullet as p
import pytest
from shapely.geometry import Polygon

from swarm.challenge_families.solar_patrol import coverage, park
from swarm.challenge_families.solar_patrol.camera import (
    THERMAL_DIAGONAL_FOV_DEG,
    WIDE_DIAGONAL_FOV_DEG,
    View,
    vertical_fov_deg,
)
from swarm.challenge_families.solar_patrol.contract import SITE_MAP_SLICES, STATE_SLICES
from swarm.challenge_families.solar_patrol.episode import SolarEpisode
from swarm.core.maps.solar.builder import SOLAR_ASSET_DIR
from validator.tests.test_solar_patrol_family import _FENCE, _action, _patrol, blank_camera, flat_park  # noqa: F401

_SOLAR_ASSETS = os.environ.get("SOLAR_ASSET_DIR", SOLAR_ASSET_DIR)
_SQUARE = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]])
_WIDE_VFOV = vertical_fov_deg(WIDE_DIAGONAL_FOV_DEG, 640, 480)
_THERMAL_VFOV = vertical_fov_deg(THERMAL_DIAGONAL_FOV_DEG, 640, 512)
# The height straight above flat ground where one pixel of the wide camera spans exactly MAX_PIXEL_M.
_WIDE_LIMIT_M = coverage.MAX_PIXEL_M * 480 / (2.0 * math.tan(math.radians(_WIDE_VFOV) / 2.0))

# Coverage reads where a frame was taken, never its pixels, so the flights skip drawing them.
pytestmark = pytest.mark.usefixtures("blank_camera")


def _slab(cli, half, centre):
    """A static box with a collision shape."""
    shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=cli)
    return p.createMultiBody(0, shape, -1, centre, physicsClientId=cli)


@pytest.fixture
def ground():
    """Flat ground at 0 m under the square fence, a table top 2 m over one corner, and a patrol with its grid laid."""
    cli = p.connect(p.DIRECT)
    terrain = _slab(cli, [80.0, 80.0, 0.5], [50.0, 50.0, -0.5])
    _slab(cli, [5.0, 5.0, 0.05], [10.0, 10.0, 2.0])
    ep = SolarEpisode(seed=4, fence=_SQUARE, terrain_uids=frozenset({terrain}), dock_position=np.array([50.0, 50.0, 0.4]))
    coverage.reset(SimpleNamespace(CLIENT=cli), ep)
    yield ep
    p.disconnect(cli)


def _down(east, north, up, feed="colour", sees=True, step=0):
    """A frame taken straight down from a point, its top edge towards the north."""
    vfov, size = (_WIDE_VFOV, (640, 480)) if feed == "colour" else (_THERMAL_VFOV, (640, 512))
    return View(feed=feed, eye=(east, north, up), forward=(0.0, 0.0, -1.0), up=(0.0, 1.0, 0.0), width=size[0],
                height=size[1], vertical_fov_deg=vfov, sees=sees, step=step)


def _looking(east, north, up, pitch_deg):
    """A wide frame looking east from a point, pitched down by the angle."""
    pitch = math.radians(pitch_deg)
    return View(feed="colour", eye=(east, north, up), forward=(math.cos(pitch), 0.0, math.sin(pitch)),
                up=(-math.sin(pitch), 0.0, math.cos(pitch)), width=640, height=480, vertical_fov_deg=_WIDE_VFOV,
                sees=True, step=0)


def _marked(ep, shot):
    """The cells one frame marks in this patrol."""
    return coverage.seen_closely(shot, ep.coverage["cells"], ep.coverage["normals"], ep.dock_position)


def _show(ep, shot):
    """Show the model a frame as the link delivers it, then let coverage read it."""
    ep.sensor_noise = {"shown": {"feed": shot, "zoom": None}}
    coverage.update(None, ep)


# ---------------------------------------------------------------- the grid


def test_the_grid_fills_the_fence_with_two_metre_cells():
    """Every cell centre stands inside the fence, neighbours stand 2 m apart, and the cells add up to the fenced
    area within one row of cells around its edge."""
    centres, _turn, _offset = coverage.grid(3, _SQUARE)
    assert np.all((centres > 0.0) & (centres < 100.0))
    gaps = np.sort(np.linalg.norm(centres[:, None] - centres[None, :50], axis=2), axis=0)[1]
    assert gaps == pytest.approx(np.full(50, coverage.CELL_M))
    assert abs(len(centres) * coverage.CELL_M ** 2 - 100.0 * 100.0) < 4 * 100.0 * coverage.CELL_M


def test_each_seed_turns_and_shifts_its_own_grid():
    """The same seed lays the same grid; across seeds the turn spreads over +-10 degrees and the offset over a whole
    cell, so no two seeds share their cell edges."""
    first = coverage.grid(7, _SQUARE)
    again = coverage.grid(7, _SQUARE)
    assert np.array_equal(first[0], again[0]) and first[1] == again[1]
    draws = [coverage.grid(seed, _SQUARE) for seed in range(200)]
    turns = np.array([d[1] for d in draws])
    offsets = np.array([d[2] for d in draws])
    assert turns.min() >= -coverage.MAX_TURN_DEG and turns.max() <= coverage.MAX_TURN_DEG
    assert turns.min() < -9.0 and turns.max() > 9.0
    assert offsets.min() >= 0.0 and offsets.max() < coverage.CELL_M and np.ptp(offsets, axis=0).min() > 1.8
    assert len({round(t, 9) for t in turns}) == len(turns)


@pytest.mark.skipif(not os.path.exists(os.path.join(_SOLAR_ASSETS, "manifest.json")),
                    reason=f"solar map not built at {_SOLAR_ASSETS}")
def test_the_park_holds_about_3200_cells_on_every_seed():
    """The park's 12,843 m2 fence holds 3,209 cells of 4 m2, give or take the cells its edge cuts, on every seed."""
    fence = park.fence_line(_SOLAR_ASSETS)
    counts = [len(coverage.grid(seed, fence)[0]) for seed in range(50)]
    assert Polygon(fence).area == pytest.approx(12843.0, rel=0.01)
    assert min(counts) > 3150 and max(counts) < 3270


def test_cells_sit_on_the_ground_even_under_a_table(ground):
    """Every cell finds the terrain under it, through the table top that stands over some of them, facing up."""
    cells, normals = ground.coverage["cells"], ground.coverage["normals"]
    assert len(cells) == len(coverage.grid(4, _SQUARE)[0])
    under_table = (np.abs(cells[:, 0] - 10.0) < 5.0) & (np.abs(cells[:, 1] - 10.0) < 5.0)
    assert under_table.sum() > 15
    assert cells[:, 2] == pytest.approx(0.0, abs=1e-6)
    assert np.allclose(normals, [0.0, 0.0, 1.0], atol=1e-6)


def test_the_ground_follows_a_slope():
    """On a 20 degree slope every cell sits on the slope and its normal leans with it."""
    cli = p.connect(p.DIRECT)
    try:
        tilt = math.radians(20.0)
        shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[200.0, 200.0, 0.5], physicsClientId=cli)
        slope = p.createMultiBody(0, shape, -1, [50.0, 50.0, -0.5 / math.cos(tilt)],
                                  p.getQuaternionFromEuler([0.0, -tilt, 0.0]), physicsClientId=cli)
        points, normals = coverage.ground(cli, np.array([[20.0, 50.0], [80.0, 50.0]]), frozenset({slope}))
        assert points[:, 2] == pytest.approx((points[:, 0] - 50.0) * math.tan(tilt), abs=1e-4)
        assert np.allclose(normals, [[-math.sin(tilt), 0.0, math.cos(tilt)]] * 2, atol=1e-6)
    finally:
        p.disconnect(cli)


# ---------------------------------------------------------------- which frames mark which cells


def test_a_frame_straight_down_from_twenty_metres_marks_its_footprint(ground):
    """From 20 m above the dock the wide lens covers 28.4 x 21.3 m of the flat ground 20.4 m below, about 150 cells,
    every cell centre in it seen closely enough."""
    marked = _marked(ground, _down(50.0, 50.0, 20.4))
    cells = ground.coverage["cells"]
    half_up = 20.4 * math.tan(math.radians(_WIDE_VFOV) / 2.0)
    half_across = half_up * 640 / 480
    in_footprint = (np.abs(cells[:, 0] - 50.0) <= half_across) & (np.abs(cells[:, 1] - 50.0) <= half_up)
    assert np.array_equal(marked, in_footprint)
    assert 130 <= marked.sum() <= 165


def test_a_frame_above_the_twenty_metre_line_marks_nothing(ground):
    """20 m above the dock is the highest a frame may be taken from; a centimetre higher marks nothing."""
    dock_z = float(ground.dock_position[2])
    assert _marked(ground, _down(50.0, 50.0, dock_z + 19.999)).any()
    assert not _marked(ground, _down(50.0, 50.0, dock_z + 20.01)).any()


def test_a_drone_in_its_dock_marks_nothing(ground):
    """A frame taken from inside the dock, below the top of its lids, marks nothing, looking down or out; once over
    the dock the same look marks the ground around it."""
    dock_z = float(ground.dock_position[2])
    assert not _marked(ground, _down(50.0, 50.0, dock_z - 0.03)).any()
    assert not _marked(ground, _looking(50.0, 50.0, dock_z + coverage.DOCK_TOP_M - 0.001, -30.0)).any()
    assert _marked(ground, _looking(50.0, 50.0, dock_z + coverage.DOCK_TOP_M + 0.01, -30.0)).any()


def test_a_frame_that_cannot_see_marks_nothing(ground):
    """A colour frame at night with night mode off sees nothing, so it marks nothing."""
    assert not _marked(ground, _down(50.0, 50.0, 15.0, sees=False)).any()


def test_a_thermal_frame_marks_its_narrower_view(ground):
    """The 45 degree thermal lens marks the smaller patch under it, every cell of it inside the wide one's."""
    wide = _marked(ground, _down(50.0, 50.0, 15.0))
    thermal = _marked(ground, _down(50.0, 50.0, 15.0, feed="thermal"))
    assert 0 < thermal.sum() < wide.sum() / 2
    assert not np.any(thermal & ~wide)


def test_ground_too_far_below_is_not_seen_closely_enough():
    """Straight down, flat ground counts while one pixel spans at most MAX_PIXEL_M: 38.4 m under the wide lens."""
    cli = p.connect(p.DIRECT)
    try:
        terrain = _slab(cli, [80.0, 80.0, 0.5], [50.0, 50.0, -0.5])
        ep = SolarEpisode(seed=4, fence=_SQUARE, terrain_uids=frozenset({terrain}),
                          dock_position=np.array([50.0, 50.0, 25.0]))
        coverage.reset(SimpleNamespace(CLIENT=cli), ep)
        assert _WIDE_LIMIT_M == pytest.approx(38.4, abs=0.1)
        assert _marked(ep, _down(50.0, 50.0, _WIDE_LIMIT_M - 0.05)).sum() > 400
        assert not _marked(ep, _down(50.0, 50.0, _WIDE_LIMIT_M + 0.05)).any()
    finally:
        p.disconnect(cli)


def test_a_far_shallow_view_marks_only_the_ground_near_the_drone(ground):
    """Looking out level-ish from 15 m, cells far off at a grazing angle stay unsearched: every marked cell lies
    within the distance where a pixel still spans MAX_PIXEL_M."""
    shot = _looking(5.0, 50.0, 15.0, -20.0)
    marked = _marked(ground, shot)
    in_view = np.isfinite(coverage.pixel_ground_m(shot, ground.coverage["cells"], ground.coverage["normals"]))
    assert marked.any() and in_view.sum() > 3 * marked.sum()
    assert ground.coverage["cells"][marked][:, 0].max() < ground.coverage["cells"][in_view][:, 0].max() - 30.0


# ---------------------------------------------------------------- the patrol's share


def test_coverage_is_the_share_of_cells_searched_so_far(ground):
    """Each frame shown adds its cells, a frame shown again adds nothing, and the outcome carries the share."""
    total = len(ground.coverage["cells"])
    first = _down(30.0, 50.0, 15.0, step=25)
    _show(ground, first)
    once = int(ground.coverage["searched"].sum())
    assert once > 0 and ground.outcome.coverage == pytest.approx(once / total)
    _show(ground, first)
    assert int(ground.coverage["searched"].sum()) == once
    _show(ground, _down(70.0, 50.0, 15.0, step=50))
    assert ground.outcome.coverage == pytest.approx(2 * once / total, rel=0.1)


def test_coverage_reads_the_frame_the_model_was_shown(ground):
    """The frame the link delivered counts, not a newer one the camera has taken but the model not yet seen; zoom
    close-ups mark nothing."""
    shown = _down(20.0, 50.0, 15.0, step=25)
    newest = _down(80.0, 50.0, 15.0, step=50)
    ground.camera = {"view": newest}
    ground.sensor_noise = {"shown": {"feed": shown, "zoom": replace(newest, feed="colour", step=51)}}
    coverage.update(None, ground)
    assert np.array_equal(ground.coverage["searched"], _marked(ground, shown))


def test_nothing_is_searched_before_a_frame_is_shown(ground):
    """Before the first observation no frame exists and coverage stays 0."""
    ground.camera = {"view": None}
    ground.sensor_noise = {"shown": {"feed": None, "zoom": None}}
    coverage.update(None, ground)
    assert ground.outcome.coverage == 0.0


# ---------------------------------------------------------------- whole patrols

# The camera straight down with night mode on auto, so night seeds see as well as day ones.
_LOOK = {"gimbal_tilt": -1.0, "night_mode": 0.9}


def _lawnmower(lane_m=18.0, height_m=19.0):
    """A pilot that takes off, sweeps the fence in north-south lanes with the camera straight down, then returns.

    The dock faces east, so the frame's short side lies across the lanes: 19.9 m of ground at 19 m, and lanes 18 m
    apart overlap a little."""
    def pilot(i, obs):
        """Follow each leg of the sweep against the wind, holding the height, and come home once it is done."""
        state = obs["state"]
        if i == 0:
            site = obs["site_map"]
            fence = site[SITE_MAP_SLICES["fence_xy"]].reshape(-1, 2)[:int(site[SITE_MAP_SLICES["fence_count"]][0])]
            low, high = fence.min(axis=0), fence.max(axis=0)
            xs = np.arange(low[0] + lane_m / 2.0, high[0], lane_m)
            ys = (low[1] + 4.0, high[1] - 4.0)
            pilot.route = [np.array([x, ys[(k + j) % 2]]) for k, x in enumerate(xs) for j in (0, 1)]
            return _action(take_off=1.0, **_LOOK)
        if int(state[STATE_SLICES["flight_phase"]][0]) != 2:
            return _action(**_LOOK)
        here = state[STATE_SLICES["position_m"]]
        if pilot.start is None:
            pilot.start = here[:2].copy()
        while pilot.route and np.hypot(*(pilot.route[0] - here[:2])) < 1.5:
            pilot.start = pilot.route.pop(0)
        if not pilot.route:
            return _action(**_LOOK, return_home=1.0)
        # Aim 6 m ahead of the nearest point on the leg, so the wind cannot bow the lane.
        end = pilot.route[0]
        leg = end - pilot.start
        length = max(float(np.hypot(*leg)), 1e-6)
        along = float(np.clip((here[:2] - pilot.start) @ leg / length, 0.0, length))
        aim = pilot.start + leg * min(along + 6.0, length) / length - here[:2]
        # Brake into each turn: from 5 m/s the drone needs 6.25 m to stop at 2 m/s2.
        speed = min(1.0, math.sqrt(2.0 * 1.5 * float(np.hypot(*(end - here[:2])))) / 5.0)
        aim = aim / max(float(np.hypot(*aim)), 1e-6) * speed
        # The dock faces east, so forward is east and right is south.
        return _action(move_forward=float(aim[0]), move_right=float(-aim[1]), **_LOOK,
                       move_up=float(np.clip(0.5 * (height_m - here[2]), -1.0, 1.0)))
    pilot.route, pilot.start = [], None
    return pilot


@pytest.mark.timeout(1200)
def test_a_full_sweep_reads_close_to_one(flat_park):  # noqa: F811
    """A lawnmower over the whole fenced square on a windy night seed, camera straight down at 19 m with night mode
    on auto, searches nearly every cell, lands, and the seed result carries the share."""
    log = _patrol(21, _lawnmower())
    outcome = log["episode"].outcome
    assert outcome.end_reason == "landed", (outcome.end_reason, outcome.coverage, log["time_s"])
    assert outcome.coverage > 0.97
    assert log["info"]["solar_outcome"]["coverage"] == outcome.coverage
    assert len(log["episode"].coverage["cells"]) == pytest.approx(Polygon(_FENCE).area / 4.0, rel=0.02)


@pytest.mark.timeout(300)
def test_a_patrol_that_never_takes_off_reads_zero(flat_park):  # noqa: F811
    """A drone left in its dock, looking straight down with a camera that can see, searches nothing."""
    log = _patrol(22, lambda i, obs: _action(**_LOOK), max_decisions=100)
    assert log["episode"].outcome.coverage == 0.0
    assert not log["episode"].coverage["searched"].any()
