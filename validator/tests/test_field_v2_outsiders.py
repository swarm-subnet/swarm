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

"""Test Field v2's people passing outside the field: how many a seed holds, the lanes they walk and the tracks they
keep beyond the edge, the ground they keep to, how they are posed and placed in a patrol, and why the 5 m rule can
never bite on them."""

from __future__ import annotations

import contextlib
import io
import math
import os
from types import SimpleNamespace

import numpy as np
import pybullet as p
import pytest
import swarm_worlds
from shapely.geometry import Point, Polygon

from swarm.challenge_families import build_benchmark_tasks
from swarm.challenge_families.solar_patrol import camera, intruders, theft_moves
from swarm.challenge_families.solar_patrol.contract import MAX_HORIZONTAL_MPS
from swarm.challenge_families.solar_patrol.score import SAFE_DISTANCE_M
from swarm.challenge_families.solar_patrol.theft_moves import GONE
from swarm.challenge_families.test_field_v2 import flight_limit, install, mix, outsiders, people, score
from swarm.challenge_families.test_field_v2.contract import FAMILY_ID
from swarm.challenge_families.test_field_v2.episode import FieldEpisode
from swarm.constants import SIM_DT
from swarm.utils.env_factory import make_env_with_initial_obs
from validator.tests import field_v2_stand_in

_MOVES = os.path.isfile(os.path.join(swarm_worlds.maps_dir(), theft_moves.ASSET_DIR, "motions.json"))
needs_moves = pytest.mark.skipif(not _MOVES, reason="the installed swarm-worlds has no moves")
_SHIPPED = _MOVES and os.path.isfile(os.path.join(swarm_worlds.maps_dir(), intruders.ASSET_DIR, "intruders.json"))
_SQUARE = np.array([[-30.0, -30.0], [30.0, -30.0], [30.0, 30.0], [-30.0, 30.0]])
# An L-shaped field, its inner corner a notch the band must round, and a long thin field turned off the axes.
_NOTCHED = np.array([[0.0, 0.0], [80.0, 0.0], [80.0, 30.0], [35.0, 30.0], [35.0, 75.0], [0.0, 75.0]])
_THIN = np.array([[0.0, 0.0], [90.0, 0.0], [90.0, 22.0], [0.0, 22.0]]) @ np.array([[0.8, 0.6], [-0.6, 0.8]])
_FPS = 30


def _shown(passer: outsiders.Pass) -> np.ndarray:
    """East and north of every frame of a passer's track that shows him in the scene."""
    return np.array([[f.x, f.y] for f in passer.frames if f.index != GONE], dtype=float)


# ---------------------------------------------------------------------------------------------------- how many

def test_a_seed_has_passers_exactly_when_the_mix_says_and_one_to_three_by_their_shares():
    """Over 20,000 seeds a seed has people passing exactly when its mix says so, and then 1, 2 or 3 of them at 60, 30
    and 10 %."""
    seeds = range(20000)
    counts = np.array([outsiders.count(seed) for seed in seeds])
    assert np.array_equal(counts > 0, [mix.present(seed).outsiders for seed in seeds])
    some = counts[counts > 0]
    for n, share in outsiders.COUNT_SHARES.items():
        assert abs(float(np.mean(some == n)) - share) < 0.02, n
    assert outsiders.count(2 ** 127 + 5) == outsiders.count(2 ** 127 + 5)


# ---------------------------------------------------------------------------------------------------- lanes and tracks

@needs_moves
@pytest.mark.parametrize("outline", [_SQUARE, _NOTCHED, _THIN], ids=["square", "notched", "thin"])
def test_every_frame_of_every_track_stays_six_to_forty_metres_outside_the_field(outline):
    """For 8 seeds of three passers each on three field shapes, every frame that shows a passer lies outside the
    outline and 6 to 40 m from it, measured apart from the planner, and within SIDE_M of his lane; each lane runs 30 to
    120 m."""
    field = Polygon(outline)
    planned = 0
    for seed in range(8):
        for n in range(3):
            passer = outsiders.plan(seed, n, outline, 1.0)
            if passer is None:
                continue
            planned += 1
            xy = _shown(passer)
            assert not any(field.contains(Point(x, y)) for x, y in xy[::5])
            gaps = np.array([field.exterior.distance(Point(x, y)) for x, y in np.unique(xy, axis=0)])
            assert gaps.min() >= outsiders.BAND_M[0] and gaps.max() <= outsiders.BAND_M[1]
            length = float(np.hypot(*np.diff(passer.lane, axis=0).T).sum())
            assert outsiders.LANE_M[0] <= length <= outsiders.LANE_M[1] + outsiders.STEP_M
    assert planned >= 22


@needs_moves
def test_a_passer_walks_at_a_normal_pace_some_stop_some_leave_and_arrivals_spread():
    """Over 30 passers: none goes faster than 1.5 m/s over any second; one who stops stands still for his stop; one
    who leaves is gone at the end, the others stand at the lane's end; a late one is away until he arrives, an early
    one is already walking at take-off."""
    stopped = left = late = early = 0
    for seed in range(10):
        for n in range(3):
            passer = outsiders.plan(seed, n, _SQUARE, 1.0)
            frames = passer.frames
            walking = [k for k, f in enumerate(frames) if f.index != GONE]
            xy = np.array([[frames[k].x, frames[k].y] for k in walking])
            assert np.max(np.hypot(*(xy[_FPS:] - xy[:-_FPS]).T)) <= 1.5
            still = (np.hypot(*np.diff(xy, axis=0).T) < 1e-9).astype(int)
            runs = np.diff(np.flatnonzero(np.diff(np.concatenate([[0], still, [0]]))))[::2] / _FPS
            if passer.stop_s:
                stopped += 1
                assert np.min(np.abs(runs - passer.stop_s)) < 3.0
            assert (frames[-1].index == GONE) == passer.leaves
            left += passer.leaves
            if passer.arrive_s > 0.0:
                late += 1
                assert walking[0] == int(round(passer.arrive_s * _FPS))
            else:
                early += 1
                assert walking[0] == 0
    assert stopped and left and late and early


@needs_moves
def test_two_passers_in_the_scene_at_once_keep_apart():
    """However their lanes cross, two passers of a seed never come within APART_M of each other while both are in the
    scene."""
    for seed in range(8):
        first = outsiders.plan(seed, 0, _SQUARE, 1.0)
        second = outsiders.plan(seed, 1, _SQUARE, 1.0, others=[first])
        assert outsiders.apart(first.frames, second.frames)
    near = outsiders.plan(1, 0, _SQUARE, 1.0)
    assert not outsiders.apart(near.frames, near.frames)


@needs_moves
def test_a_lane_is_the_same_on_every_draw_and_any_seed_width_counts():
    """The same seed plans the same passer to the bit, and seeds that differ only above 64 bits plan different
    ones."""
    a, b = outsiders.plan(7, 0, _SQUARE, 1.0), outsiders.plan(7, 0, _SQUARE, 1.0)
    assert np.array_equal(a.lane, b.lane) and a.frames == b.frames
    wide = [outsiders.plan((top << 64) + 7, 0, _SQUARE, 1.0) for top in (1, 2)]
    assert not np.array_equal(wide[0].lane, wide[1].lane)


@pytest.fixture
def ground_with_a_wall_and_a_bank():
    """Flat ground in a fresh physics client, a long wall standing on it 15 m east of the square's edge, and to the
    north a bank rising at 30 degrees."""
    cli = p.connect(p.DIRECT)
    slab = p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=[150.0, 150.0, 0.5], physicsClientId=cli),
                             basePosition=[0.0, 0.0, -0.5], physicsClientId=cli)
    p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.3, 120.0, 1.5], physicsClientId=cli),
                      basePosition=[45.0, 0.0, 1.5], physicsClientId=cli)
    bank = p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=[120.0, 20.0, 0.5], physicsClientId=cli),
                             basePosition=[0.0, 55.0, 4.0],
                             baseOrientation=p.getQuaternionFromEuler([math.radians(30.0), 0.0, 0.0]),
                             physicsClientId=cli)
    yield cli, frozenset({slab, bank})
    p.disconnect(cli)


def test_a_lane_keeps_to_open_flat_ground(ground_with_a_wall_and_a_bank):
    """No lane crosses the wall or climbs the bank: every point of its line meets only the ground, and stays low."""
    cli, terrain = ground_with_a_wall_and_a_bank
    ground = outsiders.ground_test(cli, terrain)
    drawn = 0
    for seed in range(25):
        points = outsiders.lane(np.random.default_rng(seed), _SQUARE, 120.0, ground)
        if points is None:
            continue
        drawn += 1
        line = np.concatenate([np.linspace(a, b, 17) for a, b in zip(points[:-1], points[1:])])
        heights = ground(line)
        assert np.all(np.isfinite(heights)) and np.all(np.abs(heights) < 1.0)
        assert not np.any((line[:-1, 0] - 45.0) * (line[1:, 0] - 45.0) <= 0.0)
    assert drawn >= 10


def test_the_ground_is_closed_wherever_the_site_lets_nobody_stand(ground_with_a_wall_and_a_bank):
    """With the site's own word on where someone may stand, the ground reads as closed everywhere it says no."""
    cli, terrain = ground_with_a_wall_and_a_bank
    points = np.array([[-20.0, -40.0], [20.0, -40.0], [-5.0, -45.0], [5.0, -45.0]])
    assert np.all(np.isfinite(outsiders.ground_test(cli, terrain)(points)))
    west_only = outsiders.ground_test(cli, terrain, lambda x, y: x < 0.0)
    assert np.isfinite(west_only(points)).tolist() == [True, False, True, False]


def test_a_lane_keeps_out_from_under_every_crown():
    """With crowns standing all along the band's west and south sides, no point of any lane or beside it comes under
    one."""
    cover = np.array([[x, y, 4.0] for x in np.arange(-70.0, 71.0, 6.0) for y in np.arange(-70.0, 71.0, 6.0)
                      if x < -30.0 or y < -30.0])
    drawn = 0
    for seed in range(20):
        points = outsiders.lane(np.random.default_rng(seed), _SQUARE, 80.0, cover=cover)
        if points is None:
            continue
        drawn += 1
        line = np.concatenate([np.linspace(a, b, 9) for a, b in zip(points[:-1], points[1:])])
        gap = np.hypot(line[:, None, 0] - cover[None, :, 0], line[:, None, 1] - cover[None, :, 1]) - cover[None, :, 2]
        assert gap.min() >= outsiders.COVER_CLEAR_M - 1e-9
    assert drawn >= 10


def test_a_lane_keeps_to_the_paths_when_the_band_has_them():
    """With a path three cells wide circling the field 52 to 55 m from its centre, every lane starts on it and walks
    it; with too few path cells in the band the lane is drawn as if there were none."""
    ring = np.array([[x + 0.5, y + 0.5] for x in range(-60, 60) for y in range(-60, 60)
                     if 52.0 <= math.hypot(x + 0.5, y + 0.5) <= 55.0])
    for seed in range(10):
        points = outsiders.lane(np.random.default_rng(seed), _SQUARE, 60.0, paths=ring)
        assert points is not None
        assert np.all(np.abs(np.hypot(*points.T) - 53.5) <= 2.5)
    few = outsiders.lane(np.random.default_rng(3), _SQUARE, 60.0, paths=ring[:outsiders.PATH_CELLS - 1])
    assert np.array_equal(few, outsiders.lane(np.random.default_rng(3), _SQUARE, 60.0))


# ---------------------------------------------------------------------------------------------------- the 5 m rule

@needs_moves
def test_the_five_metre_rule_never_bites_on_someone_passing(monkeypatch):
    """The patrol ends before the drone comes within EDGE_M of the outline, at most one step's flight past that line,
    so it stays more than 5 m from any frame 6 m outside it; a drone on that line, nearest the closest frame of a
    track, keeps no distance to him."""
    assert outsiders.BAND_M[0] + flight_limit.EDGE_M - MAX_HORIZONTAL_MPS * SIM_DT > SAFE_DISTANCE_M
    xy = _shown(outsiders.plan(4, 0, _SQUARE, 1.0))
    outside = np.hypot(np.maximum(np.abs(xy[:, 0]) - 30.0, 0.0), np.maximum(np.abs(xy[:, 1]) - 30.0, 0.0))
    nearest = xy[int(np.argmin(outside))]
    drone = np.clip(nearest, -30.0 + flight_limit.EDGE_M, 30.0 - flight_limit.EDGE_M)
    passer = _passer([_frame(10, *nearest)])
    ep = FieldEpisode(seed=4)
    ep.phase, ep.step = "flying", 0
    ep.people = [people.Person(index=0, pose="passing", xy=tuple(nearest), ground_z=0.0, yaw=0.0, inside=False,
                               height_m=1.93, passer=passer)]
    monkeypatch.setattr(score.sentinel_score, "update", lambda env, episode: None)
    score.update(SimpleNamespace(pos=np.array([[drone[0], drone[1], 1.0]])), ep)
    assert ep.outcome.min_threat_distance_m is None
    assert float(np.hypot(*(nearest - drone))) >= outsiders.BAND_M[0] + flight_limit.EDGE_M - 1e-9


# ---------------------------------------------------------------------------------------------------- in the patrol

class _Body:
    """A stand-in body that keeps the places it is posed at."""

    def __init__(self):
        """No pose yet."""
        self.places = []

    def pose(self, *args) -> bool:
        """Keep one pose's place."""
        self.places.append(tuple(args[2]))
        return True


def _passer(frames) -> outsiders.Pass:
    """A passer on a stand-in body over flat ground at height 0, with the frames given."""
    passer = outsiders.Pass(lane=np.array([[40.0, 0.0], [40.0, 40.0]]), frames=list(frames), body=_Body())
    for f in frames:
        passer.grounds[(f.x, f.y)] = 0.0
    return passer


def _frame(index, x, y):
    """A frame of the motion library standing at a place."""
    return theft_moves.Frame(index, False, float(x), float(y), 0.0, "walking")


@needs_moves
def test_a_passer_is_posed_only_for_a_picture_that_sees_him_and_taken_away_once_he_leaves():
    """Away, he is never posed; walking, a picture above him poses him once per new frame and one far off never does;
    once he has left he is taken out of the scene once."""
    frames = [_frame(GONE, 40.0, 0.0)] * 2 + [_frame(10, 40.0, 0.0), _frame(11, 40.0, 0.05), _frame(GONE, 40.0, 0.05)]
    passer = _passer(frames)
    above = camera.View("colour", (40.0, 0.0, 10.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0), 64, 48, 60.0, True, 0)
    away = camera.View("colour", (500.0, 500.0, 10.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0), 64, 48, 60.0, True, 0)
    at = [k / _FPS + 1e-6 for k in range(len(frames))]
    outsiders.show([passer], at[0], [above])
    assert passer.body.places == []
    outsiders.show([passer], at[2], [away])
    assert passer.body.places == []
    outsiders.show([passer], at[2], [above])
    outsiders.show([passer], at[2], [above])
    outsiders.show([passer], at[3], [above])
    assert [z for _x, _y, z in passer.body.places] == [0.0, 0.0]
    outsiders.show([passer], at[4], [away])
    outsiders.show([passer], at[4] + 5.0, [away])
    assert len(passer.body.places) == 3 and passer.body.places[-1][2] < -100.0


@needs_moves
def test_a_passers_position_reads_his_track_by_step():
    """Before he arrives his point is his lane's start and he is not in the scene; later it is the frame of the step
    asked, never past now, and past his track's end its last frame."""
    frames = [_frame(GONE, 40.0, 0.0)] * 30 + [_frame(10, 40.0, float(k) * 0.04) for k in range(60)]
    passer = _passer(frames)
    ep = FieldEpisode(seed=1)
    ep.step = 25
    assert not outsiders.in_scene(ep, passer) and outsiders.position(ep, passer).tolist() == [40.0, 0.0, 0.0]
    ep.step = 10 ** 6
    assert outsiders.in_scene(ep, passer)
    step = int(round(2.0 / SIM_DT))
    assert outsiders.position(ep, passer, step).tolist() == pytest.approx([40.0, 30 * 0.04, 0.0])
    assert outsiders.position(ep, passer).tolist() == pytest.approx([40.0, 59 * 0.04, 0.0])


@pytest.fixture
def stand_in_field(tmp_path, monkeypatch):
    """Make the stand-in area the only area a seed flies, and point the people at their everyday colours."""
    monkeypatch.setattr(install, "area_problems", lambda: [])
    field_v2_stand_in.use_area(tmp_path, monkeypatch)
    if _SHIPPED:
        field_v2_stand_in.use_people(tmp_path / "people", monkeypatch)


def test_a_patrol_puts_the_people_passing_after_everyone_on_the_field(stand_in_field):
    """A built patrol of a seed with people passing lists everyone on the field first, then each passer outside the
    field, away from it, where the area lets someone stand, and dressed apart from everyone; only the people on the
    field count as people to find, every passer's body can be told apart by the judge, and only a passer has a lane."""
    if not _SHIPPED:
        pytest.skip("the installed swarm-worlds has no intruder bodies or moves")
    seed = next(s for s in range(200) if outsiders.count(s) >= 2 and people.wanted(s) >= 2)
    task = build_benchmark_tasks(sim_dt=SIM_DT, seeds=[seed], family_id=FAMILY_ID)[0]
    with contextlib.redirect_stdout(io.StringIO()):
        env, _obs = make_env_with_initial_obs(task)
    try:
        ep = env._solar
        everyone = people.people(ep)
        inside = [person for person in everyone if person.inside]
        passing = [person for person in everyone if not person.inside]
        assert [person.index for person in everyone] == list(range(len(everyone)))
        assert everyone[:len(inside)] == inside and ep.outcome.threats == len(inside) >= 1
        assert 1 <= len(passing) <= outsiders.count(int(task.map_seed)) and len(people.passers(ep)) == len(passing)
        bodies = people.bodies(ep)
        ep.step = 10 ** 6
        assert all(person.lane is None for person in inside)
        for person in passing:
            assert {bodies[uid] for uid in person.body.bodies} == {person.index}
            assert person.lane is person.passer.lane and ep.park["open_at"](*person.lane.T).all()
            for step in range(0, 10000, 100):
                if outsiders.in_scene(ep, person.passer, step):
                    point = people.position(ep, person.index, step)
                    assert flight_limit.edge_distance(ep, point) <= -outsiders.BAND_M[0]
    finally:
        env.close()


def test_a_field_with_no_lane_round_it_has_nobody_passing(stand_in_field, monkeypatch):
    """Where no lane fits outside the field, the patrol is built all the same, with nobody passing and everyone on
    the field in place."""
    if not _SHIPPED:
        pytest.skip("the installed swarm-worlds has no intruder bodies or moves")
    seed = next(s for s in range(200) if outsiders.count(s) >= 1 and people.wanted(s) >= 1)
    monkeypatch.setattr(outsiders, "plan", lambda *args, **kwargs: None)
    task = build_benchmark_tasks(sim_dt=SIM_DT, seeds=[seed], family_id=FAMILY_ID)[0]
    with contextlib.redirect_stdout(io.StringIO()):
        env, _obs = make_env_with_initial_obs(task)
    try:
        ep = env._solar
        assert people.passers(ep) == [] and all(person.inside for person in people.people(ep))
        assert ep.outcome.threats == len(people.people(ep)) >= 1
    finally:
        env.close()
