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
"""Cover on a field that reaches into the trees: how people who stay put draw it, where they stand under and beside a
tree, the fairness rule that keeps someone under a tree only where a drone can see him, and that an open field is
peopled exactly as before."""

from __future__ import annotations

import contextlib
import hashlib
import io
import os
from itertools import combinations
from types import SimpleNamespace

import numpy as np
import pybullet as p
import pytest
import swarm_worlds

from swarm.challenge_families.solar_patrol import camera, intruders, theft_moves
from swarm.challenge_families.test_field_v2 import cover, install, people, site
from swarm.challenge_families.test_field_v2 import task as field_task
from swarm.challenge_families.test_field_v2.episode import FieldEpisode
from swarm.constants import SIM_DT
from swarm.utils.env_factory import make_env_with_initial_obs
from validator.tests import field_v2_stand_in

_BODIES = all(os.path.isfile(os.path.join(swarm_worlds.maps_dir(), folder, name)) for folder, name in
              ((intruders.ASSET_DIR, "intruders.json"), (theft_moves.ASSET_DIR, "motions.json")))
needs_bodies = pytest.mark.skipif(not _BODIES, reason="the installed swarm-worlds has no bodies or moves")
needs_raycast = pytest.mark.skipif(not camera.THERMAL, reason="the installed engine has no ray caster")
_OUTLINE = np.array([[-30.0, -30.0], [30.0, -30.0], [30.0, 30.0], [-30.0, 30.0]])
_DOCK = np.array([5.0, -5.0, 0.0])
_CROWNS = np.array([[-12.0, -12.0, 4.0], [12.0, 14.0, 3.0], [-14.0, 12.0, 2.5]])
# Where people stood on open fields before covers existed, pinned on the code of that time.
_FLAT_PINS = {0: "04209b4193ded63e", 1: "25e6568d958c1140", 2: "bbde0f475a4a0628", 3: "df7a30bc8271917d"}
_BUILT_PINS = {9: (4, "e1368efb03fec833")}


@pytest.fixture
def slab():
    """A flat ground slab, drawn and solid, in a fresh physics client, with a wall standing on it at (20, 0)."""
    cli = p.connect(p.DIRECT)
    half = [80.0, 80.0, 0.5]
    ground = p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=cli),
                               p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=[0.4, 0.5, 0.3, 1.0],
                                                   physicsClientId=cli),
                               basePosition=[0.0, 0.0, -0.5], physicsClientId=cli)
    p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.3, 30.0, 1.0], physicsClientId=cli),
                      basePosition=[20.0, 0.0, 1.0], physicsClientId=cli)
    yield cli, frozenset({ground})
    p.disconnect(cli)


@pytest.fixture
def everyday(tmp_path, monkeypatch):
    """Point the people at their everyday colours: the installed ones, or a stand-in from Sentinel's."""
    field_v2_stand_in.use_people(tmp_path / "people", monkeypatch)


def _ground(cli: int, terrain: frozenset, crowns: np.ndarray = _CROWNS) -> people.Ground:
    """The square field's ground on the slab, with crowns and no survey layer."""
    return people.Ground(cli, _OUTLINE, terrain, _DOCK[:2], crowns)


def _episode(terrain: frozenset) -> FieldEpisode:
    """A patrol on the square field with its dock, ready for people."""
    ep = FieldEpisode(seed=1)
    ep.fence, ep.terrain_uids, ep.dock_position, ep.people = _OUTLINE, terrain, _DOCK.copy(), []
    return ep


# ---------------------------------------------------------------------------------------------------- the draws

def test_covers_follow_their_shares_and_at_most_three_people_go_under_trees():
    """Over 4,000 seeds of up to ten people, about 15 % of the people who stay put stand beside a tree and a little
    under 25 % under one, never more than three a seed, and walkers always stand in the open."""
    counts = {cover.OPEN: 0, cover.BESIDE: 0, cover.UNDER: 0}
    for seed in range(4000):
        still = list(np.random.default_rng(seed).random(1 + seed % 10) < 0.75)
        covers = cover.draw(np.random.default_rng([people.PEOPLE_SEED_STREAM, seed, 4]), still)
        assert covers.count(cover.UNDER) <= cover.MOST_UNDER
        assert all(kind == cover.OPEN for kind, stays in zip(covers, still) if not stays)
        for kind, stays in zip(covers, still):
            counts[kind] += stays
    total = sum(counts.values())
    assert abs(counts[cover.BESIDE] / total - 0.15) < 0.015
    assert 0.21 < counts[cover.UNDER] / total < 0.25


def test_a_spot_under_a_tree_lies_inside_its_crown_and_one_beside_it_just_outside():
    """Spots under a crown lie from 1.2 m (or 0.4 of a small radius) off its centre out to 0.7 of its radius, and
    spots beside it 0.5 to 2 m outside its rim."""
    rng = np.random.default_rng(3)
    for radius in (1.5, 4.0):
        crowns = np.array([[10.0, -5.0, radius]])
        for _ in range(300):
            gap = np.hypot(*(cover.candidate(rng, cover.UNDER, crowns) - crowns[0, :2]))
            assert min(1.2, 0.4 * radius) - 1e-9 <= gap <= max(min(1.2, 0.4 * radius), 0.7 * radius) + 1e-9
            gap = np.hypot(*(cover.candidate(rng, cover.BESIDE, crowns) - crowns[0, :2])) - radius
            assert 0.5 - 1e-9 <= gap <= 2.0 + 1e-9
    assert cover.candidate(rng, cover.UNDER, np.zeros((0, 3))) is None


# ---------------------------------------------------------------------------------------------------- where they stand

def test_people_under_and_beside_trees_keep_their_rules(slab):
    """On a wooded field everyone under a tree stands inside a crown and off the dock and the others, everyone beside
    one stands clear of every crown, and the same seed places everyone the same way twice."""
    cli, terrain = slab
    ground = _ground(cli, terrain)
    kinds = [cover.UNDER, cover.BESIDE, cover.OPEN, cover.UNDER, cover.BESIDE, cover.OPEN]
    for seed in range(6):
        where, placed, _lead = people.layout(ground, seed, len(kinds), covers=kinds)
        again, placed_again, _lead = people.layout(ground, seed, len(kinds), covers=kinds)
        assert placed == placed_again == kinds and all(np.array_equal(a, b) for a, b in zip(where, again))
        for spot, kind in zip(where, placed):
            gaps = np.hypot(*(_CROWNS[:, :2] - spot).T) - _CROWNS[:, 2]
            assert np.all(np.abs(spot) <= 30.0 - people.PERSON_EDGE_M)
            assert np.hypot(*(spot - _DOCK[:2])) >= people.DOCK_CLEAR_M
            if kind == cover.UNDER:
                assert gaps.min() <= -0.3 + 1e-9
            else:
                assert gaps.min() >= people.CROWN_CLEAR_M
            if kind == cover.BESIDE:
                assert gaps.min() <= 2.0 + 1e-9
        for i, j in combinations(range(len(kinds)), 2):
            assert np.hypot(*(where[i] - where[j])) >= people.APART_M


def test_people_go_under_the_trees_themselves_not_a_merged_circle_round_them(slab):
    """When the site map lists one merged circle round a grove, people under a tree stand inside one of its trees'
    own circles, and people beside a tree stay clear of the merged circle."""
    cli, terrain = slab
    trees = np.array([[-12.0, -12.0, 2.0], [-6.0, -12.0, 2.0], [-9.0, -6.0, 2.0]])
    merged = np.array([[-9.0, -10.0, 6.5]])
    ground = people.Ground(cli, _OUTLINE, terrain, _DOCK[:2], merged, trees=trees)
    for seed in range(6):
        where, placed, _lead = people.layout(ground, seed, 3, covers=[cover.UNDER, cover.UNDER, cover.BESIDE])
        for spot, kind in zip(where, placed):
            if kind == cover.UNDER:
                assert np.min(np.hypot(*(trees[:, :2] - spot).T) - trees[:, 2]) <= -0.3 + 1e-9
            else:
                assert np.hypot(*(merged[0, :2] - spot)) - merged[0, 2] >= people.CROWN_CLEAR_M


def test_with_no_tree_to_stand_by_everyone_stands_in_the_open(slab):
    """A field with no crown sends everyone who drew a tree to the open."""
    cli, terrain = slab
    _where, placed, _lead = people.layout(_ground(cli, terrain, np.zeros((0, 3))), 1, 3, covers=[cover.UNDER, cover.BESIDE, cover.OPEN])
    assert placed == [cover.OPEN] * 3


def test_on_an_open_field_people_stand_where_they_always_did(slab):
    """Spots drawn with no covers match the pins taken before covers existed."""
    cli, terrain = slab
    crowns = np.array([[-10.0, -10.0, 4.0]])
    for seed, pin in _FLAT_PINS.items():
        spots = people.spots(cli, seed, _OUTLINE, terrain, np.array([5.0, -5.0]), crowns, 6, [3.0, 0, 0, 3.0, 0, 0])
        assert hashlib.sha256(np.array(spots).tobytes()).hexdigest()[:16] == pin


# ---------------------------------------------------------------------------------------------------- the fairness rule

def _sight(points: np.ndarray) -> cover.Sight:
    """A sight over given grid points on flat ground at height 0, with no world behind it."""
    sight = cover.Sight.__new__(cover.Sight)
    sight.env, sight.ep = SimpleNamespace(CLIENT=-1), SimpleNamespace(dock_position=np.zeros(3), far_m=None)
    sight.points, sight.ground = points, np.zeros(len(points))
    return sight


def test_the_rule_needs_five_percent_of_the_places_and_at_least_six():
    """The places needed are 5 % of those that may look, rounded up, and never fewer than six."""
    assert [cover.needed(n) for n in (0, 6, 100, 120, 121, 300)] == [6, 6, 6, 6, 7, 15]


def test_someone_is_kept_only_when_enough_places_see_him_clearly(monkeypatch):
    """With every place but his own neighbourhood looking, he is kept when the frames from just enough places show
    him clearly, and not with one place fewer or with one pixel fewer; the frames stop once the answer is sure."""
    grid = np.array([[x, y] for x in np.arange(-42.5, 45.0, 5.0) for y in np.arange(-42.5, 45.0, 5.0)])
    sight = _sight(grid)
    where = np.array([0.0, 0.0, 0.0])
    places = sight.places(where, where[None, :])
    assert len(places) and np.all(np.hypot(places[:, 0], places[:, 1]) >= cover.PERSON_GAP_M)
    assert np.all(np.hypot(places[:, 0], places[:, 1]) <= cover.REACH_M)
    assert set(places[:, 3]) == {0.0, 1.0, 2.0}
    least = cover.needed(len(places))

    def shown(clear_frames: int, px: int, above: bool = False):
        """An object map that shows px pixels of body 7 on the first frames, none after."""
        calls = []

        def object_map(env, ep, eye, target, size):
            """The stubbed frame; the look from straight above shows nothing of him unless told to."""
            if abs(eye[0] - target[0]) < 1.0 and abs(eye[1] - target[1]) < 1.0:
                return np.full((size, size), 7 if above else -1)
            calls.append(size)
            frame = np.full((size, size), -1)
            frame.ravel()[:px if len(calls) <= clear_frames else 0] = 7
            return frame
        monkeypatch.setattr(cover, "object_map", object_map)
        return calls

    calls = shown(least, cover.CLEAR_PX)
    assert sight.clear(where, 1.8, "standing", [7], where[None, :]) and len(calls) == least
    shown(least - 1, cover.CLEAR_PX)
    assert not sight.clear(where, 1.8, "standing", [7], where[None, :])
    calls = shown(0, cover.CLEAR_PX)
    assert not sight.clear(where, 1.8, "standing", [7], where[None, :]) and len(calls) == len(places) - least + 1
    shown(len(places), cover.CLEAR_PX - 1)
    assert not sight.clear(where, 1.8, "standing", [7], where[None, :])
    _places, counts = sight.counts(where, 1.8, "standing", [7], where[None, :])
    assert len(counts) == len(places) and counts.max() == cover.CLEAR_PX - 1
    # Kept under his tree: found from enough places, and hidden from most.
    ceiling = cover.most(len(places))
    assert cover.most(100) == 30 and least < ceiling < len(places)
    calls = shown(least, cover.CLEAR_PX)
    assert sight.kept(where, 1.8, "standing", [7], where[None, :]) and len(calls) == len(places) - (ceiling - least)
    shown(ceiling, cover.CLEAR_PX)
    assert sight.kept(where, 1.8, "standing", [7], where[None, :])
    calls = shown(ceiling + 1, cover.CLEAR_PX)
    assert not sight.kept(where, 1.8, "standing", [7], where[None, :]) and len(calls) == ceiling + 1
    shown(least - 1, cover.CLEAR_PX)
    assert not sight.kept(where, 1.8, "standing", [7], where[None, :])
    # Someone the leaves do not cover from straight above is not hidden: no other frame is drawn.
    calls = shown(least, cover.CLEAR_PX, above=True)
    assert not sight.kept(where, 1.8, "standing", [7], where[None, :]) and calls == []


@needs_bodies
def test_someone_shown_from_nowhere_tries_more_trees_then_stands_in_the_open(slab, everyday, monkeypatch):
    """Under a tree that hides him from every place, or one that hides him from none, a person is moved to three more
    places under trees and then to the open, where he keeps every rule of the open; someone the frames show from a
    few places only stays where he was put."""
    cli, terrain = slab
    ground = _ground(cli, terrain)
    for shows, kept in ((0, False), (10 ** 9, False), (40, True)):
        ep = _episode(terrain)
        rng = np.random.default_rng(4)
        where, placed, _lead = people.layout(ground, 2, 2, covers=[cover.UNDER, cover.OPEN])
        frames = {}
        for k, (spot, kind) in enumerate(zip(where, placed)):
            hub, mirrored = people.pose_hub(rng, "standing")
            index = people.pose_frame(hub)
            ep.people.append(people._stand(cli, ep, k, "standing", people.outfit(rng), 1.0, hub, mirrored, spot, 0.0))
            ep.people[-1].cover = kind
            frames[k] = (index, mirrored)
        spots = []

        def object_map(env, episode, eye, target, size):
            """The first frames of each place he stands on show all of him, the rest nothing; the look from
            straight above shows nothing."""
            if abs(eye[0] - target[0]) < 1.0 and abs(eye[1] - target[1]) < 1.0:
                return np.full((size, size), -1)
            spots.append(tuple(target[:2]))
            uid = list(ep.people[0].body.bodies)[0]
            return np.full((size, size), uid if spots.count(spots[-1]) <= shows else -1)
        monkeypatch.setattr(cover, "object_map", object_map)
        first = ep.people[0].xy
        people._fair(SimpleNamespace(CLIENT=cli), ep, ground, frames)
        person = ep.people[0]
        assert (person.cover == cover.UNDER) == kept and (person.xy == first) == kept
        if not kept:
            assert len(set(spots)) == 1 + cover.UNDER_DRAWS
            assert np.min(np.hypot(*(_CROWNS[:, :2] - person.xy).T) - _CROWNS[:, 2]) >= people.CROWN_CLEAR_M
            assert np.hypot(*(np.array(person.xy) - ep.people[1].xy)) >= people.APART_M
            assert person.ground_z == pytest.approx(0.0, abs=1e-6)


@needs_bodies
@needs_raycast
def test_a_canopy_hides_him_from_above_and_from_nowhere_only_when_it_reaches_the_ground(slab, everyday):
    """Through the real renderer: under a slab of leaves 2.2 m up he shows nothing from straight above yet clearly
    from enough low places, so he is kept; inside a box down to the ground he shows from nowhere and is not."""
    cli, terrain = slab
    ep = _episode(terrain)
    ep.fence, ep.dock_position = _OUTLINE / 2.0, np.array([5.0, 5.0, 0.0])
    rng = np.random.default_rng(6)
    hub, mirrored = people.pose_hub(rng, "standing")
    spot = np.array([-6.0, -6.0])
    ep.people.append(people._stand(cli, ep, 0, "standing", people.outfit(rng), 1.0, hub, mirrored, spot, 0.0))
    person, env = ep.people[0], SimpleNamespace(CLIENT=cli)
    uids, where = list(person.body.bodies), people.position(ep, 0)
    sight = cover.Sight(env, ep, np.zeros((0, 3)))
    target = sight.target(where, person.height_m, person.pose)
    leaves = p.createMultiBody(0, -1, p.createVisualShape(p.GEOM_BOX, halfExtents=[4.0, 4.0, 0.3], physicsClientId=cli),
                               basePosition=[spot[0], spot[1], 2.5], physicsClientId=cli)
    above = target + np.array([0.3, 0.0, 9.5 - target[2]])
    assert not np.isin(cover.object_map(env, ep, above, target, cover.FRAME_PX), uids).any()
    assert sight.clear(where, person.height_m, person.pose, uids, where[None, :])
    p.removeBody(leaves, physicsClientId=cli)
    p.createMultiBody(0, -1, p.createVisualShape(p.GEOM_BOX, halfExtents=[1.2, 1.2, 1.3], physicsClientId=cli),
                      basePosition=[spot[0], spot[1], 1.3], physicsClientId=cli)
    assert not sight.clear(where, person.height_m, person.pose, uids, where[None, :])


# ---------------------------------------------------------------------------------------------------- a whole seed

@pytest.fixture(scope="module")
def stand_in(tmp_path_factory):
    """The stand-in area as the only area and the people dressed in their everyday colours."""
    with pytest.MonkeyPatch.context() as patch:
        folder = tmp_path_factory.mktemp("stand_in")
        patch.setattr(install, "area_problems", lambda: [])
        root = field_v2_stand_in.use_area(folder, patch)
        field_v2_stand_in.use_people(folder / "people", patch)
        yield root
    site.area_tables.cache_clear()
    site.cut_area.cache_clear()


@needs_bodies
@needs_raycast
def test_an_open_seed_is_peopled_as_before_and_the_sight_frame_is_the_patrol_frames_object_map(stand_in):
    """On the stand-in (an open field) a seed puts the same people where they stood before covers existed, all in
    the open; and the fairness rule's frame gives exactly the object map a patrol's colour frame gives."""
    for seed, (count, pin) in _BUILT_PINS.items():
        with contextlib.redirect_stdout(io.StringIO()):
            env, _obs = make_env_with_initial_obs(field_task.test_field_task(seed=seed, sim_dt=SIM_DT))
        try:
            ep = env._solar
            assert ep.park["field_kind"] == "open" and not site.wooded(ep)
            folks = people.people(ep)
            rows = [(person.pose, person.xy[0], person.xy[1]) for person in folks]
            assert (len(rows), hashlib.sha256(repr(rows).encode()).hexdigest()[:16]) == (count, pin)
            assert all(person.cover == cover.OPEN for person in folks)
            sight = cover.Sight(env, ep, site.passable_outlines(ep))
            for k, person in enumerate(folks[:2]):
                where = people.position(ep, k)
                target = sight.target(where, person.height_m, person.pose)
                for eye in sight.places(where, where[None, :])[::40, :3]:
                    size = cover.FRAME_PX if np.linalg.norm(target - eye) > cover.NEAR_M else cover.NEAR_FRAME_PX
                    shot = camera.View("colour", tuple(eye), tuple((target - eye) / np.linalg.norm(target - eye)),
                                       (0.0, 0.0, 1.0), size, size, cover.FOV_DEG[size], True, 0,
                                       far_m=camera.far_plane_m(ep))
                    objects = camera.colour_frame(env, shot)[1]
                    expected = np.where(objects >= 0, objects & ((1 << 24) - 1), -1)
                    assert np.array_equal(cover.object_map(env, ep, eye, target, size), expected)
        finally:
            env.close()
