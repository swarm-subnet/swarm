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

"""Test Field v2's people: how much room a count needs, how everyone in a seed looks apart, their sizes, where they
are placed, and the walkers' strolls, stops for the drone, positions and posing; and how the reference pilot keeps off
a walker and reports him."""

from __future__ import annotations

import json
import math
import os
import shutil
from itertools import combinations
from types import SimpleNamespace

import numpy as np
import pybullet as p
import pytest
import swarm_worlds

from swarm.challenge_families.solar_patrol import camera, intruders, theft_moves
from swarm.challenge_families.solar_patrol.intruders import _quat_to_mat
from swarm.challenge_families.test_field_v2 import figures, idlers, moves, people, reference_pilot, walkers
from swarm.challenge_families.test_field_v2.episode import FieldEpisode
from swarm.constants import SIM_DT
from validator.tests import field_v2_stand_in

_BODIES = all(os.path.isfile(os.path.join(swarm_worlds.maps_dir(), folder, name)) for folder, name in
              ((intruders.ASSET_DIR, "intruders.json"), (theft_moves.ASSET_DIR, "motions.json")))
needs_bodies = pytest.mark.skipif(not _BODIES, reason="the installed swarm-worlds has no bodies or moves")
_OUTLINE = np.array([[-30.0, -30.0], [30.0, -30.0], [30.0, 30.0], [-30.0, 30.0]])
_DOCK = np.array([5.0, -5.0, 2.0])
# Every stride a walker can have: the shortest build's root scale at the smallest size to the tallest's at the largest.
_STRIDES = (0.944 * people.SIZE_RANGE[0], 1.055 * people.SIZE_RANGE[1])


@pytest.fixture
def everyday(tmp_path, monkeypatch):
    """Point the people at their package: the installed one, or a stand-in on Sentinel's skeleton."""
    field_v2_stand_in.use_people(tmp_path / "people", monkeypatch)


@pytest.fixture
def flat_ground():
    """A flat ground slab in a fresh physics client, with a wall standing on it at (20, 0)."""
    cli = p.connect(p.DIRECT)
    ground = p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=[80.0, 80.0, 0.5], physicsClientId=cli),
                               basePosition=[0.0, 0.0, -0.5], physicsClientId=cli)
    p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.3, 30.0, 1.0], physicsClientId=cli),
                      basePosition=[20.0, 0.0, 1.0], physicsClientId=cli)
    yield cli, frozenset({ground})
    p.disconnect(cli)


def _walk(seed: int, home=(0.0, 0.0), stride: float = 1.0, patch: float = walkers.PATCH_M) -> walkers.Walk:
    """A walker's planned walk from his own stream, with his track ready to be written."""
    walk = walkers.plan(np.random.default_rng([people.PEOPLE_SEED_STREAM, seed, 3, 0]), home, stride, patch)
    walkers.actor(walk)
    return walk


# ---------------------------------------------------------------------------------------------------- count and room

def test_the_room_caps_the_count_and_a_count_names_the_area_it_needs():
    """One person per AREA_PER_PERSON_M2 and ten at most: the real field's 2,755 m2 holds 7, a count's own area holds
    exactly that count, and a capped seed puts the room's worth on the field."""
    assert people.capacity(2755.0) == 7 and people.capacity(3600.0) == 10 and people.capacity(10 ** 6) == 10
    assert people.capacity(people.AREA_PER_PERSON_M2 - 1e-9) == 0
    for n in range(people.MOST_PEOPLE + 1):
        assert people.capacity(people.min_area_m2(n)) == n
    for seed in range(300):
        assert people.count(seed, 1000.0) == min(people.wanted(seed), 2)
        assert people.count(seed) == people.wanted(seed)


def test_the_fields_area_is_the_same_either_way_round():
    """The area inside an outline is exact, and the same for either winding."""
    assert people.area_m2(_OUTLINE) == 3600.0 and people.area_m2(_OUTLINE[::-1]) == 3600.0
    assert people.area_m2(np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])) == 6.0


# ---------------------------------------------------------------------------------------------------- looks and sizes

@needs_bodies
def test_any_two_people_of_a_seed_differ_in_two_of_top_head_and_trousers(everyday):
    """Over 400 seeds of ten people each, every pair differs in at least two of the three parts, and every tint worn
    is one of the package's palettes'."""
    manifest = figures.catalogue().manifest
    tints = {tuple(colour) for palette in manifest["palettes"].values() for colour in palette}
    for seed in range(400):
        rng = np.random.default_rng([people.PEOPLE_SEED_STREAM, seed, 2])
        looks = []
        for _ in range(people.MOST_PEOPLE):
            dress = people.dress_apart(rng, looks)
            looks.append(people.look(dress))
            assert all(tint is None or tuple(tint) in tints for _name, _picture, tint in dress["pieces"])
        assert all(people.apart(a, b) for a, b in combinations(looks, 2))


@needs_bodies
def test_a_look_is_the_top_or_vest_the_head_and_the_bottom(everyday):
    """A look names the colour of the top, or the vest worn over it, the head (hair with its colour, or bare) and the
    colour of the trousers or skirt; and a person is dressed for the air."""
    manifest = figures.catalogue().manifest
    rng = np.random.default_rng(1)
    dresses = [people.outfit(rng) for _ in range(400)]
    worn = lambda dress, slot: [name for name, _p, _t in dress["pieces"] if manifest["pieces"][name]["slot"] == slot]   # noqa: E731
    vested = next(d for d in dresses if worn(d, "over"))
    assert people.look(vested)[0].startswith(worn(vested, "over")[0] + ":")
    plain = next(d for d in dresses if not worn(d, "over"))
    assert people.look(plain)[0].startswith("top:")
    bare = next(d for d in dresses if not worn(d, "hair") and not worn(d, "head"))
    assert people.look(bare)[1].startswith("bare:")
    haired = next(d for d in dresses if worn(d, "hair") and not worn(d, "head"))
    assert people.look(haired)[1].startswith(worn(haired, "hair")[0] + ":hair_")
    assert all(len(worn(d, slot)) == 1 for d in dresses for slot in ("eyes", "top", "bottom", "shoes"))
    assert all(manifest["builds"][d["build"]]["sex"] == d["sex"] for d in dresses)
    cold = [people.outfit(rng, 2.0) for _ in range(200)]
    hot = [people.outfit(rng, 25.0) for _ in range(200)]
    warmth = lambda dress: {manifest["pieces"][name].get("warmth", "any") for name, _p, _t in dress["pieces"]}   # noqa: E731
    assert all("light" not in warmth(d) for d in cold) and all("warm" not in warmth(d) for d in hot)
    assert any("warm" in warmth(d) for d in cold) and any("light" in warmth(d) for d in hot)
    assert people.apart(("a", "b", "c"), ("a", "x", "y")) and not people.apart(("a", "b", "c"), ("a", "b", "y"))
    # A vest is never worn over a thick top, which would cover it; and a bottom's look is its colour alone, so
    # trousers and a skirt of one colour count as the same.
    everyone = dresses + cold + hot
    assert all(manifest["pieces"][worn(d, "top")[0]].get("bulk", "thin") == "thin" for d in everyone if worn(d, "over"))
    for dress in everyone:
        tint = next(t for name, _p, t in dress["pieces"] if manifest["pieces"][name]["slot"] == "bottom")
        if tint is not None:
            assert people.look(dress)[2] == "bottom:" + ",".join(f"{c:.3f}" for c in tint[:3])


def test_most_clothes_are_plain_and_a_picture_takes_one_draw():
    """A piece with a plain picture and patterned ones is drawn plain in 60 % of the draws and each pattern as often
    as the next; a piece with no plain picture, or only one, is drawn evenly; every choice takes exactly one draw."""
    assert (figures.PLAIN, figures.PLAIN_SHARE) == ("plain", 0.6)
    rng = np.random.default_rng(4)
    drawn = [figures._style(rng, ["checked", "plain", "striped"]) for _ in range(6000)]
    assert abs(drawn.count("plain") / 6000 - 0.6) < 0.03 and abs(drawn.count("checked") - drawn.count("striped")) < 300
    even = [figures._style(rng, ["denim", "dotted"]) for _ in range(4000)]
    assert abs(even.count("denim") / 4000 - 0.5) < 0.04
    for styles in (["plain"], ["knit"], ["checked", "plain"], ["a", "b", "c"]):
        one, other = np.random.default_rng(9), np.random.default_rng(9)
        assert figures._style(one, styles) in styles
        other.random()
        assert one.random() == other.random()


@needs_bodies
def test_a_package_nobody_can_be_drawn_from_is_named_before_any_flight(everyday, tmp_path):
    """A people package whose files are all there passes the check only if people can be drawn from it: an unreadable
    table, a sex with no skin and a slot with nothing for the cold are each named; so is a motion library with no
    walk."""
    folder = figures.catalogue().folder
    assert figures.problems(folder) == [] and moves.problems() == []
    table = json.loads(open(os.path.join(folder, "people.json"), encoding="utf-8").read())

    def check(change) -> list:
        """The problems of the package with its table changed."""
        broken = json.loads(json.dumps(table))
        change(broken)
        (tmp_path / "people.json").write_text(json.dumps(broken))
        return figures.problems(str(tmp_path))

    (tmp_path / "people.json").write_text("{ not a table")
    assert len(figures.problems(str(tmp_path))) == 1 and "cannot be read" in figures.problems(str(tmp_path))[0]

    def no_women(broken):
        """Take every woman's skin out."""
        broken["skins"] = {name: skin for name, skin in broken["skins"].items() if skin["sex"] != "f"}

    def no_tops(broken):
        """Take every top out."""
        broken["pieces"] = {name: piece for name, piece in broken["pieces"].items() if piece["slot"] != "top"}

    assert check(no_women) == ["no skin of sex f"]
    assert sum("nothing for the slot top" in line for line in check(no_tops)) == 4
    lib_folder = tmp_path / "moves"
    lib_folder.mkdir()
    manifest = json.loads(open(os.path.join(moves.MOVES_DIR, "motions.json"), encoding="utf-8").read())
    manifest["blocks"] = {name: block for name, block in manifest["blocks"].items() if not block["kind"].startswith("walk")}
    (lib_folder / "motions.json").write_text(json.dumps(manifest))
    shutil.copyfile(os.path.join(moves.MOVES_DIR, "motions.npz"), lib_folder / "motions.npz")
    assert moves.problems(str(lib_folder)) == ["the plain walk cannot be played whole"]
    (lib_folder / "motions.json").write_text("[]")
    assert "cannot be read" in moves.problems(str(lib_folder))[0]


@needs_bodies
def test_a_person_is_drawn_at_his_size_owns_every_mesh_and_a_failed_rewrite_raises(everyday):
    """A figure built from the package stands size times as tall, owns one drawn body for his skin and one for each
    piece he wears, and raises when the engine cannot rewrite one of them, instead of leaving him where he was
    built."""
    cli = p.connect(p.DIRECT)
    try:
        lib = moves.library()
        dress = people.outfit(np.random.default_rng(3), 15.0)
        pose = people.pose_args(people.pose_frame(moves.STAND), False, (0.0, 0.0, 0.0), 0.0)
        tall = {}
        for size in (1.0, 1.04):
            body = figures.Figure(figures.catalogue(), dress, pose, cli, size=size, textures={})
            v = body.vertices(*pose)
            tall[size] = float(v[:, 2].max() - v[:, 2].min())
            assert len(body.bodies) == 1 + len(dress["pieces"]) and len(set(body.bodies)) == len(body.bodies)
            assert v[:, 2].min() == pytest.approx(0.0, abs=1e-9)
        assert tall[1.04] == pytest.approx(1.04 * tall[1.0], rel=1e-6)
        assert tall[1.0] == pytest.approx(people.height_m(dress, 1.0), rel=0.06)
        low = people.pose_args(people.pose_frame(sorted(moves.low(lib))[0]), False, (0.0, 0.0, 0.0), 0.0)
        assert float(np.ptp(body.vertices(*low)[:, 2])) < 0.85 * tall[1.04]
        p.removeBody(body.bodies[-1], physicsClientId=cli)
        with pytest.raises(p.error):
            body.pose(*pose)
    finally:
        p.disconnect(cli)


@needs_bodies
def test_a_seed_with_no_look_left_apart_fails_as_its_own_world(everyday, monkeypatch):
    """When every outfit drawn comes out too close to someone already dressed, the seed's own world fails."""
    monkeypatch.setattr(people, "look", lambda dress: ("top", "head", "trousers"))
    with pytest.raises(people.NoRoom):
        people.dress_apart(np.random.default_rng(0), [("top", "head", "trousers")])


@needs_bodies
def test_a_persons_height_is_his_builds_scaled_by_his_size(everyday):
    """A body of size s stands s times his build's height."""
    builds = figures.catalogue().manifest["builds"]
    assert builds
    for build, spec in builds.items():
        assert people.height_m({"build": build}, 1.02) == pytest.approx(spec["height_m"] * 1.02)
    assert people.SIZE_RANGE[0] < 1.0 < people.SIZE_RANGE[1]


# ---------------------------------------------------------------------------------------------------- placement

def test_a_walkers_whole_patch_keeps_the_placement_rules(flat_ground):
    """Two walkers and two still people: every walker's patch lies inside the outline by 2 m, clear of the dock by
    8 m, off the wall and the crown, and every two people's patches are 3 m apart."""
    cli, terrain = flat_ground
    crowns = np.array([[-10.0, -10.0, 4.0]])
    reach = [walkers.PATCH_M, 0.0, walkers.PATCH_M, 0.0]
    for seed in range(4):
        spots = np.array(people.spots(cli, seed, _OUTLINE, terrain, _DOCK[:2], crowns, 4, reach))
        for spot, r in zip(spots, reach):
            assert np.all(np.abs(spot) <= 30.0 - people.PERSON_EDGE_M - r)
            assert np.hypot(*(spot - _DOCK[:2])) >= people.DOCK_CLEAR_M + r
            assert abs(spot[0] - 20.0) > people.OPEN_M + r
            assert np.hypot(*(spot + 10.0)) >= 4.0 + people.CROWN_CLEAR_M + r
        for i, j in combinations(range(4), 2):
            assert np.hypot(*(spots[i] - spots[j])) >= people.APART_M + reach[i] + reach[j]


def test_a_walker_with_no_room_for_his_patch_stands_still_instead(flat_ground):
    """On a field with room for a person but not for a walker's patch, the walker is made to stand and the seed
    keeps its count; with no room for anyone the seed fails."""
    cli, terrain = flat_ground
    small = np.array([[-4.5, -4.5], [4.5, -4.5], [4.5, 4.5], [-4.5, 4.5]])
    away = np.array([50.0, 50.0])
    where, poses, covers, patches, lead = people.place(people.Ground(cli, small, terrain, away, np.zeros((0, 3))), 2, ["walking"])
    assert poses == ["standing"] and len(where) == 1 and covers == ["open"] and patches == [0.0] and lead == [None]
    with pytest.raises(people.NoRoom):
        people.place(people.Ground(cli, small * 0.2, terrain, away, np.zeros((0, 3))), 2, ["walking"])


def test_a_wide_patch_with_no_room_becomes_a_strollers_before_anyone_is_made_to_stand(flat_ground):
    """On a field with room for a stroller's patch but not for a wide one, the walker keeps walking in the small
    patch, and the one who walks with him does too."""
    cli, terrain = flat_ground
    field = np.array([[-7.0, -7.0], [7.0, -7.0], [7.0, 7.0], [-7.0, 7.0]])
    ground = people.Ground(cli, field, terrain, np.array([50.0, 50.0]), np.zeros((0, 3)))
    where, poses, _covers, patches, lead = people.place(ground, 4, ["walking", "walking"], None, [8.0, 8.0], [None, 0])
    assert poses == ["walking", "walking"] and patches == [walkers.PATCH_M, walkers.PATCH_M] and lead == [None, 0]
    assert walkers.BESIDE_M[0] <= np.hypot(*(where[1] - where[0])) <= walkers.BESIDE_M[1]
    assert all(np.all(np.abs(spot) <= 7.0 - people.PERSON_EDGE_M - walkers.PATCH_M) for spot in where)


def test_who_is_with_whom_is_drawn_from_the_people_before_and_keeps_the_rules():
    """Over many draws: about 30 % of the people after the first ask for company; a walker is only ever with one
    walker who walks alone and shares his patch, someone who stays put only with a group of at most four who stay put
    in the open; nobody under or beside a tree is in a group; half the walkers' patches are wide."""
    assert (people.TOGETHER_SHARE, people.GROUP_MOST, walkers.WIDE_SHARE) == (0.30, 4, 0.5)
    assert walkers.WIDE_PATCH_M == (5.0, 8.0) and walkers.PATCH_M == 3.0
    rng = np.random.default_rng(3)
    with_someone = asked = wide = walking = 0
    for _ in range(3000):
        n = int(rng.integers(1, 11))
        poses = [str(rng.choice(["standing", "crouching", "walking"], p=[0.45, 0.30, 0.25])) for _ in range(n)]
        covers = [str(rng.choice(["open", "under", "beside"], p=[0.7, 0.15, 0.15])) if pose != "walking" else "open" for pose in poses]
        lead, patches = people.company(rng, poses, covers)
        asked += n - 1
        for k, first in enumerate(lead):
            walks = poses[k] == "walking"
            assert (patches[k] == walkers.PATCH_M or walkers.WIDE_PATCH_M[0] <= patches[k] <= walkers.WIDE_PATCH_M[1]) if walks else patches[k] == 0.0
            walking += walks
            wide += walks and patches[k] > walkers.PATCH_M
            if first is None:
                continue
            with_someone += 1
            assert first < k and lead[first] is None and (poses[first] == "walking") == walks
            assert covers[k] == covers[first] == "open"
            assert lead.count(first) <= (1 if walks else people.GROUP_MOST - 1)
    assert 0.12 < with_someone / asked < 0.30
    assert 0.46 < wide / walking < 0.54


def test_people_who_are_together_stand_close_and_everyone_else_keeps_apart(flat_ground, monkeypatch):
    """Those placed with the first of their group stand within their gap of him and never on top of each other; any
    two people of different groups keep the usual distance; with no draw allowed beside anyone, he stands alone."""
    cli, terrain = flat_ground
    ground = people.Ground(cli, _OUTLINE, terrain, _DOCK[:2], np.zeros((0, 3)))
    lead = [None, 0, 0, None, 3, None, 0]
    gaps = [people.TOGETHER_M] * 4 + [people.LOW_TOGETHER_M] + [people.TOGETHER_M] * 2
    for seed in range(6):
        where, _kinds, placed = people.layout(ground, seed, 7, None, 0, None, lead, gaps)
        assert placed == lead
        for k, first in enumerate(placed):
            if first is not None:
                assert gaps[k][0] - 1e-9 <= np.hypot(*(where[k] - where[first])) <= gaps[k][1] + 1e-9
        group = lambda k: k if placed[k] is None else placed[k]   # noqa: E731
        for i, j in combinations(range(7), 2):
            gap = np.hypot(*(where[i] - where[j]))
            assert gap >= (people.MEMBER_APART_M if group(i) == group(j) else people.APART_M)
    monkeypatch.setattr(people, "TOGETHER_TRIES", 0)
    where, _kinds, placed = people.layout(ground, 1, 7, None, 0, None, lead, gaps)
    assert placed == [None] * 7
    assert all(np.hypot(*(where[i] - where[j])) >= people.APART_M for i, j in combinations(range(7), 2))
    # A walker who finds no spot beside his leader is placed alone in his own patch, not in the one he would share.
    reach = [6.0, 6.0]
    where, kinds, placed = people.layout(ground, 2, 2, reach, 0, ["open", "beside"], [None, 0], [walkers.BESIDE_M] * 2, [6.0, 3.0])
    assert placed == [None, None] and reach == [6.0, 3.0]
    assert np.hypot(*(where[0] - where[1])) >= people.APART_M + 9.0
    monkeypatch.undo()
    where, kinds, placed = people.layout(ground, 2, 2, [0.0, 0.0], 0, ["open", "beside"], [None, 0])
    assert placed == [None, 0] and kinds == ["open", "open"]
    assert people.layout(ground, 2, 2, None, 0, None, [None, 5])[2] == [None, None]


@needs_bodies
def test_someone_in_a_group_faces_the_one_he_is_with(everyday):
    """The yaw given to a body in a base pose turns its hips along the direction asked, whatever the pose."""
    lib = moves.library()
    for hub in (moves.STAND, *sorted(moves.low(lib))):
        for mirrored in (False, True):
            index = people.pose_frame(hub)
            for toward in ((1.0, 0.0), (0.0, -2.0), (-3.0, 4.0)):
                yaw = people.facing(index, mirrored, toward)
                hips = theft_moves.hips_heading(lib.rotations[int(mirrored)][index][0]) or 0.0
                assert np.allclose(theft_moves.direction(yaw + hips), np.array(toward) / np.hypot(*toward), atol=1e-9)


def _drawn(seed: int, n: int) -> tuple:
    """The poses a day seed draws for n people and who is with whom among them."""
    rng = np.random.default_rng([people.PEOPLE_SEED_STREAM, seed, 2])
    poses = [people._pick(rng, people.POSE_SHARES) for _ in range(n)]
    lead, _patches = people.company(np.random.default_rng([people.PEOPLE_SEED_STREAM, seed, people.TOGETHER_STREAM]), poses)
    return poses, lead


@needs_bodies
def test_a_seeds_groups_stand_turned_to_each_other_and_its_pairs_walk_in_step(everyday, flat_ground, monkeypatch):
    """Put on a field, the people of a seed that draws a group and a walking pair: everyone in the group stands by
    its first, turned to the one he is with, and talks; the two walkers share one track a step apart and strides
    within a few percent, so the second's feet keep to the ground; and nobody else is nearer than the usual 3 m."""
    cli, terrain = flat_ground
    monkeypatch.setattr(people.site, "night", lambda seed: False)
    monkeypatch.setattr(people.site, "wooded", lambda ep: False)
    monkeypatch.setattr(people.site, "passable_outlines", lambda ep: np.zeros((0, 3)))
    monkeypatch.setattr(people.site, "ground_at", lambda cli, uids, x, y: 0.0)
    monkeypatch.setattr(walkers.site, "ground_at", lambda cli, uids, x, y: 0.0)
    seed = next(seed for seed in range(400)
                if any(first is not None and pose == "walking" for pose, first in zip(*_drawn(seed, 8)))
                and any(first is not None and pose != "walking" for pose, first in zip(*_drawn(seed, 8))))
    ep = FieldEpisode(seed=seed)
    ep.fence, ep.terrain_uids, ep.dock_position, ep.people, ep.park = _OUTLINE, terrain, _DOCK.copy(), [], {}
    people._on_field(SimpleNamespace(CLIENT=cli), ep, 8)
    folks = people.people(ep)
    lib = moves.library()
    together = [person for person in folks if person.beside is not None]
    assert {person.pose == "walking" for person in together} == {True, False}
    for person in together:
        first = folks[person.beside]
        assert first.company and person.company and first.beside is None
        gap = np.hypot(*(np.array(person.xy) - first.xy))
        if person.walk is not None:
            assert person.walk.leader is first.walk and first.walk.partner is person.walk
            assert walkers.BESIDE_M[0] - 1e-9 <= gap <= walkers.BESIDE_M[1] + 1e-9
            mine = figures.stride(person.body.dress, person.size)
            assert abs(mine / first.walk.scale - 1.0) < 0.12
            continue
        assert people.TOGETHER_M[0] - 1e-9 <= gap <= people.LOW_TOGETHER_M[1] + 1e-9
        assert person.idle.company and first.idle.company
        for one, other in ((person, first), (first, folks[next(q.index for q in folks if q.beside == first.index)])):
            hips = theft_moves.hips_heading(lib.rotations[int(one.idle.mirrored)][people.pose_frame(one.hub)][0]) or 0.0
            toward = (np.array(other.xy) - one.xy) / np.hypot(*(np.array(other.xy) - one.xy))
            assert float(np.dot(theft_moves.direction(one.yaw + hips), toward)) > math.cos(people.FACE_SWAY_RAD + 0.05)
    group = lambda person: person.index if person.beside is None else person.beside   # noqa: E731
    for a, b in combinations(folks, 2):
        if group(a) != group(b):
            assert np.hypot(*(np.array(a.xy) - b.xy)) >= people.APART_M + a.patch_m + b.patch_m - 1e-9


@needs_bodies
def test_a_wait_standing_is_filled_with_small_moves_and_only_the_rest_is_held(everyday):
    """Standing for some seconds plays the library's small moves while most of one still fits and holds what is
    left, so the whole lasts about as long as asked; a library with none simply holds."""
    lib = moves.library()
    kinds = moves.fidgets(lib)
    assert kinds and set(kinds) <= set(moves.FIDGETS)
    assert any(step.kind in kinds for step in moves.standing(lib, np.random.default_rng(1), 26.0, 1.0, "pause"))
    for seconds in (0.0, 0.4, 3.0, 11.0, 26.0):
        steps = list(moves.standing(lib, np.random.default_rng(1), seconds, 1.0, "pause"))
        played = sum(lib.seconds(lib.kinds({"hub": moves.STAND})[step.kind][0]) for step in steps if step.kind != "hold")
        held = sum(step.hold for step in steps if step.kind == "hold")
        assert all(step.stage == "pause" and (step.kind == "hold" or step.kind in kinds) for step in steps)
        longest = max((lib.seconds(lib.kinds({"hub": moves.STAND})[kind][0]) for kind in kinds), default=0.0)
        assert abs(played + held - seconds) <= (1.0 - moves.FIDGET_FITS) * longest + 1e-6
        assert held <= longest + 1e-6 if kinds else held == pytest.approx(seconds)
    bout = list(moves.bout("idle_talk", 2, "x"))
    assert [(step.kind, step.cycles, step.stage) for step in bout] == [("idle_talk_in", 1, "x"), ("loop", 2, "x"), ("idle_talk_to_stand", 1, "x")]


def test_what_is_done_on_one_spot_leaves_the_body_where_it_stood():
    """Steps played anchored put the track back exactly where and as it stood before them, once they are all played;
    standing played on a track does so after each small move."""
    track = SimpleNamespace(pos=np.array([1.0, 2.0]), yaw=0.5)
    steps = moves.anchored(track, iter(["a", "b"]))
    assert next(steps) == "a"
    track.pos, track.yaw = track.pos + [0.3, -0.1], 0.9
    assert next(steps) == "b"
    track.pos = track.pos + [0.2, 0.2]
    assert next(steps, None) is None
    assert np.array_equal(track.pos, [1.0, 2.0]) and track.yaw == 0.5


@needs_bodies
def test_someone_who_stands_with_others_mostly_talks(everyday):
    """Someone in a group draws what he does from the things people do together; someone alone from all of them."""
    lib = moves.library()
    assert set(moves.COMPANY_HABITS) <= set(moves.HABITS) and moves.COMPANY_HABITS["idle_talk"] == 0.50
    both = {}
    for company in (False, True):
        bouts = []
        for person in range(6):
            idle = idlers.plan(np.random.default_rng([2, person]), moves.STAND, (0.0, 0.0), 0.0, False, 1.0, company)
            track = idlers.actor(idle)
            stages = [track.at(t).stage for t in np.arange(0.0, 600.0, 1.0)]
            bouts += [b for a, b in zip(stages, stages[1:]) if a != b and b not in ("still", "greeting")]
        both[company] = bouts
    assert set(both[True]) <= set(moves.habits(lib, moves.COMPANY_HABITS)) and both[True]
    assert set(both[False]) <= set(moves.habits(lib, moves.HABITS)) and len(set(both[False])) >= 2
    talk = both[True].count("idle_talk") / len(both[True])
    assert 0.35 < talk < 0.65 and talk > 2 * both[False].count("idle_talk") / len(both[False])


# ---------------------------------------------------------------------------------------------------- walkers

@needs_bodies
@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("patch", (walkers.PATCH_M, 5.0, 8.0))
def test_a_walk_never_leaves_its_patch_nor_goes_faster_than_a_brisk_walk_and_never_ends(seed, patch):
    """Over a 200 s patrol with no drone near, every frame of a walker's track lies within his patch, narrow or wide,
    over any second he moves no faster than 1.7 m/s, a brisk walk and never a run, whatever his gait, and he still
    walks in the last minute; on a wide patch he really crosses it."""
    stride = float(np.random.default_rng(seed).uniform(*_STRIDES))
    walk = _walk(seed, home=(4.0, -7.0), stride=stride, patch=patch)
    xy = walkers.track_xy(walk, 200.0)
    assert walk.patch == patch and np.max(np.hypot(*(xy - walk.home).T)) <= patch
    fps = moves.library().fps
    assert walk.gait in moves.gaits(moves.library())
    assert np.max(np.hypot(*(xy[fps:] - xy[:-fps]).T)) <= 1.7
    last = xy[-60 * fps:]
    assert np.max(np.hypot(*(last - last[0]).T)) > 1.0
    if patch > walkers.PATCH_M:
        assert np.ptp(xy, axis=0).max() >= patch


def test_a_mark_across_a_wide_patch_lies_in_it_and_a_good_way_off():
    """Every mark drawn for a wide patch lies within the marks' radius of home and at least the distance asked from
    where he stands; with no such draw he walks straight across home to the rim of the marks."""
    rng = np.random.default_rng(2)
    home = np.array([3.0, -1.0])
    for patch in (5.0, 6.5, 8.0):
        mark = walkers.mark_m(patch)
        assert mark == patch - walkers.WIDE_EDGE_M
        for _ in range(500):
            angle, radius = rng.uniform(-math.pi, math.pi), mark * math.sqrt(rng.uniform(0.0, 1.0))
            here = home + radius * np.array([math.cos(angle), math.sin(angle)])
            end = walkers.far_mark(rng, home, here, mark, walkers.WIDE_LEG_SHARE * mark)
            assert np.hypot(*(end - home)) <= mark + 1e-9
            assert np.hypot(*(end - here)) >= walkers.WIDE_LEG_SHARE * mark - 1e-9
    across = walkers.far_mark(rng, (0.0, 0.0), (1.0, 0.0), 4.0, 99.0)
    assert np.allclose(across, [-4.0, 0.0])
    assert walkers.mark_m(walkers.PATCH_M) == walkers.MARK_M
    # A mark walked to without a stop never doubles back: the turn at the corner stays under about 135 degrees.
    assert walkers.CORNER_COS == -0.7
    for _ in range(500):
        came, here = (home + 5.0 * rng.uniform(-1.0, 1.0, 2) / math.sqrt(2.0) for _ in range(2))
        end = walkers.far_mark(rng, home, here, 6.0, 4.0, came)
        if end is not None:
            ahead, onward = (here - came) / np.hypot(*(here - came)), (end - here) / np.hypot(*(end - here))
            assert float(ahead @ onward) >= walkers.CORNER_COS - 1e-9 and np.hypot(*(end - here)) >= 4.0 - 1e-9
    assert walkers.far_mark(rng, home, home, 4.0, 99.0, home + [1.0, 0.0]) is None


@needs_bodies
def test_two_who_walk_together_keep_one_track_a_step_apart_and_walk_in_step():
    """Someone who walks with another shows the leader's frame at every moment, a fixed step to one side: in step
    with him while they walk, his mirror image while they stand."""
    leader = _walk(4, home=(2.0, 3.0), patch=6.0)
    second = walkers.plan_beside(leader, (2.9, 3.4))
    track = walkers.actor(second)
    assert second.leader is leader and leader.partner is second and second.patch == leader.patch
    assert np.allclose(second.start, np.array(leader.start) + [0.9, 0.4])
    for t in np.arange(0.0, 90.0, 0.7):
        a, b = leader.actor.at(t), track.at(t)
        assert (b.index, b.stage, b.yaw) == (a.index, a.stage, a.yaw)
        assert b.mirrored == (a.mirrored if a.stage == "walking" else not a.mirrored)
        assert (b.x, b.y) == (a.x + second.offset[0], a.y + second.offset[1])
    stages = {leader.actor.at(t).stage for t in np.arange(0.0, 90.0, 0.7)}
    assert "walking" in stages and stages - {"walking"}
    assert np.allclose(walkers.track_xy(second, 30.0), walkers.track_xy(leader, 30.0) + [0.9, 0.4])
    assert np.max(np.hypot(*(walkers.track_xy(second, 200.0) - second.home).T)) <= second.patch


@needs_bodies
def test_someone_crossing_a_wide_patch_stops_at_his_next_stride_for_the_drone_and_never_stands_a_statue():
    """Caught walking by the drone, a walker on a wide patch stands within a few seconds, inside his patch, shows
    more than one frame while it keeps him there, and walks on once it has gone."""
    walk = _walk(9, patch=8.0)
    per_s = int(round(1.0 / SIM_DT))
    caught = next(t for t in np.arange(0.0, 120.0, 0.5) if walk.actor.at(t).stage == "walking"
                  and walk.actor.at(t + 3.0).stage == "walking")
    walk = _walk(9, patch=8.0)

    def across(t):
        """The drone far, then 10 m off for a minute, then gone."""
        return 10.0 if caught <= t < caught + 60.0 else 40.0

    xy = _fly_by(walk, across, caught + 120.0)
    held = xy[int((caught + 8.0) * per_s):int((caught + 60.0) * per_s)]
    assert np.ptp(held, axis=0).max() < 0.5
    assert np.max(np.hypot(*(xy - walk.home).T)) <= walk.patch
    stages = [walk.actor.at(t).stage for t in np.arange(caught + 8.0, caught + 60.0, 0.5)]
    assert set(stages) == {"stopped"}
    assert len({walk.actor.at(t).index for t in np.arange(caught + 8.0, caught + 60.0, 0.5)}) > 3
    assert np.ptp(xy[int((caught + 60.0 + walkers.CALM_S) * per_s):], axis=0).max() > 1.0


@needs_bodies
def test_every_leg_ends_in_the_ring_a_full_leg_away():
    """From anywhere in the ring between inner_m and MARK_M, the next leg's end is a full leg away and in the ring;
    from a point the ring cannot be reached from, he heads back across home."""
    rng = np.random.default_rng(0)
    for stride in _STRIDES:
        leg, inner = walkers.leg_m(stride), walkers.inner_m(stride)
        for _ in range(2000):
            angle, radius = rng.uniform(-math.pi, math.pi), rng.uniform(inner, walkers.MARK_M)
            here = np.array([2.0, 1.0]) + radius * np.array([math.cos(angle), math.sin(angle)])
            mark = walkers.next_mark(rng, (2.0, 1.0), here, leg, inner)
            assert np.hypot(*(mark - here)) == pytest.approx(leg)
            assert inner - 1e-9 <= np.hypot(*(mark - [2.0, 1.0])) <= walkers.MARK_M + 1e-9
    back = walkers.next_mark(rng, (0.0, 0.0), (0.1, 0.0), 4.0, 1.9)
    assert np.allclose(back, [-3.9, 0.0])


def _fly_by(walk: walkers.Walk, across: callable, seconds: float) -> np.ndarray:
    """Play a walker step by step for some seconds while the airborne drone stands across(t) metres from him; his
    position at every step."""
    out = []
    for step in range(int(round(seconds / SIM_DT))):
        t = step * SIM_DT
        frame = walk.actor.at(t)
        walkers.notice(walk, t, across(t), True)
        out.append((frame.x, frame.y))
    return np.array(out)


@needs_bodies
def test_a_walker_stops_for_the_drone_and_walks_on_once_it_has_been_gone_five_seconds():
    """With the drone 10 m off from 30 s he finishes the leg he is on and stands still while it stays within 18 m; once
    it has been further for CALM_S he is free to walk on, and does."""
    walk = _walk(11)
    near, far = 30.0, 80.0

    def across(t):
        """The drone far, then 10 m off, then 15 m off, then gone."""
        return 40.0 if t < near else 10.0 if t < 60.0 else 15.0 if t < far else 40.0

    xy = _fly_by(walk, across, 160.0)
    steps = int(round(1.0 / SIM_DT))
    still = xy[int((near + 9.0) * steps):int((far + walkers.CALM_S) * steps)]
    assert np.ptp(still, axis=0).max() < 0.5
    assert np.ptp(xy[int((far + walkers.CALM_S) * steps):], axis=0).max() > 1.0


@needs_bodies
def test_a_drone_that_hovers_in_a_walkers_way_is_never_walked_into():
    """A drone holding still 12.5 m ahead of someone crossing a wide patch stops him well short of the 5 m rule: he
    never comes nearer than 12 m less the most a walker still goes once caught; and every gait a wide patch draws
    keeps that budget, a patch no gait fits being walked as a stroller's."""
    lib = moves.library()
    for seed in range(6):
        walk = _walk(seed, patch=8.0)
        if walk.patch <= walkers.PATCH_M:
            continue
        assert walkers.halt_m(walk.scale, walk.gait) <= walkers.HALT_MOST_M
        caught = next(t for t in np.arange(0.0, 150.0, 0.25) if walk.actor.at(t).stage == "walking"
                      and walk.actor.at(t + 1.0).stage == "walking" and walk.actor.at(t).x != walk.actor.at(t + 1.0).x)
        a, b = walk.actor.at(caught), walk.actor.at(caught + 1.0)
        ahead = np.array([b.x - a.x, b.y - a.y]) / math.hypot(b.x - a.x, b.y - a.y)
        walk = _walk(seed, patch=8.0)
        here = walk.actor.at(caught)
        hover = np.array([here.x, here.y]) + 12.5 * ahead
        nearest = math.inf
        for step in range(int(round((caught + 60.0) / SIM_DT))):
            t = step * SIM_DT
            frame = walk.actor.at(t)
            across = walkers.across_m(walk, frame, hover)
            walkers.notice(walk, t, across if t >= caught else 99.0, True)
            nearest = min(nearest, across) if t >= caught else nearest
        assert nearest >= walkers.STOP_M - walkers.HALT_MOST_M - 0.1
    assert walkers.STOP_M - walkers.HALT_MOST_M >= 6.0 and set(moves.gaits(lib))
    stride = 1.0
    assert all(walkers.halt_m(stride, gait) <= walkers.HALT_MOST_M for gait in moves.gaits(lib) if walkers.fits(stride, gait, 8.0))


@needs_bodies
def test_a_wide_patch_no_gait_fits_is_walked_as_a_strollers(monkeypatch):
    """With no gait that stops soon enough for a wide patch, the walker strolls in a small one."""
    monkeypatch.setattr(walkers, "HALT_MOST_M", 0.0)
    walk = _walk(3, patch=8.0)
    assert walk.patch == walkers.PATCH_M
    assert np.max(np.hypot(*(walkers.track_xy(walk, 120.0) - walk.home).T)) <= walkers.PATCH_M


def _leg_ends(walk: walkers.Walk, seconds: float, across: callable) -> np.ndarray:
    """Where each leg of a walk ended over some seconds, with the airborne drone across(t) metres off."""
    ends, before = [], None
    for step in range(int(round(seconds / SIM_DT))):
        t = step * SIM_DT
        frame = walk.actor.at(t)
        walkers.notice(walk, t, across(t), True)
        if before is not None and before.stage == "walking" and frame.stage != "walking":
            ends.append((frame.x, frame.y))
        before = frame
    return np.array(ends).reshape(-1, 2)


@needs_bodies
def test_a_fly_by_changes_when_a_stroller_walks_and_not_where():
    """Stopped by the drone for a while, a stroller looks up, stands about and is left exactly where he stopped: the
    legs he walks afterwards end on the marks he walks to with no drone near, one after another."""
    alone = _leg_ends(_walk(8), 200.0, lambda t: 99.0)
    visited = _leg_ends(_walk(8), 200.0, lambda t: 9.0 if 20.0 <= t < 55.0 else 99.0)
    assert 3 <= len(visited) < len(alone)
    assert np.allclose(visited, alone[:len(visited)], atol=0.35)
    walk = _walk(8)
    frames = [walk.actor.at(t) for t in np.arange(0.0, 200.0, 1.0 / 30.0)]
    stops = [b for a, b in zip(frames, frames[1:]) if a.stage == "walking" and b.stage != "walking"]
    starts = [b for a, b in zip(frames, frames[1:]) if a.stage != "walking" and b.stage == "walking"][1:]
    assert len(starts) >= 3
    assert all((go.x, go.y) == (stop.x, stop.y) for stop, go in zip(stops, starts))


@needs_bodies
def test_the_drone_near_either_of_two_who_walk_together_counts_for_both():
    """The distance a pair is noticed by is the nearer of the two's."""
    leader = _walk(4, home=(2.0, 3.0), patch=6.0)
    frame = leader.actor.at(0.0)
    alone = walkers.across_m(leader, frame, (frame.x + 30.0, frame.y))
    walkers.plan_beside(leader, (2.0 + 1.0, 3.0))
    assert alone == pytest.approx(30.0) and walkers.across_m(leader, frame, (frame.x + 30.0, frame.y)) == pytest.approx(29.0)
    assert walkers.across_m(leader, frame, (frame.x - 30.0, frame.y)) == pytest.approx(30.0)


@needs_bodies
def test_a_drone_in_its_dock_never_stops_anyone():
    """A grounded drone right beside a walker neither stops him nor keeps him stopped."""
    walk = _walk(3)
    walkers.notice(walk, 0.0, 1.0, False)
    assert not walk.stopped
    walkers.notice(walk, 1.0, 1.0, True)
    assert walk.stopped
    walkers.notice(walk, 2.0, 1.0, False)
    walkers.notice(walk, 2.0 + walkers.CALM_S, 1.0, False)
    assert not walk.stopped


@needs_bodies
def test_a_walkers_position_at_any_step_up_to_now_reads_his_track(flat_ground):
    """position gives the track's frame at the step asked, the same once the clock has moved on, and now for a step to
    come; the ground under it is the terrain's."""
    cli, terrain = flat_ground
    walk = _walk(5, home=(-5.0, 5.0))
    walk.cli, walk.terrain_uids = cli, terrain
    ep = FieldEpisode(seed=5)
    ep.dock_position = _DOCK.copy()
    ep.people = [people.Person(index=0, pose="walking", xy=walk.home, ground_z=0.0, yaw=0.0, walk=walk)]
    ep.step = 4000
    early, now = people.position(ep, 0, 1500), people.position(ep, 0)
    frame = walk.actor.at(1500 * SIM_DT)
    assert np.allclose(early, [frame.x, frame.y, 0.0])
    ep.step = 9000
    assert np.array_equal(people.position(ep, 0, 1500), early)
    assert np.array_equal(people.position(ep, 0, 4000), now)
    assert np.array_equal(people.position(ep, 0, 10 ** 6), people.position(ep, 0))
    assert np.allclose(people.reference(ep, 0, 1500), early[:2] - _DOCK[:2])


class _Body:
    """A stand-in body that counts the poses it is given."""

    def __init__(self):
        """No pose yet."""
        self.poses = 0

    def pose(self, *_args) -> bool:
        """Count one pose."""
        self.poses += 1
        return True


@needs_bodies
def test_a_walker_is_posed_only_for_a_picture_that_sees_him_and_only_when_his_frame_changed(flat_ground):
    """A picture looking down on him poses him once per new frame; one looking far away never does."""
    cli, terrain = flat_ground
    walk = _walk(7, home=(10.0, 10.0))
    walk.cli, walk.terrain_uids, walk.body = cli, terrain, _Body()
    above = camera.View("colour", (10.0, 10.0, 10.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0), 64, 48, 60.0, True, 0)
    away = camera.View("colour", (500.0, 500.0, 10.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0), 64, 48, 60.0, True, 0)
    walkers.show([walk], 0.0, [away])
    assert walk.body.poses == 0
    walkers.show([walk], 0.0, [above])
    walkers.show([walk], 0.0, [above])
    assert walk.body.poses == 1
    moving = next(t for t in np.arange(0.0, 60.0, 0.5) if walk.actor.at(t).x != walk.actor.at(0.0).x)
    walkers.show([walk], moving, [above])
    assert walk.body.poses == 2


@needs_bodies
def test_a_walkers_body_is_rewritten_once_when_he_is_put_on_the_field(flat_ground, monkeypatch):
    """A body reads right in the thermal camera only once it has been rewritten, so a walker's is rewritten when it
    is built, in the frame his walk begins with, and not again until that frame changes."""
    cli, terrain = flat_ground
    walk = _walk(7, home=(10.0, 10.0))
    built = []

    def figure(cat, dress, first, client, size=1.0, textures=None):
        """A stand-in body that remembers what it was built with."""
        built.append((dress, first, client, size))
        return _Body()
    monkeypatch.setattr(walkers.figures, "Figure", figure)
    monkeypatch.setattr(walkers.figures, "catalogue", lambda: None)
    ep = FieldEpisode(seed=7)
    ep.dock_position, ep.terrain_uids = _DOCK.copy(), terrain
    walkers.start(SimpleNamespace(CLIENT=cli), ep, walk, {"build": 0}, 1.0)
    assert len(built) == 1 and walk.body.poses == 1
    above = camera.View("colour", (10.0, 10.0, 10.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0), 64, 48, 60.0, True, 0)
    walkers.show([walk], 0.0, [above])
    assert walk.body.poses == 1


# ---------------------------------------------------------------------------------------------------- reference pilot

def test_the_pilot_keeps_off_a_walkers_whole_patch():
    """A walker's home is kept off by PERSON_CLEAR_M past his patch, a still person's point by PERSON_CLEAR_M."""
    persons = np.array([[-15.0, 0.0], [15.0, 0.0]])
    free = reference_pilot.Free(_OUTLINE, np.zeros((0, 3)), persons, np.array([walkers.PATCH_M, 0.0]))
    clear = reference_pilot.PERSON_CLEAR_M
    assert not free.holds(np.array([-15.0 + clear + walkers.PATCH_M - 0.6, 0.0]))
    assert free.holds(np.array([-15.0 + clear + walkers.PATCH_M + 0.6, 0.0]))
    assert free.holds(np.array([15.0 - clear - 0.6, 0.0]))


def test_the_pilot_reports_where_a_person_stood_in_the_reported_frame(monkeypatch):
    """The point a report carries is the person's reference point at the step its frame was taken on."""
    asked = []
    monkeypatch.setattr(people, "reference", lambda ep, index, step=None: asked.append((index, step)) or np.zeros(2))
    pilot = SimpleNamespace(ep=None, search=SimpleNamespace(beast=lambda n: None))
    reference_pilot.Pilot.point(pilot, 3, 250)
    assert asked == [(3, 250)]


@needs_bodies
def test_nobody_lies_on_the_ground(everyday):
    """Every base pose a seed can draw keeps the body upright: the head stands well over the hips, whether someone
    stands, squats, kneels or sits."""
    assert set(people.POSE_SHARES) == set(people.NIGHT_POSE_SHARES) == {"standing", "crouching", "walking"}
    lib, cat = moves.library(), figures.catalogue()
    rng = np.random.default_rng(7)
    hubs = {people.pose_hub(rng, pose)[0] for pose in ("standing", "crouching") for _ in range(200)}
    assert moves.STAND in hubs and hubs - {moves.STAND} == set(moves.low(lib)) and "prone" not in hubs
    for hub in sorted(hubs):
        for mirrored in (False, True):
            index = people.pose_frame(hub)
            world = cat.world(0, _quat_to_mat(lib.rotations[int(mirrored)][index]), lib.root[int(mirrored)][index])
            height = world[:, 1, 3]
            assert height[cat.joint_names.index("Head")] - height[cat.joint_names.index("Hips")] > 0.3


@needs_bodies
def test_someone_who_stays_put_is_never_a_statue_and_never_leaves_his_spot(everyday):
    """Over a 200 s patrol with no drone near, someone standing shows many different frames and more than one habit,
    someone low keeps busy in his pose, and neither's body strays from where it was set down."""
    lib = moves.library()
    for hub in (moves.STAND, *sorted(moves.low(lib))):
        idle = idlers.plan(np.random.default_rng([people.PEOPLE_SEED_STREAM, 3, people.LIFE_STREAM, 0]), hub, (4.0, -2.0), 0.3, False, 1.0)
        track = idlers.actor(idle)
        frames = [track.at(t) for t in np.arange(0.0, 200.0, 1.0 / 30.0)]
        assert len({frame.index for frame in frames}) > 20
        assert max(math.hypot(frame.x - 4.0, frame.y + 2.0) for frame in frames) < 0.6
        stages = {frame.stage for frame in frames}
        assert (len(stages - {"still"}) >= 2) if hub == moves.STAND else ("low" in stages)
        # Whatever a clip's own travel, the end of everything he does standing finds him exactly on his spot.
        back = [b for a, b in zip(frames, frames[1:]) if a.stage != "still" and b.stage == "still"]
        assert all((frame.x, frame.y, frame.yaw) == (4.0, -2.0, 0.3) for frame in back)
        assert len(back) >= 2 if hub == moves.STAND else not back


@needs_bodies
def test_someone_standing_greets_the_drone_once_and_goes_back_to_his_habits(everyday):
    """With the airborne drone 10 m off from 20 s to 40 s, someone who greets looks up or waves once within a few
    seconds and then takes his habits up again; a drone in its dock is greeted by nobody."""
    def lived(airborne: bool, until: float = 40.0, seconds: float = 120.0) -> list:
        """The stages of one person's patrol with the drone near from 20 s until a time."""
        idle = idlers.plan(np.random.default_rng(5), moves.STAND, (0.0, 0.0), 0.0, False, 1.0)
        idle.greets = True
        track = idlers.actor(idle)
        out = []
        for step in range(int(round(seconds / SIM_DT))):
            t = step * SIM_DT
            frame = track.at(t)
            idlers.notice(idle, t, 10.0 if 20.0 <= t < until else 99.0, airborne)
            out.append(frame.stage)
        return out
    stages = lived(True)
    per_s = int(round(1.0 / SIM_DT))
    assert "greeting" not in stages[:20 * per_s] and "greeting" in stages[20 * per_s:50 * per_s]
    after = stages[::-1].index("greeting")
    assert after > 5 * per_s and set(stages[-after:]) - {"greeting"}
    assert sum(a != "greeting" and b == "greeting" for a, b in zip(stages, stages[1:])) == 1
    assert "greeting" not in lived(False)
    # Under a drone that stays, he greets it again no sooner than GREET_AGAIN_S after the last time, give or take the
    # few seconds his way out of what he was doing takes.
    stays = lived(True, until=200.0, seconds=200.0)
    begun = [k * SIM_DT for k, (a, b) in enumerate(zip(stays, stays[1:])) if a != "greeting" and b == "greeting"]
    assert 2 <= len(begun) <= 5 and min(np.diff(begun)) >= idlers.GREET_AGAIN_S - 5.0


def test_at_night_fewer_walk_from_the_same_draws():
    """Read off the same draw, a night's pose is a walk for about 15 % of people against 25 % by day; only someone who
    would walk by day changes pose, to crouching; and it takes one draw as a day's does, so every later draw stays in
    its place."""
    day_rng, night_rng = np.random.default_rng(11), np.random.default_rng(11)
    day = [people._pick(day_rng, people.POSE_SHARES) for _ in range(4000)]
    night = [people._pick(night_rng, people.NIGHT_POSE_SHARES) for _ in range(4000)]
    assert day_rng.random() == night_rng.random()
    assert all(d == n or (d, n) == ("walking", "crouching") for d, n in zip(day, night))
    assert 0.13 < night.count("walking") / len(night) < 0.17 < 0.23 < day.count("walking") / len(day)
    assert math.fsum(people.NIGHT_POSE_SHARES.values()) == 1.0
