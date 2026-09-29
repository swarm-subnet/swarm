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

"""The intruders (task 4): outfits drawn per seed by the agreed rules, posing that matches plain skinning, and the
thief's visual bodies in the engine."""

from __future__ import annotations

import os
from collections import Counter

import numpy as np
import pybullet as p
import pytest
import swarm_worlds

from swarm.challenge_families.solar_patrol import intruders

FOLDER = os.path.join(swarm_worlds.maps_dir(), intruders.ASSET_DIR)
pytestmark = pytest.mark.skipif(not os.path.isfile(os.path.join(FOLDER, "intruders.json")),
                                reason="the installed swarm-worlds has no intruders yet")

STANDING = (np.tile([0.0, 0.0, 0.0, 1.0], (30, 1)), np.array([0.0, 0.95, 0.0]), (0.0, 0.0, 0.0), 0.0)


@pytest.fixture
def client():
    """A DIRECT client of its own, closed after the test."""
    cid = p.connect(p.DIRECT)
    yield cid
    p.disconnect(cid)


def _moved(seed: int) -> tuple:
    """A standing frame with every joint turned a little, the same for a given seed."""
    rng = np.random.default_rng(seed)
    axes = rng.normal(size=(30, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    half = rng.uniform(0.0, 0.25, size=(30, 1))
    local = np.concatenate([axes * np.sin(half), np.cos(half)], 1)
    return local, np.array([0.3, 0.9, -0.2]), (5.0, -3.0, 12.0), 0.7


def test_outfit_is_the_same_for_the_same_seed():
    """The same generator seed dresses the same thief."""
    assert intruders.outfit(np.random.default_rng(42)) == intruders.outfit(np.random.default_rng(42))


def test_outfit_rules_hold_over_many_thieves():
    """About 7 in 10 cover the face; a raised hood takes no hat; nothing rides on a lowered hood; fits match the top;
    one garment per slot; colours come from each piece's palette; every garment and tool turns up."""
    cat = intruders.catalogue()
    rng = np.random.default_rng(7)
    dresses = [intruders.outfit(rng) for _ in range(3000)]
    masked = np.mean([bool({"balaclava", "balaclava3", "gaiter"} & set(d["garments"])) for d in dresses])
    assert 0.66 <= masked <= 0.74
    for d in dresses:
        garments = set(d["garments"])
        slots = Counter(cat.manifest["garments"][g]["slot"] for g in garments)
        assert slots["top"] == slots["bottom"] == slots["feet"] == 1 and max(slots.values()) == 1
        assert d["build"] in cat.builds
        top = next(g for g in garments if cat.manifest["garments"][g]["slot"] == "top")
        if top == "hoodie_up":
            assert not garments & {"cap", "beanie", "hair_short"}
        carried = [g for g in garments if g.startswith(("backpack", "cable_coil"))]
        if top == "hoodie":
            assert carried == []
        fit = intruders.FIT.get(top, "")
        assert all(name in ("backpack" + fit, "cable_coil" + fit) for name in carried)
        for garment in garments:
            for piece in cat.pieces[garment]:
                assert d["colours"][piece["file"]] in piece["colours"]
    assert set().union(*(d["garments"] for d in dresses)) == set(cat.manifest["garments"])


def test_posing_matches_plain_skinning_on_the_body(client):
    """The one-product blend gives every drawn body vertex what skinning it bone by bone gives, to a micrometre."""
    cat = intruders.catalogue()
    garments = ["tshirt", "jeans", "trainers", "cap"]
    dress = {"build": "tall_heavy", "garments": garments, "skin": [0.6, 0.45, 0.3], "undershirt": [0.1, 0.1, 0.1],
             "colours": {pc["file"]: pc["colours"][0] for g in garments for pc in cat.pieces[g]}}
    frame = _moved(3)
    thief = intruders.Intruder(dress, frame, client)
    build = cat.builds.index("tall_heavy")
    joint = cat.world(build, intruders._quat_to_mat(frame[0]), frame[1]) @ np.linalg.inv(cat.build_bind[build])
    ids = thief.body_vertices
    rest = np.concatenate([cat.build_verts[build][ids], np.ones((len(ids), 1))], 1)
    reference = np.zeros((len(ids), 3))
    for k in range(cat.weight_joints.shape[1]):
        reference += cat.weights[ids, k, None] * np.einsum("nij,nj->ni", joint[cat.weight_joints[ids, k], :3], rest)
    reference = intruders._to_zup(reference)
    c, s = np.cos(frame[3]), np.sin(frame[3])
    reference = np.stack([reference[:, 0] * c - reference[:, 1] * s, reference[:, 0] * s + reference[:, 1] * c,
                          reference[:, 2]], 1)
    got = thief.vertices(*frame)[:len(ids)]
    assert len(ids) > 1000
    assert np.abs((got - got[0]) - (reference - reference[0])).max() < 1e-6


def test_thief_is_one_visual_body_per_colour(client):
    """Skin, the shirt under a long top, and one body for every colour piece worn and carried; removal clears them."""
    cat = intruders.catalogue()
    garments = ["puffer", "cargo", "work_boots", "beanie", "gaiter", "gloves", "bolt_cutters", "cable_coil_puffer"]
    dress = {"build": "stocky", "garments": garments, "skin": [0.5, 0.35, 0.25], "undershirt": [0.2, 0.2, 0.2],
             "colours": {pc["file"]: pc["colours"][0] for g in garments for pc in cat.pieces[g]}}
    thief = intruders.Intruder(dress, STANDING, client)
    bodies = len(thief.bodies)
    assert bodies == sum(len(cat.pieces[g]) for g in garments) + 2
    height = np.ptp(thief.vertices(*STANDING)[:, 2])
    assert 1.5 < height < 2.1
    before = p.getNumBodies(physicsClientId=client)
    thief.remove()
    assert thief.bodies == [] and p.getNumBodies(physicsClientId=client) == before - bodies


def test_pose_moves_the_thief_where_the_engine_can(client):
    """A new frame lands the lowest point on the place, and rewrites every piece in place or reports it cannot."""
    thief = intruders.Intruder(intruders.outfit(np.random.default_rng(5)), STANDING, client)
    after = thief.vertices(*_moved(9))
    assert np.abs(after - thief.vertices(*STANDING)).max() > 0.5
    assert np.isclose(after[:, 2].min(), 12.0)
    if thief.pose(*_moved(9)):
        # Visual-only bodies have no bounding box, so the engine's own object mask shows where they now are.
        view = p.computeViewMatrix([5.0, -7.0, 13.0], [5.0, -3.0, 12.9], [0, 0, 1], physicsClientId=client)
        proj = p.computeProjectionMatrixFOV(40, 1.0, 0.1, 20.0, physicsClientId=client)
        mask = np.asarray(p.getCameraImage(64, 64, view, proj, renderer=p.ER_TINY_RENDERER, physicsClientId=client)[4])
        assert np.isin(mask, thief.bodies).sum() > 50
    else:
        assert thief.moving is False
