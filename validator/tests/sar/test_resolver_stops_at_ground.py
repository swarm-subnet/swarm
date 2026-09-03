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

"""The resolver must never hand back a surface that sits under solid ground.

Mountain peaks and the base terrain share the SUPPORT_TERRAIN tag, so a resolver
that keeps descending past a rejected peak lands on the terrain *inside* the
mountain and buries the victim under the rock.
"""
from __future__ import annotations

import math
import random

import pybullet as p
import pytest

from swarm.core.env_builder.body_tagger import BodyTagger
from swarm.core.env_builder.sar_tagging import build_and_tag_map
from swarm.core.env_builder.sar_types import BodyCategory, SUPPORT_CATEGORIES
from swarm.core.env_builder.surface_resolver import resolve_surface


def _ground(cli, tagger, *, z=0.0):
    col = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=[40.0, 40.0, 0.1], physicsClientId=cli,
    )
    return tagger.create_body(
        BodyCategory.SUPPORT_TERRAIN,
        baseMass=0.0,
        baseCollisionShapeIndex=col,
        basePosition=[0.0, 0.0, z],
    )


def _steep_rock(cli, tagger, *, z, tilt_deg=60.0):
    col = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=[6.0, 6.0, 0.1], physicsClientId=cli,
    )
    return tagger.create_body(
        BodyCategory.SUPPORT_TERRAIN,
        baseMass=0.0,
        baseCollisionShapeIndex=col,
        basePosition=[0.0, 0.0, z],
        baseOrientation=p.getQuaternionFromEuler([0.0, math.radians(tilt_deg), 0.0]),
    )


def test_unusable_ground_does_not_expose_the_terrain_beneath(sar_pybullet):
    cli = sar_pybullet
    tagger = BodyTagger(cli)
    _ground(cli, tagger, z=0.0)
    _steep_rock(cli, tagger, z=20.0)
    assert resolve_surface(cli, 0.0, 0.0, tagger.body_tags, SUPPORT_CATEGORIES) is None


def test_the_ray_starts_above_the_tallest_peak(sar_pybullet):
    cli = sar_pybullet
    tagger = BodyTagger(cli)
    _ground(cli, tagger, z=0.0)
    _ground(cli, tagger, z=120.0)
    hit = resolve_surface(cli, 0.0, 0.0, tagger.body_tags, SUPPORT_CATEGORIES)
    assert hit is not None
    assert hit.surface_z == pytest.approx(120.1)


def test_mountain_hits_have_open_sky_above_them(sar_pybullet):
    cli = sar_pybullet
    tagger = build_and_tag_map(
        cli, seed=80_020, challenge_type=3, start=(0.0, 0.0, 1.5), goal=(8.0, 8.0, 1.5),
    )
    rng = random.Random(3)
    checked = 0
    for _ in range(400):
        x, y = rng.uniform(-30.0, 30.0), rng.uniform(-30.0, 30.0)
        hit = resolve_surface(cli, x, y, tagger.body_tags, SUPPORT_CATEGORIES)
        if hit is None:
            continue
        checked += 1
        above = p.rayTest(
            [x, y, hit.surface_z + 0.05], [x, y, 300.0], physicsClientId=cli,
        )
        blocker = int(above[0][0]) if above else -1
        assert blocker in (-1, hit.support_uid), (
            f"({x:.1f}, {y:.1f}) resolved to z={hit.surface_z:.2f} with body "
            f"{blocker} overhead"
        )
    assert checked > 100


def test_the_warehouse_floor_is_still_found_under_its_roof(sar_pybullet):
    cli = sar_pybullet
    tagger = build_and_tag_map(
        cli, seed=80_000, challenge_type=5, start=(0.0, 0.0, 1.5), goal=(6.0, 6.0, 1.5),
    )
    rng = random.Random(5)
    floors = 0
    for _ in range(200):
        hit = resolve_surface(
            cli, rng.uniform(-12.0, 12.0), rng.uniform(-12.0, 12.0),
            tagger.body_tags, {BodyCategory.SUPPORT_FLOOR},
        )
        if hit is not None and hit.category == BodyCategory.SUPPORT_FLOOR.value:
            floors += 1
    assert floors > 100
