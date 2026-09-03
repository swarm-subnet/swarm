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

"""The warehouse start pad must never be placed inside geometry.

Stacked pallets and crates are built one body per layer, each under a metre tall, so an
AABB height filter cannot see them. These tests pin the launch volume clear instead.
"""
from __future__ import annotations

import contextlib
import io

import pybullet as p
import pytest

from swarm.core.env_builder.platform import (
    PAD_MAX_RELOCATION,
    PAD_PROBE_RADIUS,
    resolve_warehouse_pad_spot,
)
from swarm.core.env_builder.sar_types import BodyCategory
from swarm.validator.task_gen import task_for_seed_and_type

# Map seeds reported as spawning the drone inside pallet stacks and a crate.
REPORTED_SEEDS = (4219133080, 3132192837, 3107369929, 1065244613)
BOUNDS = (38.0, 23.0)


def _floor(cli, half=20.0, top=0.0):
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half, half, 0.1],
                                 physicsClientId=cli)
    return p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                             basePosition=[0.0, 0.0, top - 0.1], physicsClientId=cli)


def _box(cli, x, y, half_z, *, solid=True):
    half = [0.8, 0.8, half_z]
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half,
                                 physicsClientId=cli) if solid else -1
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, physicsClientId=cli)
    return p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                             baseVisualShapeIndex=vis,
                             basePosition=[x, y, half_z], physicsClientId=cli)


def _resolve(cli, tags, sx=0.0, sy=0.0, exclude=()):
    return resolve_warehouse_pad_spot(
        cli, sx, sy, body_tags=tags,
        accepted_categories={BodyCategory.SUPPORT_FLOOR},
        exclude=set(exclude), bounds=BOUNDS,
    )


def test_a_short_solid_body_moves_the_pad(sar_pybullet):
    """A 0.30 m pallet is exactly what the old height filter ignored."""
    cli = sar_pybullet
    floor = _floor(cli)
    _box(cli, 0.0, 0.0, 0.15)
    tags = {floor: BodyCategory.SUPPORT_FLOOR.value}

    spot = _resolve(cli, tags)

    assert spot is not None
    x, y, surface_z = spot
    assert surface_z == pytest.approx(0.0, abs=1e-3)
    assert (x * x + y * y) ** 0.5 > PAD_PROBE_RADIUS, "pad stayed inside the pallet"


def test_a_visual_only_body_does_not_move_the_pad(sar_pybullet):
    """Floor markings have an AABB but no collision shape; they must not block."""
    cli = sar_pybullet
    floor = _floor(cli)
    _box(cli, 0.0, 0.0, 0.15, solid=False)
    tags = {floor: BodyCategory.SUPPORT_FLOOR.value}

    spot = _resolve(cli, tags)

    assert spot == pytest.approx((0.0, 0.0, 0.0), abs=1e-3)


def test_no_resolvable_floor_is_not_treated_as_ground(sar_pybullet):
    """An unresolved surface used to become z = 0.0 and place the pad in mid-air."""
    cli = sar_pybullet
    _floor(cli)  # present but never tagged as an accepted support

    assert _resolve(cli, {}) is None


def test_a_fully_blocked_world_reports_failure(sar_pybullet):
    cli = sar_pybullet
    floor = _floor(cli, half=3.0)
    tags = {floor: BodyCategory.SUPPORT_FLOOR.value}
    step = PAD_PROBE_RADIUS
    span = int(4.0 / step) + 1
    for i in range(-span, span + 1):
        for j in range(-span, span + 1):
            _box(cli, i * step, j * step, 0.15)

    assert _resolve(cli, tags) is None


def test_the_pad_stays_out_of_the_victim_keep_out(sar_pybullet):
    """Relocating onto the victim ends the episode on the first step."""
    cli = sar_pybullet
    floor = _floor(cli)
    _box(cli, 0.0, 0.0, 0.15)
    tags = {floor: BodyCategory.SUPPORT_FLOOR.value}
    keep_out = (2.0, 0.0, 1.4)

    spot = resolve_warehouse_pad_spot(
        cli, 0.0, 0.0, body_tags=tags,
        accepted_categories={BodyCategory.SUPPORT_FLOOR},
        exclude=set(), bounds=BOUNDS, keep_out=keep_out,
    )

    assert spot is not None
    x, y, _z = spot
    assert ((x - keep_out[0]) ** 2 + (y - keep_out[1]) ** 2) ** 0.5 >= keep_out[2]


def test_the_pad_never_relocates_past_the_limit(sar_pybullet):
    """A distant pad turns a solvable seed infeasible, so the search gives up instead."""
    cli = sar_pybullet
    floor = _floor(cli, half=40.0)
    tags = {floor: BodyCategory.SUPPORT_FLOOR.value}
    step = PAD_PROBE_RADIUS
    span = int((PAD_MAX_RELOCATION + 2.0) / step)
    for i in range(-span, span + 1):
        for j in range(-span, span + 1):
            _box(cli, i * step, j * step, 0.15)

    assert _resolve(cli, tags) is None


def test_placement_is_deterministic(sar_pybullet):
    cli = sar_pybullet
    floor = _floor(cli)
    _box(cli, 0.0, 0.0, 0.15)
    tags = {floor: BodyCategory.SUPPORT_FLOOR.value}

    assert _resolve(cli, tags) == _resolve(cli, tags)


@pytest.mark.full
@pytest.mark.timeout(900)
@pytest.mark.parametrize("map_seed", REPORTED_SEEDS)
def test_reported_seed_spawns_clear_of_obstacles(map_seed):
    """The four seeds a miner reported as an unavoidable first-tick collision."""
    from swarm.core.moving_drone import MovingDroneAviary

    task = task_for_seed_and_type(
        1 / 30, seed=map_seed, challenge_type=5, family_id="cf_search_and_rescue",
    )
    with contextlib.redirect_stdout(io.StringIO()):
        env = MovingDroneAviary(task, ctrl_freq=30, pyb_freq=30, sar_mode=True)
        env.reset(seed=map_seed)
    try:
        assert env._sar_spawn_failed is False
        cli = env.CLIENT
        exempt = set(env._platform_uids) | {int(env.DRONE_IDS[0])}
        plane_id = getattr(env, "PLANE_ID", None)
        if plane_id is not None:
            exempt.add(int(plane_id))
        contacts = [
            c for c in p.getContactPoints(bodyA=int(env.DRONE_IDS[0]),
                                          physicsClientId=cli)
            if c[2] not in exempt and c[2] != -1
        ]
        assert contacts == [], f"seed {map_seed} spawns touching {contacts[0][2]}"
    finally:
        with contextlib.suppress(Exception):
            env.close()
