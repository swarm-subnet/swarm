from __future__ import annotations

import math
import os
import random
from unittest import mock

import pybullet as p
import pytest

from swarm.core.env_builder.body_tagger import BodyTagger
from swarm.core.env_builder.sar_types import BodyCategory
from swarm.core.env_builder import victim as victim_mod
from swarm.core.env_builder.victim import spawn_victim


def _make_floor(cli, tagger, z=0.0):
    col = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=[10.0, 10.0, 0.05], physicsClientId=cli,
    )
    return tagger.create_body(
        BodyCategory.SUPPORT_TERRAIN,
        baseMass=0.0,
        baseCollisionShapeIndex=col,
        basePosition=[0.0, 0.0, z],
    )


def test_deterministic_per_seed(sar_pybullet):
    cli = sar_pybullet
    p.resetSimulation(physicsClientId=cli)
    tagger = BodyTagger(cli)
    _make_floor(cli, tagger)
    rng_a = random.Random(123)
    rng_b = random.Random(123)
    uids_a, _aabb_a, centre_a = spawn_victim(
        cli, surface_x=0.0, surface_y=0.0, surface_z=0.05,
        rng=rng_a, tagger=tagger,
    )

    p.resetSimulation(physicsClientId=cli)
    tagger_b = BodyTagger(cli)
    _make_floor(cli, tagger_b)
    uids_b, _aabb_b, centre_b = spawn_victim(
        cli, surface_x=0.0, surface_y=0.0, surface_z=0.05,
        rng=rng_b, tagger=tagger_b,
    )
    assert len(uids_a) == len(uids_b)
    assert centre_a == pytest.approx(centre_b)


def test_different_seeds_differ(sar_pybullet):
    cli = sar_pybullet
    p.resetSimulation(physicsClientId=cli)
    tagger = BodyTagger(cli)
    _make_floor(cli, tagger)
    rng_a = random.Random(1)
    rng_b = random.Random(2)
    _, aabb_a, _ = spawn_victim(
        cli, surface_x=0.0, surface_y=0.0, surface_z=0.05, rng=rng_a, tagger=tagger,
    )
    p.resetSimulation(physicsClientId=cli)
    tagger = BodyTagger(cli)
    _make_floor(cli, tagger)
    _, aabb_b, _ = spawn_victim(
        cli, surface_x=0.0, surface_y=0.0, surface_z=0.05, rng=rng_b, tagger=tagger,
    )
    assert aabb_a != aabb_b


def test_returns_uid_list_tagged_victim(sar_pybullet):
    cli = sar_pybullet
    p.resetSimulation(physicsClientId=cli)
    tagger = BodyTagger(cli)
    _make_floor(cli, tagger)
    uids, _aabb, _centre = spawn_victim(
        cli, surface_x=0.5, surface_y=-0.5, surface_z=0.05,
        rng=random.Random(7), tagger=tagger,
    )
    assert len(uids) >= 5
    for u in uids:
        assert tagger.body_tags[u] == "VICTIM"


def test_uses_prebaked_parts(sar_pybullet, monkeypatch):
    cli = sar_pybullet
    p.resetSimulation(physicsClientId=cli)
    tagger = BodyTagger(cli)
    _make_floor(cli, tagger)
    open_calls: list[str] = []
    real_open = open
    def _tracked_open(path, *args, **kwargs):
        open_calls.append(str(path))
        return real_open(path, *args, **kwargs)
    monkeypatch.setattr("swarm.core.env_builder.mesh_loader._PREBAKED_OPEN_FILE", _tracked_open)
    spawn_victim(
        cli, surface_x=0.0, surface_y=0.0, surface_z=0.05,
        rng=random.Random(0), tagger=tagger,
    )
    raw_open_count = sum(1 for p in open_calls if p.endswith("mannequin_a_raw.obj"))
    split_open_count = sum(1 for p in open_calls if "/split/" in p)
    assert raw_open_count == 0
    assert split_open_count >= 6


def test_collision_centred_on_bounds(sar_pybullet):
    cli = sar_pybullet
    p.resetSimulation(physicsClientId=cli)
    tagger = BodyTagger(cli)
    _make_floor(cli, tagger)
    uids, aabb, centre = spawn_victim(
        cli, surface_x=2.0, surface_y=-3.0, surface_z=0.05,
        rng=random.Random(11), tagger=tagger,
    )
    coll_min, coll_max = p.getAABB(uids[0], physicsClientId=cli)
    coll_centre = (
        0.5 * (coll_min[0] + coll_max[0]),
        0.5 * (coll_min[1] + coll_max[1]),
        0.5 * (coll_min[2] + coll_max[2]),
    )
    for a, b in zip(coll_centre, centre):
        assert abs(a - b) < 0.05, f"collision centre {coll_centre} != victim centre {centre}"


def test_loads_from_arbitrary_cwd(sar_pybullet, tmp_path, monkeypatch):
    cli = sar_pybullet
    p.resetSimulation(physicsClientId=cli)
    tagger = BodyTagger(cli)
    _make_floor(cli, tagger)
    monkeypatch.chdir(tmp_path)
    uids, _aabb, _centre = spawn_victim(
        cli, surface_x=0.0, surface_y=0.0, surface_z=0.05,
        rng=random.Random(99), tagger=tagger,
    )
    assert len(uids) >= 5
