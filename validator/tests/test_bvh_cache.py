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

"""Collision trees cached on disk: the epoch folder the validator owns, and the map shapes that use it."""

from __future__ import annotations

import json
import os

import pybullet as p
import pytest

from swarm.core.maps.office.builder import _asset, _collision_shape
from swarm.core.mountain_generator_parts._shared import MOUNTAIN_DIR, SNOW
from swarm.core.mountain_generator_parts.terrain import _ShapeCache

needs_cache_flag = pytest.mark.skipif(
    not hasattr(p, "GEOM_CONCAVE_BVH_CACHE"), reason="engine wheel without the BVH cache flag"
)

HILL = os.path.join(MOUNTAIN_DIR, "2.obj")
HILL_SCALE = [12.0, 12.0, 6.5]
RAYS_FROM = [[x * 3.0 - 20.0, y * 3.0 - 20.0, 80.0] for x in range(14) for y in range(14)]
RAYS_TO = [[x, y, -5.0] for x, y, _ in RAYS_FROM]


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    """A fresh cache folder handed to the engine for the duration of one test."""
    folder = tmp_path / "bvh"
    folder.mkdir()
    monkeypatch.setenv("SWARM_BVH_CACHE_DIR", str(folder))
    return folder


def _hill_hits(cli):
    """Ray hits on one hill spawned through the mountain shape cache."""
    _, col = _ShapeCache().get(cli, HILL, HILL_SCALE, SNOW)
    p.createMultiBody(0, col, -1, [0, 0, 0], physicsClientId=cli)
    return _hits(cli)


def _office_hits(cli):
    """Ray hits on one office piece spawned through the office collision helper."""
    col = _collision_shape(cli, _asset("office_shell.obj"), (1.03, 0.97, 1.01))
    p.createMultiBody(0, col, -1, [0, 0, 0], physicsClientId=cli)
    return _hits(cli)


def _hits(cli):
    """Hit fractions and points of a fixed ray grid, plus the AABB of body 0."""
    rays = p.rayTestBatch(RAYS_FROM, RAYS_TO, physicsClientId=cli)
    return [(r[2], r[3]) for r in rays], p.getAABB(0, physicsClientId=cli)


def _fresh_world(spawn):
    """Spawn in a new DIRECT client, query it, and disconnect."""
    cli = p.connect(p.DIRECT)
    try:
        return spawn(cli)
    finally:
        p.disconnect(cli)


@needs_cache_flag
@pytest.mark.parametrize("spawn", [_hill_hits, _office_hits], ids=["mountain_hill", "office_piece"])
def test_loaded_tree_matches_built_tree(cache_dir, spawn):
    """The first spawn writes one tree file, the second loads it, and both see identical hits."""
    built = _fresh_world(spawn)
    files = sorted(cache_dir.iterdir())
    assert len(files) == 1 and files[0].suffix == ".bvh"
    stamp = (files[0].stat().st_size, files[0].stat().st_mtime)
    loaded = _fresh_world(spawn)
    assert loaded == built
    assert (files[0].stat().st_size, files[0].stat().st_mtime) == stamp
    assert any(fraction < 1.0 for fraction, _ in built[0])


@needs_cache_flag
def test_no_cache_dir_means_no_files(tmp_path, monkeypatch):
    """Without the folder the engine builds as before and writes nothing."""
    monkeypatch.delenv("SWARM_BVH_CACHE_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    _fresh_world(_hill_hits)
    assert list(tmp_path.iterdir()) == []


def test_seed_manager_owns_the_epoch_cache_folder(reload_module, monkeypatch, tmp_path):
    """Loading an epoch points the engine at that epoch's folder and drops the older ones."""
    module = reload_module("swarm.validator.seed_manager")
    state_dir = tmp_path / "state"
    seeds_dir = state_dir / "epoch_seeds"
    monkeypatch.setattr(module, "STATE_DIR", state_dir)
    monkeypatch.setattr(module, "EPOCH_SEEDS_DIR", seeds_dir)
    monkeypatch.delenv(module.BVH_CACHE_ENV, raising=False)
    seeds_dir.mkdir(parents=True)
    (seeds_dir / "epoch_7.json").write_text(json.dumps({
        "epoch_number": 7,
        "family_id": module.DEFAULT_RUNTIME_FAMILY_ID,
        "seeds": list(range(module.BENCHMARK_TOTAL_SEED_COUNT)),
    }))
    stale = state_dir / "bvh_cache" / "epoch_6"
    stale.mkdir(parents=True)
    (stale / "old.bvh").write_bytes(b"x")

    manager = module.BenchmarkSeedManager()

    assert manager.epoch_number == 7
    assert os.environ[module.BVH_CACHE_ENV] == str(state_dir / "bvh_cache" / "epoch_7")
    assert (state_dir / "bvh_cache" / "epoch_7").is_dir()
    assert not stale.exists()

    manager.align_to_epoch(8)

    assert os.environ[module.BVH_CACHE_ENV] == str(state_dir / "bvh_cache" / "epoch_8")
    assert not (state_dir / "bvh_cache" / "epoch_7").exists()
