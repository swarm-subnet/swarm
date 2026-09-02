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

import math
import random

import pybullet as p
import pytest

from swarm.core.env_builder.body_tagger import BodyTagger
from swarm.core.env_builder.victim import (
    _CHALLENGE_TYPE_TO_MAP,
    _load_characters,
    select_victim_split_dir,
    spawn_victim,
)


def _folder_of(split_dir) -> str:
    return str(split_dir).split("/")[-4]


def test_selection_deterministic_and_map_scoped():
    characters = _load_characters()
    for ct, map_name in _CHALLENGE_TYPE_TO_MAP.items():
        pool = {c["folder"] for c in characters if map_name in c.get("maps", [])}
        assert pool, f"no victims registered for {map_name}"
        used = set()
        for seed in range(400):
            first = select_victim_split_dir(seed, ct)
            second = select_victim_split_dir(seed, ct)
            assert first is not None
            assert str(first) == str(second)
            folder = _folder_of(first)
            assert folder in pool
            used.add(folder)
        assert used == pool


def test_unknown_challenge_type_falls_back_to_default():
    assert select_victim_split_dir(123, 99) is None


def test_selection_respects_slope():
    characters = _load_characters()
    steep_ok = {
        c["folder"] for c in characters
        if "mountain" in c.get("maps", []) and c.get("max_slope_deg", 30.0) >= 45.0
    }
    assert steep_ok
    for seed in range(200):
        steep = select_victim_split_dir(seed, 3, slope_deg=45.0)
        assert steep is not None
        assert _folder_of(steep) in steep_ok
    flat_used = {_folder_of(select_victim_split_dir(s, 3, slope_deg=0.0)) for s in range(200)}
    assert flat_used - steep_ok


@pytest.mark.full
@pytest.mark.parametrize("ct", sorted(_CHALLENGE_TYPE_TO_MAP))
def test_selected_victim_spawns_grounded(ct):
    cli = p.connect(p.DIRECT)
    try:
        for seed in range(0, 56, 7):
            split_dir = select_victim_split_dir(seed, ct)
            assert split_dir is not None
            tagger = BodyTagger(cli)
            uids, (mn, mx), centre = spawn_victim(
                cli, surface_x=0.0, surface_y=0.0, surface_z=0.0,
                rng=random.Random(seed), tagger=tagger, split_dir=split_dir,
            )
            assert uids
            assert abs(mn[2] - 0.005) < 0.02
            assert mx[2] - mn[2] > 0.05
            assert all(math.isfinite(v) for v in centre)
            for u in uids:
                p.removeBody(u, physicsClientId=cli)
    finally:
        p.disconnect(cli)
