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

from __future__ import annotations

import random
from typing import Optional, Tuple

import pybullet as p

from swarm.constants import SAR_MAX_VICTIM_DISTANCE_M

from .sar_tagging import build_and_tag_map, enumerate_bodies, tag_world_after_build
from .sar_types import SafetyPatch, SARWorld
from .search_clue import sample_search_centre
from .spawn_pipeline import SARSpawnError, find_spawn_xy
from .victim import select_victim_split_dir, spawn_victim, terrain_slope_deg, victim_scale_for


def build_sar_world(
    cli: int,
    *,
    seed: int,
    challenge_type: int,
    start: Optional[Tuple[float, float, float]] = None,
    goal: Optional[Tuple[float, float, float]] = None,
) -> SARWorld:
    n_before = p.getNumBodies(physicsClientId=cli)
    tagger = build_and_tag_map(
        cli, seed=seed, challenge_type=challenge_type,
        start=start, goal=goal, sar_mode=True,
    )

    spawn_x, spawn_y, hit = find_spawn_xy(
        cli,
        map_seed=seed,
        challenge_type=challenge_type,
        body_tags=tagger.body_tags,
        near=(float(start[0]), float(start[1])) if start is not None else None,
        max_dist=SAR_MAX_VICTIM_DISTANCE_M,
    )

    rng = random.Random(seed ^ 0xA5A5A5A5)
    slope_deg = terrain_slope_deg(cli, spawn_x, spawn_y, hit.surface_z)
    split_dir = select_victim_split_dir(seed, challenge_type, slope_deg=slope_deg)
    if split_dir is None:
        raise SARSpawnError(
            f"no victim asset available for challenge_type={challenge_type}"
        )
    victim_uids, union_aabb, victim_centre = spawn_victim(
        cli,
        surface_x=spawn_x,
        surface_y=spawn_y,
        surface_z=hit.surface_z,
        rng=rng,
        tagger=tagger,
        split_dir=split_dir,
        scale=victim_scale_for(challenge_type),
    )

    n_after = p.getNumBodies(physicsClientId=cli)
    new_uids = enumerate_bodies(cli)[n_before:n_after]
    tag_world_after_build(
        cli,
        tagger,
        challenge_type=challenge_type,
        body_range=new_uids,
        victim_uids=victim_uids,
        support_uid=hit.support_uid,
    )

    safety_patch = SafetyPatch(
        support_uid=hit.support_uid,
        xy=(spawn_x, spawn_y),
        surface_z=hit.surface_z,
    )

    sc_rng = random.Random(seed ^ 0x5A5A5A5A)
    search_centre = sample_search_centre(sc_rng, (victim_centre[0], victim_centre[1]))

    return SARWorld(
        victim_uids=list(victim_uids),
        victim_aabb=union_aabb,
        victim_centre=victim_centre,
        support_uid=hit.support_uid,
        support_category=hit.category,
        surface_z=hit.surface_z,
        safety_patch=safety_patch,
        body_tags=dict(tagger.body_tags),
        adjusted_start=start,
        search_centre=search_centre,
    )
