from __future__ import annotations

import random
from typing import Optional, Tuple

import pybullet as p

from .body_tagger import BodyTagger
from .sar_tagging import build_and_tag_map, enumerate_bodies, tag_world_after_build
from .sar_types import SafetyPatch, SARWorld
from .search_clue import sample_search_centre
from .spawn_pipeline import find_spawn_xy
from .victim import spawn_victim


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
    )

    rng = random.Random(seed ^ 0xA5A5A5A5)
    victim_uids, union_aabb, victim_centre = spawn_victim(
        cli,
        surface_x=spawn_x,
        surface_y=spawn_y,
        surface_z=hit.surface_z,
        rng=rng,
        tagger=tagger,
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
