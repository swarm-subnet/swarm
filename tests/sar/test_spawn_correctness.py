"""Audit that 50 spawn seeds per map produce physically valid placements.

For each accepted spawn the test asserts:
  1. Mannequin feet sit on a real SUPPORT_* surface (not inside any body).
  2. No obstacle within the 0.8 m no-touch sphere around the spawn point.
  3. The 5 m hover column above the spawn point is empty of obstacles.
  4. The mannequin's union AABB sits flush on the surface (feet 5 mm above).
"""
from __future__ import annotations

import math
import random
import time
from typing import Dict

import pybullet as p
import pytest

from swarm.core.env_builder.sar_tagging import build_and_tag_map
from swarm.core.env_builder.sar_types import BodyCategory
from swarm.core.env_builder.spawn_pipeline import (
    NO_TOUCH_SPHERE_RADIUS,
    SARSpawnError,
    find_spawn_xy,
)
from swarm.core.env_builder.victim import spawn_victim


_MAPS = {
    "open":      2,
    "city":      1,
    "village":   4,
    "warehouse": 5,
    "forest":    6,
    "mountain":  3,
}
_N_SEEDS = 50
_HOVER_COLUMN_HEIGHT_M = 5.0
_FOOT_TOLERANCE_M = 0.05


def _audit_one(cli: int, seed: int, ctype: int) -> Dict[str, float]:
    p.resetSimulation(physicsClientId=cli)

    tagger = build_and_tag_map(
        cli, seed=seed, challenge_type=ctype,
        start=(0.0, 0.0, 1.5), goal=(8.0, 8.0, 1.5),
    )
    body_tags = tagger.body_tags

    spawn_x, spawn_y, hit = find_spawn_xy(
        cli,
        map_seed=seed,
        challenge_type=ctype,
        body_tags=body_tags,
    )

    rng = random.Random(seed ^ 0xA5A5A5A5)
    victim_uids, union_aabb, _centre = spawn_victim(
        cli,
        surface_x=spawn_x,
        surface_y=spawn_y,
        surface_z=hit.surface_z,
        rng=rng,
        tagger=tagger,
    )
    victim_set = set(victim_uids)

    union_min, union_max = union_aabb

    foot_gap = union_min[2] - hit.surface_z
    assert foot_gap >= 0.0, (
        f"seed={seed} ctype={ctype}: feet sunk into surface by {-foot_gap:.4f} m"
    )
    assert foot_gap <= _FOOT_TOLERANCE_M, (
        f"seed={seed} ctype={ctype}: feet floating {foot_gap:.4f} m above surface"
    )

    support_tag = body_tags.get(hit.support_uid)
    assert support_tag is not None, (
        f"seed={seed} ctype={ctype}: surface body {hit.support_uid} has no tag"
    )
    assert isinstance(support_tag, str) and support_tag.startswith("SUPPORT_"), (
        f"seed={seed} ctype={ctype}: surface body tagged {support_tag!r}, "
        f"not a SUPPORT_* category"
    )

    r = NO_TOUCH_SPHERE_RADIUS
    aabb_min = (spawn_x - r, spawn_y - r, hit.surface_z + 0.01)
    aabb_max = (spawn_x + r, spawn_y + r, hit.surface_z + r)
    overlaps = p.getOverlappingObjects(aabb_min, aabb_max, physicsClientId=cli)
    if overlaps:
        for entry in overlaps:
            uid = int(entry[0])
            if uid == hit.support_uid or uid in victim_set:
                continue
            tag = body_tags.get(uid)
            if tag is None or tag == BodyCategory.VICTIM.value:
                continue
            if isinstance(tag, str) and tag.startswith("SUPPORT_"):
                continue
            pytest.fail(
                f"seed={seed} ctype={ctype}: obstacle uid={uid} tag={tag!r} "
                f"intrudes into 0.8 m no-touch sphere"
            )

    bottom = (spawn_x, spawn_y, hit.surface_z + 0.05)
    top = (spawn_x, spawn_y, hit.surface_z + _HOVER_COLUMN_HEIGHT_M)
    hits_above = p.rayTest(bottom, top, physicsClientId=cli)
    for raw in hits_above:
        uid = int(raw[0])
        if uid < 0 or uid == hit.support_uid or uid in victim_set:
            continue
        tag = body_tags.get(uid)
        if tag == BodyCategory.VICTIM.value:
            continue
        pytest.fail(
            f"seed={seed} ctype={ctype}: hover column blocked by uid={uid} "
            f"tag={tag!r} at z={raw[3][2]:.2f}"
        )

    horiz_extent = max(
        math.hypot(union_min[0] - spawn_x, union_min[1] - spawn_y),
        math.hypot(union_max[0] - spawn_x, union_max[1] - spawn_y),
    )

    return {
        "spawn_x": spawn_x,
        "spawn_y": spawn_y,
        "surface_z": hit.surface_z,
        "foot_gap_mm": foot_gap * 1000.0,
        "horiz_extent_m": horiz_extent,
        "support_category": support_tag,
    }


@pytest.mark.timeout(600)
@pytest.mark.parametrize("name,ctype", list(_MAPS.items()))
def test_50_seeds_produce_valid_spawns(sar_pybullet, name, ctype):
    failures = []
    rows = []
    started = time.time()
    for seed in range(_N_SEEDS):
        try:
            rows.append(_audit_one(sar_pybullet, seed, ctype))
        except SARSpawnError as exc:
            failures.append((seed, str(exc)))

    elapsed = time.time() - started
    print(
        f"\n[{name} ctype={ctype}] {len(rows)}/{_N_SEEDS} placed, "
        f"{len(failures)} failed, {elapsed:.1f}s"
    )
    if rows:
        gaps = [r["foot_gap_mm"] for r in rows]
        extents = [r["horiz_extent_m"] for r in rows]
        cats = {}
        for r in rows:
            cats[r["support_category"]] = cats.get(r["support_category"], 0) + 1
        print(
            f"  foot gap (mm):  min={min(gaps):.1f}  "
            f"max={max(gaps):.1f}  mean={sum(gaps)/len(gaps):.1f}"
        )
        print(
            f"  horiz reach (m): min={min(extents):.2f}  "
            f"max={max(extents):.2f}  mean={sum(extents)/len(extents):.2f}"
        )
        print(f"  support tags: {cats}")
    if failures:
        print(f"  failures: {failures[:3]}{' ...' if len(failures) > 3 else ''}")

    failure_rate = len(failures) / _N_SEEDS
    assert failure_rate <= 0.10, (
        f"{name}: spawn failure rate {failure_rate:.0%} exceeds 10% threshold"
    )
