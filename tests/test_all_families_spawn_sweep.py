from __future__ import annotations

import os
import random

import pybullet as p
import pytest

from swarm.constants import SIM_DT, SAR_MAX_VICTIM_DISTANCE_M
from swarm.core.env_builder.sar_tagging import build_and_tag_map
from swarm.core.env_builder.spawn_pipeline import SARSpawnError, find_spawn_xy
from swarm.domain_model import CHALLENGE_FAMILY_IDS
from swarm.validator.task_gen import random_task

pytestmark = pytest.mark.full


_N_SEEDS = int(os.environ.get("SWARM_SPAWN_SWEEP_SEEDS", "100"))
_SAR_FAMILIES = {"cf_search_and_rescue", "cf_swarm_sar"}


def _seeds(family_id: str) -> list[int]:
    rng = random.Random(f"spawn-sweep:{family_id}")
    return [rng.randrange(1, 2**32) for _ in range(_N_SEEDS)]


@pytest.mark.parametrize("family_id", CHALLENGE_FAMILY_IDS)
def test_every_seed_builds_a_runnable_task(family_id):
    """A seed the validator hands out must always produce a task, and for the SAR
    families a placed victim; anything else is scored as a miner failure."""
    build_failures: list[tuple[int, str]] = []
    spawn_failures: list[tuple[int, int, str]] = []

    for seed in _seeds(family_id):
        try:
            task = random_task(sim_dt=SIM_DT, seed=seed, family_id=family_id)
        except Exception as exc:  # noqa: BLE001
            build_failures.append((seed, f"{type(exc).__name__}: {exc}"))
            continue

        if family_id not in _SAR_FAMILIES:
            continue

        challenge_type = int(task.challenge_type)
        # only the multi-drone families populate starts/goals
        start = tuple(float(c) for c in (task.starts[0] if task.starts else task.start))
        goal = tuple(float(c) for c in (task.goals[0] if task.goals else task.goal))
        cli = p.connect(p.DIRECT)
        try:
            tagger = build_and_tag_map(
                cli, seed=task.map_seed, challenge_type=challenge_type,
                start=start, goal=goal, sar_mode=True,
            )
            find_spawn_xy(
                cli, map_seed=task.map_seed, challenge_type=challenge_type,
                body_tags=tagger.body_tags,
                near=(start[0], start[1]), max_dist=SAR_MAX_VICTIM_DISTANCE_M,
            )
        except SARSpawnError as exc:
            spawn_failures.append((seed, challenge_type, str(exc)))
        finally:
            p.disconnect(cli)

    assert not build_failures, (
        f"{family_id}: {len(build_failures)}/{_N_SEEDS} seeds produced no task: "
        f"{build_failures[:3]}"
    )
    assert not spawn_failures, (
        f"{family_id}: {len(spawn_failures)}/{_N_SEEDS} seeds could not place a victim: "
        f"{spawn_failures[:3]}"
    )
