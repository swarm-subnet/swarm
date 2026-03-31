from __future__ import annotations

import pybullet as p
import pytest

from swarm.constants import SIM_DT
from swarm.core.env_builder.generation import _build_static_world
from swarm.validator.task_gen import task_for_seed_and_type


_DETERMINISM_SEEDS = {
    1: 2,
    2: 2,
    3: 2,
    4: 208280,
    5: 2,
    6: 2,
}


def _body_layout_fingerprint(cli: int) -> tuple[tuple[float, ...], ...]:
    fingerprint: list[tuple[float, ...]] = []
    for body_id in range(p.getNumBodies(physicsClientId=cli)):
        pos, orn = p.getBasePositionAndOrientation(body_id, physicsClientId=cli)
        visual_shapes = p.getVisualShapeData(body_id, physicsClientId=cli) or []
        fingerprint.append(
            (
                round(float(pos[0]), 9),
                round(float(pos[1]), 9),
                round(float(pos[2]), 9),
                round(float(orn[0]), 9),
                round(float(orn[1]), 9),
                round(float(orn[2]), 9),
                round(float(orn[3]), 9),
                float(len(visual_shapes)),
            )
        )
    return tuple(fingerprint)


def _fresh_static_world_snapshot(challenge_type: int, seed: int) -> tuple[tuple[float, ...], ...]:
    task = task_for_seed_and_type(sim_dt=SIM_DT, seed=seed, challenge_type=challenge_type)
    cli = p.connect(p.DIRECT)
    try:
        _build_static_world(
            seed=task.map_seed,
            cli=cli,
            start=task.start,
            goal=task.goal,
            challenge_type=task.challenge_type,
        )
        return _body_layout_fingerprint(cli)
    finally:
        p.disconnect(cli)


@pytest.mark.parametrize("challenge_type", [1, 2, 3, 4, 5, 6])
def test_static_world_same_seed_is_deterministic(challenge_type: int) -> None:
    seed = _DETERMINISM_SEEDS[challenge_type]
    first = _fresh_static_world_snapshot(challenge_type, seed)
    second = _fresh_static_world_snapshot(challenge_type, seed)
    assert first == second, (challenge_type, seed)
