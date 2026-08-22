"""A drone must not carry a safety penalty for the spot it was placed in.

The spawn sits START_PLATFORM_TAKEOFF_BUFFER above its pad, below
SAFETY_DISTANCE_DANGER, and the episode keeps the minimum clearance, so anything
scored before the drone flies clear is locked in for the whole run.
"""
from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest

from types import SimpleNamespace

from swarm.constants import (
    SAFETY_DISTANCE_DANGER,
    SAFETY_DISTANCE_SAFE,
    SAFETY_DISTANCE_SAFE_BY_TYPE,
    TYPE_6_SAFETY_DISTANCE_SAFE,
)
from swarm.core import moving_drone as moving_drone_mod
from swarm.core.moving_drone import MovingDroneAviary
from swarm.validator.task_gen import random_task

_MAPS = {1: "city", 2: "open", 3: "mountain", 4: "village", 5: "warehouse", 6: "forest"}
_SIM_DT = 1.0 / 30.0
_CLEAR_LIFT = SAFETY_DISTANCE_SAFE + 0.5


def _gate_probe(origin, position, num_drones=1, challenge_type=1):
    env = MovingDroneAviary.__new__(MovingDroneAviary)
    env.task = SimpleNamespace(challenge_type=challenge_type)
    env.NUM_DRONES = num_drones
    env._takeoff_origins = np.array(origin, dtype=float).reshape(num_drones, 3)
    env._takeoff_cleared = np.zeros(num_drones, dtype=bool)
    env.pos = np.array(position, dtype=float).reshape(num_drones, 3)
    return env


def test_grace_holds_until_the_drone_has_flown_clear():
    env = _gate_probe([[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]])
    assert env._in_takeoff_grace(0)

    env.pos[0, 2] = SAFETY_DISTANCE_SAFE - 0.01
    assert env._in_takeoff_grace(0)

    env.pos[0, 2] = SAFETY_DISTANCE_SAFE + 0.01
    assert not env._in_takeoff_grace(0)


def test_the_latch_never_closes_again():
    env = _gate_probe([[0.0, 0.0, 0.0]], [[0.0, 0.0, 5.0]])
    assert not env._in_takeoff_grace(0)

    # Coming home must not buy a second grace period.
    env.pos[0, 2] = 0.0
    assert not env._in_takeoff_grace(0)


def test_each_drone_clears_on_its_own():
    env = _gate_probe(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        [[0.0, 0.0, 5.0], [10.0, 0.0, 0.0]],
        num_drones=2,
    )
    assert not env._in_takeoff_grace(0)
    assert env._in_takeoff_grace(1)


def test_a_collision_still_zeroes_clearance_during_grace(monkeypatch):
    env = MovingDroneAviary.__new__(MovingDroneAviary)
    env.NUM_DRONES = 1
    env._collision = True
    env._min_clearance_episode = SAFETY_DISTANCE_SAFE
    env.task = SimpleNamespace(challenge_type=1)
    env._takeoff_origins = np.zeros((1, 3))
    env._takeoff_cleared = np.zeros(1, dtype=bool)
    env.pos = np.zeros((1, 3))

    def _boom(*a, **k):  # the scan must never be reached
        raise AssertionError("collision path should return before scanning")

    monkeypatch.setattr(moving_drone_mod.p, "getAABB", _boom)
    env._update_min_clearance()

    assert env._min_clearance_episode == 0.0


def _tasks_for(ctype: int, count: int, family: str):
    found = []
    for seed in range(1, 4000):
        if len(found) >= count:
            break
        with contextlib.suppress(Exception):
            task = random_task(sim_dt=_SIM_DT, seed=seed, family_id=family)
            if int(task.challenge_type) == ctype:
                found.append(task)
    return found


def _spawn_then_climb(task, *, sar: bool):
    """Clearance recorded at the spawn pose, and again once the drone is clear."""
    with contextlib.redirect_stdout(io.StringIO()):
        env = MovingDroneAviary(task, ctrl_freq=30, pyb_freq=30, sar_mode=sar)
        env.reset(seed=task.map_seed)
    try:
        env._update_min_clearance()
        at_spawn = float(env._min_clearance_episode)

        lifted = np.array(env.pos[0, :3], dtype=float)
        lifted[2] += _CLEAR_LIFT
        moving_drone_mod.p.resetBasePositionAndOrientation(
            int(env.DRONE_IDS[0]), lifted.tolist(), [0, 0, 0, 1],
            physicsClientId=env.CLIENT,
        )
        env.pos[0, :3] = lifted
        env._update_min_clearance()
        return at_spawn, float(env._min_clearance_episode)
    finally:
        with contextlib.suppress(Exception):
            env.close()


@pytest.mark.parametrize("family,sar", [("cf_search_and_rescue", True), ("cf_autopilot", False)])
def test_warehouse_spawn_is_not_scored(family, sar):
    """The flat warehouse floor is the worst case: it capped every SAR run at 0.9."""
    for task in _tasks_for(5, 1, family):
        at_spawn, after_climb = _spawn_then_climb(task, sar=sar)
        assert at_spawn == SAFETY_DISTANCE_SAFE
        assert after_climb > SAFETY_DISTANCE_DANGER


@pytest.mark.full
@pytest.mark.parametrize("ctype", sorted(_MAPS))
def test_no_map_is_scored_before_the_drone_flies_clear(ctype):
    for family, sar in (("cf_search_and_rescue", True), ("cf_autopilot", False)):
        for task in _tasks_for(ctype, 2, family):
            at_spawn, after_climb = _spawn_then_climb(task, sar=sar)
            assert at_spawn == SAFETY_DISTANCE_SAFE, (
                f"{family} {_MAPS[ctype]} seed {task.map_seed} scored {at_spawn:.3f} at spawn"
            )
            assert after_climb > SAFETY_DISTANCE_DANGER, (
                f"{family} {_MAPS[ctype]} seed {task.map_seed} still at {after_climb:.3f} once clear"
            )


def test_forest_grace_ends_at_its_own_relaxed_bar():
    """Forest scores full safety at 0.6 m, so its grace must not run past that."""
    assert SAFETY_DISTANCE_SAFE_BY_TYPE[6] == TYPE_6_SAFETY_DISTANCE_SAFE
    env = _gate_probe([[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]], challenge_type=6)

    env.pos[0, 2] = TYPE_6_SAFETY_DISTANCE_SAFE - 0.01
    assert env._in_takeoff_grace(0)

    env.pos[0, 2] = TYPE_6_SAFETY_DISTANCE_SAFE + 0.01
    assert not env._in_takeoff_grace(0)
