from __future__ import annotations

import math

import numpy as np
import pytest

from swarm.challenge_families import runtime_family_for_task
from swarm.constants import SWARM_MAX_DRONES, SWARM_MIN_DRONES, SWARM_SEARCH_RADIUS
from swarm.utils.env_factory import make_env
from swarm.validator import task_gen


@pytest.mark.parametrize("challenge_type", [1, 2, 3, 4, 5, 6])
def test_swarm_task_gen_deterministic_and_distinct(challenge_type):
    kwargs = dict(sim_dt=0.02, seed=4242, challenge_type=challenge_type, family_id="cf_swarm_autopilot")
    t1 = task_gen.task_for_seed_and_type(**kwargs)
    t2 = task_gen.task_for_seed_and_type(**kwargs)

    assert t1 == t2
    assert SWARM_MIN_DRONES <= t1.num_drones <= SWARM_MAX_DRONES
    assert len(t1.starts) == t1.num_drones
    assert len(t1.goals) == t1.num_drones
    assert tuple(t1.start) == tuple(t1.starts[0])
    assert tuple(t1.goal) == tuple(t1.goals[0])
    assert t1.moving_platform is False

    for i in range(t1.num_drones):
        for j in range(i + 1, t1.num_drones):
            d = math.hypot(t1.starts[i][0] - t1.starts[j][0], t1.starts[i][1] - t1.starts[j][1])
            assert d > 0.5, f"starts {i},{j} nearly coincident on type {challenge_type}"


def test_swarm_count_varies_with_seed_and_stays_in_range():
    counts = {
        task_gen.task_for_seed_and_type(
            sim_dt=0.02, seed=s, challenge_type=2, family_id="cf_swarm_autopilot",
        ).num_drones
        for s in range(50)
    }
    assert len(counts) > 1, "drone count never varies across seeds"
    assert all(SWARM_MIN_DRONES <= c <= SWARM_MAX_DRONES for c in counts)


def test_swarm_task_gen_seed_changes_layout():
    a = task_gen.task_for_seed_and_type(sim_dt=0.02, seed=1, challenge_type=2, family_id="cf_swarm_autopilot")
    b = task_gen.task_for_seed_and_type(sim_dt=0.02, seed=2, challenge_type=2, family_id="cf_swarm_autopilot")
    assert a.starts != b.starts


def _rollout(seed, steps=40):
    task = task_gen.task_for_seed_and_type(
        sim_dt=1 / 30, seed=seed, challenge_type=2, family_id="cf_swarm_autopilot",
    )
    family = runtime_family_for_task(task)
    env = make_env(task, gui=False)
    try:
        obs, info = env.reset(seed=task.map_seed)
        n = env.NUM_DRONES
        assert SWARM_MIN_DRONES <= n <= SWARM_MAX_DRONES
        assert obs["depth"].shape[0] == n
        assert obs["state"].ndim == 2 and obs["state"].shape[0] == n
        state_width = int(obs["state"].shape[1])
        for _ in range(steps):
            _o, _r, term, trunc, info = env.step(np.zeros((n, 5), dtype=np.float32))
            if term or trunc:
                break
        return family.score_swarm(task, info), state_width, n
    finally:
        env.close()


def test_swarm_rollout_scores_for_random_count():
    result, _width, n = _rollout(2025)
    assert 0.0 <= result["final_score"] <= 1.0
    assert len(result["per_drone_final_score"]) == n
    assert math.isclose(
        result["final_score"], sum(result["per_drone_final_score"]) / n, rel_tol=1e-9,
    )


def test_swarm_obs_row_width_is_count_invariant():
    seed_for_n = {}
    for s in range(60):
        t = task_gen.task_for_seed_and_type(
            sim_dt=1 / 30, seed=s, challenge_type=2, family_id="cf_swarm_autopilot",
        )
        seed_for_n.setdefault(t.num_drones, s)
    counts = sorted(seed_for_n)
    assert len(counts) >= 2, "need two different drone counts to compare"

    _r_low, width_low, n_low = _rollout(seed_for_n[counts[0]])
    _r_high, width_high, n_high = _rollout(seed_for_n[counts[-1]])
    assert n_low != n_high
    assert width_low == width_high, "observation row width must not depend on drone count"


def test_swarm_smoke_obs_batches_and_action_validates():
    from swarm.domain_model import get_policy_interface_contract
    from swarm.policy_interface import (
        PolicyInterfaceError,
        build_smoke_test_observation,
        validate_action_output,
    )

    contract = get_policy_interface_contract("cf_swarm_autopilot", "submission_zip.v1")
    action_space = contract["action_space"]
    for n in (SWARM_MIN_DRONES, SWARM_MAX_DRONES):
        obs = build_smoke_test_observation("cf_swarm_autopilot", "submission_zip.v1", num_drones=n)
        assert obs["depth"].shape == (n, 128, 128, 1)
        assert obs["state"].shape == (n, 190)
        validate_action_output(np.zeros((n, 5), dtype=np.float32), action_space, num_drones=n)

    with pytest.raises(PolicyInterfaceError):
        validate_action_output(np.zeros((3, 5), dtype=np.float32), action_space, num_drones=SWARM_MAX_DRONES)


def test_score_single_drone_matches_autopilot_formula():
    from swarm.challenge_families.autopilot import AutopilotChallengeFamily
    from swarm.protocol import MapTask
    from swarm.validator.reward import _calculate_target_time, _score_single_drone

    ap = AutopilotChallengeFamily()
    task = MapTask(
        map_seed=1, start=(0.0, 0.0, 0.0), goal=(3.0, 4.0, 0.0),
        sim_dt=0.02, horizon=60.0, challenge_type=1, family_id="cf_autopilot",
    )
    target = _calculate_target_time(task)
    cases = [
        dict(success=True, t=0.1, min_clearance=2.0, collision=False, failure_reason="NONE"),
        dict(success=True, t=0.1, min_clearance=0.6, collision=False, failure_reason="NONE"),
        dict(success=True, t=40.0, min_clearance=2.0, collision=False, failure_reason="NONE"),
        dict(success=True, t=1.0, min_clearance=2.0, collision=True, failure_reason="NONE"),
        dict(success=False, t=2.0, min_clearance=1.0, collision=False, failure_reason="OBSTACLE_COLLISION"),
        dict(success=False, t=0.0, min_clearance=1.0, collision=False, failure_reason="TIMEOUT"),
        dict(success=True, t=0.1, min_clearance=None, collision=False, failure_reason="NONE"),
    ]
    for c in cases:
        single = _score_single_drone(
            success=c["success"], t=c["t"], horizon=60.0, target_time=target,
            min_clearance=c["min_clearance"], collision=c["collision"],
            challenge_type=task.challenge_type, legitimate_model=True,
            failure_reason=c["failure_reason"],
        )
        metrics = ap.build_rollout_metrics(
            task=task, success=c["success"], t=c["t"], horizon=60.0,
            min_clearance=c["min_clearance"], collision=c["collision"],
            legitimate_model=True, failure_reason=c["failure_reason"],
        )
        expected = ap.normalize_rollout_metrics(task=task, metrics=metrics)["final_score"]
        assert single == expected, (c, single, expected)


def test_swarm_uses_shared_search_clue_and_platform_pool():
    task = task_gen.task_for_seed_and_type(
        sim_dt=1 / 30, seed=2025, challenge_type=2, family_id="cf_swarm_autopilot",
    )
    env = make_env(task, gui=False)
    try:
        env.reset(seed=task.map_seed)
        n = env.NUM_DRONES
        assert hasattr(env, "_search_area_center")
        assert len(env._swarm_platform_groups) == n     # N logical platforms in the shared pool
        assert len(env._swarm_claimed) == 0             # nothing claimed at the start
        centroid = np.asarray(env.GOAL_POSES, dtype=float).mean(axis=0)
        off = float(np.hypot(env._search_area_center[0] - centroid[0], env._search_area_center[1] - centroid[1]))
        assert off <= SWARM_SEARCH_RADIUS * 1.5         # noisy clue, within the (bigger) radius
    finally:
        env.close()


def test_swarm_rollout_is_deterministic():
    r1, _w1, _n1 = _rollout(2025)
    r2, _w2, _n2 = _rollout(2025)
    assert r1["final_score"] == r2["final_score"]
    assert r1["per_drone_final_score"] == r2["per_drone_final_score"]
