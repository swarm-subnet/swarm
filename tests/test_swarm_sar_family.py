from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from swarm.challenge_families import runtime_family_for_task
from swarm.challenge_families.swarm_sar import SwarmSarChallengeFamily
from swarm.constants import (
    SAR_DWELL_SEC,
    SAR_NO_TOUCH_RADIUS,
    SWARM_MAX_DRONES,
    SWARM_MIN_DRONES,
    SWARM_SAR_SEARCH_RADIUS,
)
from swarm.protocol import MapTask
from swarm.utils.env_factory import make_env
from swarm.validator import task_gen


@pytest.mark.parametrize("challenge_type", [1, 2, 3, 4, 5, 6])
def test_swarm_sar_task_gen_deterministic_and_distinct(challenge_type):
    kwargs = dict(sim_dt=0.02, seed=4242, challenge_type=challenge_type, family_id="cf_swarm_sar")
    t1 = task_gen.task_for_seed_and_type(**kwargs)
    t2 = task_gen.task_for_seed_and_type(**kwargs)

    assert t1 == t2
    assert SWARM_MIN_DRONES <= t1.num_drones <= SWARM_MAX_DRONES
    assert len(t1.starts) == t1.num_drones
    assert tuple(t1.start) == tuple(t1.starts[0])
    assert t1.family_id == "cf_swarm_sar"
    assert t1.moving_platform is False

    for i in range(t1.num_drones):
        for j in range(i + 1, t1.num_drones):
            d = math.hypot(t1.starts[i][0] - t1.starts[j][0], t1.starts[i][1] - t1.starts[j][1])
            assert d > 0.5, f"starts {i},{j} nearly coincident on type {challenge_type}"


def test_swarm_sar_count_varies_with_seed_and_stays_in_range():
    counts = {
        task_gen.task_for_seed_and_type(
            sim_dt=0.02, seed=s, challenge_type=2, family_id="cf_swarm_sar",
        ).num_drones
        for s in range(50)
    }
    assert len(counts) > 1
    assert all(SWARM_MIN_DRONES <= c <= SWARM_MAX_DRONES for c in counts)


def test_swarm_sar_seed_changes_layout():
    a = task_gen.task_for_seed_and_type(sim_dt=0.02, seed=1, challenge_type=2, family_id="cf_swarm_sar")
    b = task_gen.task_for_seed_and_type(sim_dt=0.02, seed=2, challenge_type=2, family_id="cf_swarm_sar")
    assert a.starts != b.starts


def _rollout(seed, steps=50):
    task = task_gen.task_for_seed_and_type(
        sim_dt=1 / 30, seed=seed, challenge_type=2, family_id="cf_swarm_sar",
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
            _o, _r, term, trunc, info = env.step(
                np.zeros((n, env.action_space.shape[-1]), dtype=np.float32)
            )
            if term or trunc:
                break
        return family.score_swarm(task, info), state_width, n
    finally:
        env.close()


def test_swarm_sar_rollout_scores_for_random_count():
    result, _width, n = _rollout(2025)
    assert 0.0 <= result["final_score"] <= 1.0
    assert len(result["per_drone_final_score"]) == n


def test_swarm_sar_obs_row_width_is_count_invariant():
    seed_for_n = {}
    for s in range(60):
        t = task_gen.task_for_seed_and_type(
            sim_dt=1 / 30, seed=s, challenge_type=2, family_id="cf_swarm_sar",
        )
        seed_for_n.setdefault(t.num_drones, s)
    counts = sorted(seed_for_n)
    assert len(counts) >= 2

    _r_low, width_low, n_low = _rollout(seed_for_n[counts[0]])
    _r_high, width_high, n_high = _rollout(seed_for_n[counts[-1]])
    assert n_low != n_high
    assert width_low == width_high, "observation row width must not depend on drone count"


def test_swarm_sar_smoke_obs_batches_and_action_validates():
    from swarm.domain_model import get_policy_interface_contract
    from swarm.policy_interface import (
        PolicyInterfaceError,
        build_smoke_test_observation,
        validate_action_output,
    )

    contract = get_policy_interface_contract("cf_swarm_sar", "submission_zip.v1")
    action_space = contract["action_space"]
    for n in (SWARM_MIN_DRONES, SWARM_MAX_DRONES):
        obs = build_smoke_test_observation("cf_swarm_sar", "submission_zip.v1", num_drones=n)
        assert obs["depth"].shape == (n, 256, 256, 1)
        assert obs["rgb"].shape == (n, 256, 256, 3)
        assert obs["state"].shape == (n, 214)
        validate_action_output(np.zeros((n, 6), dtype=np.float32), action_space, num_drones=n)

    with pytest.raises(PolicyInterfaceError):
        validate_action_output(np.zeros((3, 6), dtype=np.float32), action_space, num_drones=SWARM_MAX_DRONES)


def test_swarm_sar_shared_clue_and_single_victim():
    task = task_gen.task_for_seed_and_type(
        sim_dt=1 / 30, seed=2025, challenge_type=2, family_id="cf_swarm_sar",
    )
    env = make_env(task, gui=False)
    try:
        env.reset(seed=task.map_seed)
        n = env.NUM_DRONES
        assert env.sar_world is not None
        assert hasattr(env, "_search_area_center")
        # one shared victim; GOAL_POSES are all the same victim centre
        assert env.GOAL_POSES.shape[0] == n
        assert np.allclose(env.GOAL_POSES - env.GOAL_POSES[0], 0.0)
        vc = np.asarray(env.sar_world.victim_centre, dtype=float)
        off = float(np.hypot(env._search_area_center[0] - vc[0], env._search_area_center[1] - vc[1]))
        assert off <= SWARM_SAR_SEARCH_RADIUS + 1e-6
    finally:
        env.close()


def test_swarm_sar_rollout_is_deterministic():
    r1, _w1, _n1 = _rollout(2025)
    r2, _w2, _n2 = _rollout(2025)
    assert r1["final_score"] == r2["final_score"]
    assert r1["per_drone_final_score"] == r2["per_drone_final_score"]


def _sar_task(n=4):
    starts = tuple((float(i), 0.0, 1.0) for i in range(n))
    goals = tuple((10.0, 0.0, 0.0) for _ in range(n))
    t = MapTask(
        map_seed=1, start=starts[0], goal=goals[0], sim_dt=0.02, horizon=60.0,
        challenge_type=2, family_id="cf_swarm_sar", num_drones=n, starts=starts, goals=goals,
    )
    t.search_centre = (5.0, 0.0)
    return t


def _base_info(n, **over):
    info = {
        "num_drones": n,
        "per_drone_success": [False] * n,
        "per_drone_collision": [False] * n,
        "per_drone_min_clearance": [2.0] * n,
        "per_drone_failure_reason": ["NONE"] * n,
        "sar_team_success": False,
        "sar_team_t": None,
        "sar_mission_failed": False,
        "sar_team_failure_reason": "TIMEOUT",
        "swarm_adjusted_starts": [list((float(i), 0.0, 1.0)) for i in range(n)],
    }
    info.update(over)
    return info


def test_score_swarm_team_find_clean():
    fam = SwarmSarChallengeFamily()
    task = _sar_task(4)
    info = _base_info(4, sar_team_success=True, sar_team_t=5.0)
    out = fam.score_swarm(task, info)
    assert out["success"] is True
    # clean find: success + time + positive safety
    assert out["final_score"] > 0.9
    assert all(s == out["final_score"] for s in out["per_drone_final_score"])


def test_score_swarm_crash_zeros_safety_but_keeps_find():
    fam = SwarmSarChallengeFamily()
    task = _sar_task(4)
    col = [False, False, True, False]
    info = _base_info(4, sar_team_success=True, sar_team_t=5.0, per_drone_collision=col)
    out = fam.score_swarm(task, info)
    assert out["success"] is True
    # crash forces the 0.10 safety term to zero -> max 0.90
    assert out["final_score"] <= 0.90 + 1e-9
    assert out["final_score"] >= 0.45


def test_score_swarm_no_touch_fails_mission():
    fam = SwarmSarChallengeFamily()
    task = _sar_task(4)
    info = _base_info(
        4, sar_team_success=True, sar_team_t=5.0,
        sar_mission_failed=True, sar_team_failure_reason="NO_TOUCH_SPHERE",
    )
    out = fam.score_swarm(task, info)
    assert out["success"] is False
    assert out["final_score"] == pytest.approx(0.01)  # participation only


def test_score_swarm_timeout_participation():
    fam = SwarmSarChallengeFamily()
    task = _sar_task(4)
    info = _base_info(4)  # never found
    out = fam.score_swarm(task, info)
    assert out["success"] is False
    assert out["failure_reason"] == "TIMEOUT"
    assert out["final_score"] == pytest.approx(0.01)


def test_score_swarm_all_crash_reports_collision():
    fam = SwarmSarChallengeFamily()
    task = _sar_task(3)
    info = _base_info(
        3, per_drone_collision=[True, True, True],
        per_drone_failure_reason=["OBSTACLE_COLLISION", "OBSTACLE_COLLISION", "OBSTACLE_COLLISION"],
    )
    out = fam.score_swarm(task, info)
    assert out["success"] is False
    assert out["failure_reason"] == "OBSTACLE_COLLISION"
    assert out["final_score"] == pytest.approx(0.01)


def test_score_swarm_spawn_failure_reports_reason():
    fam = SwarmSarChallengeFamily()
    task = _sar_task(4)
    info = _base_info(4, sar_team_failure_reason="SPAWN_FAILURE")
    out = fam.score_swarm(task, info)
    assert out["success"] is False
    assert out["failure_reason"] == "SPAWN_FAILURE"
    assert out["final_score"] == pytest.approx(0.01)


class _FakeEnv:
    def __init__(self, n, victim_centre, victim_top_z, drone_pos, drone_vel):
        self.NUM_DRONES = n
        self._sim_dt = 0.1
        self._time_alive = 0.0
        self.sar_world = SimpleNamespace(
            victim_centre=np.asarray(victim_centre, dtype=float),
            victim_aabb=(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, float(victim_top_z)])),
        )
        self._d_sar_predicate_active = [False] * n
        self._d_sar_dwell_time = [0.0] * n
        self._d_sar_min_horizontal = [float("inf")] * n
        self._d_sar_min_sphere = [float("inf")] * n
        self._sar_team_success = False
        self._sar_team_t = None
        self._sar_mission_failed = False
        self._sar_team_failure_reason = "TIMEOUT"
        self._d_success = [False] * n
        self._d_t_to_goal = [None] * n
        self._d_failure_reason = ["NONE"] * n
        self._pos = np.asarray(drone_pos, dtype=float)
        self._vel = np.asarray(drone_vel, dtype=float)

    def _getDroneStateVector(self, i):
        s = np.zeros(20, dtype=float)
        s[0:3] = self._pos
        s[10:13] = self._vel
        return s


def test_update_sar_dwell_multi_confirms_after_dwell():
    fam = SwarmSarChallengeFamily()
    # hover 3 m above a victim whose top is z=2 (centre z=1), within 2 m, near-zero speed
    env = _FakeEnv(2, victim_centre=(0.0, 0.0, 1.0), victim_top_z=2.0,
                   drone_pos=(0.0, 0.0, 5.0), drone_vel=(0.0, 0.0, 0.0))
    steps = int(SAR_DWELL_SEC / env._sim_dt) + 1
    for k in range(steps):
        env._time_alive = k * env._sim_dt
        fam.update_sar_dwell_multi(env, 0)
    assert env._sar_team_success is True
    assert env._d_success[0] is True
    assert env._sar_mission_failed is False


def test_update_sar_dwell_multi_no_touch_fails():
    fam = SwarmSarChallengeFamily()
    # inside the 0.8 m no-touch sphere of the victim
    env = _FakeEnv(2, victim_centre=(0.0, 0.0, 1.0), victim_top_z=1.2,
                   drone_pos=(0.0, 0.0, 1.0 + SAR_NO_TOUCH_RADIUS / 2.0), drone_vel=(0.0, 0.0, 0.0))
    fam.update_sar_dwell_multi(env, 0)
    assert env._sar_mission_failed is True
    assert env._sar_team_failure_reason == "NO_TOUCH_SPHERE"
    assert env._sar_team_success is False


def test_no_touch_overrides_same_step_confirm_regardless_of_order():
    # A teammate may have already confirmed earlier in the same step; a no-touch
    # breach by any drone must still fail the mission (no order dependence).
    fam = SwarmSarChallengeFamily()
    env = _FakeEnv(2, victim_centre=(0.0, 0.0, 1.0), victim_top_z=1.2,
                   drone_pos=(0.0, 0.0, 1.0 + SAR_NO_TOUCH_RADIUS / 2.0), drone_vel=(0.0, 0.0, 0.0))
    env._sar_team_success = True  # drone 0 confirmed first this step
    fam.update_sar_dwell_multi(env, 1)
    assert env._sar_mission_failed is True
    assert env._sar_team_failure_reason == "NO_TOUCH_SPHERE"
    task = _sar_task(2)
    info = _base_info(2, sar_team_success=True, sar_team_t=5.0,
                      sar_mission_failed=True, sar_team_failure_reason="NO_TOUCH_SPHERE")
    out = fam.score_swarm(task, info)
    assert out["success"] is False
    assert out["failure_reason"] == "NO_TOUCH_SPHERE"
