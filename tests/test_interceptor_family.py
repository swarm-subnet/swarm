import math

import numpy as np
import pybullet as p
import pytest

from swarm.challenge_families import get_challenge_family, list_registered_challenge_families
from swarm.constants import (
    INTERCEPTOR_HORIZON_SEC,
    INTERCEPTOR_MAX_START_DISTANCE_M,
    INTERCEPTOR_MIN_START_DISTANCE_M,
)
from swarm.protocol import FailureReason
from swarm.validator import task_gen
from swarm.validator.reward import _calculate_interceptor_target_time

SIM_DT = 1.0 / 30.0


def _task(seed):
    return task_gen.random_task(sim_dt=SIM_DT, seed=seed, family_id="cf_interceptor")


def test_registered():
    assert "cf_interceptor" in list_registered_challenge_families()


def test_task_open_map_distance_horizon():
    for seed in range(20):
        t = _task(seed)
        assert t.family_id == "cf_interceptor"
        assert t.challenge_type == 2  # open map only
        assert t.horizon == INTERCEPTOR_HORIZON_SEC
        gap = math.dist(t.start, t.goal)
        assert INTERCEPTOR_MIN_START_DISTANCE_M - 1 <= gap <= INTERCEPTOR_MAX_START_DISTANCE_M + 1


def test_task_deterministic():
    a, b = _task(11), _task(11)
    assert a.start == b.start and a.goal == b.goal


def test_target_time_under_horizon():
    for seed in range(40):
        t = _task(seed)
        assert _calculate_interceptor_target_time(t) < t.horizon


def test_scoring_is_50_50():
    fam = get_challenge_family("cf_interceptor")
    t = _task(3)
    base = dict(task=t, min_clearance=1.0, legitimate_model=True, collision=False)
    # caught quickly -> high score driven by 0.5 success + 0.5 time
    m = fam.build_rollout_metrics(success=True, t=2.0, horizon=t.horizon,
                                  failure_reason="NONE", **base)
    norm = fam.normalize_rollout_metrics(task=t, metrics=m)
    tt = norm["time_term"]
    assert abs(norm["final_score"] - (0.5 * 1.0 + 0.5 * tt)) < 1e-6
    assert norm["safety_term"] == 0.0
    assert 0.5 <= norm["final_score"] <= 1.0
    # not caught -> participation only
    m2 = fam.build_rollout_metrics(success=False, t=5.0, horizon=t.horizon,
                                   failure_reason="TIMEOUT", **base)
    norm2 = fam.normalize_rollout_metrics(task=t, metrics=m2)
    assert norm2["final_score"] == pytest.approx(0.01)


# ---- env-backed checks (render-light) ----
def _env(seed):
    from swarm.utils.env_factory import make_env_with_initial_obs
    return make_env_with_initial_obs(_task(seed))


def test_env_obs_and_airborne_target():
    env, obs = _env(7)
    assert obs["depth"].shape == (1024, 1024, 1)
    assert obs["state"].ndim == 1
    # target airborne, above its resolved floor band
    assert env._target_pos[2] > env._target_floor_z + 0.5
    assert env._target_uid is not None


def test_catch_on_contact():
    env, _ = _env(8)
    cli = env.CLIENT
    p.resetBasePositionAndOrientation(env.DRONE_IDS[0], list(env._target_pos),
                                      p.getQuaternionFromEuler([0, 0, 0]), physicsClientId=cli)
    env._updateAndStoreKinematicInformation()
    env._success = False
    env.family_runtime.post_step_update(env)
    assert env._success and env._intercept_caught


def test_ram_is_a_catch_not_a_chaser_crash():
    env, _ = _env(9)
    cli = env.CLIENT
    # drop the chaser onto the target so the bodies physically collide
    p.resetBasePositionAndOrientation(env.DRONE_IDS[0], list(env._target_pos),
                                      p.getQuaternionFromEuler([0, 0, 0]), physicsClientId=cli)
    p.stepSimulation(physicsClientId=cli)
    cps = p.getContactPoints(bodyA=int(env.DRONE_IDS[0]), bodyB=int(env._target_uid),
                             physicsClientId=cli)
    assert len(cps) > 0  # collisions enabled -> a real crash
    # the engine must NOT score that ram as a chaser obstacle-collision
    env._updateAndStoreKinematicInformation()
    env._collision = False
    env._check_collision()
    assert env._collision is False


def test_target_not_culled_and_protected():
    env, _ = _env(10)
    assert int(env._target_uid) not in [t[0] for t in env._cull_targets]
    assert int(env._target_uid) in env.family_runtime.protected_body_uids(env)


def test_rollout_determinism():
    def trace(seed, steps):
        env, _ = _env(seed)
        out = []
        for _ in range(steps):
            cpos = env._getDroneStateVector(0)[0:3]
            d = env._target_pos - cpos
            n = np.linalg.norm(d)
            u = d / n if n > 1e-6 else np.zeros(3)
            act = np.array([[u[0], u[1], u[2], 1.0, 0.0]], dtype=np.float32)
            env.step(act)
            out.append(tuple(np.round(env._target_pos, 5)))
        return out
    assert trace(13, 6) == trace(13, 6)


def test_other_families_unaffected_drone_size():
    # cf_autopilot still uses the 12 cm drone (no interceptor scaling leak)
    from swarm.utils.env_factory import make_env_with_initial_obs
    t = task_gen.random_task(sim_dt=SIM_DT, seed=1, family_id="cf_autopilot")
    env, _ = make_env_with_initial_obs(t)
    assert abs(float(env.M) - 0.027) < 1e-4
