import numpy as np

from swarm.constants import SAR_RGB_REQUEST_CAP, SAR_RGB_RES
from swarm.utils.env_factory import make_env_with_initial_obs
from swarm.validator import task_gen

SIM_DT = 1 / 50


def _env(family_id, seed=7, challenge_type=2):
    task = task_gen._build_task_for_type(
        sim_dt=SIM_DT, seed=seed, challenge_type=challenge_type, family_id=family_id,
    )
    return make_env_with_initial_obs(task)


def _act(env, request):
    a = np.zeros((env.NUM_DRONES, 6), dtype=np.float32)
    a[:, 5] = request
    return a


def test_rgb_present_and_blank_without_request():
    env, obs = _env("cf_search_and_rescue")
    try:
        assert obs["rgb"].shape == (SAR_RGB_RES, SAR_RGB_RES, 3)
        assert not np.any(obs["rgb"])
        obs, *_ = env.step(_act(env, 0.0))
        assert not np.any(obs["rgb"])
    finally:
        env.close()


def test_rgb_renders_on_request_in_range():
    env, _ = _env("cf_search_and_rescue")
    try:
        obs, *_ = env.step(_act(env, 0.9))
        assert np.any(obs["rgb"])
        assert obs["rgb"].min() >= 0.0 and obs["rgb"].max() <= 1.0
    finally:
        env.close()


def test_rgb_render_is_deterministic():
    env, _ = _env("cf_search_and_rescue")
    try:
        env.step(_act(env, 0.9))
        a = env._render_onboard_rgb(0)
        b = env._render_onboard_rgb(0)
        assert np.array_equal(a, b)  # same state -> identical frame (consensus-safe)
        assert np.any(a)             # and a real frame, not zeros == zeros
    finally:
        env.close()


def test_swarm_rgb_is_per_drone():
    env, obs = _env("cf_swarm_sar")
    try:
        n = env.NUM_DRONES
        assert n >= 2
        assert obs["rgb"].shape == (n, SAR_RGB_RES, SAR_RGB_RES, 3)
        a = np.zeros((n, 6), dtype=np.float32)
        a[0, 5] = 0.9  # only drone 0 asks
        obs, *_ = env.step(a)
        assert np.any(obs["rgb"][0])
        assert not np.any(obs["rgb"][1])
        assert int(env._rgb_request_count[0]) == 1
        assert int(env._rgb_request_count[1]) == 0
    finally:
        env.close()


def test_rgb_request_cap_is_enforced():
    env, _ = _env("cf_search_and_rescue")
    try:
        for _ in range(SAR_RGB_REQUEST_CAP + 5):
            obs, *_ = env.step(_act(env, 1.0))
        assert int(env._rgb_request_count[0]) == SAR_RGB_REQUEST_CAP  # never exceeds the budget
        assert not np.any(obs["rgb"])  # the final over-cap step returns a blank frame
    finally:
        env.close()
