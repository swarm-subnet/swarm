"""Office interceptor family: RC action contract, conversion math, and env behavior."""

import math

import numpy as np
import pytest

from swarm.constants import (
    OFFICE_CHALLENGE_TYPE,
    OFFICE_MAX_START_DISTANCE_M,
    OFFICE_MIN_START_DISTANCE_M,
    OFFICE_RC_DEAD_ZONE,
    OFFICE_RC_SLEW_PER_SEC,
    OFFICE_TELEM_DELAY_STEPS,
    OFFICE_TELEM_PERIOD_STEPS,
)
from swarm.core.maps.office import OFFICE_CEILING_M, OFFICE_X_RANGE, OFFICE_Y_RANGE
from swarm.core.moving_drone import rc_sticks_to_world_velocity
from swarm.domain_model import get_policy_interface_contract
from swarm.utils.env_factory import make_env
from swarm.validator import task_gen


@pytest.fixture(scope="module")
def office_env():
    task = task_gen.screening_task(
        1 / 50, 3, challenge_type=OFFICE_CHALLENGE_TYPE,
        distance_range=(OFFICE_MIN_START_DISTANCE_M, OFFICE_MAX_START_DISTANCE_M),
        family_id="cf_office_interceptor",
    )
    env = make_env(task)
    yield env
    env.close()


def test_office_contract_action_space():
    contract = get_policy_interface_contract("cf_office_interceptor", "submission_zip.v1")
    action = contract["action_space"]
    assert action["shape"] == [4]
    assert action["component_names"] == ["lr", "fb", "ud", "yaw"]
    assert action["lower_bound"] == [-1.0, -1.0, -1.0, -1.0]
    assert action["upper_bound"] == [1.0, 1.0, 1.0, 1.0]


def test_rc_body_to_world_math():
    # Facing +X: forward stick moves +X, right stick moves -Y.
    v = rc_sticks_to_world_velocity([0.0, 1.0, 0.0], 0.0, 3.0)
    assert np.allclose(v, [3.0, 0.0, 0.0], atol=1e-9)
    v = rc_sticks_to_world_velocity([1.0, 0.0, 0.0], 0.0, 3.0)
    assert np.allclose(v, [0.0, -3.0, 0.0], atol=1e-9)
    # Facing +Y (yaw 90 deg): forward moves +Y, right moves +X.
    v = rc_sticks_to_world_velocity([0.0, 1.0, 0.0], math.pi / 2, 3.0)
    assert np.allclose(v, [0.0, 3.0, 0.0], atol=1e-9)
    v = rc_sticks_to_world_velocity([1.0, 0.0, 0.0], math.pi / 2, 3.0)
    assert np.allclose(v, [3.0, 0.0, 0.0], atol=1e-9)
    # Up stick is yaw-independent.
    v = rc_sticks_to_world_velocity([0.0, 0.0, 0.5], 1.234, 3.0)
    assert np.allclose(v, [0.0, 0.0, 1.5], atol=1e-9)


def test_office_env_action_space_and_state(office_env):
    env = office_env
    assert env.action_space.shape == (1, 4)
    assert np.all(env.action_space.low == -1.0)
    assert np.all(env.action_space.high == 1.0)
    obs, _ = env.reset(seed=env.task.map_seed)
    contract = get_policy_interface_contract("cf_office_interceptor", "submission_zip.v1")
    assert obs["state"].shape == tuple(contract["smoke_test_observation"]["state"]["shape"])
    assert obs["depth"].shape == (256, 256, 1)
    assert all(a.shape == (1, 4) for a in env.action_buffer)


def test_office_zero_action_hovers(office_env):
    env = office_env
    env.reset(seed=env.task.map_seed)
    start = env.pos[0].copy()
    for _ in range(50):
        env.step(np.zeros((1, 4), dtype=np.float32))
    drift = float(np.linalg.norm(env.pos[0][:2] - start[:2]))
    assert drift < 0.15, f"hover drifted {drift:.3f} m"


def test_office_dead_zone_and_slew(office_env):
    env = office_env
    env.reset(seed=env.task.map_seed)
    tiny = np.full((1, 4), OFFICE_RC_DEAD_ZONE * 0.5, dtype=np.float32)
    env.step(tiny)
    assert np.allclose(env._rc_command, 0.0)
    env.step(np.ones((1, 4), dtype=np.float32))
    max_step = OFFICE_RC_SLEW_PER_SEC * env.CTRL_TIMESTEP
    assert np.all(np.abs(env._rc_command) <= max_step + 1e-9)


def test_office_stale_visual_cuts_forward(office_env):
    env = office_env
    env.reset(seed=env.task.map_seed)
    env._visual_stale = True
    for _ in range(5):
        env.step(np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32))
    assert env._rc_command[0, 1] <= 0.0
    env._visual_stale = False


def test_office_forward_matches_heading(office_env):
    env = office_env
    env.reset(seed=env.task.map_seed)
    # Climb clear of ground effect, gently enough to stay under the 3 m ceiling.
    for _ in range(60):
        env.step(np.array([[0.0, 0.0, 0.3, 0.0]], dtype=np.float32))
    start = env.pos[0].copy()
    for _ in range(40):
        env.step(np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32))
    delta = env.pos[0] - start
    heading = math.atan2(delta[1], delta[0])
    # Spawn yaw is 0, so forward stick must move the drone along +X.
    assert abs(delta[0]) > 0.3
    assert abs(heading) < math.radians(20), f"moved at {math.degrees(heading):.1f} deg"


def test_office_contract_is_telemetry_only():
    contract = get_policy_interface_contract("cf_office_interceptor", "submission_zip.v1")
    assert contract["observation_assembly"]["state"] == [
        "tello_attitude", "tello_velocity", "tello_acceleration",
        "tello_altitude", "action_history",
    ]
    channels = contract["observation_space"]["fields"]["state"]["semantic_channels"]
    # No ground truth in the state vector: everything must exist on the real SDK.
    assert "position_xyz" not in channels
    assert "angular_velocity_xyz" not in channels
    assert "search_clue_offset_xy" not in channels


def test_office_telemetry_cadence_and_age(office_env):
    env = office_env
    obs, _ = env.reset(seed=env.task.map_seed)
    telem = obs["state"][:15]
    assert np.all(telem[:13] == 0.0)
    assert telem[14] == 0.0, "telemetry must be invalid before the first packet"
    hover = np.zeros((1, 4), dtype=np.float32)
    dt = env.CTRL_TIMESTEP
    for i in range(1, OFFICE_TELEM_PERIOD_STEPS):
        obs, *_ = env.step(hover)
        assert obs["state"][14] == 0.0, f"no packet should arrive at step {i}"
    obs, *_ = env.step(hover)
    telem = obs["state"][:15]
    assert telem[14] == 1.0, "first packet must arrive on the period boundary"
    assert telem[13] == pytest.approx(OFFICE_TELEM_DELAY_STEPS * dt)
    # Between packets the readings hold while only the age advances.
    held = telem[:13].copy()
    obs, *_ = env.step(hover)
    assert np.array_equal(obs["state"][:13], held)
    assert obs["state"][13] == pytest.approx(telem[13] + dt)


def test_office_telemetry_tracks_flight(office_env):
    env = office_env
    env.reset(seed=env.task.map_seed)
    # Gentle climb: a full-stick climb would reach the 3 m ceiling and tumble.
    for _ in range(60):
        obs, *_ = env.step(np.array([[0.0, 0.0, 0.3, 0.0]], dtype=np.float32))
    telem = obs["state"][:15]
    assert telem[2] ** 2 + telem[3] ** 2 == pytest.approx(1.0, abs=1e-5)
    assert telem[6] > 0.2, "climb must show as positive body up-velocity"
    assert telem[10] > 0.5, "ToF must grow as the drone leaves the floor"
    assert abs(telem[11] - (env.pos[0][2] - 0.05)) < 0.3, "fused height must track altitude"
    for _ in range(40):
        obs, *_ = env.step(np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32))
    telem = obs["state"][:15]
    assert telem[4] > 0.5, "forward flight must show as positive body forward-velocity"
    assert abs(telem[5]) < 0.5, "sideways velocity must stay near zero"


def test_office_telemetry_deterministic():
    task = task_gen.random_task(1 / 50, 42, family_id="cf_office_interceptor")
    streams = []
    for _ in range(2):
        env = make_env(task)
        env.reset(seed=task.map_seed)
        states = []
        for i in range(3 * OFFICE_TELEM_PERIOD_STEPS):
            obs, *_ = env.step(np.array([[0.2, 0.5, 0.3, 0.1]], dtype=np.float32))
            states.append(obs["state"].copy())
        env.close()
        streams.append(np.stack(states))
    assert np.array_equal(streams[0], streams[1])


def test_office_task_generation_deterministic():
    a = task_gen.random_task(1 / 50, 77, family_id="cf_office_interceptor")
    b = task_gen.random_task(1 / 50, 77, family_id="cf_office_interceptor")
    assert a == b
    assert a.challenge_type == OFFICE_CHALLENGE_TYPE
    assert OFFICE_X_RANGE[0] < a.start[0] < OFFICE_X_RANGE[1]
    assert OFFICE_Y_RANGE[0] < a.start[1] < OFFICE_Y_RANGE[1]
    assert 0.0 < a.goal[2] < OFFICE_CEILING_M
