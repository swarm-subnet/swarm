from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest

from swarm.constants import MAX_RAY_DISTANCE
from swarm.core.observation import (
    assemble,
    observation_space,
    observation_vector_dim,
    smoke_observation,
)
from swarm.core.observation.channels import action_buffer_size
from swarm.policy_interface import (
    build_artifact_policy_contract,
    build_smoke_test_observation,
)
from swarm.protocol import MapTask


def _task(family_id: str):
    return MapTask(
        map_seed=99,
        start=(0.0, 0.0, 1.5),
        goal=(8.0, 8.0, 1.5),
        sim_dt=1 / 30,
        horizon=60.0,
        challenge_type=2,
        family_id=family_id,
        version="5.0.0",
    )


def _build_env(family_id: str, sar_mode: bool):
    from gym_pybullet_drones.utils.enums import ActionType
    from swarm.core.moving_drone import MovingDroneAviary

    with contextlib.redirect_stdout(io.StringIO()):
        env = MovingDroneAviary(
            _task(family_id), act=ActionType.VEL, ctrl_freq=30, pyb_freq=30, sar_mode=sar_mode,
        )
        env.reset(seed=_task(family_id).map_seed)
    return env


def _expected_v1_state(env, family_id):
    sv = env._getDroneStateVector(0)
    parts = [sv[0:3], sv[7:10], sv[10:13], sv[13:16]]
    for i in range(env.ACTION_BUFFER_SIZE):
        parts.append(env.action_buffer[i][0, :])
    parts.append([env._get_altitude_distance() / MAX_RAY_DISTANCE])
    if family_id == "cf_autopilot":
        parts.append(env.GOAL_POS - sv[0:3])
    else:
        parts.append((env._search_area_center - sv[0:3])[:2])
    return np.concatenate([np.asarray(p, dtype=np.float32).reshape(-1) for p in parts])


def test_artifact_contract_unchanged_for_v1():
    for family_id in ("cf_autopilot", "cf_search_and_rescue"):
        art = build_artifact_policy_contract(family_id, "submission_zip.v1")
        assert set(art) == {
            "contract_version",
            "family_id",
            "interface_version",
            "entry_point",
            "observation_space",
            "action_space",
        }
        assert "observation_assembly" not in art
        assert art["observation_space"]["fields"]["depth"]["shape"] == [128, 128, 1]
        assert art["observation_space"]["fields"]["state"]["shape"] == ["dynamic"]


def test_smoke_observation_lengths_match_production_runtime():
    autopilot = build_smoke_test_observation("cf_autopilot", "submission_zip.v1")
    sar = build_smoke_test_observation("cf_search_and_rescue", "submission_zip.v1")
    assert autopilot["depth"].shape == (128, 128, 1)
    assert sar["depth"].shape == (128, 128, 1)
    # 50 Hz: 12 + 25*5 action history + 1 altitude + clue (3 autopilot / 2 SAR)
    assert autopilot["state"].shape == (141,)
    assert sar["state"].shape == (140,)


@pytest.mark.timeout(180)
@pytest.mark.parametrize(
    "family_id,sar_mode", [("cf_autopilot", False), ("cf_search_and_rescue", True)]
)
def test_v1_observation_matches_documented_layout(family_id, sar_mode):
    env = _build_env(family_id, sar_mode)
    try:
        obs = env._computeObs()
        assert set(obs) == {"depth", "state"}
        assert obs["depth"].shape == (128, 128, 1)
        expected = _expected_v1_state(env, family_id)
        assert obs["state"].shape == expected.shape
        assert np.allclose(obs["state"], expected, atol=1e-6)
        # smoke must match the live runtime state length
        action_dim = int(env.action_space.shape[-1])
        smoke = smoke_observation(
            env._obs_layout, ctrl_freq=30, action_dim=action_dim
        )
        assert smoke["state"].shape[0] == env._state_dim == obs["state"].shape[0]
    finally:
        with contextlib.suppress(Exception):
            env.close()


@pytest.mark.timeout(180)
def test_real_env_runs_gpsless_layout():
    env = _build_env("cf_autopilot", sar_mode=False)
    try:
        gpsless = {
            "depth": ["depth_camera"],
            "state": ["orientation", "angular_velocity", "altitude_norm", "action_history"],
        }
        env._obs_layout = gpsless
        env.observation_space = observation_space(gpsless, env)

        obs = env._computeObs()
        assert obs["depth"].shape == (128, 128, 1)

        sv = env._getDroneStateVector(0)
        parts = [sv[7:10], sv[13:16], [env._get_altitude_distance() / MAX_RAY_DISTANCE]]
        for i in range(env.ACTION_BUFFER_SIZE):
            parts.append(env.action_buffer[i][0, :])
        expected = np.concatenate(
            [np.asarray(p, dtype=np.float32).reshape(-1) for p in parts]
        )
        assert obs["state"].shape == expected.shape
        assert np.allclose(obs["state"], expected, atol=1e-6)
        assert obs["state"].shape[0] == env.observation_space["state"].shape[0]
        assert "position" not in gpsless["state"]
        assert "goal_offset" not in gpsless["state"]
    finally:
        with contextlib.suppress(Exception):
            env.close()


class _FakeEnv:
    def __init__(self):
        self.ACTION_BUFFER_SIZE = 2
        self.action_buffer = [
            np.array([[0.1, 0.2, 0.3, 0.4, 0.5]], dtype=np.float32),
            np.array([[1.1, 1.2, 1.3, 1.4, 1.5]], dtype=np.float32),
        ]
        self.action_space = type("Space", (), {"shape": (5,)})()
        self.IMG_RES = np.array([128, 128])

    def _get_altitude_distance(self):
        return 5.0


def test_gpsless_layout_drops_global_position():
    layout = {
        "depth": ["depth_camera"],
        "state": ["orientation", "angular_velocity", "altitude_norm", "action_history"],
    }
    env = _FakeEnv()
    sv = np.arange(16, dtype=np.float32) + 0.5
    depth = np.full((128, 128, 1), 0.3, dtype=np.float32)

    obs = assemble(layout, env, sv, {"depth": depth})

    expected_state = np.concatenate(
        [
            sv[7:10],
            sv[13:16],
            [5.0 / MAX_RAY_DISTANCE],
            env.action_buffer[0][0, :],
            env.action_buffer[1][0, :],
        ]
    ).astype(np.float32)
    assert np.array_equal(obs["state"], expected_state)
    assert obs["state"].shape == (17,)
    assert np.array_equal(obs["depth"], depth)

    # the layout exposes no global-position-derived channel
    assert "position" not in layout["state"]
    assert "goal_offset" not in layout["state"]
    assert "search_clue_offset" not in layout["state"]

    space = observation_space(layout, env)
    assert space["state"].shape == (17,)
    assert space["depth"].shape == (128, 128, 1)
    assert observation_vector_dim(layout, env) == 17

    smoke = smoke_observation(layout, ctrl_freq=4, action_dim=5)
    assert smoke["state"].shape == (17,)
    assert action_buffer_size(4) == 2
