# The MIT License (MIT)
# Copyright © 2026 Swarm

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""The v1 observation contract: the published artifact, the smoke observation and the live environment all agree."""
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
    """A challenge_type 2 MapTask on map seed 99 for the named family, version 5.0.0."""
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
    """A reset MovingDroneAviary for the family at 30 Hz, with the simulator's startup chatter swallowed."""
    from gym_pybullet_drones.utils.enums import ActionType

    from swarm.core.moving_drone import MovingDroneAviary

    with contextlib.redirect_stdout(io.StringIO()):
        env = MovingDroneAviary(
            _task(family_id), act=ActionType.VEL, ctrl_freq=30, pyb_freq=30, sar_mode=sar_mode,
        )
        env.reset(seed=_task(family_id).map_seed)
    return env


def _expected_v1_state(env, family_id):
    """The v1 state vector assembled by hand: position, orientation, velocities, action history, altitude, then the search clue offset."""
    sv = env._getDroneStateVector(0)
    parts = [sv[0:3], sv[7:10], sv[10:13], sv[13:16]]
    for i in range(env.ACTION_BUFFER_SIZE):
        parts.append(env.action_buffer[i][0, :])
    parts.append([env._get_altitude_distance() / MAX_RAY_DISTANCE])
    if family_id == "cf_autopilot":
        parts.append(env._search_area_center - sv[0:3])
    else:
        parts.append((env._search_area_center - sv[0:3])[:2])
    return np.concatenate([np.asarray(p, dtype=np.float32).reshape(-1) for p in parts])


def test_artifact_contract_shape_for_v1():
    """The published v1 contract pins the miner entry point, the leading state channels and the per-family depth shape, and its minimum_length fits the smoke state."""
    expected_depth = {"cf_autopilot": [128, 128, 1], "cf_search_and_rescue": [256, 256, 1]}
    for family_id in ("cf_autopilot", "cf_search_and_rescue"):
        art = build_artifact_policy_contract(family_id, "submission_zip.v1")
        assert art["entry_point"]["module"] == "drone_agent"
        assert art["entry_point"]["class_name"] == "DroneFlightController"
        state_field = art["observation_space"]["fields"]["state"]
        assert state_field["semantic_channels"][:4] == [
            "position_xyz",
            "orientation_rpy",
            "linear_velocity_xyz",
            "angular_velocity_xyz",
        ]
        assert art["observation_space"]["fields"]["depth"]["shape"] == expected_depth[family_id]
        expected_state = 141 if family_id == "cf_autopilot" else 165
        smoke = build_smoke_test_observation(family_id, "submission_zip.v1")
        assert smoke["state"].shape == (expected_state,)
        assert state_field["minimum_length"] <= expected_state


def test_smoke_observation_lengths_match_production_runtime():
    """A miner offline sees the shapes production hands over: 128 depth for autopilot, 256 depth plus RGB for SAR, state widths 141 and 165."""
    autopilot = build_smoke_test_observation("cf_autopilot", "submission_zip.v1")
    sar = build_smoke_test_observation("cf_search_and_rescue", "submission_zip.v1")
    assert autopilot["depth"].shape == (128, 128, 1)
    assert sar["depth"].shape == (256, 256, 1)
    assert sar["rgb"].shape == (256, 256, 3)
    # 50 Hz: 12 + 25*action_dim action history + 1 altitude + clue (3 autopilot / 2 SAR)
    assert autopilot["state"].shape == (141,)   # action_dim 5
    assert sar["state"].shape == (165,)          # action_dim 6 (RGB request value)


@pytest.mark.timeout(180)
@pytest.mark.parametrize(
    "family_id,sar_mode", [("cf_autopilot", False), ("cf_search_and_rescue", True)]
)
def test_v1_observation_matches_documented_layout(family_id, sar_mode):
    """A live environment's observation equals the hand-built v1 layout for both families, and smoke_observation reports the same state width."""
    env = _build_env(family_id, sar_mode)
    try:
        obs = env._computeObs()
        if family_id == "cf_search_and_rescue":
            assert set(obs) == {"depth", "rgb", "state"}
            assert obs["depth"].shape == (256, 256, 1)
            assert obs["rgb"].shape == (256, 256, 3)
        else:
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
    """Swapping a live environment to the GPS-less channel set yields orientation, rates, altitude and action history only, matching the rebuilt space."""
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
    """A minimal stand-in exposing an action buffer, a 5-wide action space, 128x128 image resolution and a constant altitude ray."""
    def __init__(self):
        """Set the action buffer, action space and image resolution the assemble path reads."""
        self.ACTION_BUFFER_SIZE = 2
        self.action_buffer = [
            np.array([[0.1, 0.2, 0.3, 0.4, 0.5]], dtype=np.float32),
            np.array([[1.1, 1.2, 1.3, 1.4, 1.5]], dtype=np.float32),
        ]
        self.action_space = type("Space", (), {"shape": (5,)})()
        self.IMG_RES = np.array([128, 128])

    def _get_altitude_distance(self, nth_drone=0):
        """A constant 5.0 m ray reading, so the normalised altitude channel is predictable."""
        return 5.0


def test_gpsless_layout_drops_global_position():
    """The GPS-less layout assembles a 17-wide state with no position, goal or clue channel, and space, dim and smoke all agree on 17."""
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
