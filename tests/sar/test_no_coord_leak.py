"""D.1.3 — victim true XY must never reach the miner.

The miner only sees the per-step observation passed across the cap'n proto
boundary by `_serialize_observation`. We build a SAR scene, query reset's
observation, and assert no float in any observation array equals the
victim's true XY or victim_centre 3D components within float32 precision.

Also asserts the state vector's search-clue slice is 2D and is `(Δx, Δy)`
relative to the noisy `search_centre`, not the raw victim XY.
"""
from __future__ import annotations

import contextlib
import io
import math

import numpy as np
import pytest

from swarm.protocol import MapTask


_FLOAT_EPS = 1e-3


def _task():
    return MapTask(
        map_seed=8181,
        start=(0.0, 0.0, 1.5),
        goal=(8.0, 8.0, 1.5),
        sim_dt=1 / 30,
        horizon=60.0,
        challenge_type=2,
        version="5.0.0",
    )


@pytest.fixture
def sar_env():
    from swarm.core.moving_drone import MovingDroneAviary

    with contextlib.redirect_stdout(io.StringIO()):
        env = MovingDroneAviary(
            _task(), ctrl_freq=30, pyb_freq=30, sar_mode=True,
        )
        env.reset(seed=_task().map_seed)
    yield env
    try:
        env.close()
    except Exception:
        pass


def _approx_eq(a: float, b: float) -> bool:
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= _FLOAT_EPS


def _contains_value(arr, target: float) -> bool:
    flat = np.asarray(arr, dtype=np.float64).reshape(-1)
    return bool(np.any(np.abs(flat - target) <= _FLOAT_EPS))


@pytest.mark.timeout(180)
def test_first_obs_does_not_contain_victim_xy(sar_env):
    env = sar_env
    obs = env._computeObs()
    vx, vy, vz = env.sar_world.victim_centre

    for key, arr in obs.items():
        assert not _contains_value(arr, vx), (
            f"obs['{key}'] contains victim x={vx:.4f}"
        )
        assert not _contains_value(arr, vy), (
            f"obs['{key}'] contains victim y={vy:.4f}"
        )


@pytest.mark.timeout(180)
def test_state_clue_slice_is_2d_and_offset(sar_env):
    env = sar_env
    obs = env._computeObs()
    state = obs["state"]
    clue_dim = env._clue_dim
    base_len = state.shape[0] - 1 - clue_dim
    clue_slice = state[base_len + 1 : base_len + 1 + clue_dim]
    assert clue_slice.shape == (2,), "SAR mode must expose a 2D clue slice"

    drone_pos = state[0:3]
    search_centre = env.sar_world.search_centre
    expected_dx = float(search_centre[0]) - float(drone_pos[0])
    expected_dy = float(search_centre[1]) - float(drone_pos[1])
    assert _approx_eq(float(clue_slice[0]), expected_dx)
    assert _approx_eq(float(clue_slice[1]), expected_dy)


@pytest.mark.timeout(180)
def test_serialize_observation_payload_omits_victim_xy(sar_env):
    """Round-trip the obs through the same cap'n proto serializer the docker
    miner sees, then assert no float in any tensor blob matches victim XY."""
    pytest.importorskip("capnp")
    from swarm.validator.docker.docker_evaluator_parts.submission import (
        _serialize_observation,
    )
    from pathlib import Path
    import capnp

    schema_path = (
        Path(__file__).resolve().parents[2]
        / "swarm" / "validator" / "docker" / "agent.capnp"
    )
    if not schema_path.is_file():
        pytest.skip(f"capnp schema not found at {schema_path}")
    agent_capnp = capnp.load(str(schema_path))

    env = sar_env
    obs = env._computeObs()
    vx, vy, _ = env.sar_world.victim_centre

    message = _serialize_observation(agent_capnp, obs)
    for entry in message.entries:
        raw = np.frombuffer(entry.tensor.data, dtype=np.float32).astype(np.float64)
        assert not bool(np.any(np.abs(raw - vx) <= _FLOAT_EPS)), (
            f"capnp tensor '{entry.key}' contains victim x={vx:.4f}"
        )
        assert not bool(np.any(np.abs(raw - vy) <= _FLOAT_EPS)), (
            f"capnp tensor '{entry.key}' contains victim y={vy:.4f}"
        )


@pytest.mark.timeout(180)
def test_runtime_telemetry_does_not_expose_victim_xy(sar_env):
    """Telemetry stays validator-side, but we still guarantee the info dict
    that feeds runtime_telemetry never carries the victim true XY."""
    env = sar_env
    info = env._computeInfo()
    vx, vy, _ = env.sar_world.victim_centre

    for key, value in info.items():
        try:
            f = float(value)
        except (TypeError, ValueError):
            continue
        assert not _approx_eq(f, vx), f"info['{key}']={f} matches victim x={vx}"
        assert not _approx_eq(f, vy), f"info['{key}']={f} matches victim y={vy}"
