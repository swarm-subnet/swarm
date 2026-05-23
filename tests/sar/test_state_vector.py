from __future__ import annotations

import contextlib
import io

import pytest

from swarm.protocol import MapTask


def _task():
    return MapTask(
        map_seed=99,
        start=(0.0, 0.0, 1.5),
        goal=(8.0, 8.0, 1.5),
        sim_dt=1 / 30,
        horizon=60.0,
        challenge_type=2,
        version="5.0.0",
    )


def _build_aviary(sar_mode):
    from swarm.core.moving_drone import MovingDroneAviary

    with contextlib.redirect_stdout(io.StringIO()):
        env = MovingDroneAviary(
            _task(), ctrl_freq=30, pyb_freq=30, sar_mode=sar_mode,
        )
        env.reset(seed=_task().map_seed)
    return env


def _close(env):
    try:
        env.close()
    except Exception:
        pass


@pytest.mark.timeout(180)
def test_sar_state_dim_uses_2d_clue():
    """SAR-only after D.4: state vector ends with a 2-element (Δx, Δy) clue
    slice; total dim is 12 + ACTION_BUFFER_SIZE * action_dim + 1 + 2."""
    env = _build_aviary(sar_mode=True)
    try:
        assert env._clue_dim == 2
        action_dim = env.action_space.shape[-1]
        expected = 12 + env.ACTION_BUFFER_SIZE * action_dim + 1 + 2
        assert env._state_dim == expected
    finally:
        _close(env)


@pytest.mark.timeout(180)
def test_sar_clue_offset_is_2d():
    env = _build_aviary(sar_mode=True)
    try:
        obs = env._computeObs()
        state = obs["state"]
        assert state.shape[0] == env._state_dim
        clue_dim = env._clue_dim
        assert clue_dim == 2
        base_len = state.shape[0] - 1 - clue_dim
        clue_slice = state[base_len + 1:base_len + 1 + clue_dim]
        assert clue_slice.shape == (2,)
    finally:
        _close(env)
