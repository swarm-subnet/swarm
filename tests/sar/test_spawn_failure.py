from __future__ import annotations

import contextlib
import io

import pytest

from swarm.protocol import FailureReason, MapTask


def _task():
    return MapTask(
        map_seed=4242,
        start=(0.0, 0.0, 1.5),
        goal=(8.0, 8.0, 1.5),
        sim_dt=1 / 30,
        horizon=60.0,
        challenge_type=2,
        version="5.0.0",
    )


def _build_env(monkeypatch):
    from swarm.core import moving_drone as md
    from swarm.core.env_builder.spawn_pipeline import SARSpawnError

    def _always_raise(*args, **kwargs):
        raise SARSpawnError("forced for test")

    monkeypatch.setattr(
        "swarm.core.env_builder.sar_world.find_spawn_xy",
        _always_raise,
    )

    with contextlib.redirect_stdout(io.StringIO()):
        env = md.MovingDroneAviary(
            _task(), ctrl_freq=30, pyb_freq=30, sar_mode=True,
        )
        env.reset(seed=_task().map_seed)
    return env


def _close(env):
    try:
        env.close()
    except Exception:
        pass


@pytest.mark.timeout(180)
def test_env_reset_returns_valid_obs_on_spawn_failure(monkeypatch):
    env = _build_env(monkeypatch)
    try:
        assert env._sar_spawn_failed is True
        assert env.sar_world is None
        assert env._failure_reason == FailureReason.SPAWN_FAILURE.value
        obs = env._computeObs()
        assert obs["state"].shape[0] == env._state_dim
    finally:
        _close(env)


@pytest.mark.timeout(180)
def test_first_step_terminates_on_spawn_failure(monkeypatch):
    env = _build_env(monkeypatch)
    try:
        import numpy as np
        action = np.zeros(env.action_space.shape[-1], dtype=np.float32)
        action = action.reshape(env.action_space.shape)
        obs, _r, terminated, _trunc, info = env.step(action)
        assert terminated is True
        assert info["failure_reason"] == FailureReason.SPAWN_FAILURE.value
    finally:
        _close(env)


@pytest.mark.timeout(180)
def test_score_is_participation_on_spawn_failure(monkeypatch):
    env = _build_env(monkeypatch)
    try:
        from swarm.validator.reward import flight_reward
        score = flight_reward(
            success=False, t=0.0, horizon=env.EP_LEN_SEC, task=env.task,
            failure_reason=FailureReason.SPAWN_FAILURE.value,
            sar_mode=True, min_clearance=None,
        )
        assert score == 0.01
    finally:
        _close(env)
