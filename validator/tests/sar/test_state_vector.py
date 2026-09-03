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
        family_id="cf_search_and_rescue",
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
