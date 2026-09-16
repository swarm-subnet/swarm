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

"""A SAR world that cannot spawn: the episode still observes, ends at once, and pays participation."""
from __future__ import annotations

import contextlib
import io

import pytest

from swarm.protocol import FailureReason, MapTask


def _task():
    """A search-and-rescue MapTask on seed 4242, running 60 seconds at 30 Hz."""
    return MapTask(
        map_seed=4242,
        start=(0.0, 0.0, 1.5),
        goal=(8.0, 8.0, 1.5),
        sim_dt=1 / 30,
        horizon=60.0,
        challenge_type=2,
        family_id="cf_search_and_rescue",
        version="5.0.0",
    )


def _build_env(monkeypatch):
    """A reset SAR environment whose victim spawn search always raises, so no world is built."""
    from swarm.core import moving_drone as md
    from swarm.core.env_builder.spawn_pipeline import SARSpawnError

    def _always_raise(*args, **kwargs):
        """Raise SARSpawnError in place of the real spawn search."""
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
    """Shut the environment down, swallowing whatever PyBullet raises on the way out."""
    try:
        env.close()
    except Exception:
        pass


@pytest.mark.timeout(180)
def test_env_reset_returns_valid_obs_on_spawn_failure(monkeypatch):
    """With no world built, the reason is recorded and reset still hands back a correctly shaped state."""
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
    """The very first step ends the episode and reports SPAWN_FAILURE in the info dict."""
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
    """A world that would not build scores 0.01, so a miner is never punished for a bad seed."""
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


def test_all_steep_map_spawns_on_flattest_spot(monkeypatch):
    """When every candidate surface is too steep, the search settles on the flattest rather than giving up."""
    from swarm.core.env_builder import spawn_pipeline as sp

    slopes = {}

    def _fake_resolve(cli, x, y, body_tags, accepted):
        """Answer every probe with the same support hit at z = 1.0, flagged as sloped."""
        return sp.SurfaceHit(
            surface_z=1.0, support_uid=7, category="SUPPORT_TERRAIN",
            normal=(0.0, 0.0, 1.0), is_slope=True,
        )

    def _fake_slope(cli, x, y, surface_z, radius=0.4):
        """Give every probe an angle of at least 30 degrees, over the 22 degree cap, and remember it per (x, y)."""
        key = (round(x, 6), round(y, 6))
        slopes.setdefault(key, 30.0 + (len(slopes) % 17))
        return slopes[key]

    monkeypatch.setattr(sp, "resolve_surface", _fake_resolve)
    monkeypatch.setattr(sp, "terrain_slope_deg", _fake_slope)
    monkeypatch.setattr(sp, "_hover_column_clear", lambda *a, **k: True)
    monkeypatch.setattr(sp, "_sphere_obstacle_clear", lambda *a, **k: True)

    x, y, hit = sp.find_spawn_xy(
        0, map_seed=1234, challenge_type=3, body_tags={},
    )
    assert hit.surface_z == 1.0
    flattest = min(slopes.items(), key=lambda kv: kv[1])[0]
    assert (round(x, 6), round(y, 6)) == flattest


def test_spawn_still_fails_without_any_valid_surface(monkeypatch):
    """When nothing resolves under the raycast at all, the search raises rather than inventing a spot."""
    from swarm.core.env_builder import spawn_pipeline as sp

    monkeypatch.setattr(sp, "resolve_surface", lambda *a, **k: None)
    with pytest.raises(sp.SARSpawnError):
        sp.find_spawn_xy(0, map_seed=1234, challenge_type=3, body_tags={})
