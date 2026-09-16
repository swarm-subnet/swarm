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

"""Env construction: GUI overlays, the observation from the first reset, and per-family kwargs."""

from __future__ import annotations

from types import SimpleNamespace

import gymnasium.spaces as spaces
import numpy as np

from swarm.utils import env_factory


class _DummyPyBullet:
    """PyBullet stub that logs every visualizer call and answers the rest with nothing."""
    COV_ENABLE_RENDERING = 7
    COV_ENABLE_SHADOWS = 2
    COV_ENABLE_GUI = 1
    COV_ENABLE_RGB_BUFFER_PREVIEW = 3
    COV_ENABLE_DEPTH_BUFFER_PREVIEW = 4
    COV_ENABLE_SEGMENTATION_MARK_PREVIEW = 5
    COV_ENABLE_WIREFRAME = 6

    def __init__(self) -> None:
        """Start with an empty log of visualizer calls."""
        self.calls: list[tuple[int, int, int]] = []

    def setAdditionalSearchPath(self, _path: str) -> None:
        """Accept the pybullet_data directory and do nothing with it."""
        return None

    def configureDebugVisualizer(self, flag: int, value: int, physicsClientId: int) -> None:
        """Append the flag, the value and the client id to the call log."""
        self.calls.append((flag, value, physicsClientId))

    def setPhysicsEngineParameter(self, **_kwargs) -> None:
        """Swallow the solver settings the factory applies after the reset."""
        return None

    def getNumBodies(self, physicsClientId: int) -> int:
        """Report an empty world: zero bodies loaded."""
        _ = physicsClientId
        return 0


class _DummyEnv:
    """Aviary stub exposing the kwargs it was built with and a tiny depth-plus-state space."""
    def __init__(self, task, **_kwargs) -> None:
        """Keep the task and every runtime kwarg, and declare the 4x4 depth and 4-vector state."""
        self.task = task
        self.kwargs = dict(_kwargs)
        self._state_dim = 4
        self.DRONE_IDS = np.array([0])
        self.observation_space = {
            "depth": spaces.Box(low=0.0, high=1.0, shape=(4, 4, 1), dtype=np.float32),
            "state": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
        }

    def getPyBulletClient(self) -> int:
        """Return 99 as the physics client id the visualizer calls must carry."""
        return 99

    def reset(self, seed: int):
        """Hand back zeroed depth and state arrays with an empty info dict."""
        _ = seed
        return {
            "depth": np.zeros((4, 4, 1), dtype=np.float32),
            "state": np.zeros((4,), dtype=np.float32),
        }, {}


def test_make_env_hides_gui_rendering_during_reset(monkeypatch) -> None:
    """A viewer world is built with overlays off and redraw suspended, then redraw comes back."""
    dummy_p = _DummyPyBullet()
    monkeypatch.setattr(env_factory, "MovingDroneAviary", _DummyEnv)
    monkeypatch.setattr(env_factory, "p", dummy_p)
    monkeypatch.setattr(env_factory.pybullet_data, "getDataPath", lambda: "/tmp")
    monkeypatch.setattr(env_factory.time, "sleep", lambda _seconds: None)

    task = SimpleNamespace(
        sim_dt=0.1,
        map_seed=123,
        version="5.0.0",
        family_id="cf_search_and_rescue",
    )
    env_factory.make_env(task, gui=True)

    assert dummy_p.calls == [
        (dummy_p.COV_ENABLE_SHADOWS, 0, 99),
        (dummy_p.COV_ENABLE_GUI, 0, 99),
        (dummy_p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0, 99),
        (dummy_p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0, 99),
        (dummy_p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0, 99),
        (dummy_p.COV_ENABLE_WIREFRAME, 0, 99),
        (dummy_p.COV_ENABLE_RENDERING, 0, 99),
        (dummy_p.COV_ENABLE_RENDERING, 1, 99),
    ]


def test_make_env_with_initial_obs_returns_first_observation(monkeypatch) -> None:
    """The env comes back paired with the depth and state produced by its own reset."""
    dummy_p = _DummyPyBullet()
    monkeypatch.setattr(env_factory, "MovingDroneAviary", _DummyEnv)
    monkeypatch.setattr(env_factory, "p", dummy_p)
    monkeypatch.setattr(env_factory.pybullet_data, "getDataPath", lambda: "/tmp")

    task = SimpleNamespace(
        sim_dt=0.1,
        map_seed=123,
        version="5.0.0",
        family_id="cf_search_and_rescue",
    )
    env, obs = env_factory.make_env_with_initial_obs(task, gui=False)

    assert isinstance(env, _DummyEnv)
    assert tuple(obs["depth"].shape) == (4, 4, 1)
    assert tuple(obs["state"].shape) == (4,)


def test_make_env_with_initial_obs_uses_family_runtime_kwargs(monkeypatch) -> None:
    """A search-and-rescue task reaches the aviary with sar_mode switched on."""
    dummy_p = _DummyPyBullet()
    monkeypatch.setattr(env_factory, "MovingDroneAviary", _DummyEnv)
    monkeypatch.setattr(env_factory, "p", dummy_p)
    monkeypatch.setattr(env_factory.pybullet_data, "getDataPath", lambda: "/tmp")

    task = SimpleNamespace(
        sim_dt=0.1,
        map_seed=123,
        version="5.0.0",
        family_id="cf_search_and_rescue",
    )
    env, _obs = env_factory.make_env_with_initial_obs(task, gui=False)

    assert env.kwargs["sar_mode"] is True


def test_make_env_with_initial_obs_uses_autopilot_runtime_kwargs(monkeypatch) -> None:
    """An autopilot task reaches the aviary with sar_mode switched off."""
    dummy_p = _DummyPyBullet()
    monkeypatch.setattr(env_factory, "MovingDroneAviary", _DummyEnv)
    monkeypatch.setattr(env_factory, "p", dummy_p)
    monkeypatch.setattr(env_factory.pybullet_data, "getDataPath", lambda: "/tmp")

    task = SimpleNamespace(
        sim_dt=0.1,
        map_seed=123,
        version="4.9.0",
        family_id="cf_autopilot",
    )
    env, _obs = env_factory.make_env_with_initial_obs(task, gui=False)

    assert env.kwargs["sar_mode"] is False
