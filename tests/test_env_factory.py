from __future__ import annotations

from types import SimpleNamespace

import gymnasium.spaces as spaces
import numpy as np

from swarm.utils import env_factory


class _DummyPyBullet:
    COV_ENABLE_RENDERING = 7
    COV_ENABLE_SHADOWS = 2
    COV_ENABLE_GUI = 1
    COV_ENABLE_RGB_BUFFER_PREVIEW = 3
    COV_ENABLE_DEPTH_BUFFER_PREVIEW = 4
    COV_ENABLE_SEGMENTATION_MARK_PREVIEW = 5
    COV_ENABLE_WIREFRAME = 6

    def __init__(self) -> None:
        self.calls: list[tuple[int, int, int]] = []

    def setAdditionalSearchPath(self, _path: str) -> None:
        return None

    def configureDebugVisualizer(self, flag: int, value: int, physicsClientId: int) -> None:
        self.calls.append((flag, value, physicsClientId))


class _DummyEnv:
    def __init__(self, task, **_kwargs) -> None:
        self.task = task
        self._state_dim = 4
        self.observation_space = {
            "depth": spaces.Box(low=0.0, high=1.0, shape=(4, 4, 1), dtype=np.float32),
            "state": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
        }

    def getPyBulletClient(self) -> int:
        return 99

    def reset(self, seed: int):
        _ = seed
        return {
            "depth": np.zeros((4, 4, 1), dtype=np.float32),
            "state": np.zeros((4,), dtype=np.float32),
        }, {}


def test_make_env_hides_gui_rendering_during_reset(monkeypatch) -> None:
    dummy_p = _DummyPyBullet()
    monkeypatch.setattr(env_factory, "MovingDroneAviary", _DummyEnv)
    monkeypatch.setattr(env_factory, "p", dummy_p)
    monkeypatch.setattr(env_factory.pybullet_data, "getDataPath", lambda: "/tmp")
    monkeypatch.setattr(env_factory.time, "sleep", lambda _seconds: None)

    task = SimpleNamespace(sim_dt=0.1, map_seed=123)
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
