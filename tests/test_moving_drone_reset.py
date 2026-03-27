from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import gymnasium.spaces as spaces
import numpy as np

from swarm.core import moving_drone as moving_drone_mod


def _build_stub_env() -> moving_drone_mod.MovingDroneAviary:
    env = moving_drone_mod.MovingDroneAviary.__new__(moving_drone_mod.MovingDroneAviary)
    env.task = SimpleNamespace(map_seed=123, goal=(1.0, 2.0, 3.0), search_radius=10.0)
    env.GOAL_POS = np.array([1.0, 2.0, 3.0], dtype=float)
    env.EP_LEN_SEC = 60.0
    env.NUM_DRONES = 1
    env.ACTION_BUFFER_SIZE = 2
    env._state_dim = 4
    env.CLIENT = 7
    env.action_space = spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(1, 4),
        dtype=np.float32,
    )
    env.action_buffer = deque(
        [
            np.ones((1, 4), dtype=np.float32),
            np.full((1, 4), 2.0, dtype=np.float32),
        ],
        maxlen=env.ACTION_BUFFER_SIZE,
    )
    env.observation_space = {
        "depth": spaces.Box(low=0.0, high=1.0, shape=(2, 2, 1), dtype=np.float32),
        "state": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
    }
    env._housekeeping = lambda: None
    env._updateAndStoreKinematicInformation = lambda: None
    env._startVideoRecording = lambda: None
    return env


def test_reset_generates_search_area_after_world_spawn(monkeypatch) -> None:
    env = _build_stub_env()

    monkeypatch.setattr(moving_drone_mod, "flight_reward", lambda **kwargs: 0.0)
    monkeypatch.setattr(moving_drone_mod.p, "resetSimulation", lambda **kwargs: None)
    monkeypatch.setattr(
        moving_drone_mod.p,
        "setPhysicsEngineParameter",
        lambda **kwargs: None,
    )

    def _spawn_task_world() -> None:
        env.GOAL_POS = np.array([100.0, 200.0, 300.0], dtype=float)

    env._spawn_task_world = _spawn_task_world
    env._computeObs = lambda: {
        "depth": np.zeros((2, 2, 1), dtype=np.float32),
        "state": np.zeros((4,), dtype=np.float32),
    }
    env._computeInfo = lambda: {
        "goal": tuple(env.GOAL_POS.tolist()),
        "success": env._success,
        "collision": env._collision,
    }
    env._generate_search_area_center = lambda seed=None: env.GOAL_POS.copy()

    obs, info = moving_drone_mod.MovingDroneAviary.reset(env, seed=123)

    assert np.array_equal(env._search_area_center, np.array([100.0, 200.0, 300.0]))
    assert np.array_equal(obs["state"], np.zeros((4,), dtype=np.float32))
    assert info == {
        "goal": (100.0, 200.0, 300.0),
        "success": False,
        "collision": False,
    }


def test_reset_clears_action_history(monkeypatch) -> None:
    env = _build_stub_env()

    monkeypatch.setattr(moving_drone_mod, "flight_reward", lambda **kwargs: 0.0)
    monkeypatch.setattr(moving_drone_mod.p, "resetSimulation", lambda **kwargs: None)
    monkeypatch.setattr(
        moving_drone_mod.p,
        "setPhysicsEngineParameter",
        lambda **kwargs: None,
    )

    env._spawn_task_world = lambda: None
    env._generate_search_area_center = lambda seed=None: env.GOAL_POS.copy()
    env._computeObs = lambda: {
        "depth": np.zeros((2, 2, 1), dtype=np.float32),
        "state": env.action_buffer[0][0].copy(),
    }
    env._computeInfo = lambda: {}

    obs, _ = moving_drone_mod.MovingDroneAviary.reset(env, seed=123)

    assert np.array_equal(obs["state"], np.zeros((4,), dtype=np.float32))
    assert all(np.count_nonzero(buf) == 0 for buf in env.action_buffer)


def test_spawn_task_world_sinks_default_plane_for_open_maps(monkeypatch) -> None:
    env = moving_drone_mod.MovingDroneAviary.__new__(moving_drone_mod.MovingDroneAviary)
    env.task = SimpleNamespace(
        map_seed=123,
        challenge_type=2,
        start=(1.0, 2.0, 3.0),
        goal=(4.0, 5.0, 6.0),
    )
    env._original_start = env.task.start
    env._original_goal = env.task.goal
    env.GOAL_POS = np.array(env.task.goal, dtype=float)
    env.CLIENT = 7
    env.DRONE_IDS = np.array([99], dtype=np.int64)
    env.PLANE_ID = 1234
    env._build_cull_targets = lambda: None

    monkeypatch.setattr(
        moving_drone_mod,
        "build_world",
        lambda **kwargs: ([], [], None, None, None, None),
    )
    monkeypatch.setattr(
        moving_drone_mod.p,
        "getQuaternionFromEuler",
        lambda euler: (0.0, 0.0, 0.0, 1.0),
    )

    reset_calls: list[tuple[int, tuple[float, float, float]]] = []

    def _reset_base_position_and_orientation(
        body_id,
        position,
        orientation,
        physicsClientId=None,
    ) -> None:
        reset_calls.append((int(body_id), tuple(float(v) for v in position)))

    monkeypatch.setattr(
        moving_drone_mod.p,
        "resetBasePositionAndOrientation",
        _reset_base_position_and_orientation,
    )
    monkeypatch.setattr(
        moving_drone_mod.p,
        "changeVisualShape",
        lambda *args, **kwargs: None,
    )

    moving_drone_mod.MovingDroneAviary._spawn_task_world(env)

    assert (99, (1.0, 2.0, 3.0)) in reset_calls
    assert (1234, (0.0, 0.0, -1000.0)) in reset_calls
