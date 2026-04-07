from __future__ import annotations

import gymnasium.spaces as spaces
import numpy as np

from swarm.core import moving_drone as moving_drone_mod


def test_step_runs_collision_bookkeeping_after_physics(monkeypatch) -> None:
    env = moving_drone_mod.MovingDroneAviary.__new__(moving_drone_mod.MovingDroneAviary)
    env.RECORD = False
    env.GUI = False
    env.USER_DEBUG = False
    env.USE_GUI_RPM = False
    env.VISION_ATTR = False
    env.step_counter = 0
    env.PYB_STEPS_PER_CTRL = 1
    env.PHYSICS = moving_drone_mod.Physics.PYB
    env.NUM_DRONES = 1
    env.CLIENT = 99
    env.CTRL_FREQ = 50
    env._time_alive = 0.0
    env._step_processed = False
    env._collision = False
    env._success = False
    env.action_space = spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(1, 4),
        dtype=np.float32,
    )
    env.last_clipped_action = np.zeros((1, 4), dtype=np.float32)

    order: list[str] = []
    flags = {"stepped": False}

    monkeypatch.setattr(
        moving_drone_mod.p,
        "stepSimulation",
        lambda **kwargs: (order.append("step_sim"), flags.__setitem__("stepped", True)),
    )

    env._preprocessAction = lambda action: action
    env._update_moving_platform = lambda: order.append("platform")
    env._physics = lambda rpm, i: order.append("physics_apply")
    env._updateAndStoreKinematicInformation = lambda: order.append("store_kin")

    def _check_collision():
        assert flags["stepped"] is True
        order.append("collision_check")
        return False, False

    env._check_collision = _check_collision
    env._update_landing_state = lambda platform_hit: order.append("landing")
    env._update_min_clearance = lambda: order.append("clearance")
    env._apply_distance_cull = lambda: order.append("cull")
    env._computeObs = lambda: {}
    env._computeReward = lambda: 0.0
    env._computeTerminated = lambda: False
    env._computeTruncated = lambda: False
    env._computeInfo = lambda: {}

    moving_drone_mod.MovingDroneAviary.step(env, np.zeros((1, 4), dtype=np.float32))

    assert order.index("step_sim") < order.index("collision_check")
    assert abs(env._time_alive - (1.0 / env.CTRL_FREQ)) < 1e-9


def test_advance_simulation_runs_platform_and_bookkeeping(monkeypatch) -> None:
    env = moving_drone_mod.MovingDroneAviary.__new__(moving_drone_mod.MovingDroneAviary)
    env.step_counter = 0
    env.PYB_STEPS_PER_CTRL = 1
    env.PHYSICS = moving_drone_mod.Physics.PYB
    env.CLIENT = 99
    env.CTRL_FREQ = 50
    env._time_alive = 0.0
    env._step_processed = False
    env._collision = False
    env._success = False

    order: list[str] = []
    flags = {"stepped": False}

    monkeypatch.setattr(
        moving_drone_mod.p,
        "stepSimulation",
        lambda **kwargs: (order.append("step_sim"), flags.__setitem__("stepped", True)),
    )

    env._update_moving_platform = lambda: order.append("platform")
    env._updateAndStoreKinematicInformation = lambda: order.append("store_kin")

    def _check_collision():
        assert flags["stepped"] is True
        order.append("collision_check")
        return False, False

    env._check_collision = _check_collision
    env._update_landing_state = lambda platform_hit: order.append("landing")
    env._update_min_clearance = lambda: order.append("clearance")
    env._apply_distance_cull = lambda: order.append("cull")

    moving_drone_mod.MovingDroneAviary.advance_simulation(env)

    assert order == [
        "platform",
        "step_sim",
        "store_kin",
        "collision_check",
        "landing",
        "clearance",
        "cull",
    ]
    assert env.step_counter == 1
    assert abs(env._time_alive - (1.0 / env.CTRL_FREQ)) < 1e-9
