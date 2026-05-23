from __future__ import annotations

import contextlib
import io

import numpy as np
import pybullet as p
import pytest

from swarm.constants import (
    SAR_CONFIRM_HORIZ_RADIUS,
    SAR_CONFIRM_SPEED_MAX,
    SAR_DWELL_SEC,
    SAR_HOVER_BAND,
    SAR_HYSTERESIS_GRACE,
    SAR_NO_TOUCH_RADIUS,
)
from swarm.protocol import MapTask


def _task():
    return MapTask(
        map_seed=4096,
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
            _task(),
            ctrl_freq=30,
            pyb_freq=30,
            sar_mode=True,
        )
        env.reset(seed=_task().map_seed)
    yield env
    try:
        env.close()
    except Exception:
        pass


def _place_drone(env, x, y, z, *, vel=(0.0, 0.0, 0.0)):
    cli = env.CLIENT
    p.resetBasePositionAndOrientation(
        env.DRONE_IDS[0],
        [float(x), float(y), float(z)],
        p.getQuaternionFromEuler([0, 0, 0]),
        physicsClientId=cli,
    )
    p.resetBaseVelocity(
        env.DRONE_IDS[0],
        linearVelocity=list(vel),
        angularVelocity=[0, 0, 0],
        physicsClientId=cli,
    )
    env._updateAndStoreKinematicInformation()


def test_predicate_true_when_centred_above_victim(sar_env):
    env = sar_env
    vx, vy, _ = env.sar_world.victim_centre
    top_z = env.sar_world.victim_aabb[1][2]
    target_z = top_z + (SAR_HOVER_BAND[0] + SAR_HOVER_BAND[1]) / 2.0
    _place_drone(env, vx, vy, target_z, vel=(0.0, 0.0, 0.0))
    assert env._sar_check_predicate() is True


def test_predicate_false_outside_horizontal_radius(sar_env):
    env = sar_env
    vx, vy, _ = env.sar_world.victim_centre
    top_z = env.sar_world.victim_aabb[1][2]
    target_z = top_z + 3.0
    _place_drone(env, vx + SAR_CONFIRM_HORIZ_RADIUS + 1.0, vy, target_z)
    assert env._sar_check_predicate() is False


def test_predicate_false_above_hover_band(sar_env):
    env = sar_env
    vx, vy, _ = env.sar_world.victim_centre
    top_z = env.sar_world.victim_aabb[1][2]
    _place_drone(env, vx, vy, top_z + SAR_HOVER_BAND[1] + 1.0)
    assert env._sar_check_predicate() is False


def test_predicate_false_in_no_touch_sphere(sar_env):
    env = sar_env
    vc = np.asarray(env.sar_world.victim_centre)
    _place_drone(env, vc[0], vc[1], vc[2])
    assert env._sar_check_predicate() is False


def test_predicate_false_above_speed_limit(sar_env):
    env = sar_env
    vx, vy, _ = env.sar_world.victim_centre
    top_z = env.sar_world.victim_aabb[1][2]
    target_z = top_z + 3.0
    _place_drone(env, vx, vy, target_z, vel=(SAR_CONFIRM_SPEED_MAX + 0.5, 0.0, 0.0))
    assert env._sar_check_predicate() is False


def test_hysteresis_keeps_active_at_2_05m(sar_env):
    env = sar_env
    vx, vy, _ = env.sar_world.victim_centre
    top_z = env.sar_world.victim_aabb[1][2]
    target_z = top_z + 3.0
    _place_drone(env, vx, vy, target_z)
    assert env._sar_check_predicate() is True
    env._sar_predicate_active = True
    _place_drone(env, vx + SAR_CONFIRM_HORIZ_RADIUS + 0.05, vy, target_z)
    assert env._sar_check_predicate() is True
    _place_drone(env, vx + SAR_CONFIRM_HORIZ_RADIUS + 0.25, vy, target_z)
    assert env._sar_check_predicate() is False


def test_dwell_accumulates_and_resets(sar_env):
    env = sar_env
    vx, vy, _ = env.sar_world.victim_centre
    top_z = env.sar_world.victim_aabb[1][2]
    target_z = top_z + 3.0

    _place_drone(env, vx, vy, target_z)
    for _ in range(int(round(1.5 * 30))):
        env._step_processed = False
        env._sar_step_update()
        env._time_alive += env._sim_dt
    assert env._sar_dwell_time >= 1.4
    assert not env._success

    _place_drone(env, vx + 5.0, vy, target_z)
    env._step_processed = False
    env._sar_step_update()
    assert env._sar_dwell_time == 0.0

    _place_drone(env, vx, vy, target_z)
    for _ in range(int(round(SAR_DWELL_SEC * 30)) + 3):
        env._step_processed = False
        env._sar_step_update()
        env._time_alive += env._sim_dt
    assert env._success is True


def test_terminated_on_no_touch_sphere(sar_env):
    env = sar_env
    vc = np.asarray(env.sar_world.victim_centre)
    _place_drone(env, vc[0], vc[1], vc[2] + 0.2)
    terminated = env._computeTerminated()
    assert terminated is True
    from swarm.protocol import FailureReason
    assert env._failure_reason == FailureReason.NO_TOUCH_SPHERE.value


def test_terminated_on_dwell_success(sar_env):
    env = sar_env
    vx, vy, _ = env.sar_world.victim_centre
    top_z = env.sar_world.victim_aabb[1][2]
    target_z = top_z + 3.0
    _place_drone(env, vx, vy, target_z)
    for _ in range(int(round(SAR_DWELL_SEC * 30)) + 3):
        env._step_processed = False
        env._sar_step_update()
        env._time_alive += env._sim_dt
    assert env._success
    terminated = env._computeTerminated()
    assert terminated is True
    from swarm.protocol import FailureReason
    assert env._failure_reason == FailureReason.NONE.value
