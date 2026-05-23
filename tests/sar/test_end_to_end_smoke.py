from __future__ import annotations

import contextlib
import io

import numpy as np
import pybullet as p
import pytest

from swarm.constants import HORIZON_SEC, SAR_DWELL_SEC, SAR_HOVER_BAND
from swarm.protocol import FailureReason, MapTask, ValidationResult
from swarm.validator.reward import flight_reward


def _task():
    return MapTask(
        map_seed=2718,
        start=(0.0, 0.0, 1.5),
        goal=(8.0, 8.0, 1.5),
        sim_dt=1 / 30,
        horizon=HORIZON_SEC,
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


def _build_upload_item(env, uid: int) -> dict:
    info = env._computeInfo()
    score = flight_reward(
        success=bool(info["success"]),
        t=float(env._time_alive),
        horizon=env.EP_LEN_SEC,
        task=env.task,
        min_clearance=info.get("min_clearance"),
        collision=info.get("collision", False),
        failure_reason=info["failure_reason"],
        sar_mode=True,
    )
    vr = ValidationResult(uid, info["success"], env._time_alive, score, failure_reason=info["failure_reason"])
    return {
        "seed_index": 0,
        "score": float(vr.score),
        "map_type": "open",
        "failure_reason": vr.failure_reason,
    }


def _place(env, x, y, z, vel=(0.0, 0.0, 0.0)):
    p.resetBasePositionAndOrientation(
        env.DRONE_IDS[0], [float(x), float(y), float(z)],
        p.getQuaternionFromEuler([0, 0, 0]),
        physicsClientId=env.CLIENT,
    )
    p.resetBaseVelocity(
        env.DRONE_IDS[0], linearVelocity=list(vel),
        angularVelocity=[0, 0, 0], physicsClientId=env.CLIENT,
    )
    env._updateAndStoreKinematicInformation()


@pytest.mark.timeout(180)
def test_confirmed_success_path(sar_env):
    env = sar_env
    vx, vy, _ = env.sar_world.victim_centre
    top_z = env.sar_world.victim_aabb[1][2]
    target_z = top_z + (SAR_HOVER_BAND[0] + SAR_HOVER_BAND[1]) / 2.0

    _place(env, vx, vy, target_z)
    for _ in range(int(round(SAR_DWELL_SEC * 30)) + 3):
        env._step_processed = False
        env._sar_step_update()
        env._time_alive += env._sim_dt
    assert env._success is True
    item = _build_upload_item(env, uid=1)
    assert item["failure_reason"] == FailureReason.NONE.value
    assert item["score"] > 0.0


@pytest.mark.timeout(180)
def test_no_touch_sphere_failure_path(sar_env):
    env = sar_env
    vc = env.sar_world.victim_centre
    _place(env, vc[0], vc[1], vc[2] + 0.2)
    env._step_processed = False
    env._sar_step_update()
    env._time_alive += env._sim_dt
    env._computeTerminated()
    item = _build_upload_item(env, uid=1)
    assert item["failure_reason"] == FailureReason.NO_TOUCH_SPHERE.value
    assert item["score"] == pytest.approx(0.01)


@pytest.mark.timeout(180)
def test_infeasible_failure_path(sar_env):
    env = sar_env
    _place(env, 200.0, 200.0, 5.0)
    env._time_alive = env.EP_LEN_SEC - 1.0
    env._step_processed = False
    env._sar_step_update()
    env._computeTruncated()
    item = _build_upload_item(env, uid=1)
    assert item["failure_reason"] == FailureReason.INFEASIBLE.value
    assert item["score"] == pytest.approx(0.01)
