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

"""The SAR confirm rules: the hover predicate, its boundary grace, and the dwell that ends the episode."""
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
)
from swarm.protocol import MapTask


def _task():
    """A search-and-rescue MapTask on seed 4096, running 60 seconds at 30 Hz."""
    return MapTask(
        map_seed=4096,
        start=(0.0, 0.0, 1.5),
        goal=(8.0, 8.0, 1.5),
        sim_dt=1 / 30,
        horizon=60.0,
        challenge_type=2,
        family_id="cf_search_and_rescue",
        version="5.0.0",
    )


@pytest.fixture
def sar_env():
    """A reset MovingDroneAviary in SAR mode on the sample task, closed again afterwards."""
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
    """Teleport the drone body to (x, y, z) at the given velocity and refresh the cached kinematics."""
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
    """Hovering still in the middle of the height band directly over the victim satisfies the confirm."""
    env = sar_env
    vx, vy, _ = env.sar_world.victim_centre
    top_z = env.sar_world.victim_aabb[1][2]
    target_z = top_z + (SAR_HOVER_BAND[0] + SAR_HOVER_BAND[1]) / 2.0
    _place_drone(env, vx, vy, target_z, vel=(0.0, 0.0, 0.0))
    assert env._sar_check_predicate() is True


def test_predicate_false_outside_horizontal_radius(sar_env):
    """A metre beyond the confirm cylinder breaks the predicate however good the height is."""
    env = sar_env
    vx, vy, _ = env.sar_world.victim_centre
    top_z = env.sar_world.victim_aabb[1][2]
    target_z = top_z + 3.0
    _place_drone(env, vx + SAR_CONFIRM_HORIZ_RADIUS + 1.0, vy, target_z)
    assert env._sar_check_predicate() is False


def test_predicate_false_above_hover_band(sar_env):
    """Sitting a metre over the top of the band fails the confirm even when perfectly centred."""
    env = sar_env
    vx, vy, _ = env.sar_world.victim_centre
    top_z = env.sar_world.victim_aabb[1][2]
    _place_drone(env, vx, vy, top_z + SAR_HOVER_BAND[1] + 1.0)
    assert env._sar_check_predicate() is False


def test_predicate_false_in_no_touch_sphere(sar_env):
    """Closing right onto the victim never counts as a confirm, however well centred the drone is."""
    env = sar_env
    vc = np.asarray(env.sar_world.victim_centre)
    _place_drone(env, vc[0], vc[1], vc[2])
    assert env._sar_check_predicate() is False


def test_predicate_false_above_speed_limit(sar_env):
    """Crossing the victim faster than the confirm cap fails even from an otherwise perfect pose."""
    env = sar_env
    vx, vy, _ = env.sar_world.victim_centre
    top_z = env.sar_world.victim_aabb[1][2]
    target_z = top_z + 3.0
    _place_drone(env, vx, vy, target_z, vel=(SAR_CONFIRM_SPEED_MAX + 0.5, 0.0, 0.0))
    assert env._sar_check_predicate() is False


def test_hysteresis_keeps_active_at_2_05m(sar_env):
    """Once the confirm is active the 0.1 m grace holds it at 2.05 m from the victim, but not at 2.25 m."""
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
    """Hover time builds while the predicate holds, drops to zero the moment it breaks, and flips success once the full hold is served."""
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
    """Breaking into the sphere around the victim ends the episode and records NO_TOUCH_SPHERE."""
    env = sar_env
    vc = np.asarray(env.sar_world.victim_centre)
    _place_drone(env, vc[0], vc[1], vc[2] + 0.2)
    terminated = env._computeTerminated()
    assert terminated is True
    from swarm.protocol import FailureReason
    assert env._failure_reason == FailureReason.NO_TOUCH_SPHERE.value


def test_terminated_on_dwell_success(sar_env):
    """A completed hold ends the episode with no failure reason attached to it."""
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
