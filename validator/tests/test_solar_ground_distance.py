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

"""Solar Patrol ground distance (task 13): the reading straight down from the sensors under the drone.

Each test stands a small aircraft body with a foot below its belly over flat ground in its own physics world.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pybullet as p
import pytest

from swarm.challenge_families.solar_patrol import ground_distance
from swarm.challenge_families.solar_patrol.contract import DECISION_STEPS, STATE_SLICES, new_state
from swarm.challenge_families.solar_patrol.episode import SolarEpisode


@pytest.fixture
def world():
    """Flat ground at z = 0 and an aircraft whose foot hangs straight under the sensor; returns (env, episode)."""
    cli = p.connect(p.DIRECT)
    ground = p.createCollisionShape(p.GEOM_BOX, halfExtents=[50.0, 50.0, 0.5], physicsClientId=cli)
    p.createMultiBody(0, ground, -1, [0.0, 0.0, -0.5], physicsClientId=cli)
    body = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.05], physicsClientId=cli)
    foot = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.03, 0.03, 0.03], physicsClientId=cli)
    aircraft = p.createMultiBody(0, body, -1, [0.0, 0.0, 5.0], linkMasses=[0.0], linkCollisionShapeIndices=[foot],
                                 linkVisualShapeIndices=[-1], linkPositions=[[0.0, 0.0, -0.15]],
                                 linkOrientations=[[0, 0, 0, 1]], linkInertialFramePositions=[[0, 0, 0]],
                                 linkInertialFrameOrientations=[[0, 0, 0, 1]], linkParentIndices=[0],
                                 linkJointTypes=[p.JOINT_FIXED], linkJointAxis=[[0, 0, 1]], physicsClientId=cli)
    yield SimpleNamespace(CLIENT=cli, DRONE_IDS=[aircraft]), SolarEpisode(seed=0)
    p.disconnect(cli)


def _place(env, sensor_height, roll_deg=0.0):
    """Stand the aircraft so its sensor is sensor_height above the ground, rolled by roll_deg."""
    orn = p.getQuaternionFromEuler([math.radians(roll_deg), 0.0, 0.0])
    rot = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
    base = np.array([0.0, 0.0, sensor_height]) - rot @ ground_distance.SENSOR
    p.resetBasePositionAndOrientation(env.DRONE_IDS[0], base.tolist(), orn, physicsClientId=env.CLIENT)


def _reading(env, ep):
    """The ground distance and the downward-sensing flag the model is shown."""
    state = new_state()
    ground_distance.observe(env, ep, state)
    return float(state[STATE_SLICES["ground_distance_m"]][0]), float(state[STATE_SLICES["downward_sensing_ok"]][0])


def _reading_at(env, ep, sensor_height, roll_deg=0.0):
    """The ground distance read on a fresh patrol with the sensor sensor_height above the ground."""
    _place(env, sensor_height, roll_deg)
    ground_distance.reset(env, ep)
    return _reading(env, ep)[0]


@pytest.mark.parametrize("height", [0.5, 1.0, 5.0, 12.3, 15.9])
def test_reads_the_true_distance_inside_the_range(world, height):
    """From 0.5 to 16 m the model reads the true distance to the ground, not the drone's own foot."""
    assert _reading_at(*world, height) == pytest.approx(height, abs=1e-3)


@pytest.mark.parametrize("height", [16.1, 20.0, 45.0])
def test_beyond_the_range_reads_out_of_range(world, height):
    """Past 16 m the sensor sees no ground and gives the out-of-range value."""
    assert _reading_at(*world, height) == ground_distance.OUT_OF_RANGE_M


@pytest.mark.parametrize("height", [0.1, 0.36])
def test_closer_than_the_range_reads_its_nearest_distance(world, height):
    """Closer than 0.5 m, as in the dock or at touch-down, the reading holds at 0.5 m."""
    assert _reading_at(*world, height) == ground_distance.MIN_RANGE_M


def test_reads_the_first_thing_below(world):
    """A panel under the drone is what the sensor measures, not the ground beneath it."""
    env, ep = world
    panel = p.createCollisionShape(p.GEOM_BOX, halfExtents=[2.0, 1.0, 0.05], physicsClientId=env.CLIENT)
    p.createMultiBody(0, panel, -1, [0.0, 0.0, 1.95], physicsClientId=env.CLIENT)
    assert _reading_at(env, ep, 10.0) == pytest.approx(8.0, abs=1e-3)


def test_measures_straight_down_when_the_drone_tilts(world):
    """Rolled 25 degrees, the reading is still the vertical distance from the sensor to the ground."""
    assert _reading_at(*world, 6.0, roll_deg=25.0) == pytest.approx(6.0, abs=1e-3)


def test_a_new_reading_ten_times_a_second(world):
    """The reading changes only every DECISION_STEPS control steps and is held in between."""
    env, ep = world
    _place(env, 5.0)
    ground_distance.reset(env, ep)
    _place(env, 8.0)
    seen = []
    for step in range(1, DECISION_STEPS + 1):
        ep.step = step
        ground_distance.update(env, ep)
        seen.append(_reading(env, ep)[0])
    assert seen[:-1] == [pytest.approx(5.0, abs=1e-3)] * (DECISION_STEPS - 1)
    assert seen[-1] == pytest.approx(8.0, abs=1e-3)


def test_downward_sensing_reports_working(world):
    """The dock's working flag reaches the model as 1."""
    env, ep = world
    _place(env, 3.0)
    ground_distance.reset(env, ep)
    assert _reading(env, ep)[1] == 1.0
