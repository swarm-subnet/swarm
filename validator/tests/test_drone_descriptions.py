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

"""Drones come from swarm-description: each is staged whole where the gym loads bodies from, and the numbers the code
turns into control steps keep the exact values the families have always flown with."""

from __future__ import annotations

from pathlib import Path

import gym_pybullet_drones.envs.BaseAviary as base_aviary
import pytest
import swarm_description

from swarm.challenge_families.solar_patrol import airframe, sensor_noise
from swarm.constants import OFFICE_TELEM_DELAY_STEPS, OFFICE_TELEM_PERIOD_STEPS
from swarm.core.moving_drone import DEFAULT_DRONE
from swarm.utils import gym_assets


@pytest.fixture
def stage(monkeypatch, tmp_path):
    """A fresh staging folder and registry of the test's own; returns the folder."""
    monkeypatch.setattr(gym_assets, "stage_dir", lambda: str(tmp_path))
    monkeypatch.setattr(gym_assets, "_STAGED", {})
    return tmp_path


@pytest.mark.parametrize("name", swarm_description.drones())
def test_a_drone_is_staged_whole_and_registered(stage, name):
    """Every URDF and mesh of a drone lands byte for byte in its own staged folder, and each URDF resolves there."""
    body = gym_assets.stage_drone(name)
    source = Path(swarm_description.drone_dir(name))
    files = swarm_description.get_params(name)["files"]
    assert body == files["body"]
    for urdf in files.values():
        assert gym_assets.staged_path(urdf) == str(stage / name / urdf)
        assert (stage / name / urdf).read_bytes() == (source / urdf).read_bytes()
    for mesh in sorted((source / "meshes").glob("*")):
        assert (stage / name / "meshes" / mesh.name).read_bytes() == mesh.read_bytes()


def test_the_default_drone_resolves_to_its_staged_copy(stage):
    """A family that names no drone flies cf2x, and the gym's own lookup of it lands on the staged copy."""
    gym_assets.stage_drone(DEFAULT_DRONE)
    resolved = base_aviary.pkg_resources.resource_filename("gym_pybullet_drones", "assets/cf2x.urdf")
    assert resolved == str(stage / "cf2x" / "cf2x.urdf")


def test_link_times_turn_into_the_control_steps_flown_before():
    """The M4TD's and Tello's link timings, kept in milliseconds and hertz, give the step counts the families shipped."""
    assert sensor_noise.COMMAND_DELAY_STEPS == 1
    assert sensor_noise.DATA_DELAY_STEPS == (3, 4)
    assert (OFFICE_TELEM_PERIOD_STEPS, OFFICE_TELEM_DELAY_STEPS) == (5, 2)


def test_the_m4td_tables_keep_their_shape():
    """Values the code reads as tuples and integer-keyed tables come out of the params file in those shapes."""
    assert airframe.SPIN == (1, -1, 1, -1)
    assert airframe.MOTOR_XY[0] == (0.17065, -0.1915)
    assert list(airframe.HEAT) == ["motor", "battery", "gimbal"]
