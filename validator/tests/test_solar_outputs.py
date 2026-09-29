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

"""Solar Patrol model outputs: every value the model sends stays inside the real aircraft's limits."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from swarm.challenge_families import get_challenge_family
from swarm.challenge_families.solar_patrol import camera, outputs, reports, sensor_noise, zoom
from swarm.challenge_families.solar_patrol.contract import (
    ACTION_DIM,
    ACTION_HIGH,
    ACTION_INDEX,
    ACTION_LOW,
    FAMILY_ID,
    MAX_ZOOMS,
    decode_action,
)
from swarm.challenge_families.solar_patrol.episode import SolarEpisode
from swarm.constants import SIM_DT


def _action(**values):
    """An action vector at rest, with the named fields set."""
    a = np.zeros(ACTION_DIM, dtype=np.float32)
    for name, value in values.items():
        a[ACTION_INDEX[name]] = value
    return a


def _command(**values):
    """The command a fresh press of these action values decodes to."""
    return decode_action(_action(**values), None)


def _patrol():
    """An aircraft resting in its dock, with the parts the outputs hand values to."""
    env = SimpleNamespace(NUM_DRONES=1, CTRL_TIMESTEP=SIM_DT, rpy=np.zeros((1, 3)), action_buffer=[])
    env._solar = SolarEpisode(seed=0)
    for part in (camera, zoom, reports, outputs, sensor_noise):
        part.reset(env, env._solar)
    return env, env._solar


def _asked(command):
    """Speed across the ground, climb rate and turn rate in degrees a second that a command asks for."""
    target = outputs.setpoint(SimpleNamespace(rpy=np.zeros((1, 3))), None, command)
    vx, vy, vz = target.velocity_mps
    return math.hypot(vx, vy), vz, -math.degrees(target.yaw_rate_rad_s)


def test_a_diagonal_flies_no_faster_than_a_straight_line():
    """Full forward and full right together ask for 5 m/s across the ground, still half-way between the two."""
    target = outputs.setpoint(SimpleNamespace(rpy=np.zeros((1, 3))), None,
                              _command(move_forward=1.0, move_right=1.0))
    vx, vy, _ = target.velocity_mps
    assert math.hypot(vx, vy) == pytest.approx(5.0)
    assert vx == pytest.approx(-vy)


def test_a_partial_stick_keeps_its_share_of_the_speed():
    """Inside the cap a stick asks for its own share of 5 m/s, in any direction."""
    assert _asked(_command(move_forward=0.6))[0] == pytest.approx(3.0)
    assert _asked(_command(move_forward=-0.3, move_right=0.4))[0] == pytest.approx(2.5)


def test_climb_descent_and_turn_limits():
    """Full sticks ask for 3 m/s up, 2 m/s down and 90 degrees a second either way."""
    assert _asked(_command(move_up=1.0))[1] == pytest.approx(3.0)
    assert _asked(_command(move_up=-1.0))[1] == pytest.approx(-2.0)
    assert _asked(_command(turn=1.0))[2] == pytest.approx(90.0)
    assert _asked(_command(turn=-1.0))[2] == pytest.approx(-90.0)


def test_values_outside_the_contract_are_held_to_its_bounds():
    """A raw action far outside the bounds is applied as its nearest in-bounds value, and never crashes the step."""
    env, ep = _patrol()
    raw = np.where(np.arange(ACTION_DIM) % 2 == 0, 40.0, -40.0).astype(np.float32)
    # The aircraft acts on a command one control step after it is sent.
    for _ in range(2):
        get_challenge_family(FAMILY_ID).preprocess_action(env, raw)
    held = env.action_buffer[-1].reshape(-1)
    assert np.all(held >= np.asarray(ACTION_LOW)) and np.all(held <= np.asarray(ACTION_HIGH))
    for name in ("move_forward", "move_right", "move_up", "turn", "gimbal_tilt"):
        assert abs(getattr(ep.command, name)) <= 1.0
    assert ep.command.night_mode == "auto"
    assert decode_action(outputs.clip(_action(night_mode=-40.0)), None).night_mode == "off"


def test_the_gimbal_turns_at_most_100_degrees_a_second():
    """From straight down, asking for straight up moves the camera 2 degrees per control step and takes 1.8 s."""
    env, ep = _patrol()
    down, up = _command(gimbal_tilt=-1.0), _command(gimbal_tilt=1.0)
    for _ in range(90):
        outputs.apply(env, ep, down)
    assert ep.camera["tilt_deg"] == pytest.approx(-90.0)
    tilts = [ep.camera["tilt_deg"]]
    while tilts[-1] < 90.0 - 1e-6 and len(tilts) < 1000:
        outputs.apply(env, ep, up)
        tilts.append(ep.camera["tilt_deg"])
    assert max(np.diff(tilts)) == pytest.approx(100.0 * SIM_DT)
    assert (len(tilts) - 1) * SIM_DT == pytest.approx(1.8)


def test_night_vision_works_only_through_the_7x_lens():
    """Night vision stays off with no zoom and on the 3x lens, and comes on once the 7x lens is in use."""
    env, ep = _patrol()
    outputs.apply(env, ep, _command(night_vision=1.0))
    assert not ep.zoom["night_vision"]
    outputs.apply(env, ep, _command(night_vision=1.0, zoom=1.0, zoom_lens=0.0))
    assert not ep.zoom["night_vision"]
    outputs.apply(env, ep, _command(night_vision=1.0, zoom=1.0, zoom_lens=1.0))
    assert ep.zoom["night_vision"]
    outputs.apply(env, ep, _command(night_vision=1.0, zoom=1.0, zoom_lens=0.0))
    assert not ep.zoom["night_vision"]


def test_a_refused_zoom_leaves_the_lens_in_use():
    """Once the patrol's zooms are spent, a 7x press changes no lens, so night vision stays off on the 3x view."""
    env, ep = _patrol()
    outputs.apply(env, ep, _command(zoom=1.0, zoom_lens=0.0))
    ep.outcome.zooms_used = MAX_ZOOMS
    outputs.apply(env, ep, _command(night_vision=1.0, zoom=1.0, zoom_lens=1.0))
    assert not ep.zoom["night_vision"]
    assert ep.outcome.zooms_used == MAX_ZOOMS
