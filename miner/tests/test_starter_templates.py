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

"""The starter controllers a miner copies before writing their own.

They ship as the first thing a miner runs, so they have to import on their own
and return an action of the shape their family declares.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "src" / "submission_template"


@pytest.mark.parametrize("name", ["drone_agent.py", "sentinel_drone_agent.py"])
def test_starter_imports_on_its_own(name, tmp_path):
    """A miner copies the file out of the package, so it cannot rely on it.

    Run from elsewhere under -I, or the repo on sys.path answers the imports and
    a starter that reaches back into swarm passes anyway."""
    copied = tmp_path / name
    copied.write_bytes((TEMPLATE_DIR / name).read_bytes())
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PWD")}
    result = subprocess.run(
        [sys.executable, "-I", "-c",
         f"import importlib.util;"
         f"spec=importlib.util.spec_from_file_location('m', r'{copied}');"
         f"m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);"
         f"assert hasattr(m,'DroneFlightController')"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
    )
    assert result.returncode == 0, result.stderr[-1500:]


def _load(name):
    """Execute one starter file from the template directory and return a fresh DroneFlightController out of it."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("starter", TEMPLATE_DIR / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DroneFlightController()


def test_sar_starter_returns_the_six_element_action_its_family_declares():
    """[dir_x, dir_y, dir_z, speed, yaw, rgb_request], speed within [0, 1]."""
    controller = _load("drone_agent.py")
    action = np.asarray(
        controller.act({"state": np.zeros(64, dtype=np.float32)}), dtype=np.float32
    )
    assert action.shape == (6,)
    assert 0.0 <= action[3] <= 1.0


def _sentinel_observation(phase, xy=(0.0, 0.0), time_left=390.0, fence=None):
    """A Swarm Sentinel observation with the state fields the starter reads, and the site map when a fence is given."""
    state = np.zeros(31, dtype=np.float32)
    state[0:2], state[7], state[9], state[19], state[23] = xy, 20.0, time_left, 60.0, phase
    observation = {"state": state}
    if fence is not None:
        site_map = np.zeros(660, dtype=np.float32)
        site_map[0] = len(fence)
        site_map[1:1 + 2 * len(fence)] = np.asarray(fence, dtype=np.float32).reshape(-1)
        observation["site_map"] = site_map
    return observation


SQUARE_FENCE = [(-60.0, -40.0), (60.0, -40.0), (60.0, 40.0), (-60.0, 40.0)]


def test_sentinel_starter_returns_the_24_value_action_and_presses_take_off_once():
    """In the dock it presses take_off, releases it on the next decision, and every value stays in its bounds."""
    controller = _load("sentinel_drone_agent.py")
    first = np.asarray(controller.act({"state": np.zeros(64, dtype=np.float32)}), dtype=np.float32)
    second = np.asarray(controller.act({"state": np.zeros(64, dtype=np.float32)}), dtype=np.float32)
    assert first.shape == (24,)
    assert first[21] == 1.0 and second[21] == 0.0
    assert np.all(first[:5] >= -1.0) and np.all(first[:5] <= 1.0)
    assert np.all(first[5:] >= 0.0) and np.all(first[5:] <= 1.0)


def test_sentinel_starter_sweeps_inside_the_fence_then_presses_return_home():
    """Its lanes keep 4 m inside the fence, it flies at the first one, and it presses return home when time runs short."""
    controller = _load("sentinel_drone_agent.py")
    controller.act(_sentinel_observation(0, fence=SQUARE_FENCE))
    route = controller.route
    assert len(route) >= 4
    assert np.all(np.abs(route[:, 0]) <= 56.0) and np.all(np.abs(route[:, 1]) <= 36.0)
    flying = np.asarray(controller.act(_sentinel_observation(2)), dtype=np.float32)
    assert np.hypot(flying[0], flying[1]) > 0.0 and flying[22] == 0.0
    late = np.asarray(controller.act(_sentinel_observation(2, time_left=20.0)), dtype=np.float32)
    assert late[22] == 1.0


def test_sentinel_starter_lanes_reach_both_far_edges_of_the_park():
    """Lanes are spread evenly across the park, so none sits more than half a spacing from either edge."""
    starter = type(_load("sentinel_drone_agent.py")).act.__globals__
    wide_fence = np.array([(-55.0, -40.0), (55.0, -40.0), (55.0, 40.0), (-55.0, 40.0)])
    lanes = np.unique(np.round(starter["_lanes"](wide_fence, 0)[:, 0], 6))
    half = starter["LANE_SPACING_M"] / 2.0
    assert lanes.min() + 55.0 <= half and 55.0 - lanes.max() <= half
    assert np.all(np.diff(lanes) <= 2.0 * half + 1e-9)


def test_sentinel_starter_sweeps_both_arms_of_a_u_shaped_park():
    """When no lane heading keeps every leg inside, the starter goes round the fence's corners instead of planning nothing."""
    starter = type(_load("sentinel_drone_agent.py")).act.__globals__
    u_fence = np.array([(-60, -20), (60, -20), (60, 80), (30, 80), (30, 10), (-30, 10), (-30, 80), (-60, 80)], float)
    route = starter["plan_route"](u_fence)
    assert len(route) > 0
    assert starter["_legs_clear"](u_fence, route)
    assert np.any((route[:, 0] < -30) & (route[:, 1] > 30)) and np.any((route[:, 0] > 30) & (route[:, 1] > 30))


def test_sentinel_starter_holds_return_home_until_the_straight_line_home_stays_inside():
    """Around a bend in the fence it flies back along its lanes instead of pressing return home across the corner."""
    bent_fence = [(-5.0, -10.0), (100.0, -10.0), (100.0, 100.0), (80.0, 100.0), (80.0, 10.0), (-5.0, 10.0)]
    controller = _load("sentinel_drone_agent.py")
    controller.act(_sentinel_observation(0, fence=bent_fence))
    controller.route = np.array([(40.0, 0.0), (90.0, 0.0), (90.0, 90.0)])
    controller.waypoint = len(controller.route)
    far_arm = np.asarray(controller.act(_sentinel_observation(2, xy=(90.0, 90.0))), dtype=np.float32)
    assert far_arm[22] == 0.0 and far_arm[0] < 0.0
    at_bend = np.asarray(controller.act(_sentinel_observation(2, xy=(90.0, 0.0))), dtype=np.float32)
    assert at_bend[22] == 1.0
