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

"""The field's 10 m ceiling and its solid tree crowns: Sentinel keeps its own heights, the field's climb stops at the
ceiling, a crown ends the patrol as a collision, and the ground under a crown is no part of the coverage."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from swarm.challenge_families.solar_patrol import coverage as sentinel_coverage
from swarm.challenge_families.solar_patrol import dock as sentinel_dock
from swarm.challenge_families.solar_patrol import outputs
from swarm.challenge_families.solar_patrol.contract import HEIGHT_LIMIT_M, PATROL_HEIGHT_M, SITE_MAP_SLICES, new_site_map
from swarm.challenge_families.solar_patrol.episode import Setpoint, SolarEpisode
from swarm.challenge_families.test_field_v2 import coverage, crowns, site
from swarm.challenge_families.test_field_v2.contract import CEILING_M, TAKE_OFF_M
from swarm.challenge_families.test_field_v2.episode import FieldEpisode

_CROWNS = np.array([[10.0, 0.0, 3.0], [-20.0, 5.0, 4.5]])


class _Controller:
    """Records the velocity the outputs ask the flight controller for."""

    def __init__(self):
        """Nothing asked yet."""
        self.asked = None

    def computeControl(self, **kwargs):
        """Keep the target velocity and spin no rotor."""
        self.asked = np.array(kwargs["target_vel"], dtype=float)
        return np.zeros(4), None, None


def _env(z):
    """An environment whose drone hangs at a height, with a recording flight controller."""
    return SimpleNamespace(pos=np.array([[0.0, 0.0, z]]), quat=np.zeros((1, 4)), vel=np.zeros((1, 3)),
                           ang_v=np.zeros((1, 3)), rpy=np.zeros((1, 3)), CTRL_TIMESTEP=0.02, NUM_DRONES=1,
                           ctrl=[_Controller()])


def _climbing(ep, z, steps=200):
    """The upward speed the outputs end up asking for after the climb stick is held at a height."""
    env = _env(z)
    outputs.reset(env, ep)
    for _ in range(steps):
        outputs.fly(env, ep, Setpoint((0.0, 0.0, 3.0)))
    return float(env.ctrl[0].asked[2])


def test_sentinel_keeps_its_45_m_limit_and_its_20_m_take_off_and_return():
    """Sentinel's episode still stops at 45 m and flies its take-off and far return at 20 m above the dock."""
    ep = SolarEpisode(seed=1)
    assert (ep.height_limit_m, ep.patrol_height_m) == (HEIGHT_LIMIT_M, PATROL_HEIGHT_M) == (45.0, 20.0)
    far = np.array([80.0, 0.0, 3.0])
    assert sentinel_dock._return_height(ep, far) == pytest.approx(20.0)
    assert _climbing(ep, 30.0) == pytest.approx(3.0)
    assert _climbing(ep, 45.0) == 0.0


def test_the_field_stops_rising_at_its_ceiling():
    """The field's drone climbs freely under the ceiling, slows into it, and is never asked to rise past 10 m."""
    ep = FieldEpisode(seed=1)
    assert (ep.height_limit_m, ep.patrol_height_m) == (CEILING_M, TAKE_OFF_M) == (10.0, 9.5)
    assert _climbing(ep, 5.0) == pytest.approx(3.0)
    assert 0.0 < _climbing(ep, 9.8) < 1.0
    assert _climbing(ep, 10.0) == 0.0 and _climbing(ep, 10.3) == 0.0


@pytest.mark.parametrize("pos, height", [((80.0, 0.0, 4.0), TAKE_OFF_M), ((30.0, 0.0, 7.0), 7.0), ((3.0, 0.0, 8.0), 8.0)])
def test_the_docks_return_never_climbs_past_the_ceiling(pos, height):
    """Pressed far away the return climbs to the take-off height under the ceiling; nearer it keeps its height."""
    assert sentinel_dock._return_height(FieldEpisode(seed=1), np.array(pos)) == pytest.approx(height)


def _crowned(phase="flying"):
    """A field episode with two crowns, its dock at the map origin."""
    ep = FieldEpisode(seed=2)
    ep.phase, ep.crowns = phase, _CROWNS.copy()
    return ep


def test_flying_into_a_crown_ends_the_patrol_as_a_collision():
    """Outside every crown the closest gap is kept; inside one the patrol ends as a collision, at any height."""
    ep = _crowned()
    crowns.update(SimpleNamespace(pos=np.array([[10.0, 4.0, 9.0]])), ep)
    assert ep.outcome.end_reason == "" and ep.outcome.min_crown_m == pytest.approx(1.0)
    crowns.update(SimpleNamespace(pos=np.array([[10.0, 2.5, 2.0]])), ep)
    assert ep.outcome.end_reason == "collision" and ep.outcome.min_crown_m == pytest.approx(-0.5)


def test_a_crown_is_no_obstacle_to_a_drone_in_its_dock():
    """A drone resting in the dock is never judged against the crowns."""
    ep = _crowned("docked")
    crowns.update(SimpleNamespace(pos=np.array([[10.0, 0.0, 1.0]])), ep)
    assert ep.outcome.end_reason == "" and ep.outcome.min_crown_m is None


def test_the_site_map_carries_the_crowns_in_the_table_rows():
    """The table rows give each crown's centre from the dock and its diameter twice, nearest first."""
    ep = _crowned()
    ep.dock_position = np.array([2.0, 0.0, 1.0])
    vector = new_site_map()
    crowns.site_map(None, ep, vector)
    rows = vector[SITE_MAP_SLICES["tables"]].reshape(-1, 5)
    assert vector[SITE_MAP_SLICES["table_count"]][0] == 2
    assert rows[:2].tolist() == [[8.0, 0.0, 6.0, 6.0, 0.0], [-22.0, 5.0, 9.0, 9.0, 0.0]]


def test_the_ground_under_a_crown_is_no_part_of_the_coverage(monkeypatch):
    """Sentinel's cells are laid over the field and every one under a crown is dropped, with its normal and its flag."""
    cells = np.array([[10.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-20.0, 7.0, 0.0], [30.0, 30.0, 0.0]])

    def laid(env, ep):
        """Sentinel's grid, four cells."""
        ep.coverage = {"cells": cells, "normals": np.tile([0.0, 0.0, 1.0], (4, 1)), "searched": np.zeros(4, bool)}

    monkeypatch.setattr(sentinel_coverage, "reset", laid)
    monkeypatch.setattr(site, "passable_outlines", lambda ep: _CROWNS)
    ep = FieldEpisode(seed=3)
    coverage.reset(None, ep)
    assert ep.coverage["cells"][:, :2].tolist() == [[0.0, 0.0], [30.0, 30.0]]
    assert len(ep.coverage["normals"]) == len(ep.coverage["searched"]) == 2
