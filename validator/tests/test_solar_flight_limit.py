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

"""Solar Patrol's flight limit (task 15): the line each seed draws around the fence, what the dock reports about it,
and the stop line that ends the patrol outside the fence.

The limit is drawn on a square fence here, and on the park's own fence when the map is installed.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest
from shapely.geometry import Point, Polygon

from swarm.challenge_families.solar_patrol import flight_limit, park
from swarm.challenge_families.solar_patrol.contract import (
    MAX_LIMIT_POINTS,
    SITE_MAP_SLICES,
    STATE_SLICES,
    new_site_map,
    new_state,
)
from swarm.challenge_families.solar_patrol.episode import SolarEpisode
from swarm.core.maps.solar.builder import SOLAR_ASSET_DIR

_SOLAR_ASSETS = os.environ.get("SOLAR_ASSET_DIR", SOLAR_ASSET_DIR)
_SQUARE = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]])
_SEEDS = range(200)
_FURTHEST = flight_limit.STOP_LINE_M + flight_limit.MAX_OUTSIDE_M


def _episode(seed=0, phase="flying", fence=_SQUARE):
    """A patrol with its limit drawn around the fence and a dock in the middle of the square."""
    ep = SolarEpisode(seed=seed, phase=phase, fence=fence, dock_position=np.array([50.0, 50.0, 30.0]))
    flight_limit.reset(None, ep)
    return ep


def _at(east, north):
    """An environment stand-in holding the drone at a point, 20 m up."""
    return SimpleNamespace(pos=np.array([[east, north, 20.0]]))


def _keeps_the_rules(limit, fence):
    """One outline whose stop line holds the whole fence and whose corners stand at most 10 m out."""
    outline = Polygon(limit)
    stop = outline.buffer(-flight_limit.STOP_LINE_M)
    furthest = max(Polygon(fence).exterior.distance(Point(corner)) for corner in limit)
    return outline.is_valid and stop.buffer(1e-9).covers(Polygon(fence)) and furthest <= _FURTHEST + 1e-9


# ---------------------------------------------------------------- the line each seed draws


def test_each_side_is_pushed_out_by_its_own_distance():
    """On a square fence each side of the limit stands out from its own side by that side's draw, 5 to 10 m: south,
    east, north and west are four different distances."""
    ep = _episode(seed=3)
    limit, push = ep.flight_limit["polygon"], ep.flight_limit["pushes"]
    assert limit[:, 1].min() == pytest.approx(-push[0])
    assert limit[:, 0].max() == pytest.approx(100.0 + push[1])
    assert limit[:, 1].max() == pytest.approx(100.0 + push[2])
    assert limit[:, 0].min() == pytest.approx(-push[3])
    assert len(set(np.round(push, 6))) == 4


def test_the_stop_line_stands_outside_the_fence_on_every_seed():
    """Every seed's stop line holds the whole fence, so every spot inside it can be flown, and the limit stands at
    most 10 m out."""
    for seed in _SEEDS:
        assert _keeps_the_rules(_episode(seed=seed).flight_limit["polygon"], _SQUARE)


def test_each_seed_draws_its_own_limit_and_keeps_it():
    """The same seed draws the same limit; across seeds every side's stop line draws over the whole 0 to 5 m outside
    the fence."""
    draws = np.array([_episode(seed=seed).flight_limit["pushes"] for seed in _SEEDS]) - flight_limit.STOP_LINE_M
    assert np.array_equal(draws[7] + flight_limit.STOP_LINE_M, _episode(seed=7).flight_limit["pushes"])
    assert len({tuple(np.round(d, 6)) for d in draws}) == len(draws)
    assert draws.min() >= 0.0 and draws.max() <= flight_limit.MAX_OUTSIDE_M
    assert np.all(draws.min(axis=0) < 0.25) and np.all(draws.max(axis=0) > 4.75)


@pytest.mark.skipif(not os.path.exists(os.path.join(_SOLAR_ASSETS, "manifest.json")),
                    reason=f"solar map not built at {_SOLAR_ASSETS}")
def test_the_parks_limit_keeps_every_rule_on_every_seed():
    """On the park's 28-sided fence every seed's limit is one outline, its stop line holds the whole fence, it stands
    at most 10 m out, and its corners fit the site map."""
    fence = park.fence_line(_SOLAR_ASSETS)
    assert len(fence) == 28
    for seed in _SEEDS:
        limit = _episode(seed=seed, fence=fence).flight_limit["polygon"]
        assert _keeps_the_rules(limit, fence)
        assert len(limit) <= MAX_LIMIT_POINTS


# ---------------------------------------------------------------- the stop line


@pytest.mark.parametrize("gap, ends", [(5.1, False), (4.9, True)])
def test_the_stop_line_ends_an_airborne_patrol(gap, ends):
    """A drone 4.9 m from the limit ends the patrol at the flight limit; 5.1 m from it flies on."""
    ep = _episode(seed=5)
    south = -ep.flight_limit["pushes"][0]
    flight_limit.update(_at(50.0, south + gap), ep)
    assert (ep.outcome.end_reason == "flight_limit") == ends


def test_a_drone_over_the_fence_flies_on():
    """The stop line stands outside the fence, so a drone right over it is still patrolling."""
    ep = _episode(seed=5)
    flight_limit.update(_at(50.0, 0.0), ep)
    assert not ep.outcome.end_reason


@pytest.mark.parametrize("phase", ["docked", "landed"])
def test_a_drone_on_the_ground_is_never_stopped(phase):
    """Only a drone in the air can reach the stop line; resting on the ground it cannot end the patrol."""
    ep = _episode(seed=5, phase=phase)
    flight_limit.update(_at(50.0, 1.0 - ep.flight_limit["pushes"][0]), ep)
    assert not ep.outcome.end_reason


def test_leaving_the_limit_ends_the_patrol():
    """A drone outside the limit, however it got there, ends the patrol."""
    ep = _episode(seed=5)
    flight_limit.update(_at(50.0, -30.0), ep)
    assert ep.outcome.end_reason == "flight_limit"


# ---------------------------------------------------------------- what the model is told


@pytest.mark.parametrize("north, inside", [(20.0, 1.0), (-30.0, 0.0)])
def test_the_dock_reports_the_distance_to_the_limit_and_whether_the_drone_is_inside(north, inside):
    """As DJI's dock reports a custom flight area: the distance to its edge, and whether the drone is inside it."""
    ep = _episode(seed=5)
    state = new_state()
    flight_limit.observe(_at(50.0, north), ep, state)
    distance = Polygon(ep.flight_limit["polygon"]).exterior.distance(Point(50.0, north))
    assert state[STATE_SLICES["flight_limit_distance_m"]][0] == pytest.approx(distance, abs=1e-4)
    assert state[STATE_SLICES["inside_flight_limit"]][0] == inside


def test_the_site_map_carries_the_limits_corners_from_the_dock():
    """The model gets the limit's shape at the start: every corner, in metres from this seed's dock."""
    ep = _episode(seed=5)
    site = new_site_map()
    flight_limit.site_map(None, ep, site)
    limit = ep.flight_limit["polygon"]
    count = int(site[SITE_MAP_SLICES["limit_count"]][0])
    corners = site[SITE_MAP_SLICES["limit_xy"]].reshape(-1, 2)
    assert count == len(limit)
    assert np.allclose(corners[:count], limit - ep.dock_position[:2], atol=1e-4)
    assert not corners[count:].any()
