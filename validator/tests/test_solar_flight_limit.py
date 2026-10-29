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

"""Solar Patrol's flight limit (tasks 15 and 40): the line each seed draws around the fence, what the dock reports
about it, and the stop line that ends the patrol outside the fence and clear of every panel table.

The limit is drawn on a square fence here, and on the park's own fence and tables when the map is installed.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pybullet as p
import pytest
from shapely.geometry import MultiPoint, Point, Polygon

from swarm.challenge_families.solar_patrol import flight_limit, park
from swarm.challenge_families.solar_patrol.contract import (
    MAX_LIMIT_POINTS,
    SITE_MAP_SLICES,
    STATE_SLICES,
    new_site_map,
    new_state,
)
from swarm.challenge_families.solar_patrol.episode import SolarEpisode
from swarm.core.maps.solar.builder import (
    SOLAR_ASSET_DIR,
    _footprint,
    build_solar_map,
    solar_manifest,
    solar_shifts,
)

_SOLAR_ASSETS = os.environ.get("SOLAR_ASSET_DIR", SOLAR_ASSET_DIR)
_SQUARE = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]])
_L_SHAPE = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 50.0], [50.0, 50.0], [50.0, 100.0], [0.0, 100.0]])
_SEEDS = range(200)
_FURTHEST = flight_limit.STOP_LINE_M + flight_limit.MAX_OUTSIDE_M
_PARK_SEEDS = range(40)


def _episode(seed=0, phase="flying", fence=_SQUARE, tables=()):
    """A patrol with its limit drawn around the fence, a dock in the middle of the square, and the tables given."""
    ep = SolarEpisode(seed=seed, phase=phase, fence=fence, dock_position=np.array([50.0, 50.0, 30.0]))
    ep.park = {"world": {"tables": list(tables)}}
    flight_limit.reset(None, ep)
    return ep


def _table(west, south, east, north):
    """A flat table's eight corners seen from above, a box between the two corners given."""
    return np.array([[x, y] for x in (west, east) for y in (south, north) for _ in range(2)])


def _clearance(limit, tables):
    """How far the stop line stands from the nearest table, measured as the patrol does, negative past it."""
    stop = Polygon(limit).buffer(-flight_limit.STOP_LINE_M, quad_segs=64)
    shapes = [MultiPoint(corners).convex_hull for corners in tables]
    return min(stop.exterior.distance(shape) * (1.0 if stop.contains(shape) else -1.0) for shape in shapes)


def _at_lowest(fence, tables):
    """The limit with every side at its lowest draw for these tables, the nearest any seed stands its stop line."""
    low = flight_limit.lowest(fence, tables)
    return flight_limit.outline(fence, flight_limit.STOP_LINE_M + low, low)


def _park_tables(seed):
    """Every piece of every panel table where the seed stands it, read from the survey and the seed's shifts."""
    manifest = solar_manifest(_SOLAR_ASSETS)
    shifts = solar_shifts(seed, _SOLAR_ASSETS)
    tables = []
    for index, place in enumerate(manifest["placements"]):
        if "row" in place:
            moved = dict(place, **{key: shifts[index][key] for key in ("position", "quaternion", "scale")})
            tables.append(_footprint(manifest["items"][place["item"]], moved))
    return np.array(tables)


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


# ---------------------------------------------------------------- the gap to the panel tables


def test_a_table_near_one_side_raises_only_that_sides_lowest_draw():
    """A table 0.4 m inside the south side lifts that side's stop line to at least 0.6 m out on every seed; the other
    three sides still draw over the whole 0 to 5 m."""
    table = _table(40.0, 0.4, 60.0, 4.0)
    assert flight_limit.lowest(_SQUARE, [table]) == pytest.approx([0.6, 0.0, 0.0, 0.0])
    draws = np.array([_episode(seed=seed, tables=[table]).flight_limit["pushes"] for seed in _SEEDS])
    draws -= flight_limit.STOP_LINE_M
    assert draws[:, 0].min() >= 0.6 and draws[:, 0].max() > 4.75
    assert np.all(draws[:, 1:].min(axis=0) < 0.25)


def test_tables_clear_of_every_side_leave_the_draws_as_they_were():
    """Tables at least a metre in from every side, however near a corner, leave every seed's limit exactly as it is
    drawn with no tables at all."""
    tables = [_table(20.0, 20.0, 40.0, 24.0), _table(1.0, 1.0, 21.0, 5.0), _table(60.0, 95.0, 99.0, 99.0)]
    for seed in _SEEDS:
        assert np.array_equal(_episode(seed=seed, tables=tables).flight_limit["pushes"],
                              _episode(seed=seed).flight_limit["pushes"])


@pytest.mark.parametrize("fence, table", [
    (_SQUARE, _table(98.0, 0.3, 99.8, 3.0)),
    (_SQUARE, _table(0.2, 96.0, 3.0, 99.9)),
    (_SQUARE, _table(0.01, 40.0, 2.0, 60.0)),
    (_SQUARE, _table(97.0, 98.0, 99.95, 99.99)),
    (_L_SHAPE, _table(45.0, 45.0, 49.9, 49.95)),
    (_L_SHAPE, _table(40.0, 52.0, 49.95, 60.0)),
    (_L_SHAPE, _table(55.0, 49.2, 70.0, 49.9)),
])
def test_the_lowest_draws_keep_the_stop_line_clear_of_a_table_by_a_side_or_a_corner(fence, table):
    """With every side at its lowest draw, the nearest a seed can stand the stop line, it still clears a table hard
    by a side, tucked into an outward corner or by an inward one by the full metre, and never comes inside the fence."""
    limit = _at_lowest(fence, [table])
    assert _clearance(limit, [table]) >= flight_limit.PANEL_CLEAR_M - 1e-9
    assert _keeps_the_rules(limit, fence)


@pytest.mark.skipif(not os.path.exists(os.path.join(_SOLAR_ASSETS, "manifest.json")),
                    reason=f"solar map not built at {_SOLAR_ASSETS}")
def test_the_parks_stop_line_clears_every_table_where_the_seed_stands_it():
    """On the park, with its tables where each seed shifts them and every side at its lowest draw, the stop line
    stands at least a metre off every piece of every table and holds the whole fence; any draw stands it further out."""
    fence = park.fence_line(_SOLAR_ASSETS)
    for seed in _PARK_SEEDS:
        tables = _park_tables(seed)
        assert len(tables) == 3 * 58
        limit = _at_lowest(fence, tables)
        assert _clearance(limit, tables) >= flight_limit.PANEL_CLEAR_M - 1e-9, seed
        assert _keeps_the_rules(limit, fence)


@pytest.mark.skipif(not os.path.exists(os.path.join(_SOLAR_ASSETS, "manifest.json")),
                    reason=f"solar map not built at {_SOLAR_ASSETS}")
@pytest.mark.timeout(600)
def test_the_built_park_hands_the_limit_every_table_where_the_seed_stands_it():
    """The park the patrol flies hands the flight limit the same table outlines the seed's shifts stand, so the gap is
    kept to the tables as built, not as surveyed."""
    cli = p.connect(p.DIRECT)
    try:
        world = build_solar_map(seed=6, cli=cli, asset_dir=_SOLAR_ASSETS, groups=("park",))
    finally:
        p.disconnect(cli)
    ep = SolarEpisode(seed=6, park={"world": world})
    assert park.table_footprints(ep) == pytest.approx(_park_tables(6), abs=1e-6)


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


@pytest.mark.parametrize("heading_deg", [0.0, 100.0, 225.0, 290.0])
def test_a_drone_flying_out_is_stopped_on_the_first_step_past_the_stop_line(heading_deg):
    """Checked every step on a straight flight out of the middle, the patrol ends on the first point whose distance to
    the limit is the stop line's or less, whichever checks were skipped while it was far inside."""
    ep = _episode(seed=11)
    edge = Polygon(ep.flight_limit["polygon"]).exterior
    heading = np.radians(heading_deg)
    path = [(50.0 + d * np.cos(heading), 50.0 + d * np.sin(heading)) for d in np.arange(0.0, 80.0, 0.37)]
    first = next(i for i, xy in enumerate(path) if edge.distance(Point(xy)) <= flight_limit.STOP_LINE_M)
    for i, (east, north) in enumerate(path):
        flight_limit.update(_at(east, north), ep)
        if ep.outcome.end_reason:
            break
    assert ep.outcome.end_reason == "flight_limit" and i == first


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
