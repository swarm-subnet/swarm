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

"""Solar Patrol drone state: position, heading, velocity, clock and battery every step, and the site map."""

from __future__ import annotations

import json
import math
import os
from types import SimpleNamespace

import numpy as np
import pybullet as p
import pytest

from swarm.challenge_families.solar_patrol import drone_state
from swarm.challenge_families.solar_patrol.contract import (
    HORIZON_S,
    MAX_BUILDINGS,
    MAX_TABLES,
    SITE_MAP_SLICES,
    STATE_SLICES,
    new_site_map,
    new_state,
)
from swarm.challenge_families.solar_patrol.episode import SolarEpisode
from swarm.constants import SIM_DT
from swarm.core.maps.solar.builder import SOLAR_ASSET_DIR

_SOLAR_ASSETS = os.environ.get("SOLAR_ASSET_DIR", SOLAR_ASSET_DIR)
_DOCK = np.array([60.0, 97.0, 31.5])
_HOVER = np.full((1, 4), 3800.0)


def _patrol(pos=(0.0, 0.0, 0.0), yaw=0.0, vel=(0.0, 0.0, 0.0)):
    """A patrol with its dock at _DOCK and the drone at pos from the dock, facing yaw, moving at vel."""
    env = SimpleNamespace(pos=(_DOCK + np.asarray(pos))[None, :], rpy=np.array([[0.0, 0.0, yaw]]),
                          vel=np.asarray(vel, dtype=float)[None, :], CTRL_TIMESTEP=SIM_DT,
                          last_clipped_action=np.zeros((1, 4)))
    ep = SolarEpisode(seed=0, dock_position=_DOCK.copy())
    drone_state.reset(env, ep)
    return env, ep


def _read(env, ep, name):
    """One named state field as the model would receive it."""
    state = new_state()
    drone_state.observe(env, ep, state)
    value = state[STATE_SLICES[name]]
    return value if value.size > 1 else float(value[0])


def _place(item, position, yaw_deg=0.0, pitch_deg=0.0, scale=(1.0, 1.0, 1.0)):
    """A manifest placement of item at position, turned yaw_deg counter-clockwise and tilted pitch_deg."""
    quaternion = p.getQuaternionFromEuler([0.0, math.radians(pitch_deg), math.radians(yaw_deg)])
    return {"item": item, "position": list(position), "quaternion": list(quaternion), "scale": list(scale)}


def _item(low, high):
    """A manifest item with these bounds."""
    return {"bounds_min": list(low), "bounds_max": list(high)}


def test_the_dock_reads_zero_zero_zero():
    """A drone resting on the pad is at 0, 0, 0 and 0 m above take-off, wherever the seed stands the dock."""
    env, ep = _patrol()
    assert _read(env, ep, "position_m") == pytest.approx([0.0, 0.0, 0.0])
    assert _read(env, ep, "height_above_takeoff_m") == 0.0


def test_position_is_metres_east_north_up_from_the_dock():
    """A drone 12 m east, 5 m south and 20 m up of the dock reads exactly that, and 20 m above take-off."""
    env, ep = _patrol(pos=(12.0, -5.0, 20.0))
    assert _read(env, ep, "position_m") == pytest.approx([12.0, -5.0, 20.0])
    assert _read(env, ep, "height_above_takeoff_m") == pytest.approx(20.0)


@pytest.mark.parametrize("yaw_deg, compass", [(90.0, 0.0), (0.0, 90.0), (-90.0, 180.0), (180.0, -90.0),
                                              (45.0, 45.0), (135.0, -45.0)])
def test_heading_is_a_compass_heading(yaw_deg, compass):
    """Facing north reads 0, east 90, west -90 and south 180, clockwise positive as DJI reports it."""
    env, ep = _patrol(yaw=math.radians(yaw_deg))
    heading = _read(env, ep, "heading_deg")
    assert -180.0 <= heading < 180.0
    assert (heading - compass + 180.0) % 360.0 - 180.0 == pytest.approx(0.0, abs=1e-4)


def test_velocity_is_east_north_up():
    """The velocity is passed on in the same east, north, up metres a second as the position."""
    env, ep = _patrol(vel=(4.0, -3.0, 1.5))
    assert _read(env, ep, "velocity_mps") == pytest.approx([4.0, -3.0, 1.5])


def test_time_left_counts_down_from_390():
    """The clock starts at 390 s, loses 0.1 s a decision and stops at 0."""
    env, ep = _patrol()
    assert _read(env, ep, "time_left_s") == HORIZON_S == 390.0
    ep.step = int(round(100.0 / SIM_DT))
    assert _read(env, ep, "time_left_s") == pytest.approx(290.0)
    ep.step = int(round((HORIZON_S + 5.0) / SIM_DT))
    assert _read(env, ep, "time_left_s") == 0.0


def test_the_battery_holds_its_charge_while_the_rotors_are_still():
    """The patrol starts on the dock's 95 % and a drone waiting in its dock uses none of it."""
    env, ep = _patrol()
    for _ in range(int(round(60.0 / SIM_DT))):
        drone_state.update(env, ep)
    assert _read(env, ep, "battery_pct") == 95.0


def test_a_whole_patrol_in_the_air_uses_about_14_percent():
    """390 s with the rotors turning drain 390 s of the M4TD's 47 min hover, from 95 % to 81 %."""
    env, ep = _patrol()
    env.last_clipped_action = _HOVER
    for _ in range(int(round(HORIZON_S / SIM_DT))):
        drone_state.update(env, ep)
    assert ep.drone_state["battery_pct"] == pytest.approx(95.0 - 100.0 * HORIZON_S / (47 * 60))
    assert _read(env, ep, "battery_pct") == 81.0


def test_the_battery_reads_in_whole_percent_and_never_below_zero():
    """The reading is the charge rounded to a whole percent, as the aircraft reports it, and stops at 0."""
    env, ep = _patrol()
    ep.drone_state["battery_pct"] = 94.6
    assert _read(env, ep, "battery_pct") == 95.0
    ep.drone_state["battery_pct"] = 94.4
    assert _read(env, ep, "battery_pct") == 94.0
    ep.drone_state["battery_pct"] = 0.0005
    env.last_clipped_action = _HOVER
    drone_state.update(env, ep)
    assert ep.drone_state["battery_pct"] == 0.0


def test_a_footprint_is_the_outline_seen_from_above():
    """A 20 x 4 m piece with off-centre bounds, turned 30 degrees and tilted, gives its real centre, its outline
    across the ground and the compass heading of its long side."""
    place = _place("table", (10.0, 20.0, 5.0), yaw_deg=30.0, pitch_deg=10.0)
    east, north, length, width, heading = drone_state._footprint(np.array([-8.0, -2.0, -1.0]),
                                                                 np.array([12.0, 2.0, 1.0]), place)
    along = np.array([math.cos(math.radians(30.0)), math.sin(math.radians(30.0))])
    assert (east, north) == pytest.approx(tuple(np.array([10.0, 20.0]) + 2.0 * math.cos(math.radians(10.0)) * along))
    assert length == pytest.approx(20.0 * math.cos(math.radians(10.0)) + 2.0 * math.sin(math.radians(10.0)))
    assert width == pytest.approx(4.0)
    assert heading == pytest.approx(60.0)


def test_the_long_side_sets_the_heading_whichever_axis_it_lies_on():
    """A piece longer along its own y than its x reports that side as its length, and its heading turns with it."""
    place = _place("hut", (0.0, 0.0, 0.0))
    _, _, length, width, heading = drone_state._footprint(np.array([-1.0, -3.0, 0.0]), np.array([1.0, 3.0, 2.0]),
                                                          place)
    assert (length, width, heading) == pytest.approx((6.0, 2.0, 0.0))


def test_the_survey_reads_tables_and_buildings_from_the_manifest(tmp_path):
    """A table is one outline over its three pieces, the buildings come as listed, and nothing else is read."""
    parts = {"frame": ([-10.0, -2.0, -1.0], [10.0, 2.0, 1.0]), "glass": ([-10.1, -2.0, -0.9], [10.1, 2.0, 0.9]),
             "racking": ([-10.0, -2.0, -1.9], [10.0, 2.1, 0.9])}
    items = {f"{size}_table_{part}": _item(*bounds) for size in ("full", "half") for part, bounds in parts.items()}
    items.update({name: _item([-2.0, -1.5, 0.0], [2.0, 1.5, 3.0]) for name in drone_state.BUILDINGS})
    items["fence_post"] = _item([-0.1, -0.1, 0.0], [0.1, 0.1, 2.0])
    placements = [_place(f"full_table_{part}", (50.0, 100.0, 40.0)) for part in parts]
    placements += [_place("white_unit_north", (5.0, -50.0, 25.0)), _place("fence_post", (0.0, 0.0, 0.0))]
    (tmp_path / "manifest.json").write_text(json.dumps({"items": items, "placements": placements}))
    tables, buildings = drone_state.survey(str(tmp_path))
    assert tables == pytest.approx(np.array([[50.0, 100.05, 20.2, 4.1, 90.0]]))
    assert buildings == pytest.approx(np.array([[5.0, -50.0, 4.0, 3.0, 90.0]]))


def test_the_site_map_places_the_survey_from_the_dock(monkeypatch):
    """Fence, tables and buildings arrive in metres from this seed's dock, each list counted and zero padded, and
    the cached survey itself is left in world metres for the next patrol."""
    tables = np.array([[70.0, 110.0, 20.0, 4.0, 89.0], [40.0, 90.0, 10.0, 4.0, 88.0]])
    buildings = np.array([[6.8, -52.1, 3.8, 3.7, 0.0]])
    monkeypatch.setattr(drone_state, "survey", lambda asset_dir: (tables, buildings))
    env, ep = _patrol()
    ep.park = {"world": {"asset_dir": "survey"}}
    ep.fence = np.array([[0.0, 40.0], [120.0, 40.0], [120.0, 160.0]])
    site = new_site_map()
    drone_state.site_map(env, ep, site)
    shift = np.array([_DOCK[0], _DOCK[1], 0.0, 0.0, 0.0])
    assert site[SITE_MAP_SLICES["fence_count"]][0] == 3
    assert site[SITE_MAP_SLICES["fence_xy"]][:6] == pytest.approx((ep.fence - _DOCK[:2]).reshape(-1))
    assert site[SITE_MAP_SLICES["table_count"]][0] == 2
    assert site[SITE_MAP_SLICES["tables"]].reshape(MAX_TABLES, 5)[:2] == pytest.approx(tables - shift)
    assert not site[SITE_MAP_SLICES["tables"]][10:].any()
    assert site[SITE_MAP_SLICES["building_count"]][0] == 1
    assert site[SITE_MAP_SLICES["buildings"]].reshape(MAX_BUILDINGS, 5)[:1] == pytest.approx(buildings - shift,
                                                                                              abs=1e-4)
    assert tables[0, 0] == 70.0


@pytest.mark.skipif(not os.path.exists(os.path.join(_SOLAR_ASSETS, "manifest.json")),
                    reason=f"solar map not built at {_SOLAR_ASSETS}")
def test_the_real_park_survey_holds_58_tables_and_3_buildings():
    """The Manolia survey gives 45 full and 13 half tables in rows running east to west, and the two white units
    and the service cabin, all inside the site map's room."""
    tables, buildings = drone_state.survey(_SOLAR_ASSETS)
    assert len(tables) == 58 <= MAX_TABLES
    assert len(buildings) == 3 <= MAX_BUILDINGS
    assert np.sum(tables[:, 2] > 15.0) == 45
    assert np.all((tables[:, 2] > 9.5) & (tables[:, 2] < 20.5))
    assert np.all(np.abs(tables[:, 4] - 90.0) < 2.0)
    assert np.all((buildings[:, 2] > 3.0) & (buildings[:, 2] < 4.5))
