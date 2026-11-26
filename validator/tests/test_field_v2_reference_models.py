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

"""Test Field v2's reference models and starter agent: what each cheap pilot presses, how a run is tallied, and the
starter's sweep staying clear of the field's edge."""
from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest

from swarm.challenge_families.solar_patrol.contract import STATE_SLICES, new_site_map, new_state
from swarm.challenge_families.test_field_v2.contract import ACTION_DIM, ACTION_INDEX, HORIZON_S
from validator.scripts import field_v2_reference_models as models
from validator.tests import field_v2_stand_in

_SQUARE = np.array(field_v2_stand_in.OUTLINE, dtype=float) + np.array([10.0, 0.0])   # the dock 10 m west of the centre


def _starter():
    """A fresh starter agent, loaded from its file as a miner's zip holds it."""
    spec = importlib.util.spec_from_file_location("starter_under_test", models.STARTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _observation(phase, xy=(0.0, 0.0), time_left=HORIZON_S, with_map=False):
    """An observation in a flight phase at a place, the site map holding the square outline when asked."""
    state = new_state()
    state[STATE_SLICES["flight_phase"]] = phase
    state[STATE_SLICES["position_m"]] = [xy[0], xy[1], 20.0]
    state[STATE_SLICES["height_above_takeoff_m"]] = 20.0
    state[STATE_SLICES["time_left_s"]] = time_left
    state[STATE_SLICES["ground_distance_m"]] = 60.0
    site = new_site_map()
    if with_map:
        site[0] = len(_SQUARE)
        site[1:1 + 2 * len(_SQUARE)] = _SQUARE.reshape(-1)
    return {"state": state, "site_map": site}


def test_the_starters_sweep_stays_well_inside_the_edge():
    """Every point and leg of the starter's sweep keeps its inset from the field's edge, far over the 1 m rule."""
    module = _starter()
    route = module.plan_route(_SQUARE)
    assert len(route) >= 4
    assert np.all(module._edge_distance(_SQUARE, route) >= module.EDGE_INSET_M - 1e-6)
    assert module._legs_clear(_SQUARE, route) and module.LEG_MARGIN_M > 1.0


def test_the_starters_sweep_keeps_off_the_crowns_the_site_map_gives():
    """The starter reads the crowns from the site map's table rows and plans every point and leg of its sweep its
    margin off each one, and the ones far from the field are left out."""
    module = _starter()
    agent = module.DroneFlightController()
    site = _observation(0, with_map=True)["site_map"]
    rows = np.array([[10.0, 0.0, 8.0, 8.0, 0.0], [-5.0, 12.0, 6.0, 6.0, 0.0], [300.0, 0.0, 10.0, 10.0, 0.0]])
    site[258] = len(rows)
    site[259:259 + rows.size] = rows.reshape(-1)
    route = agent._read_route(site)
    assert agent.crowns.tolist() == [[10.0, 0.0, 4.0], [-5.0, 12.0, 3.0]]
    path = np.vstack([[0.0, 0.0], route])
    samples = np.concatenate([np.linspace(p, q, 50) for p, q in zip(path[:-1], path[1:])])
    for east, north, radius in agent.crowns:
        assert np.min(np.hypot(samples[:, 0] - east, samples[:, 1] - north)) >= radius + module.CROWN_MARGIN_M - 0.1


def test_the_starter_takes_off_sweeps_and_comes_home_without_reporting():
    """In the dock it presses take-off; flying it moves; with time short it presses return home; it never reports
    and always sends 26 values."""
    agent = _starter().DroneFlightController()
    first = agent.act(_observation(0, with_map=True))
    assert first.shape == (ACTION_DIM,) and first[ACTION_INDEX["take_off"]] == 1.0
    flying = agent.act(_observation(2))
    assert np.any(flying[:2] != 0.0) and flying[ACTION_INDEX["report"]] == 0.0
    late = agent.act(_observation(2, time_left=5.0))
    assert late[ACTION_INDEX["return_home"]] == 1.0 and late[ACTION_INDEX["report"]] == 0.0


def test_hover_takes_off_and_presses_return_home_with_forty_seconds_left():
    """The hover pilot presses take-off on the first decision and return home once 40 s are left."""
    ep = SimpleNamespace(phase="flying", time_s=HORIZON_S - models.HOVER_HOME_LEFT_S - 1.0)
    pilot = models.Hover(SimpleNamespace(_solar=ep, EP_LEN_SEC=HORIZON_S))
    assert pilot.action(0)[ACTION_INDEX["take_off"]] == 1.0
    assert pilot.action(5)[ACTION_INDEX["return_home"]] == 0.0
    ep.time_s += 2.0
    assert pilot.action(6)[ACTION_INDEX["return_home"]] == 1.0


def test_the_tally_counts_each_pilots_people_and_false_alarms():
    """A pilot's line holds its mean seed score, the people it reported out of those on the field, and its ends."""
    rows = [{"pilot": "perfect", "score": 1.0, "outcome": {"end_reason": "landed", "threats": 3, "valid_reports": 3,
                                                          "false_alarms": 0}},
            {"pilot": "perfect", "score": 0.5, "outcome": {"end_reason": "timeout", "threats": 1, "valid_reports": 0,
                                                          "false_alarms": 2}}]
    table = models.tally(rows)["perfect"]
    assert table == {"seeds": 2, "mean_score": 0.75, "people": 4, "reported": 3, "false_alarms": 2,
                     "outside_reports": 0, "ends": {"landed": 1, "timeout": 1}, "penalised_score": 0.0}


def test_the_tally_counts_reports_on_people_outside_and_zeroes_five_seeds_for_each_mistake():
    """A report on someone passing outside the field is one false alarm, counted apart, and costs five of the
    pilot's best seeds."""
    rows = [{"pilot": "all_people", "score": 1.0 - k / 100.0,
             "outcome": {"end_reason": "landed", "threats": 1, "valid_reports": 1, "false_alarms": int(k == 9),
                         "outside_reports": int(k == 9), "missed_threats": 0, "edge_trips": 0}} for k in range(10)]
    table = models.tally(rows)["all_people"]
    assert table["false_alarms"] == 1 and table["outside_reports"] == 1
    assert table["penalised_score"] == pytest.approx(sum(1.0 - k / 100.0 for k in range(5, 10)) / 10.0, abs=1e-4)


def test_every_reference_model_is_listed():
    """The script flies the six models the family is judged against, the position-free one, the one that reports
    every warm body and the one that reports people outside the field."""
    assert set(models.PILOTS) == {"silent", "everything", "boxes_only", "hover", "random", "starter", "perfect",
                                  "warm_bodies", "all_people"}


def test_the_boxes_only_pilot_names_the_point_under_the_drone_for_any_frame():
    """The position-free pilot answers the reference pilot's call with a frame's step, naming the point under the
    drone."""
    pilot = SimpleNamespace(env=SimpleNamespace(pos=np.array([[12.0, -3.0, 9.0]])),
                            ep=SimpleNamespace(dock_position=np.array([2.0, 1.0, 0.0])))
    assert models.BoxesOnly.point(pilot, 4, 250).tolist() == [10.0, -4.0]


@pytest.mark.parametrize("name", ["hover", "random"])
def test_the_simple_pilots_send_the_full_action(name):
    """The pilots that do not search send 26 values inside the contract's bounds."""
    env = SimpleNamespace(_solar=SimpleNamespace(phase="docked", time_s=0.0, seed=3), EP_LEN_SEC=HORIZON_S)
    a = models.PILOTS[name](env).action(0)
    assert a.shape == (ACTION_DIM,)
    assert np.all(a >= np.asarray(models.ACTION_LOW)) and np.all(a <= np.asarray(models.ACTION_HIGH))


def test_the_warm_body_model_reports_people_first_then_animals_where_they_stood(monkeypatch):
    """Every warm body a frame shows is reported once, the person before the dog, the dog at its place then."""
    objects = np.full((480, 640), -1, dtype=np.int32)
    objects[100:140, 100:120], objects[300:320, 400:440] = 7, 9
    view = SimpleNamespace(objects=objects, sees=True, step=40, eye=(0.0, 0.0, 9.0), width=640, height=480)
    ep = SimpleNamespace(phase="flying", dock_position=np.array([1.0, 2.0, 0.0]),
                         animals=SimpleNamespace(members=[SimpleNamespace(uid=9)]))
    pilot = object.__new__(models.WarmBodies)
    pilot.env, pilot.ep, pilot.warm, pilot.told, pilot.reported, pilot.pressed = None, ep, {}, set(), set(), -2
    pilot.search = SimpleNamespace(action=lambda decision: np.zeros(ACTION_DIM, dtype=np.float32))
    pilot.eyes = SimpleNamespace(last=None)
    monkeypatch.setattr(models.sensor_noise, "shown_view", lambda episode, image: view)
    monkeypatch.setattr(models.coverage, "height_above_ground", lambda env, episode, eye: 9.0)
    monkeypatch.setattr(models.reports, "candidates", lambda episode: {7: ("person", 0), 9: ("animal", 9)})
    monkeypatch.setattr(models.people, "reference", lambda episode, index, step: np.array([3.0, 4.0]))
    monkeypatch.setattr(models.animals, "positions", lambda episode, step: np.array([[21.0, 12.0, 0.0]]) if step == 40 else None)
    first, idle, second = pilot.action(0), pilot.action(1), pilot.action(2)
    assert first[ACTION_INDEX["report"]] == 1.0 and idle[ACTION_INDEX["report"]] == 0.0
    assert second[ACTION_INDEX["report"]] == 1.0 and pilot.reported == {0}
    assert second[ACTION_INDEX["report_cx"]] == pytest.approx(420.0 / 640.0)
    reach = models.reference_pilot.REPORT_REACH_M
    assert second[ACTION_INDEX["report_east"]] == pytest.approx(20.0 / reach)
    assert second[ACTION_INDEX["report_north"]] == pytest.approx(10.0 / reach)
