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

"""Solar Patrol per-seed score (task 23): detection, coverage and flight, and the two rules that zero a seed.

The score cases start from a patrol that took off, flew at 20 m and landed after its own return home, and change
one thing each. The distance tests stand flat terrain, a panel and a thief in their own physics world.
"""

from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pybullet as p
import pytest

from swarm.challenge_families import get_challenge_family
from swarm.challenge_families.solar_patrol import score, theft
from swarm.challenge_families.solar_patrol.contract import FAMILY_ID, HORIZON_S, SCORE_TERMS, Outcome
from swarm.challenge_families.solar_patrol.episode import SolarEpisode

_GROUND_Z = 30.0                     # terrain height at the thief, well above the dock, as on the sloping park
_THIEF_XY = (10.0, 0.0)


def _metrics(**changes):
    """A finished patrol's metrics: took off, peak 20 m, landed after its own return, then the given changes."""
    outcome = dict(asdict(Outcome()), took_off=True, landed_in_dock=True, returned_by_model=True, max_height_m=20.0,
                   end_reason="landed")
    return {**outcome, "time_sec": 300.0, "horizon_sec": HORIZON_S, "success": True, **changes}


def _final(**changes):
    """The seed score of a patrol with the given changes."""
    return score.score(None, _metrics(**changes))["final_score"]


@pytest.mark.parametrize("changes, expected", [
    ({"threats": 3, "valid_reports": 3, "coverage": 0.93}, 0.99),
    ({"coverage": 0.95}, 0.955),
    ({"coverage": 1.0}, 1.00),
    ({"threats": 3, "valid_reports": 2, "coverage": 0.93}, 0.75),
    ({"threats": 3, "valid_reports": 3, "false_alarms": 3, "coverage": 0.93}, 0.64),
    ({"took_off": False, "landed_in_dock": False, "returned_by_model": False, "max_height_m": 0.0,
      "end_reason": "timeout"}, 0.0),
])
def test_the_cards_cases_score_as_shown(changes, expected):
    """All 3 found = 0.99, empty 95% = 0.955, empty 100% = 1.00, 2 of 3 = 0.75, 3 + 3 false alarms = 0.64, no
    take-off = 0; the unstated parts of a case are those of the first one."""
    assert round(_final(**changes), 3) == pytest.approx(expected, abs=0.005)


def test_the_terms_are_each_between_0_and_1_and_weighted_70_20_10():
    """Each term is on its own 0 to 1 scale, and the seed score is 0.70, 0.20 and 0.10 of them."""
    terms = score.score(None, _metrics(threats=2, valid_reports=1, coverage=0.6, max_height_m=26.0))
    assert set(terms) == set(SCORE_TERMS)
    assert terms["detection_term"] == pytest.approx(0.5)
    assert terms["coverage_term"] == pytest.approx(0.6)
    assert terms["flight_term"] == pytest.approx(0.75)
    assert terms["final_score"] == pytest.approx(0.70 * 0.5 + 0.20 * 0.6 + 0.10 * 0.75)


def test_an_empty_seed_pays_only_for_the_share_searched():
    """With no threat, detection is the share searched: half the park scores about 0.55, no search earns nothing."""
    assert _final(coverage=0.5) == pytest.approx(0.55)
    assert score.score(None, _metrics(coverage=0.0))["detection_term"] == 0.0


def test_a_false_alarm_on_an_empty_seed_cuts_the_search():
    """Each false alarm on an empty seed divides the searched share: one halves it, three quarter it."""
    assert score.score(None, _metrics(coverage=1.0, false_alarms=1))["detection_term"] == pytest.approx(0.5)
    assert score.score(None, _metrics(coverage=1.0, false_alarms=3))["detection_term"] == pytest.approx(0.25)


def test_a_miss_and_a_false_alarm_cost_the_same():
    """Two of three threats found scores the same detection as two of two found with one false alarm."""
    missed = score.score(None, _metrics(threats=3, valid_reports=2))["detection_term"]
    wrong = score.score(None, _metrics(threats=2, valid_reports=2, false_alarms=1))["detection_term"]
    assert missed == pytest.approx(wrong) == pytest.approx(2 / 3)


def test_valid_reports_never_count_past_the_threats():
    """More valid reports than threats cannot lift detection past 1."""
    assert score.score(None, _metrics(threats=2, valid_reports=5))["detection_term"] == 1.0


@pytest.mark.parametrize("peak, height_part", [
    (0.5, 1.0), (20.0, 1.0), (22.0, 1.0), (24.0, 0.75), (26.0, 0.5), (29.0, 0.125), (30.0, 0.0), (45.0, 0.0),
])
def test_height_is_free_to_22_m_and_falls_in_a_straight_line_to_0_at_30_m(peak, height_part):
    """The height half of the flight term is full up to 22 m above the dock, low flying included, and 0 from 30 m."""
    assert score.score(None, _metrics(max_height_m=peak))["flight_term"] == pytest.approx(0.5 + 0.5 * height_part)


@pytest.mark.parametrize("end", [
    {"end_reason": "timeout", "landed_in_dock": False},
    {"end_reason": "timeout", "landed_in_dock": False, "returned_by_model": True},
    {"end_reason": "flight_limit", "landed_in_dock": False, "returned_by_model": False},
    {"end_reason": "landed", "landed_in_dock": True, "returned_by_model": False},
])
def test_only_the_models_own_return_that_lands_earns_the_landing_half(end):
    """The clock running out, a return still in the air at the end, or the flight limit's stop earn no landing."""
    assert score.score(None, _metrics(**end))["flight_term"] == pytest.approx(0.5)


def test_a_collision_keeps_detection_coverage_and_height_but_not_the_landing():
    """A crash ends the patrol with what it earned so far, less the landing half of the flight term."""
    crashed = _metrics(threats=1, valid_reports=1, coverage=0.4, end_reason="collision", landed_in_dock=False,
                       success=False)
    terms = score.score(None, crashed)
    assert (terms["detection_term"], terms["coverage_term"], terms["flight_term"]) == (1.0, 0.4, 0.5)
    assert terms["final_score"] == pytest.approx(0.70 + 0.20 * 0.4 + 0.10 * 0.5)


def test_a_patrol_that_never_took_off_scores_0_whatever_else_it_carries():
    """Staying in the dock earns nothing, not even the height half it trivially kept."""
    terms = score.score(None, _metrics(took_off=False, landed_in_dock=False, returned_by_model=False, max_height_m=0.0))
    assert terms["flight_term"] == 0.0 and terms["final_score"] == 0.0


@pytest.mark.parametrize("closest, zeroed", [(None, False), (0.0, True), (4.99, True), (5.0, False), (18.2, False)])
def test_closer_than_5_m_to_a_threat_zeroes_the_seed(closest, zeroed):
    """Coming within 5 m of a threat scores 0 however well the rest went; exactly 5 m is allowed."""
    final = _final(threats=1, valid_reports=1, coverage=1.0, min_threat_distance_m=closest)
    assert (final == 0.0) is zeroed


def test_the_family_scores_a_rollout_from_the_outcome_its_last_step_carried():
    """The family's evaluation hands the outcome to the scorer and reports its final score as the seed's score."""
    outcome = dict(asdict(Outcome()), took_off=True, landed_in_dock=True, returned_by_model=True, max_height_m=20.0,
                   end_reason="landed", coverage=0.95)
    evaluation = get_challenge_family(FAMILY_ID).evaluate_rollout(
        task=None, success=True, t=300.0, horizon=HORIZON_S, min_clearance=None, collision=False,
        failure_reason="NONE", info={"solar_outcome": outcome})
    assert evaluation.score == pytest.approx(0.955)
    assert evaluation.normalized_metrics["final_score"] == evaluation.score


@pytest.fixture
def world(monkeypatch):
    """Terrain whose top is _GROUND_Z, a 3 m panel over the thief's spot, and one thief inside the fence; returns
    (env, episode, the list of people theft reports)."""
    cli = p.connect(p.DIRECT)
    slab = p.createCollisionShape(p.GEOM_BOX, halfExtents=[50.0, 50.0, 0.5], physicsClientId=cli)
    terrain = p.createMultiBody(0, slab, -1, [0.0, 0.0, _GROUND_Z - 0.5], physicsClientId=cli)
    panel = p.createCollisionShape(p.GEOM_BOX, halfExtents=[2.0, 1.0, 0.05], physicsClientId=cli)
    p.createMultiBody(0, panel, -1, [_THIEF_XY[0], _THIEF_XY[1], _GROUND_Z + 3.0], physicsClientId=cli)
    people = [{"bodies": [], "xy": _THIEF_XY, "stage": "work", "threat": True}]
    monkeypatch.setattr(theft, "people", lambda ep, step=None: people)
    ep = SolarEpisode(seed=0, phase="flying", dock_position=np.zeros(3), terrain_uids=frozenset({terrain}))
    env = SimpleNamespace(CLIENT=cli, pos=np.zeros((1, 3)))
    score.reset(env, ep)
    yield env, ep, people
    p.disconnect(cli)


def _fly(env, ep, x, y, height_above_ground):
    """Put the drone at (x, y), height_above_ground over the thief's terrain, and run one score update."""
    env.pos[0] = (x, y, _GROUND_Z + height_above_ground)
    score.update(env, ep)


def test_the_closest_distance_is_measured_to_the_body_under_the_panels(world):
    """Straight over a thief the distance is to his head, found on the terrain past the panel above him."""
    env, ep, _ = world
    _fly(env, ep, *_THIEF_XY, 20.0)
    assert ep.outcome.min_threat_distance_m == pytest.approx(20.0 - score.PERSON_HEIGHT_M)


def test_a_drone_beside_a_standing_thief_is_measured_across(world):
    """Level with his body, the distance is the horizontal one, and 3 m zeroes the seed."""
    env, ep, _ = world
    _fly(env, ep, _THIEF_XY[0] + 3.0, _THIEF_XY[1], 1.0)
    assert ep.outcome.min_threat_distance_m == pytest.approx(3.0)
    closest = ep.outcome.min_threat_distance_m
    assert _final(threats=1, valid_reports=1, coverage=1.0, min_threat_distance_m=closest) == 0.0


def test_the_closest_distance_only_ever_falls(world):
    """Flying away after a close pass keeps the closest distance."""
    env, ep, _ = world
    _fly(env, ep, *_THIEF_XY, 8.0)
    _fly(env, ep, *_THIEF_XY, 20.0)
    assert ep.outcome.min_threat_distance_m == pytest.approx(8.0 - score.PERSON_HEIGHT_M)


def test_people_who_are_not_threats_and_threats_5_m_away_across_are_not_measured(world):
    """A man outside the fence, or a threat 5 m or more across, leaves the distance unset."""
    env, ep, people = world
    _fly(env, ep, _THIEF_XY[0] + 5.0, _THIEF_XY[1], 0.5)
    people[0]["threat"] = False
    _fly(env, ep, *_THIEF_XY, 1.0)
    assert ep.outcome.min_threat_distance_m is None


@pytest.mark.parametrize("phase", ["docked", "landed"])
def test_a_drone_in_the_dock_is_never_close_to_a_threat(world, phase):
    """A thief walking past the closed dock does not count against the patrol."""
    env, ep, _ = world
    ep.phase = phase
    _fly(env, ep, *_THIEF_XY, 1.0)
    assert ep.outcome.min_threat_distance_m is None


def test_the_peak_height_is_measured_above_the_dock():
    """Heights are above the take-off point, not above the ground under the drone, and only the highest is kept."""
    ep = SolarEpisode(seed=0, dock_position=np.array([0.0, 0.0, 14.0]))
    env = SimpleNamespace(CLIENT=-1, pos=np.zeros((1, 3)))
    score.reset(env, ep)
    for z in (14.0, 34.0, 37.5, 30.0):
        env.pos[0] = (0.0, 0.0, z)
        score.update(env, ep)
    assert ep.outcome.max_height_m == pytest.approx(23.5)
