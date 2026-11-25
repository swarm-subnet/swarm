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

"""Test Field v2's rules, each on its own: the edge rule, reports with a position, the score, the people, the site map,
the dock's spot and the daytime wind."""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pybullet as p
import pytest

from swarm.challenge_families.solar_patrol import camera, coverage, sensor_noise
from swarm.challenge_families.solar_patrol import dock as sentinel_dock
from swarm.challenge_families.solar_patrol import wind as sentinel_wind
from swarm.challenge_families.solar_patrol.contract import MAX_BUILDINGS, SITE_MAP_SLICES, STATE_SLICES, Box, new_site_map, new_state
from swarm.challenge_families.test_field_v2 import figures, flight_limit, mix, people, reports, score, site, site_map, wind
from swarm.challenge_families.test_field_v2.contract import HORIZON_S, FieldOutcome, FieldReport
from swarm.challenge_families.test_field_v2.episode import FieldEpisode
from swarm.challenge_families.test_field_v2.family import end_on_collision, write_clock
from swarm.constants import SIM_DT

_OUTLINE = np.array([[-30.0, -30.0], [30.0, -30.0], [30.0, 30.0], [-30.0, 30.0]])
_DOCK = np.array([5.0, -5.0, 2.0])


def _episode(phase="flying"):
    """An episode on the square stand-in outline, its dock at _DOCK and its edge rule reset."""
    ep = FieldEpisode(seed=3)
    ep.fence, ep.dock_position, ep.phase = _OUTLINE.copy(), _DOCK.copy(), phase
    flight_limit.reset(None, ep)
    return ep


def _env(x, y, z=20.0):
    """An environment whose drone stands at a point."""
    return SimpleNamespace(pos=np.array([[x, y, z]]))


@pytest.mark.parametrize("x, ends", [(29.0, True), (29.5, True), (28.98, False), (28.5, False), (31.0, True)])
def test_the_edge_rule_trips_within_one_metre_of_the_outline_or_outside_it(x, ends):
    """A drone out of the dock ends the patrol on the flight limit once it is 1 m or less inside the outline, or out."""
    ep = _episode()
    flight_limit.update(_env(x, 0.0), ep)
    assert (ep.outcome.end_reason == "flight_limit") is ends
    assert ep.outcome.min_edge_m == pytest.approx(30.0 - x)


@pytest.mark.parametrize("phase", ["taking_off", "returning", "landing"])
def test_the_edge_rule_holds_in_the_docks_own_flights(phase):
    """The rule also ends a take-off, a return or a landing that strays to the edge."""
    ep = _episode(phase)
    flight_limit.update(_env(29.5, 0.0), ep)
    assert ep.outcome.end_reason == "flight_limit"


def test_the_edge_rule_sleeps_in_the_dock():
    """A drone in the dock is never judged, wherever the numbers say it stands."""
    ep = _episode("docked")
    flight_limit.update(_env(40.0, 0.0), ep)
    assert ep.outcome.end_reason == "" and ep.outcome.min_edge_m is None


def test_the_state_reports_the_distance_to_the_outline_itself():
    """The state's flight-limit distance is the distance to the outline, and inside is 1 only inside it."""
    ep = _episode()
    state = new_state()
    flight_limit.observe(_env(25.0, 0.0), ep, state)
    assert state[STATE_SLICES["flight_limit_distance_m"]][0] == pytest.approx(5.0)
    assert state[STATE_SLICES["inside_flight_limit"]][0] == 1.0
    flight_limit.observe(_env(32.0, 0.0), ep, state)
    assert state[STATE_SLICES["flight_limit_distance_m"]][0] == pytest.approx(2.0)
    assert state[STATE_SLICES["inside_flight_limit"]][0] == 0.0


def test_the_site_map_gives_the_outline_as_fence_and_limit_and_the_nearest_houses(monkeypatch):
    """Fence and limit are both the outline from the dock; the houses come nearest the dock first, as many as fit."""
    ep = _episode()
    houses = np.array([[100.0 + k, 0.0, 8.0, 6.0, 10.0] for k in range(MAX_BUILDINGS + 4)][::-1])
    monkeypatch.setattr(site, "buildings", lambda episode: houses)
    vector = new_site_map()
    site_map.write(None, ep, vector)
    fence = vector[SITE_MAP_SLICES["fence_xy"]].reshape(-1, 2)[:4]
    assert vector[SITE_MAP_SLICES["fence_count"]][0] == 4 == vector[SITE_MAP_SLICES["limit_count"]][0]
    assert np.allclose(fence, _OUTLINE - _DOCK[:2])
    assert np.allclose(vector[SITE_MAP_SLICES["limit_xy"]], vector[SITE_MAP_SLICES["fence_xy"]])
    assert vector[SITE_MAP_SLICES["table_count"]][0] == 0
    rows = vector[SITE_MAP_SLICES["buildings"]].reshape(-1, 5)
    assert vector[SITE_MAP_SLICES["building_count"]][0] == MAX_BUILDINGS
    assert rows[0, 0] == pytest.approx(100.0 - _DOCK[0]) and np.all(np.diff(rows[:, 0]) > 0)


# ---------------------------------------------------------------------------------------------------- reports

_PERSON_UID = 7


@pytest.fixture
def frame(monkeypatch):
    """A 640 x 480 frame from 20 m whose object map shows one person, index 2, at pixels 300 to 340 by 200 to 260,
    standing 12 m east and 3 m north of the dock."""
    objects = np.full((480, 640), -1, dtype=np.int32)
    objects[200:260, 300:340] = _PERSON_UID
    view = camera.View("colour", (0.0, 0.0, 20.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0), 640, 480, 50.0, True, 0,
                       objects=objects)
    ep = _episode()
    ep.people = [people.Person(index=k, pose="standing", xy=(0.0, 0.0), ground_z=0.0, yaw=0.0) for k in range(3)]
    reports.reset(None, ep)
    monkeypatch.setattr(sensor_noise, "shown_view", lambda episode, image: view)
    monkeypatch.setattr(coverage, "height_above_ground", lambda env, episode, eye: 20.0)
    monkeypatch.setattr(people, "bodies", lambda episode: {_PERSON_UID: 2})
    monkeypatch.setattr(people, "reference", lambda episode, index, step=None: np.array([12.0, 3.0]))
    return ep


_ON_HIM = Box(cx=320.0 / 640.0, cy=230.0 / 480.0, w=40.0 / 640.0, h=60.0 / 480.0)


def _report(east=12.0, north=3.0, box=_ON_HIM, kind="person"):
    """A report on the feed frame."""
    return FieldReport(kind, "feed", box, east_m=east, north_m=north)


def test_a_box_on_the_person_with_his_position_is_valid(frame):
    """The right box and a position 0.9 m off his reference point is valid; reporting him again is a repeat."""
    reports.submit(None, frame, _report(east=12.0 + 0.9))
    reports.submit(None, frame, _report())
    assert [verdict for _step, _r, verdict in frame.reports["log"]] == ["valid", "repeat"]
    assert frame.outcome.valid_reports == 1 and frame.outcome.false_alarms == 1


def test_a_position_more_than_two_metres_off_is_a_false_alarm_and_he_can_still_be_reported(frame):
    """A wrong position is its own verdict and a false alarm, and leaves the person to a later right report."""
    reports.submit(None, frame, _report(east=12.0 + 1.5, north=3.0 + 1.5))
    reports.submit(None, frame, _report())
    assert [verdict for _step, _r, verdict in frame.reports["log"]] == ["wrong_position", "valid"]


@pytest.mark.parametrize("change, verdict", [({"box": Box(0.1, 0.1, 0.05, 0.05)}, "no_object"),
                                             ({"kind": "vehicle"}, "wrong_class")])
def test_sentinels_checks_still_come_first(frame, change, verdict):
    """A box on nothing and the vehicle class fail as in Sentinel, whatever the position says."""
    reports.submit(None, frame, _report(**change))
    assert frame.reports["log"][0][2] == verdict


def test_a_frame_from_over_24_m_carries_no_valid_report(frame, monkeypatch):
    """A frame taken higher than 24 m above the ground under the camera is too high for a report."""
    monkeypatch.setattr(coverage, "height_above_ground", lambda env, episode, eye: reports.MAX_REPORT_HEIGHT_M + 0.5)
    reports.submit(None, frame, _report())
    assert frame.reports["log"][0][2] == "too_high"


# ---------------------------------------------------------------------------------------------------- score

def _metrics(**changes):
    """A patrol that reported 2 of 2 people, searched the whole field, landed after its own return, flew at 20 m."""
    metrics = dict(vars(FieldOutcome(end_reason="landed", took_off=True, landed_in_dock=True, returned_by_model=True,
                                     threats=2, valid_reports=2, coverage=1.0, max_height_m=20.0)))
    metrics.update(changes)
    return metrics


def test_a_perfect_patrol_scores_one():
    """Everyone reported, the field searched, a landing after the model's own return: 1."""
    assert score.score(None, _metrics())["final_score"] == pytest.approx(1.0)


@pytest.mark.parametrize("end", ["flight_limit", "collision"])
def test_tripping_the_edge_or_any_collision_scores_zero(end):
    """A patrol ended on the edge rule or by any collision scores 0, whatever it had earned."""
    assert score.score(None, _metrics(end_reason=end, landed_in_dock=False))["final_score"] == 0.0


def test_the_other_terms_are_sentinels():
    """One person of two missed and one false alarm is Sentinel's detection, 1 / 3, and a timeout keeps it."""
    terms = score.score(None, _metrics(valid_reports=1, false_alarms=1, end_reason="timeout", landed_in_dock=False))
    assert terms["detection_term"] == pytest.approx(1.0 / 3.0)
    assert terms["final_score"] == pytest.approx(0.7 / 3.0 + 0.2 + 0.05)


def test_coming_within_five_metres_of_a_person_is_kept(monkeypatch):
    """The closest the drone came to a person out of the dock is kept, measured to a 1.8 m tall person, and it zeroes
    the seed."""
    ep = _episode()
    ep.terrain_uids = frozenset()
    ep.people = [people.Person(index=0, pose="standing", xy=(3.0, 4.0), ground_z=0.0, yaw=0.0)]
    monkeypatch.setattr(score.sentinel_score, "update", lambda env, episode: None)
    score.update(_env(0.0, 0.0, 3.0), ep)
    assert ep.outcome.min_threat_distance_m is None
    score.update(_env(0.0, 1.0, 3.0), ep)
    assert ep.outcome.min_threat_distance_m == pytest.approx(math.hypot(math.hypot(3.0, 3.0), 3.0 - 1.8))
    assert score.score(None, _metrics(min_threat_distance_m=ep.outcome.min_threat_distance_m))["final_score"] == 0.0


# ---------------------------------------------------------------------------------------------------- people

def test_one_seed_in_twenty_has_nobody_and_the_rest_hold_one_to_ten_by_the_decided_shares():
    """Over 20,000 seeds, about 5 % hold nobody, exactly those the mix leaves without people, every count from 0 to 10
    happens, and each count of 1 to 10 takes its decided share of the seeds that hold someone."""
    counts = np.array([people.wanted(seed) for seed in range(20000)])
    assert set(counts.tolist()) == set(range(11))
    assert abs(float(np.mean(counts == 0)) - (1.0 - mix.PEOPLE_SHARE)) < 0.006
    assert np.array_equal(counts == 0, [not mix.present(seed).people for seed in range(20000)])
    present = counts[counts > 0]
    for n, share in people.COUNT_SHARES.items():
        assert abs(float(np.mean(present == n)) - share) < 0.012, n


@pytest.fixture
def flat_ground():
    """A flat ground slab in a fresh physics client, with a wall standing on it at (20, 0)."""
    cli = p.connect(p.DIRECT)
    ground = p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=[80.0, 80.0, 0.5], physicsClientId=cli),
                               basePosition=[0.0, 0.0, -0.5], physicsClientId=cli)
    p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.3, 30.0, 1.0], physicsClientId=cli),
                      basePosition=[20.0, 0.0, 1.0], physicsClientId=cli)
    yield cli, frozenset({ground})
    p.disconnect(cli)


def test_people_stand_on_open_field_ground_apart_from_the_dock_and_each_other(flat_ground):
    """Six people stand inside the outline by 2 m, 8 m from the dock, 3 m apart, off the wall and every crown."""
    cli, terrain = flat_ground
    crowns = np.array([[-10.0, -10.0, 4.0]])
    for seed in range(5):
        spots = np.array(people.spots(cli, seed, _OUTLINE, terrain, _DOCK[:2], crowns, 6))
        assert len(spots) == 6
        assert np.all(np.abs(spots) <= 30.0 - people.PERSON_EDGE_M)
        assert np.all(np.hypot(*(spots - _DOCK[:2]).T) >= people.DOCK_CLEAR_M)
        gaps = np.hypot(*(spots[:, None] - spots[None]).transpose(2, 0, 1))
        assert np.all(gaps[~np.eye(6, dtype=bool)] >= people.APART_M)
        assert np.all(np.abs(spots[:, 0] - 20.0) > people.OPEN_M)
        assert np.all(np.hypot(*(spots + 10.0).T) >= 4.0 + people.CROWN_CLEAR_M)
    assert np.array_equal(people.spots(cli, 3, _OUTLINE, terrain, _DOCK[:2], crowns, 6),
                          people.spots(cli, 3, _OUTLINE, terrain, _DOCK[:2], crowns, 6))


def test_a_field_with_no_room_fails_its_seed(flat_ground):
    """When the people cannot all be placed, the seed's own world fails."""
    cli, terrain = flat_ground
    with pytest.raises(people.NoRoom):
        people.spots(cli, 1, np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]]), terrain, np.array([50.0, 50.0]),
                     np.zeros((0, 3)), 3)


def test_everyday_clothes_hold_no_mask_tool_or_glove():
    """The package the people draw from holds eyes, hair, clothes, shoes, hats and packs only: nothing a thief wears
    or carries to steal."""
    pieces = figures.catalogue().manifest["pieces"]
    assert {piece["slot"] for piece in pieces.values()} <= {"eyes", "hair", "top", "bottom", "shoes", "over", "head", "back"}
    assert not set(pieces) & {"balaclava", "balaclava3", "gaiter", "gloves", "bolt_cutters", "cable_coil"}


# ---------------------------------------------------------------------------------------------------- dock and wind

def test_the_docks_spot_keeps_its_margin_and_sentinels_default_is_unchanged(flat_ground):
    """The field's dock stands 5 m inside the outline, and Sentinel's own call draws the same spot as before."""
    cli, terrain = flat_ground
    fence = np.array([[-10.0, -30.0], [12.0, -30.0], [12.0, 30.0], [-10.0, 30.0]])
    spot = np.array(sentinel_dock.find_site(cli, 4, fence, terrain, np.zeros((0, 3)), margin_m=5.0, max_slope_deg=10.0))
    assert -5.0 <= spot[0] <= 7.0 and abs(spot[1]) <= 25.0
    default = sentinel_dock.find_site(cli, 4, fence, terrain, np.zeros((0, 3)))
    explicit = sentinel_dock.find_site(cli, 4, fence, terrain, np.zeros((0, 3)), margin_m=sentinel_dock.FENCE_MARGIN_M,
                                       max_slope_deg=sentinel_dock.MAX_SLOPE_DEG)
    assert default == explicit


def test_the_wind_follows_the_fields_own_night_and_sentinels_is_unchanged():
    """Every seed's wind is Sentinel's draw for the field's own day or night, a day seed's the daytime draw it always
    was, and Sentinel without the override keeps its own night draw."""
    seeds = range(200)
    assert all(wind.strength(s) == sentinel_wind.strength(s, night=site.night(s)) for s in seeds)
    assert all(wind.for_seed(s) == sentinel_wind.for_seed(s, night=False) for s in seeds if not site.night(s))
    assert any(site.night(s) for s in seeds)
    assert all(sentinel_wind.strength(s) == sentinel_wind.strength(s, night=sentinel_wind.is_night(s)) for s in seeds)
    assert any(sentinel_wind.is_night(s) for s in seeds)


@pytest.mark.parametrize("flown_s, left_s", [(0.0, HORIZON_S), (190.0, 10.0), (HORIZON_S + 5.0, 0.0)])
def test_the_state_counts_down_the_fields_own_clock(flown_s, left_s):
    """The seconds left run from the field's 200 s horizon to 0, not from Sentinel's longer patrol."""
    ep = _episode()
    ep.step = int(round(flown_s / SIM_DT))
    state = new_state()
    state[STATE_SLICES["time_left_s"]] = 390.0 - flown_s
    write_clock(ep, state)
    assert float(state[STATE_SLICES["time_left_s"]][0]) == pytest.approx(left_s)


def test_a_collision_on_the_landing_step_is_not_hidden_by_the_landing():
    """A hit on the very step the patrol ends as landed closes it as a collision, which scores 0."""
    ep = _episode("landed")
    ep.end("landed")
    end_on_collision(ep, ended_before=False)
    assert ep.outcome.end_reason == "collision"
    assert score.score(None, _metrics(end_reason=ep.outcome.end_reason))["final_score"] == 0.0


def test_a_collision_after_the_patrol_ended_changes_nothing():
    """Once an earlier step ended the patrol, a later hit leaves its end reason as it was."""
    ep = _episode("landed")
    ep.end("landed")
    end_on_collision(ep, ended_before=True)
    assert ep.outcome.end_reason == "landed"
