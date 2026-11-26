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
"""Test Field v2's rules when people move and the field has company: a report is judged where the person stood in
the frame it names, its box can land on someone outside the field or on an animal, those false alarms are counted
apart, the 5 m rule follows each person, and the score's formula is unchanged."""
from __future__ import annotations

import math
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from swarm.challenge_families import get_challenge_family
from swarm.challenge_families.solar_patrol import camera, coverage, sensor_noise
from swarm.challenge_families.solar_patrol import reports as sentinel_reports
from swarm.challenge_families.solar_patrol.contract import Box
from swarm.challenge_families.test_field_v2 import animals, contract, people, reports, score
from swarm.challenge_families.test_field_v2.contract import FAMILY_ID, HORIZON_S, RESULT_METRICS, FieldOutcome, FieldReport
from swarm.challenge_families.test_field_v2.episode import FieldEpisode
from swarm.constants import SIM_DT
from swarm.domain_model import get_challenge_family_definition, get_score_schema

_DOCK = np.array([5.0, -5.0, 2.0])
_WALKER, _OUTSIDER, _DOG = 7, 8, 9           # drawn body ids
_FRAME_STEP = 100
_PACE_M = 1.0 * SIM_DT                       # the walker's metres per control step, at 1 m/s
_PRESS_STEP = _FRAME_STEP + 75               # 1.5 s after the frame, 1.5 m further east
_OUTSIDER_AT = np.array([60.0, 30.0, 2.0])

# Pixel rectangles (left, top, right, bottom) of each body in the stand-in frames.
_WALKER_PX = (300, 200, 340, 260)
_OUTSIDER_PX = (100, 100, 140, 160)
_DOG_PX = (400, 300, 460, 330)


def _walker_at(step: int) -> np.ndarray:
    """The walker's reference point and the ground under it at a step: he heads east at 1 m/s from (17, -2)."""
    return np.array([17.0 + _PACE_M * (step - _FRAME_STEP), -2.0, 2.0])


def _box(px: tuple) -> Box:
    """A report box exactly on a pixel rectangle of a 640 x 480 frame."""
    left, top, right, bottom = px
    return Box(cx=(left + right) / 1280.0, cy=(top + bottom) / 960.0, w=(right - left) / 640.0, h=(bottom - top) / 480.0)


def _frame(step: int, sees: bool = True, drawn=((_WALKER_PX, _WALKER), (_OUTSIDER_PX, _OUTSIDER), (_DOG_PX, _DOG))):
    """A 640 x 480 frame taken at a step whose object map holds the given bodies, later ones drawn over earlier."""
    objects = np.full((480, 640), -1, dtype=np.int32)
    for (left, top, right, bottom), uid in drawn:
        objects[top:bottom, left:right] = uid
    return camera.View("colour", (0.0, 0.0, 9.5), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0), 640, 480, 50.0, sees, step,
                       objects=objects)


@pytest.fixture
def field(monkeypatch):
    """A patrol at _PRESS_STEP with a walker on the field (index 0, 1.85 m tall), a man passing outside it (index 1)
    and a dog, its reports reset; set field.shown to the frame the model was shown."""
    ep = FieldEpisode(seed=3)
    ep.dock_position, ep.phase, ep.step = _DOCK.copy(), "flying", _PRESS_STEP
    ep.people = [people.Person(index=0, pose="standing", xy=(17.0, -2.0), ground_z=2.0, yaw=0.0, height_m=1.85),
                 people.Person(index=1, pose="standing", xy=tuple(_OUTSIDER_AT[:2]), ground_z=2.0, yaw=0.0,
                               inside=False)]
    ep.animals = {_DOG: "dog"}
    ep.shown = _frame(_FRAME_STEP)
    reports.reset(None, ep)

    def position(episode, index, step=None):
        """Each person's point at a step, now when none is given."""
        at = episode.step if step is None else step
        return _walker_at(at) if index == 0 else _OUTSIDER_AT.copy()

    monkeypatch.setattr(people, "position", position)
    monkeypatch.setattr(people, "bodies", lambda episode: {_WALKER: 0, _OUTSIDER: 1})
    monkeypatch.setattr(sensor_noise, "shown_view", lambda episode, image: episode.shown)
    monkeypatch.setattr(coverage, "height_above_ground", lambda env, episode, eye: 9.5)
    return ep


def _report(ep, px, east, north, kind="person"):
    """Press report on the shown feed frame with a box on a pixel rectangle and a named point; returns the verdict."""
    reports.submit(None, ep, FieldReport(kind, "feed", _box(px), east_m=east, north_m=north))
    return ep.reports["log"][-1][2]


_DOG_AT = (30.0, 10.0)                       # where the stood animal is, east and north
_DOG_NAMED = (_DOG_AT[0] - _DOCK[0], _DOG_AT[1] - _DOCK[1])


class _Still:
    """A track that stays on one spot."""

    def place(self, t: float) -> tuple:
        """East, north and heading at any time."""
        return _DOG_AT[0], _DOG_AT[1], 1.0, 0.0


class _Trotting:
    """A track that heads east at 2 m/s and passes the still animal's spot when the test's frame is taken."""

    def place(self, t: float) -> tuple:
        """East, north and heading at a time."""
        return _DOG_AT[0] + 2.0 * (t - _FRAME_STEP * SIM_DT), _DOG_AT[1], 1.0, 0.0


def _stand(ep, kind: str, body: str, on_field: bool, track=None) -> None:
    """Make the frame's animal body one planned animal of a kind and a body, living on or off the field, still unless
    given a track."""
    ep.animals = animals.Animals([animals.Animal(kind=kind, body=body, coat="", size=1.0, pace=1.0,
                                                 track=track or _Still(), on_field=on_field, uid=_DOG)])


def _beasts(ep) -> tuple:
    """The patrol's reports that named an animal: valid, failed, and on one not asked for."""
    return ep.outcome.valid_animal_reports, ep.outcome.wrong_animal_reports, ep.outcome.unasked_animal_reports


def _apart(ep):
    """The patrol's false alarms counted apart: on animals, on people outside, repeats."""
    return ep.outcome.animal_reports, ep.outcome.outside_reports, ep.outcome.repeat_reports


# ---------------------------------------------------------------------------------------------------- judging

def test_a_walker_is_judged_where_he_stood_in_the_frame_the_report_names(field):
    """He walked 1.5 m east between the frame and the press. A point 1 m west of where the frame showed him is valid;
    the same report judged at the press step's point, 2.5 m away, would be a wrong position."""
    frame_point = people.reference(field, 0, step=_FRAME_STEP)
    assert np.allclose(frame_point, [12.0, 3.0])
    assert float(np.hypot(*(people.reference(field, 0) - frame_point))) == pytest.approx(1.5)
    field.shown = _frame(_PRESS_STEP)
    assert _report(field, _WALKER_PX, 11.0, 3.0) == "wrong_position"
    field.shown = _frame(_FRAME_STEP)
    assert _report(field, _WALKER_PX, 11.0, 3.0) == "valid"
    assert field.outcome.valid_reports == 1 and field.outcome.false_alarms == 1 and _apart(field) == (0, 0, 0)


def test_someone_outside_the_field_is_never_a_person_to_find(field):
    """A box on the man passing outside, with his exact point, is outside_field every time, never valid nor a repeat,
    and each one is a false alarm counted apart."""
    east, north = _OUTSIDER_AT[:2] - _DOCK[:2]
    assert [_report(field, _OUTSIDER_PX, east, north) for _ in range(2)] == ["outside_field", "outside_field"]
    assert field.outcome.valid_reports == 0 and field.outcome.false_alarms == 2 and _apart(field) == (0, 2, 0)


def test_an_animal_named_a_person_is_a_false_alarm(field):
    """A box on the dog in a report that names a person is an animal false alarm, counted apart, whether or not the
    dog is one to report."""
    assert _report(field, _DOG_PX, 30.0, 0.0) == "animal"
    _stand(field, "dog", "husky", on_field=True)
    assert _report(field, _DOG_PX, *_DOG_NAMED) == "animal"
    assert field.outcome.false_alarms == 2 and _apart(field) == (2, 0, 0)
    assert _beasts(field) == (0, 0, 0)


def test_an_animal_to_report_is_valid_once_where_it_stood_and_never_a_false_alarm(field):
    """A goat, a dog or a grown boar living on the field, named an animal with a box on it and its place within 1 m,
    is a valid animal report; a second one is a repeat and a place 3 m off a wrong position, both failed animal
    reports. None of them is a false alarm, and none counts as a report of a person."""
    for kind, body in (("goat", "goat"), ("dog", "husky"), ("boar", "boar")):
        _stand(field, kind, body, on_field=True)
        reports.reset(None, field)
        assert _report(field, _DOG_PX, _DOG_NAMED[0] + 3.0, _DOG_NAMED[1], kind="animal") == "wrong_position"
        assert _report(field, _DOG_PX, _DOG_NAMED[0] + 0.9, _DOG_NAMED[1], kind="animal") == "valid"
        assert _report(field, _DOG_PX, *_DOG_NAMED, kind="animal") == "repeat"
    assert _beasts(field) == (3, 6, 0)
    assert field.outcome.false_alarms == 0 and field.outcome.valid_reports == 0 and _apart(field) == (0, 0, 0)


def test_an_animal_off_the_field_fails_and_one_not_asked_for_is_neither_right_nor_wrong(field):
    """A dog living off the field, named an animal, is outside the field: a failed animal report. A piglet, a bird, a
    rat or a body that is no known animal's is not asked for: the report counts as unasked, neither valid nor
    failed."""
    _stand(field, "dog", "husky", on_field=False)
    assert _report(field, _DOG_PX, *_DOG_NAMED, kind="animal") == "outside_field"
    for kind, body in (("boar", "piglet"), ("bird", "crow"), ("rat", "rat")):
        _stand(field, kind, body, on_field=True)
        assert _report(field, _DOG_PX, *_DOG_NAMED, kind="animal") == "not_asked"
    field.animals = {_DOG: "dog"}
    assert _report(field, _DOG_PX, *_DOG_NAMED, kind="animal") == "not_asked"
    assert _beasts(field) == (0, 1, 4) and field.outcome.false_alarms == 0


def test_a_moving_animal_is_judged_where_it_stood_in_the_frame_the_report_names(field):
    """A dog trots 3 m between the frame and the press. The place it had in the frame is a valid animal report; the
    place it has at the press is a wrong position."""
    _stand(field, "dog", "husky", on_field=True, track=_Trotting())
    assert animals.where(animals.member(field, _DOG), _PRESS_STEP)[0] - _DOG_AT[0] == pytest.approx(3.0)
    assert _report(field, _DOG_PX, _DOG_NAMED[0] + 3.0, _DOG_NAMED[1], kind="animal") == "wrong_position"
    assert _report(field, _DOG_PX, *_DOG_NAMED, kind="animal") == "valid"
    assert _beasts(field) == (1, 1, 0)


def test_a_report_on_an_animal_nobody_asks_for_still_uses_one_of_the_patrols_reports(field):
    """Nineteen reports that name an animal on a bird count as unasked and as reports made; the twentieth press is
    still judged, and the twenty-first does nothing at all."""
    _stand(field, "bird", "crow", on_field=True)
    for _ in range(sentinel_reports.MAX_REPORTS - 1):
        assert _report(field, _DOG_PX, *_DOG_NAMED, kind="animal") == "not_asked"
    assert field.outcome.reports_made == sentinel_reports.MAX_REPORTS - 1 == 19
    assert _report(field, _WALKER_PX, 12.0, 3.0) == "valid"
    reports.submit(None, field, FieldReport("person", "feed", _box(_OUTSIDER_PX), east_m=0.0, north_m=0.0))
    assert len(field.reports["log"]) == field.outcome.reports_made == 20
    assert field.outcome.valid_reports == 1 and field.outcome.false_alarms == 0 and _beasts(field) == (0, 0, 19)


def test_the_class_switch_names_a_person_up_to_the_threshold_and_an_animal_above_it():
    """A pressed report whose class switch is at the button threshold or under it names a person, one just over it an
    animal, and both carry the place the model gave, in metres from the dock."""
    def press(switch: float) -> FieldReport:
        """The report an action with this class switch makes."""
        action = np.zeros(contract.ACTION_DIM, dtype=np.float32)
        for name, value in (("report", 1.0), ("report_class", switch), ("report_east", 0.25), ("report_north", -0.5)):
            action[contract.ACTION_INDEX[name]] = value
        return contract.decode_action(action, None).report

    threshold = np.float32(contract.sentinel.BUTTON_THRESHOLD)
    assert [press(switch).kind for switch in (0.0, float(threshold), float(np.nextafter(threshold, np.float32(1.0))), 1.0)] == \
        ["person", "person", "animal", "animal"]
    assert (press(1.0).east_m, press(1.0).north_m) == (0.25 * contract.REPORT_REACH_M, -0.5 * contract.REPORT_REACH_M)


def test_a_person_named_an_animal_fails_and_is_still_to_report(field):
    """A box on the walker in a report that names an animal is a wrong class: a failed animal report, no false alarm,
    and he can still be reported as the person he is."""
    assert _report(field, _WALKER_PX, 12.0, 3.0, kind="animal") == "wrong_class"
    assert _beasts(field) == (0, 1, 0) and field.outcome.false_alarms == 0
    assert _report(field, _WALKER_PX, 12.0, 3.0) == "valid"
    assert field.outcome.valid_reports == 1


def test_an_animal_report_on_nothing_fails_as_an_animal_report(field):
    """A report that names an animal with a box on empty ground, or on a frame that shows nothing, is a failed animal
    report and never a false alarm."""
    assert _report(field, (10, 10, 40, 40), 0.0, 0.0, kind="animal") == "no_object"
    field.shown = _frame(_FRAME_STEP, sees=False)
    assert _report(field, _DOG_PX, 0.0, 0.0, kind="animal") == "no_frame"
    assert _beasts(field) == (0, 2, 0) and field.outcome.false_alarms == 0


def _metrics(**changed) -> dict:
    """A flawless patrol's metrics on a field of two people, with what a test changes."""
    metrics = {"coverage": 1.0, "threats": 2, "valid_reports": 2, "false_alarms": 0, "landed_in_dock": True,
               "returned_by_model": True, "max_height_m": 9.5, "took_off": True, "min_threat_distance_m": None,
               "end_reason": "landed", "animals": 0, "valid_animal_reports": 0, "wrong_animal_reports": 0}
    metrics.update(changed)
    return metrics


def test_a_seed_scores_half_for_its_people_and_a_fifth_for_its_animals():
    """seed = 0.50 people + 0.20 animals + 0.20 coverage + 0.10 flight. With animals to report their term is the
    valid animal reports over the animals and the failed animal reports; with none it is the people's term over one
    and the failed animal reports, so such a seed scores as it did before animals were reported. A patrol the edge
    rule ended, or one that came too close to someone, scores 0 whatever it reported."""
    assert score.score(None, _metrics())["final_score"] == pytest.approx(1.0)
    half = score.score(None, _metrics(valid_reports=1))
    assert half["final_score"] == pytest.approx(0.70 * 0.5 + 0.20 + 0.10) and half["animal_term"] == pytest.approx(0.5)
    assert score.score(None, _metrics(wrong_animal_reports=1))["final_score"] == pytest.approx(0.50 + 0.20 * 0.5 + 0.30)
    herd = score.score(None, _metrics(animals=2, valid_animal_reports=1))
    assert herd["animal_term"] == pytest.approx(0.5) and herd["final_score"] == pytest.approx(0.50 + 0.10 + 0.30)
    assert score.score(None, _metrics(animals=2, valid_animal_reports=2))["final_score"] == pytest.approx(1.0)
    assert score.score(None, _metrics(animals=2, valid_animal_reports=2, wrong_animal_reports=2))["animal_term"] == pytest.approx(0.5)
    assert score.score(None, _metrics(animals=2))["final_score"] == pytest.approx(0.80)
    missed = score.score(None, _metrics(animals=2, valid_animal_reports=2, valid_reports=1, false_alarms=1))
    assert missed["detection_term"] == pytest.approx(1 / 3) and missed["final_score"] == pytest.approx(0.50 / 3 + 0.20 + 0.30)
    for ended in ({"end_reason": "flight_limit"}, {"end_reason": "collision"}, {"min_threat_distance_m": 4.9}, {"took_off": False}):
        assert score.score(None, _metrics(animals=2, valid_animal_reports=2, **ended))["final_score"] == 0.0


def test_the_registry_names_every_term_the_scorer_gives():
    """The family's score record in the registry lists exactly the terms a scored patrol carries, the animals' among
    them, so nothing that reads the registry shows a score without one of its parts."""
    record = get_score_schema(get_challenge_family_definition(FAMILY_ID)["score_schema_id"])
    assert {metric["metric_id"] for metric in record["metrics"]} == set(score.score(None, _metrics()))
    assert record["primary_metric"] == "final_score" and "animal" in record["description"]


def test_a_box_that_overlaps_an_animal_more_than_a_person_is_on_the_animal(field):
    """The dog stands over most of the walker: a box on the dog also overlaps the walker by more than half, which a
    judge that only knew people would have counted on him. A dog is a candidate too,
    so the box lands on the dog, the one judged."""
    dog_px = (305, 205, 345, 260)
    field.shown = _frame(_FRAME_STEP, drawn=((_WALKER_PX, _WALKER), (dog_px, _DOG)))
    boxes_of_people = sentinel_reports.object_boxes(field.shown, reports.owners(field))
    assert sentinel_reports.best_match(field.shown, _box(dog_px), boxes_of_people) == ("person", 0)
    assert _report(field, dog_px, 12.0, 3.0) == "animal"
    assert field.outcome.valid_reports == 0 and _apart(field) == (1, 0, 0)


def test_a_second_report_of_a_walker_after_he_moved_is_a_repeat(field):
    """Reported once, he walks 3 m on; a box on him in a later frame with his new point is a repeat, counted apart."""
    assert _report(field, _WALKER_PX, 12.0, 3.0) == "valid"
    later = _FRAME_STEP + 150
    field.step, field.shown = later, _frame(later)
    east, north = people.reference(field, 0, step=later)
    assert east == pytest.approx(15.0)
    assert _report(field, _WALKER_PX, east, north) == "repeat"
    assert field.outcome.valid_reports == 1 and field.outcome.false_alarms == 1 and _apart(field) == (0, 0, 1)


def test_a_second_report_of_someone_with_a_point_off_him_is_still_a_repeat(field):
    """Reported once, a second box on him that names a point more than the tolerance away is a repeat, not a wrong
    position: who is already reported is checked before where."""
    assert _report(field, _WALKER_PX, 12.0, 3.0) == "valid"
    assert _report(field, _WALKER_PX, 12.0 + 2.0 * reports.POSITION_M, 3.0) == "repeat"
    assert field.outcome.valid_reports == 1 and field.outcome.false_alarms == 1 and _apart(field) == (0, 0, 1)


def test_a_report_on_a_frame_that_shows_nothing_is_no_frame(field):
    """A colour frame in the dark with night mode off still carries an object map, but the model saw nothing in it:
    a report on it is no_frame, a plain false alarm."""
    field.shown = _frame(_FRAME_STEP, sees=False)
    assert _report(field, _WALKER_PX, 12.0, 3.0) == "no_frame"
    assert field.outcome.false_alarms == 1 and _apart(field) == (0, 0, 0)


def test_the_verdicts_come_in_the_planned_order():
    """The judge's checks, first to last; every verdict but valid is a false alarm."""
    assert reports.VERDICTS == ("valid", "no_frame", "too_high", "no_object", "animal", "wrong_class", "outside_field",
                                "repeat", "wrong_position", "not_asked")
    assert set(reports.COUNTED_APART) == {"animal", "outside_field", "repeat"}


# ---------------------------------------------------------------------------------------------------- the 5 m rule

def test_bodies_whose_boxes_coincide_judge_a_report_in_its_favour(field):
    """A dog and the walker fill the same box in the frame, so a box cannot tell them apart: a report that names the
    walker's own place is valid whichever was drawn last, and one that names a wrong place fails as a report on him."""
    for drawn in (((_WALKER_PX, _WALKER), (_WALKER_PX, _DOG)), ((_WALKER_PX, _DOG), (_WALKER_PX, _WALKER))):
        left, top, right, bottom = _WALKER_PX
        objects = np.full((480, 640), -1, dtype=np.int32)
        # Opposite corners of one box: each body's tight box is the whole rectangle.
        first, second = drawn[0][1], drawn[1][1]
        objects[top:bottom, left:right] = first
        objects[top, left] = objects[bottom - 1, right - 1] = second
        objects[top, right - 1] = objects[bottom - 1, left] = first
        field.shown = camera.View("colour", (0.0, 0.0, 9.5), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0), 640, 480, 50.0, True,
                                  _FRAME_STEP, objects=objects)
        reports.reset(None, field)
        boxes = sentinel_reports.object_boxes(field.shown, reports.candidates(field))
        assert boxes[("person", 0)] == boxes[("animal", _DOG)]
        assert reports.matches(field.shown, _box(_WALKER_PX), boxes) == [("person", 0), ("animal", _DOG)]
        east, north = _walker_at(_FRAME_STEP)[:2] - _DOCK[:2]
        assert _report(field, _WALKER_PX, east + 9.0, north) == "wrong_position"
        assert _report(field, _WALKER_PX, east, north) == "valid"
        assert _report(field, _WALKER_PX, east, north) == "repeat"


def _tied(field) -> None:
    """Show a frame in which the walker and the animal body fill one and the same box."""
    left, top, right, bottom = _WALKER_PX
    objects = np.full((480, 640), -1, dtype=np.int32)
    objects[top:bottom, left:right] = _WALKER
    objects[top, left] = objects[bottom - 1, right - 1] = _DOG
    field.shown = camera.View("colour", (0.0, 0.0, 9.5), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0), 640, 480, 50.0, True,
                              _FRAME_STEP, objects=objects)


def test_a_report_that_names_an_animal_on_coinciding_boxes_is_judged_on_the_animal(field):
    """The walker and an animal fill the same box. A report that names an animal is judged on the animal, not as a
    person named an animal: a dog on the field is valid at its own place and a wrong position elsewhere, a bird is not
    asked for, and only with no known animal under the box at all is the report a wrong class. The walker can still
    be reported as the person he is."""
    _tied(field)
    _stand(field, "dog", "husky", on_field=True)
    assert _report(field, _WALKER_PX, _DOG_NAMED[0] + 9.0, _DOG_NAMED[1], kind="animal") == "wrong_position"
    assert _report(field, _WALKER_PX, *_DOG_NAMED, kind="animal") == "valid"
    _stand(field, "bird", "crow", on_field=True)
    assert _report(field, _WALKER_PX, *_DOG_NAMED, kind="animal") == "not_asked"
    _stand(field, "dog", "husky", on_field=False)
    assert _report(field, _WALKER_PX, *_DOG_NAMED, kind="animal") == "outside_field"
    assert _beasts(field) == (1, 2, 1) and field.outcome.false_alarms == 0
    east, north = _walker_at(_FRAME_STEP)[:2] - _DOCK[:2]
    assert _report(field, _WALKER_PX, east, north) == "valid"


def test_two_animals_on_coinciding_boxes_judge_an_animal_report_in_its_favour(field, monkeypatch):
    """A dog off the field and a bird fill one box: a report that names an animal is not asked for, the kinder of
    the two verdicts, whichever body comes first."""
    left, top, right, bottom = _DOG_PX
    bird = _DOG + 1
    for first, second in ((_DOG, bird), (bird, _DOG)):
        objects = np.full((480, 640), -1, dtype=np.int32)
        objects[top:bottom, left:right] = first
        objects[top, left] = objects[bottom - 1, right - 1] = second
        field.shown = camera.View("colour", (0.0, 0.0, 9.5), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0), 640, 480, 50.0, True,
                                  _FRAME_STEP, objects=objects)
        field.animals = animals.Animals([
            animals.Animal(kind="dog", body="husky", coat="", size=1.0, pace=1.0, track=_Still(), on_field=False, uid=_DOG),
            animals.Animal(kind="bird", body="crow", coat="", size=1.0, pace=1.0, track=_Still(), on_field=True, uid=bird)])
        assert _report(field, _DOG_PX, *_DOG_NAMED, kind="animal") == "not_asked"
    assert _beasts(field) == (0, 0, 2)


def test_a_tied_person_report_credits_the_person_at_the_point_named_only_once(field, monkeypatch):
    """Two people whose boxes coincide: a report credits the one standing at the point it names, the same report
    again is a repeat that credits nobody, and a report at the other's point credits the other."""
    field.people = [people.Person(index=index, pose="standing", xy=(17.0, -2.0), ground_z=2.0, yaw=0.0)
                    for index in (0, 1)]
    points = {0: np.array([1.5, 0.0]), 1: np.array([0.0, 0.0])}
    monkeypatch.setattr(people, "reference", lambda episode, index, step=None: points[index])
    monkeypatch.setattr(sentinel_reports, "frame_boxes",
                        lambda *args: {("person", 0): _WALKER_PX, ("person", 1): _WALKER_PX})
    assert _report(field, _WALKER_PX, 0.0, 0.0) == "valid"
    assert field.reports["reported"] == {1}
    assert _report(field, _WALKER_PX, 0.0, 0.0) == "repeat"
    assert field.reports["reported"] == {1} and field.outcome.false_alarms == 1
    assert _report(field, _WALKER_PX, 1.5, 0.0) == "valid"
    assert field.reports["reported"] == {0, 1} and field.outcome.valid_reports == 2


@pytest.mark.parametrize("order", [(11, 12, 13), (13, 12, 11)])
def test_a_tied_animal_report_credits_the_nearest_goat_only_once(field, monkeypatch, order):
    """Two equal-box goats and a bird give credit only to the goat at the reported point."""
    points = {11: (1.5, 0.0), 12: (0.0, 0.0), 13: (0.0, 0.0)}
    field.animals = animals.Animals([
        animals.Animal(kind="bird" if uid == 13 else "goat", body="", coat="", size=1.0, pace=1.0,
                       track=SimpleNamespace(), on_field=True, uid=uid)
        for uid in order
    ])
    monkeypatch.setattr(animals, "where", lambda animal, step: _DOCK[:2] + points[animal.uid])
    boxes = {("animal", uid): _DOG_PX for uid in order}
    monkeypatch.setattr(sentinel_reports, "frame_boxes", lambda *args: boxes)
    assert _report(field, _DOG_PX, 0.0, 0.0, kind="animal") == "valid"
    assert field.reports["animals"] == {12}
    assert _report(field, _DOG_PX, 0.0, 0.0, kind="animal") == "repeat"
    assert field.reports["animals"] == {12} and _beasts(field) == (1, 1, 0)
    assert _report(field, _DOG_PX, 9.0, 0.0, kind="animal") == "not_asked"
    reports.reset(None, field)
    assert _report(field, _DOG_PX, 0.75, 0.0, kind="animal") == "valid"
    assert field.reports["animals"] == {11}


def test_the_five_metre_rule_measures_to_a_crouching_persons_own_head(field, monkeypatch):
    """Someone crouching reaches 1.1 m, not the 1.85 m he stands: a drone 5.5 m over his head's real top keeps the
    rule, where his standing height would have broken it."""
    monkeypatch.setattr(score.sentinel_score, "update", lambda env, episode: None)
    field.people[0].pose, field.people[0].top_m = "crouching", 1.1
    assert field.people[0].head_m == 1.1 and field.people[1].head_m == field.people[1].height_m
    field.step = _FRAME_STEP
    east, north, ground = _walker_at(_FRAME_STEP)
    score.update(_drone_at(east, north, ground + 1.1 + 4.5), field)
    assert field.outcome.min_threat_distance_m == pytest.approx(4.5)
    field.outcome.min_threat_distance_m = None
    score.update(_drone_at(east, north, ground + 1.1 + 5.5), field)
    assert field.outcome.min_threat_distance_m == pytest.approx(5.5)


def _drone_at(x, y, z):
    """An environment whose drone stands at a point."""
    return SimpleNamespace(pos=np.array([[x, y, z]]))


def test_the_five_metre_rule_follows_a_walker_under_the_drone(field, monkeypatch):
    """At the frame step the walker is 8 m away; 400 steps later he has walked under the drone hovering 6 m above
    the ground under him. The distance is measured to where he is now and to the top of his own 1.85 m, and it zeroes
    the seed, though the spot he started from stays more than 5 m away."""
    monkeypatch.setattr(score.sentinel_score, "update", lambda env, episode: None)
    drone = _drone_at(25.0, -1.0, 8.0)
    field.step = _FRAME_STEP
    score.update(drone, field)
    assert field.outcome.min_threat_distance_m is None
    field.step = _FRAME_STEP + 400
    assert np.allclose(people.position(field, 0)[:2], [25.0, -2.0])
    assert math.hypot(25.0 - 17.0, -1.0 + 2.0) > 5.0
    score.update(drone, field)
    assert field.outcome.min_threat_distance_m == pytest.approx(math.hypot(1.0, 8.0 - 2.0 - 1.85))
    metrics = dict(asdict(FieldOutcome(took_off=True, coverage=1.0)), min_threat_distance_m=field.outcome.min_threat_distance_m)
    assert score.score(None, metrics)["final_score"] == 0.0


def test_the_five_metre_rule_holds_for_someone_outside_the_field(field, monkeypatch):
    """The man passing outside is protected like everyone else, though he is not a person to find."""
    monkeypatch.setattr(score.sentinel_score, "update", lambda env, episode: None)
    score.update(_drone_at(60.0, 33.0, 6.0), field)
    assert field.outcome.min_threat_distance_m == pytest.approx(math.hypot(3.0, 6.0 - 2.0 - 1.8))


# ---------------------------------------------------------------------------------------------------- score

def _empty_seed(**changes):
    """A seed with nobody on the field: 90 % searched, landed after the model's own return, flown at 9 m."""
    metrics = dict(asdict(FieldOutcome(end_reason="landed", took_off=True, landed_in_dock=True, returned_by_model=True,
                                       threats=0, coverage=0.9, max_height_m=9.0)))
    metrics.update(changes)
    return metrics


@pytest.mark.parametrize("false_alarms, detection", [(0, 0.9), (1, 0.45), (2, 0.3)])
def test_a_seed_with_nobody_on_the_field_scores_by_the_same_formula(false_alarms, detection):
    """With nobody to find, detection is the share searched over 1 + false alarms, as before; a report on an animal
    or on someone outside costs exactly what any false alarm costs, and the counts kept apart change nothing."""
    plain = score.score(None, _empty_seed(false_alarms=false_alarms))
    apart = score.score(None, _empty_seed(false_alarms=false_alarms, animal_reports=false_alarms, outside_reports=0))
    assert plain == apart
    assert plain["detection_term"] == pytest.approx(detection)
    assert plain["final_score"] == pytest.approx(0.70 * detection + 0.20 * 0.9 + 0.10 * 1.0)


def test_the_counts_kept_apart_reach_the_seeds_result():
    """A patrol's animal, outside and repeat counts are raw metrics of its result, beside the false alarms."""
    assert {"animal_reports", "outside_reports", "repeat_reports"} <= set(RESULT_METRICS)
    outcome = asdict(FieldOutcome(end_reason="landed", took_off=True, threats=1, valid_reports=1, false_alarms=3,
                                  animal_reports=1, outside_reports=1, repeat_reports=1))
    evaluation = get_challenge_family(FAMILY_ID).evaluate_rollout(
        task=None, success=True, t=120.0, horizon=HORIZON_S, min_clearance=None, collision=False,
        failure_reason="NONE", info={"field_outcome": outcome})
    assert set(RESULT_METRICS) <= set(evaluation.metrics)
    assert (evaluation.metrics["animal_reports"], evaluation.metrics["outside_reports"],
            evaluation.metrics["repeat_reports"]) == (1, 1, 1)
