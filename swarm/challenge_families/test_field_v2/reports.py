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

"""Reports: every report judged against the simulator's truth for its frame, and where it says its subject stood.

A report is Sentinel's (solar's reports module) and names a person or an animal. It is checked against the object
map of the frame the model was shown, in this order. The frame must exist and show something: a colour frame in the
dark with night mode off shows nothing. It was taken no higher than MAX_REPORT_HEIGHT_M above the ground under its
camera. The box sits on a drawn body (intersection over union of at least 0.5 with the tight box around its visible
pixels), matched against every body the frame can hold: the people on the field, the people passing outside it and
the animals. Bodies whose boxes coincide in the frame cannot be told apart by a box, so such a report is judged on the
one of the class it names that stands nearest the point it names.

A report that names a person: the body is a person, not an animal, and he is on the field, a person to find. It is
the first valid report of him. Last, the report names where he stood, metres east and north of the dock, and that
point must lie within POSITION_M across the ground of his reference point (people module) at the step the frame was
taken, so a walker is judged where the model saw him. It adds to valid_reports when it passes and to false_alarms
when it fails; a report on an animal, on someone outside the field or on someone already reported is also counted
apart.

A report that names an animal: the body is an animal, of a kind and a body that is asked for (the animals module: a
goat, a dog, a grown boar), it lives on the field, it is the first valid report of it, and the point named lies
within POSITION_M of its middle at the step the frame was taken. It adds to valid_animal_reports when it passes and
to wrong_animal_reports when it fails, never to false_alarms: an animal is a lesser thing to get wrong than a
person. One on an animal that is not asked for (a piglet, a bird, a rat, a mouse) is neither: the model saw an
animal, and counts as unasked.

A report never ends the patrol, and a press past MAX_REPORTS in one patrol does nothing.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..solar_patrol import coverage, sensor_noise
from ..solar_patrol import reports as sentinel_reports
from ..solar_patrol.reports import MAX_REPORT_HEIGHT_M, MAX_REPORTS
from . import animals, people
from .contract import ANIMAL_CLASS, PERSON_CLASS, FieldReport
from .episode import FieldEpisode

POSITION_M = 1.0
VERDICTS = ("valid", "no_frame", "too_high", "no_object", "animal", "wrong_class", "outside_field", "repeat",
            "wrong_position", "not_asked")
# The false alarms a patrol also counts apart, each in its own outcome field.
COUNTED_APART = {"animal": "animal_reports", "outside_field": "outside_reports", "repeat": "repeat_reports"}
UNASKED = "not_asked"                  # the verdict on a report that names an animal no patrol is asked for

update = sentinel_reports.update


def reset(env: Any, ep: FieldEpisode) -> None:
    """Empty the report log and mark nobody and no animal as reported."""
    sentinel_reports.reset(env, ep)
    ep.reports["animals"] = set()


def submit(env: Any, ep: FieldEpisode, report: FieldReport) -> None:
    """Judge one report the moment its button is pressed, and count it as valid or as a false alarm; once the
    patrol has made MAX_REPORTS, a press is not judged and counts nowhere."""
    if ep.outcome.reports_made >= MAX_REPORTS:
        return
    verdict = judge(env, ep, report)
    ep.reports["log"].append((ep.step, report, verdict))
    ep.outcome.reports_made += 1
    if report.kind == ANIMAL_CLASS:
        name = {"valid": "valid_animal_reports", UNASKED: "unasked_animal_reports"}.get(verdict, "wrong_animal_reports")
        setattr(ep.outcome, name, getattr(ep.outcome, name) + 1)
        return
    if verdict == "valid":
        ep.outcome.valid_reports += 1
        return
    ep.outcome.false_alarms += 1
    if verdict in COUNTED_APART:
        name = COUNTED_APART[verdict]
        setattr(ep.outcome, name, getattr(ep.outcome, name) + 1)


def owners(ep: FieldEpisode) -> dict:
    """Each drawn body of a person, on the field or outside it, mapped to the object it belongs to: ("person", his
    index)."""
    return {uid: ("person", index) for uid, index in people.bodies(ep).items()}


def candidates(ep: FieldEpisode) -> dict:
    """Each drawn body a report's box can sit on, mapped to the object it belongs to: ("person", his index) for every
    person, and ("animal", the body id) for every animal."""
    found = {uid: ("animal", uid) for uid in animals.bodies(ep)}
    found.update(owners(ep))
    return found


def judge(env: Any, ep: FieldEpisode, report: FieldReport) -> str:
    """The verdict on one report: valid, or the first check it fails. A valid report marks its person, or its
    animal, as reported; among bodies whose boxes coincide it is the one nearest the point named, and with none in
    range a report that names an animal is not asked for when one of them is an animal nobody asks for."""
    view = sensor_noise.shown_view(ep, report.image)
    if view is None or view.objects is None or not view.sees:
        return "no_frame"
    # Written so a frame with no terrain under it, a NaN height, is too high.
    if not coverage.height_above_ground(env, ep, view.eye) <= MAX_REPORT_HEIGHT_M:
        return "too_high"
    keys = matches(view, report.box, sentinel_reports.frame_boxes(ep, report.image, view, candidates(ep)))
    if not keys:
        return "no_object"
    # Bodies whose boxes coincide cannot be told apart by a box: the point the report names says which one it means.
    if report.kind == ANIMAL_CLASS:
        keys = sorted(keys, key=lambda key: key[0] != "animal")
    checked = [(_candidate(ep, report, view, key), key) for key in keys]
    in_range = [(distance, index) for (distance, _verdict), (_kind, index) in checked if distance <= POSITION_M]
    if in_range:
        _distance, index = min(in_range)
        reported = ep.reports["animals" if report.kind == ANIMAL_CLASS else "reported"]
        if index in reported:
            return "repeat"
        reported.add(index)
        return "valid"
    if any(verdict == UNASKED for (_distance, verdict), _key in checked):
        return UNASKED
    return checked[0][0][1]


def matches(view: Any, box: Any, boxes: dict) -> list:
    """Every object a report's box overlaps as much as the one it overlaps most, when that overlap reaches the minimum
    IoU (0.5): one object, or several whose boxes coincide; people before animals, each kind in its own order."""
    left, right = (box.cx - box.w / 2.0) * view.width, (box.cx + box.w / 2.0) * view.width
    top, bottom = (box.cy - box.h / 2.0) * view.height, (box.cy + box.h / 2.0) * view.height
    overlaps = {key: sentinel_reports._overlap((left, top, right, bottom), pixels) for key, pixels in boxes.items()}
    best = max(overlaps.values(), default=0.0)
    if best < sentinel_reports.MIN_OVERLAP:
        return []
    return sorted((key for key, overlap in overlaps.items() if overlap == best), key=lambda key: (key[0] != "person", key[1]))


def _candidate(ep: FieldEpisode, report: FieldReport, view: Any, key: tuple) -> tuple[float, str]:
    """How far the point a report names lies from one body its box sits on, and the verdict when that body cannot
    be credited; a body the report cannot be about at all (the other class, outside the field, not asked for) is
    infinitely far."""
    what, index = key
    unavailable = float("inf")
    if what == "animal":
        if report.kind == PERSON_CLASS:
            return unavailable, "animal"
        animal = animals.member(ep, index)
        if animal is None or not animal.asked:
            return unavailable, UNASKED
        if not animal.on_field:
            return unavailable, "outside_field"
        stood = np.asarray(animals.where(animal, view.step)) - ep.dock_position[:2]
    else:
        if report.kind != PERSON_CLASS:
            return unavailable, "wrong_class"
        if not people.people(ep)[index].inside:
            return unavailable, "outside_field"
        stood = people.reference(ep, index, step=view.step)
    distance = float(np.hypot(report.east_m - stood[0], report.north_m - stood[1]))
    again = index in ep.reports["animals" if what == "animal" else "reported"]
    return distance, "repeat" if again else "wrong_position"
