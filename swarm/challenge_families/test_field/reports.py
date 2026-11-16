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

"""Reports: every report judged against the simulator's truth for its frame, and where it says the person stands.

A report is Sentinel's (solar's reports module), checked the same way against the object map of the frame the model
was shown: the frame was taken no higher than MAX_REPORT_HEIGHT_M above the ground under its camera, the box sits on a
real person (intersection over union of at least 0.5 with the tight box around his visible pixels), the class
is a person, and it is the first valid report of him. On the field it also names where he stands, metres east and
north of the dock, and that point must lie within POSITION_M across the ground of his reference point (people
module). A report that passes adds to valid_reports; one that fails, a repeat or a wrong position included, adds to
false_alarms. A report never ends the patrol, and a press past MAX_REPORTS in one patrol does nothing.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..solar_patrol import coverage, sensor_noise
from ..solar_patrol import reports as sentinel_reports
from ..solar_patrol.reports import MAX_REPORT_HEIGHT_M, MAX_REPORTS
from . import people
from .contract import FieldReport
from .episode import FieldEpisode

POSITION_M = 2.0
PERSON_CLASS = "person"
VERDICTS = ("valid", "no_frame", "too_high", "no_object", "wrong_class", "repeat", "wrong_position")

reset = sentinel_reports.reset
update = sentinel_reports.update


def submit(env: Any, ep: FieldEpisode, report: FieldReport) -> None:
    """Judge one report the moment its button is pressed, and count it as valid or as a false alarm; once the
    patrol has made MAX_REPORTS, a press is not judged and counts nowhere."""
    if ep.outcome.reports_made >= MAX_REPORTS:
        return
    verdict = judge(env, ep, report)
    ep.reports["log"].append((ep.step, report, verdict))
    ep.outcome.reports_made += 1
    if verdict == "valid":
        ep.outcome.valid_reports += 1
    else:
        ep.outcome.false_alarms += 1


def owners(ep: FieldEpisode) -> dict:
    """Each drawn body of a person, mapped to the object it belongs to: ("person", his index)."""
    return {uid: ("person", index) for uid, index in people.bodies(ep).items()}


def judge(env: Any, ep: FieldEpisode, report: FieldReport) -> str:
    """The verdict on one report: valid, or the first check it fails. A valid report marks its person as reported."""
    view = sensor_noise.shown_view(ep, report.image)
    if view is None or view.objects is None:
        return "no_frame"
    # Written so a frame with no terrain under it, a NaN height, is too high.
    if not coverage.height_above_ground(env, ep, view.eye) <= MAX_REPORT_HEIGHT_M:
        return "too_high"
    key = sentinel_reports.best_match(view, report.box, sentinel_reports.frame_boxes(ep, report.image, view, owners(ep)))
    if key is None:
        return "no_object"
    if report.kind != PERSON_CLASS:
        return "wrong_class"
    _, index = key
    if index in ep.reports["reported"]:
        return "repeat"
    named = np.array([report.east_m, report.north_m], dtype=float)
    if not float(np.hypot(*(named - people.reference(ep, index)))) <= POSITION_M:
        return "wrong_position"
    ep.reports["reported"].add(index)
    return "valid"
