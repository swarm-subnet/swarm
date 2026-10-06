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

"""Reports and the object-map check (task 21): every report judged against the simulator's truth for its frame.

A report names a class and draws a box on the feed or zoom frame the model was shown. That frame's view carries the
object map from its own draw, which body each pixel shows, so the check reads exactly what the model saw even when
the thieves have moved since. The frame must have been taken no higher than MAX_REPORT_HEIGHT_M above the dock; then
four checks, in order: the box sits on a real object, the class is right, the object is a threat (a thief inside the
fence at that frame's step, never a decoy), and it is the first report of that thief.

The box sits on the object it overlaps most, when the overlap with the tight box around that object's visible pixels
(intersection over union) is at least MIN_OVERLAP. A report that passes adds to valid_reports; one that fails, a
repeat included, adds to false_alarms. A report never ends the patrol.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from . import decoys, sensor_noise, theft
from .camera import View
from .contract import Box, Report
from .episode import SolarEpisode

MIN_OVERLAP = 0.5
MAX_REPORT_HEIGHT_M = 24.0              # over the dock: a frame taken from higher carries no valid report
BODY_BITS = (1 << 24) - 1               # an object map code keeps the body id in its low 24 bits, the link above
THREAT_CLASS = "person"
VERDICTS = ("valid", "no_frame", "too_high", "no_object", "wrong_class", "not_threat", "repeat")


def reset(env: Any, ep: SolarEpisode) -> None:
    """No report made yet and no thief reported."""
    ep.reports = {"log": [], "reported": set()}


def submit(env: Any, ep: SolarEpisode, report: Report) -> None:
    """Judge one report the moment its button is pressed, and count it as valid or as a false alarm."""
    verdict = judge(ep, report)
    ep.reports["log"].append((ep.step, report, verdict))
    ep.outcome.reports_made += 1
    if verdict == "valid":
        ep.outcome.valid_reports += 1
    else:
        ep.outcome.false_alarms += 1


def judge(ep: SolarEpisode, report: Report) -> str:
    """The verdict on one report: valid, or the first check it fails. A valid report marks its thief as reported."""
    view = sensor_noise.shown_view(ep, report.image)
    if view is None or view.objects is None:
        return "no_frame"
    if view.eye[2] - float(ep.dock_position[2]) > MAX_REPORT_HEIGHT_M:
        return "too_high"
    owners: dict[int, tuple] = {uid: ("thief", thief) for uid, thief in theft.bodies(ep).items()}
    kinds: dict[tuple, str] = {key: THREAT_CLASS for key in owners.values()}
    for uid, kind in decoys.bodies(ep).items():
        owners[uid] = ("decoy", uid)
        kinds[("decoy", uid)] = kind
    key = boxed_object(view, report.box, owners)
    if key is None:
        return "no_object"
    if report.kind != THREAT_CLASS or kinds[key] != THREAT_CLASS:
        return "wrong_class"
    _, thief = key
    if not theft.inside(ep, thief, view.step):
        return "not_threat"
    if thief in ep.reports["reported"]:
        return "repeat"
    ep.reports["reported"].add(thief)
    return "valid"


def boxed_object(view: View, box: Box, owners: dict[int, tuple]) -> Optional[tuple]:
    """The object a box sits on in a frame: the one whose visible pixels' tight box it overlaps most, if that
    overlap reaches MIN_OVERLAP. owners maps each body id to the object it belongs to."""
    if not owners:
        return None
    ids = np.where(view.objects >= 0, view.objects & BODY_BITS, -1)
    ys, xs = np.nonzero(np.isin(ids, np.fromiter(owners, dtype=np.int64), kind="table"))
    uids = ids[ys, xs]
    left, right = (box.cx - box.w / 2.0) * view.width, (box.cx + box.w / 2.0) * view.width
    top, bottom = (box.cy - box.h / 2.0) * view.height, (box.cy + box.h / 2.0) * view.height
    best, best_overlap = None, MIN_OVERLAP
    for key in dict.fromkeys(owners.values()):
        mine = np.isin(uids, [uid for uid, owner in owners.items() if owner == key])
        if not mine.any():
            continue
        x0, x1 = float(xs[mine].min()), float(xs[mine].max()) + 1.0
        y0, y1 = float(ys[mine].min()), float(ys[mine].max()) + 1.0
        overlap = _overlap((left, top, right, bottom), (x0, y0, x1, y1))
        if overlap >= best_overlap:
            best, best_overlap = key, overlap
    return best


def _overlap(a: tuple, b: tuple) -> float:
    """Intersection over union of two (left, top, right, bottom) boxes in pixels."""
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    shared = width * height
    union = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - shared
    return shared / union if union > 0.0 else 0.0


def update(env: Any, ep: SolarEpisode) -> None:
    """Keep the count of threats still unreported."""
    ep.outcome.missed_threats = max(0, ep.outcome.threats - ep.outcome.valid_reports)
