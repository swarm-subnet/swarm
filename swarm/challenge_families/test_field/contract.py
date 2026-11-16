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

"""The Swarm Test Field contract: what the model receives, what it sends, and what one seed's result carries.

The drone, its cameras, its state and its buttons are Sentinel's, so the observation keys, shapes and the first 24
action values keep Sentinel's meaning and order. Two values follow them: where the reported person stands, east and
north of the dock. The registry entry for the family mirrors what is declared here, and a test holds the two together.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from ..solar_patrol import contract as sentinel
from ..solar_patrol.contract import ActionField, Command, Outcome, Report

FAMILY_ID = "cf_test_field"
CHALLENGE_TYPE = 9
INTERFACE_VERSION = sentinel.INTERFACE_VERSION
DECISION_STEPS = sentinel.DECISION_STEPS

# The field is 2,755 m2, six lanes under the 10 m ceiling; the take-off hands over in about 10 s and the slowest
# measured return and landing took 29 s.
TIME_BUDGET_S = {"take_off": 15.0, "people": 30.0, "sweep": 95.0, "landing": 40.0}
HORIZON_S = sum(TIME_BUDGET_S.values())

# The real field's drone flies no higher than this above its dock, and the dock's take-off ends half a metre under it,
# where the dock's arrival band of 0.5 m ends the climb before the limit's braking curve slows it to a crawl.
CEILING_M = 10.0
TAKE_OFF_M = CEILING_M - 0.5
REPORT_REACH_M = 100.0               # report_east and report_north are shares of this, from the dock
ACTION_FIELDS = sentinel.ACTION_FIELDS + (
    ActionField("report_east", -1.0, 1.0),     # where the reported person stands: share of REPORT_REACH_M east of the dock
    ActionField("report_north", -1.0, 1.0),    # and north of it
)
ACTION_DIM = len(ACTION_FIELDS)
ACTION_INDEX = {f.name: i for i, f in enumerate(ACTION_FIELDS)}
ACTION_LOW = tuple(f.low for f in ACTION_FIELDS)
ACTION_HIGH = tuple(f.high for f in ACTION_FIELDS)


@dataclass(frozen=True)
class FieldReport(Report):
    """A report with where the person stands, metres east and north of the dock."""

    east_m: float = 0.0
    north_m: float = 0.0


def decode_action(action: np.ndarray, previous: Optional[np.ndarray]) -> Command:
    """Read a clipped action vector into a Command as Sentinel does, a report carrying the position the model gave."""
    a = np.asarray(action, dtype=np.float32).reshape(-1)
    before = None if previous is None else np.asarray(previous, dtype=np.float32).reshape(-1)[:sentinel.ACTION_DIM]
    command = sentinel.decode_action(a[:sentinel.ACTION_DIM], before)
    if command.report is None:
        return command
    east = float(a[ACTION_INDEX["report_east"]]) * REPORT_REACH_M
    north = float(a[ACTION_INDEX["report_north"]]) * REPORT_REACH_M
    report = FieldReport(command.report.kind, command.report.image, command.report.box, east_m=east, north_m=north)
    return replace(command, report=report)


@dataclass
class FieldOutcome(Outcome):
    """Sentinel's outcome, where threats counts the people on the field, plus how close the drone came to the field's
    edge and to a tree crown while out of the dock, and whether the edge rule ended the patrol."""

    min_edge_m: Optional[float] = None
    min_crown_m: Optional[float] = None   # how close it came to a tree crown's edge, negative inside one
    edge_trips: int = 0                   # 1 when the patrol ended on the edge rule: a mistake under the ×5 rule


# The raw metrics of one seed's result: the patrol outcome plus the clock.
RESULT_METRICS = tuple(FieldOutcome.__dataclass_fields__) + ("time_sec", "horizon_sec", "success")
