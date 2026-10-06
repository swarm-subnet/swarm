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

"""Swarm Sentinel cheat checks: actions and reports a model might send to get something for nothing.

A broken action must be flown as an action at rest, never crash the patrol and never press a button, and a box drawn
over the whole frame must never pass for a report of the thief somewhere inside it. The drone stops rising at the
dock's height limit, and a frame taken from higher than reports may be made from never carries a valid one.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from swarm.challenge_families.solar_patrol import reports
from swarm.challenge_families.solar_patrol.contract import ACTION_DIM, HEIGHT_LIMIT_M, STATE_SLICES, Box, Report
from validator.tests.test_solar_patrol_family import (  # noqa: F401
    _action,
    _patrol,
    blank_camera,
    flat_park,
)
from validator.tests.test_solar_reports import _report, park  # noqa: F401

_CLIMBED = 160                      # decisions: the dock's take-off is done well inside this
_BROKEN = 20                        # broken decisions sent once the model flies
_CLIMBING = 300                     # decisions of full climb asked for once the model flies
_LIMIT_SLACK_M = 0.5


@pytest.mark.parametrize("broken", [
    np.full(ACTION_DIM, np.nan, dtype=np.float32),
    np.full(ACTION_DIM, np.inf, dtype=np.float32),
    np.full(ACTION_DIM, -np.inf, dtype=np.float32),
    np.zeros(ACTION_DIM - 1, dtype=np.float32),
    np.ones(ACTION_DIM + 1, dtype=np.float32),
    np.zeros(0, dtype=np.float32),
], ids=["nan", "inf", "minus_inf", "short", "long", "empty"])
@pytest.mark.usefixtures("blank_camera")
def test_a_broken_action_is_flown_as_rest(flat_park, broken):  # noqa: F811
    """NaN, infinity or a vector of the wrong length holds the drone where it is, presses nothing, and the patrol
    goes on."""
    def pilot(i, _obs):
        """Take off, then send the broken action, every button value in it included."""
        if i < _CLIMBED:
            return _action(take_off=1.0 if i == 0 else 0.0)
        return broken

    log = _patrol(0, pilot, max_decisions=_CLIMBED + _BROKEN)
    ep = log["episode"]
    assert log["decisions"] == _CLIMBED + _BROKEN
    assert ep.phase == "flying" and not ep.outcome.end_reason
    assert ep.outcome.reports_made == 0 and ep.outcome.zooms_used == 0
    assert np.abs(log["observations"][-1]["state"][STATE_SLICES["velocity_mps"]]).max() < 0.5


@pytest.mark.usefixtures("blank_camera")
def test_the_drone_stops_rising_at_the_height_limit(flat_park):  # noqa: F811
    """Full climb held for 30 s takes the drone up to the dock's height limit and no further, and it flies on."""
    def pilot(i, _obs):
        """Take off, then ask for the top climb on every decision."""
        if i < _CLIMBED:
            return _action(take_off=1.0 if i == 0 else 0.0)
        return _action(move_up=1.0)

    log = _patrol(0, pilot, max_decisions=_CLIMBED + _CLIMBING)
    heights = [float(obs["state"][STATE_SLICES["height_above_takeoff_m"]][0]) for obs in log["observations"]]
    assert HEIGHT_LIMIT_M - _LIMIT_SLACK_M < max(heights) <= HEIGHT_LIMIT_M + _LIMIT_SLACK_M
    assert log["episode"].phase == "flying" and not log["episode"].outcome.end_reason


@pytest.mark.parametrize("eye_m, verdict", [(reports.MAX_REPORT_HEIGHT_M, "valid"),
                                            (reports.MAX_REPORT_HEIGHT_M + 1.0, "too_high")])
def test_a_report_from_too_high_is_a_false_alarm(park, eye_m, verdict):  # noqa: F811
    """The same box on the same thief counts when the frame was taken at the report height limit, and is a false
    alarm when it was taken from higher."""
    park.shown["feed"] = replace(park.view, eye=(0.0, 0.0, eye_m))
    assert _report(park, park.inside) == verdict
    assert park.ep.outcome.false_alarms == (verdict != "valid")


def test_a_box_over_the_whole_frame_is_a_false_alarm(park):  # noqa: F811
    """A thief inside the fence somewhere under a box drawn over the whole frame is not boxed: the report fails."""
    reports.submit(None, park.ep, Report("person", "feed", Box(0.5, 0.5, 1.0, 1.0)))
    assert park.ep.reports["log"][-1][2] == "no_object"
    assert park.ep.outcome.false_alarms == 1 and park.ep.outcome.valid_reports == 0
