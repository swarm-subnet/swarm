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

A report that passes adds to valid_reports; one that fails, a repeat included, adds to false_alarms. A report
never ends the patrol.

Stand-in: no thief stands in the park yet, so every report is a false alarm.
"""

from __future__ import annotations

from typing import Any

from .contract import Report
from .episode import SolarEpisode


def reset(env: Any, ep: SolarEpisode) -> None:
    """No report made yet."""
    ep.reports = []


def submit(env: Any, ep: SolarEpisode, report: Report) -> None:
    """Judge one report the moment its button is pressed."""
    ep.reports.append((ep.step, report))
    ep.outcome.reports_made += 1
    ep.outcome.false_alarms += 1


def update(env: Any, ep: SolarEpisode) -> None:
    """Keep the count of threats still unreported."""
    ep.outcome.missed_threats = max(0, ep.outcome.threats - ep.outcome.valid_reports)
