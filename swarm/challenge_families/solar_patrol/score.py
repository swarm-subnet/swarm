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

"""Per-seed score (task 23): one score per patrol, from detection, coverage and flight.

update() keeps the flight facts the score needs while the patrol runs; score() turns a finished patrol's metrics
into the terms named in SCORE_TERMS.

Stand-in: the height record is kept, and every term is 0.
"""

from __future__ import annotations

from typing import Any

from .contract import SCORE_TERMS
from .episode import SolarEpisode


def reset(env: Any, ep: SolarEpisode) -> None:
    """Nothing recorded yet."""
    ep.score = None
    ep.outcome.max_height_m = 0.0


def update(env: Any, ep: SolarEpisode) -> None:
    """Keep the highest the drone has flown above its take-off point."""
    height = float(env.pos[0][2]) - float(ep.dock_position[2])
    ep.outcome.max_height_m = max(ep.outcome.max_height_m, height)


def score(task: Any, metrics: dict[str, Any]) -> dict[str, float]:
    """The normalised terms of one patrol, final_score among them."""
    return {term: 0.0 for term in SCORE_TERMS}
