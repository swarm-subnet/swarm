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

"""Theft scenarios (task 19): the thieves one seed writes, the stage they are at, and how they move.

The number of threats the patrol can report goes into the outcome, so the score knows what was there to find.

Stand-in: no seed has thieves.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .episode import SolarEpisode


def reset(env: Any, ep: SolarEpisode) -> None:
    """Stage this seed's theft, if it has one."""
    ep.theft = []
    ep.outcome.threats = 0


def advance(env: Any, ep: SolarEpisode) -> None:
    """Move the thieves and their vehicles to the step physics is about to run."""


def show(env: Any, ep: SolarEpisode, view: Any) -> None:
    """Pose the thieves a picture of the view can see, as they are at the moment it is taken."""


def people(ep: SolarEpisode, step: Optional[int] = None) -> List[Dict[str, Any]]:
    """Every thief in the scene at a step: his body ids, where he stands, what he is doing, and whether he is a
    threat, which is a man inside the fence."""
    return []


def bodies(ep: SolarEpisode) -> Dict[int, int]:
    """Every drawn body of every thief, mapped to the thief's index in the crew."""
    return {}
