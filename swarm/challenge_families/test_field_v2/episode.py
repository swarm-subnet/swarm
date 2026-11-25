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

"""The state one Swarm Test Field v2 patrol keeps between steps: Sentinel's episode, with the field's outcome,
its people and its animals."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from swarm.core.maps.andorra.rendering import FAR_PLANE_M

from ..solar_patrol.episode import SolarEpisode
from .contract import CEILING_M, TAKE_OFF_M, FieldOutcome


@dataclass
class FieldEpisode(SolarEpisode):
    """One patrol of the field. The shared fields and the parts' slots are Sentinel's: park holds the field's world, the
    field itself and its temperatures, fence the field's outline, and theft stays empty, so Sentinel's cameras pose no
    thief. The drone stops rising CEILING_M above the dock, the dock's take-off and far return fly TAKE_OFF_M above it,
    and the cameras draw as far as the mountains round the area. The colour camera keeps the frame as drawn: the field's
    white balance table was measured under another light than the areas are matched under."""

    outcome: FieldOutcome = field(default_factory=FieldOutcome)
    people: Any = None
    animals: Any = None
    crowns: Any = None
    height_limit_m: float = CEILING_M
    patrol_height_m: float = TAKE_OFF_M
    white_balance: Optional[tuple] = None
    far_m: Optional[float] = FAR_PLANE_M
