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

"""The state one Test Field patrol keeps between steps: Sentinel's episode, with the field's outcome and its people."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..solar_patrol.episode import SolarEpisode
from .contract import CEILING_M, TAKE_OFF_M, FieldOutcome
from .white_balance import WHITE_BALANCE


@dataclass
class FieldEpisode(SolarEpisode):
    """One patrol of the field. The shared fields and the parts' slots are Sentinel's: park holds the field's world and
    its temperatures, fence the field's outline, and theft stays empty, so Sentinel's cameras pose no thief. The drone
    stops rising CEILING_M above the dock, the dock's take-off and far return fly TAKE_OFF_M above it, and the colour
    camera takes the low sun's cast out with the field's white balance."""

    outcome: FieldOutcome = field(default_factory=FieldOutcome)
    people: Any = None
    crowns: Any = None
    height_limit_m: float = CEILING_M
    patrol_height_m: float = TAKE_OFF_M
    white_balance: tuple = WHITE_BALANCE
