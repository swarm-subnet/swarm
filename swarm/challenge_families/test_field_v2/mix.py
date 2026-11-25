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

"""What a seed puts on and around the field, beside the place itself: six switches, each drawn on its own.

People to find stand on the field in most seeds. Animals are near in under half of them, and only a few at a time. People walk past
outside the field in one seed in four. Birds come down on or round the field in about three seeds in ten. Wild boars
and rats or mice are rare by day and come to one night in five: both are abroad mostly after dark. The switches
are independent, so a seed can hold nobody and a herd, or people and someone passing, or nothing at all, and no switch
tells a model anything about another.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MIX_SEED_STREAM = 0x31C5               # the mix's own stream, so no other draw of the seed moves it
PEOPLE_SHARE = 0.95                    # seeds with at least one person to find on the field
ANIMALS_SHARE = 0.45                   # seeds with animals on or near the field
OUTSIDERS_SHARE = 0.25                 # seeds with people walking past outside the field
BIRDS_SHARE = 0.30                     # seeds with birds on or round the field by day
BOARS_SHARE = (0.05, 0.20)             # seeds with wild boars near the field, by day and at night
RODENTS_SHARE = (0.06, 0.20)           # seeds with rats or mice near the field, by day and at night


@dataclass(frozen=True)
class Mix:
    """Which of the six a seed holds."""

    people: bool
    animals: bool
    outsiders: bool
    birds: bool
    boars: bool = False
    rodents: bool = False


def present(seed: int, night: bool = False) -> Mix:
    """The seed's mix, each switch from its own draw, so a switch added later moves none before it. The boars' and the
    rodents' shares follow the seed's light, which the caller that asks about them gives."""
    draws = [float(np.random.default_rng([MIX_SEED_STREAM, int(seed), n]).random()) for n in range(6)]
    return Mix(people=draws[0] < PEOPLE_SHARE, animals=draws[1] < ANIMALS_SHARE, outsiders=draws[2] < OUTSIDERS_SHARE,
               birds=draws[3] < BIRDS_SHARE, boars=draws[4] < BOARS_SHARE[bool(night)],
               rodents=draws[5] < RODENTS_SHARE[bool(night)])
