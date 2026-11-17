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

"""Wind per seed: Sentinel's wind, its readings and the air the aircraft feels, in its daytime shares only.

The field is flown by day only, so every seed draws its strength, and the dock's gauge its share of the 20 m wind,
from Sentinel's daytime figures, whatever a night share would have made of the seed.
"""

from __future__ import annotations

from typing import Any

from ..solar_patrol import wind as sentinel_wind
from ..solar_patrol.episode import SolarEpisode

update = sentinel_wind.update
felt = sentinel_wind.felt
observe = sentinel_wind.observe


def strength(seed: int) -> str:
    """The seed's wind strength: none, light or strong, in the daytime shares."""
    return sentinel_wind.strength(seed, night=False)


def for_seed(seed: int) -> dict[str, Any]:
    """The task's wind fields for a seed, the ones the environment's seeded wind reads."""
    return sentinel_wind.for_seed(seed, night=False)


def reset(env: Any, ep: SolarEpisode) -> None:
    """Draw this seed's reading errors and push clocks, the dock's gauge on its daytime share."""
    sentinel_wind.reset(env, ep, night=False)
