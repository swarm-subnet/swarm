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

from ._shared import *
from .mountains_only import _build_mountains_only
from .terrain import _cache, get_global_scale
from .village import _build_ski_village


def get_mountain_subtype(seed: int) -> int:
    subtype_rng = random.Random(seed + 666666)
    subtypes = list(MOUNTAIN_SUBTYPE_DISTRIBUTION.keys())
    weights = list(MOUNTAIN_SUBTYPE_DISTRIBUTION.values())
    return subtype_rng.choices(subtypes, weights=weights, k=1)[0]


def build_mountains(
    cli: int,
    seed: int,
    safe_zones: List[Tuple[float, float]],
    safe_zone_radius: float,
    *,
    forced_subtype: Optional[int] = None,
) -> Tuple[Callable, List, float]:
    _cache.clear()

    gs = get_global_scale(seed)
    if forced_subtype in (1, 2):
        chosen = int(forced_subtype)
    else:
        chosen = get_mountain_subtype(seed)

    if chosen == 1:
        get_z, peaks = _build_mountains_only(
            cli, seed, gs, safe_zones, safe_zone_radius
        )
    else:
        get_z, peaks = _build_ski_village(cli, seed, gs, safe_zones, safe_zone_radius)

    return get_z, peaks, gs
