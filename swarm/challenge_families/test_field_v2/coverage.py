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

"""Coverage of the field: Sentinel's cells and its seeing rule, over the field less the ground under the tree crowns.

The cells are Sentinel's (solar's coverage module): 2 m cells inside the outline, each searched once a frame shows it
from 20 m or lower above the ground under the camera, sharply enough. A cell under a crown circle is dropped: no
camera above the crown can see the ground there.
"""

from __future__ import annotations

from typing import Any

from ..solar_patrol import coverage as sentinel_coverage
from . import crowns, site
from .episode import FieldEpisode

update = sentinel_coverage.update
height_above_ground = sentinel_coverage.height_above_ground


def reset(env: Any, ep: FieldEpisode) -> None:
    """Lay Sentinel's grid over the field and drop every cell under a crown."""
    sentinel_coverage.reset(env, ep)
    keep = crowns.outside(site.passable_outlines(ep), ep.coverage["cells"][:, :2])
    for key in ("cells", "normals", "searched"):
        ep.coverage[key] = ep.coverage[key][keep]
