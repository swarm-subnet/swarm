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

"""The dock on the field: Sentinel's Dock 3, its take-off, return home and landing, on a spot each seed draws.

Everything the dock does is Sentinel's (solar's dock module). Only where it stands is the field's: a spot on the field
MARGIN_M inside its outline, clear of everything standing and of every tree crown the survey lists by DJI's
installation distance, on ground no steeper than MAX_SLOPE_DEG, where the concrete base stands it upright.
"""

from __future__ import annotations

from typing import Any

from ..solar_patrol import dock as sentinel_dock
from . import site
from .episode import FieldEpisode

# The edge rule trips at 1 m; 4 m more keeps a take-off or landing pushed off its line by a gust well clear of it.
MARGIN_M = 5.0
# The field slopes about 6 degrees (9 % east, 6 % south); steeper ground is a bank or a bump, not the field.
MAX_SLOPE_DEG = 10.0

NoDockGround = sentinel_dock.NoDockGround
command = sentinel_dock.command
autopilot = sentinel_dock.autopilot
update = sentinel_dock.update
closing = sentinel_dock.closing
observe = sentinel_dock.observe
LIDS_OPEN = sentinel_dock.LIDS_OPEN


def reset(env: Any, ep: FieldEpisode) -> None:
    """Stand the dock on this seed's spot on the field and set the drone down on its pad."""
    x, y = sentinel_dock.find_site(env.CLIENT, ep.seed, ep.fence, ep.terrain_uids, site.passable_outlines(ep),
                                   margin_m=MARGIN_M, max_slope_deg=MAX_SLOPE_DEG)
    sentinel_dock.stand(env, ep, x, y)
