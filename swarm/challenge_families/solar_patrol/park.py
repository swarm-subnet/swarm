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

"""Park shifts, day and night (task 17): the Manolia solar park built for a seed, and the light it is flown in.

The park also publishes the fence line, which never moves: the flight limit and the site map are drawn from it.

Stand-in: the map exactly as it ships, its movers running with the goats walking stiff, every seed in today's light.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from swarm.core.maps.solar.builder import build_solar_map, build_solar_movers, solar_manifest

from .episode import SolarEpisode

SEEDED_SUN = False
NIGHT_SHARE = 0.0
FENCE_TOLERANCE_M = 1.0                # a post this close to a straight side stays on it


def _straight_sides(points: np.ndarray, tolerance: float) -> np.ndarray:
    """The corners of an open run of points once every post within tolerance of a straight side is dropped."""
    first, last = points[0], points[-1]
    chord = last - first
    length = float(np.hypot(*chord))
    offsets = points - first
    if length > 0.0:
        distances = np.abs(chord[0] * offsets[:, 1] - chord[1] * offsets[:, 0]) / length
    else:
        distances = np.hypot(offsets[:, 0], offsets[:, 1])
    worst = int(np.argmax(distances))
    if len(points) < 3 or distances[worst] <= tolerance:
        return np.vstack([first, last])
    return np.vstack([_straight_sides(points[:worst + 1], tolerance)[:-1], _straight_sides(points[worst:], tolerance)])


@lru_cache(maxsize=2)
def fence_line(asset_dir: str) -> np.ndarray:
    """The fence as the survey draws it, world metres: its posts joined in order and reduced to straight sides."""
    posts = np.array([place["position"][:2] for place in solar_manifest(asset_dir)["placements"]
                      if place["item"] == "fence_post"], dtype=float)
    order, left = [0], set(range(1, len(posts)))
    while left:
        last = posts[order[-1]]
        nearest = min(left, key=lambda k: (float(np.hypot(*(posts[k] - last))), k))
        order.append(nearest)
        left.remove(nearest)
    ring = posts[order]
    far = int(np.argmax(np.hypot(*(ring - ring[0]).T)))
    return np.vstack([_straight_sides(ring[:far + 1], FENCE_TOLERANCE_M)[:-1],
                      _straight_sides(np.vstack([ring[far:], ring[:1]]), FENCE_TOLERANCE_M)[:-1]])


def reset(env: Any, ep: SolarEpisode) -> None:
    """Build the park for the seed, start its movers, and publish the fence and the terrain."""
    world = build_solar_map(seed=ep.seed, cli=env.CLIENT)
    movers = build_solar_movers(world, seed=ep.seed, cli=env.CLIENT)
    # The herd leaves Solar Patrol with the decoys; until then it walks stiff, as the mesh rewrite costs a patrol hours.
    movers.goat_mesh = False
    ep.park = {"world": world, "movers": movers}
    ep.fence = fence_line(world["asset_dir"])
    ep.terrain_uids = frozenset(int(uid) for uid in world["bodies"].get("terrain", ()))


def advance(env: Any, ep: SolarEpisode) -> None:
    """Move the park's movers to the step physics is about to run."""
    ep.park["movers"].advance(ep.step + 1)


def moving_bodies(ep: SolarEpisode) -> frozenset:
    """Every body the park moves itself, kept out of the clearance metric."""
    return ep.park["movers"].body_uids


def asset_dir(ep: SolarEpisode) -> str:
    """The folder this patrol's park was built from, whose manifest is the site's survey."""
    return ep.park["world"]["asset_dir"]


def passable_outlines(ep: SolarEpisode) -> np.ndarray:
    """East, north and radius, world metres, of each standing piece the simulator lets a drone pass through but a
    real one would hit: a tree's crown."""
    return np.asarray(ep.park["world"].get("passable", ()), dtype=float).reshape(-1, 3)
