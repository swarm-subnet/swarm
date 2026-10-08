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

"""Park shifts, day and night (task 17): the solar park built for a seed, and the light it is flown in.

Each seed stands the real site with every row of tables, building and tree shifted a little (the map builder draws
the shifts), flies it by day or by night, half and half, under the seed's own sun or moon, and sets the air, the
clear sky and, by day, the panel glass to the temperatures the thermal camera reads.

The park also publishes the fence line, which never moves: the flight limit is drawn from it, and the site map
reads the survey rather than this seed's shifts.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Dict, List

import numpy as np
import pybullet as p

from swarm.core.daylight import SunLight, max_elevation_deg
from swarm.core.maps.solar.builder import build_solar_map, build_solar_movers, solar_fence

from .episode import SolarEpisode

SEEDED_SUN = True
NIGHT_SHARE = 0.5
# The seed's sky is drawn from its sun, and colour frames use the daylight model, on an engine that has both.
SKY_FROM_SUN = hasattr(p, "ER_SWARM_SKY_SUN")
DAYLIGHT = SKY_FROM_SUN and hasattr(p, "ER_SWARM_RAYCAST") and hasattr(p, "ER_SWARM_DAYLIGHT")
THERMAL = hasattr(p, "ER_SWARM_THERMAL")
FENCE_TOLERANCE_M = 1.0                # a post this close to a straight side stays on it
FLOOR_DEPTH_M = -1000.0                # the environment's flat floor goes here, under the map's lowest valley

HEAT_SEED_STREAM = 0x4EA7              # the park's own stream, so its temperatures never move another part's draws
AIR_DAY_C = (15.0, 31.0)               # the air on a clear day at the site's latitude, spring to late summer
AIR_NIGHT_C = (8.0, 22.0)              # the air on a clear night in the same months
SKY_BELOW_AIR_C = (30.0, 45.0)         # how much colder the clear sky straight up reads than the air
PANEL_RISE_C = 32.0                    # panel glass over the air under the noon sun: 45 to 65 C with the spread (task 2)
PANEL_SPREAD_C = 2.0                   # one table's glass against another's under the same sun


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
    ring = solar_fence(asset_dir)
    far = int(np.argmax(np.hypot(*(ring - ring[0]).T)))
    return np.vstack([_straight_sides(ring[:far + 1], FENCE_TOLERANCE_M)[:-1],
                      _straight_sides(np.vstack([ring[far:], ring[:1]]), FENCE_TOLERANCE_M)[:-1]])


def heat(sun: SunLight | None, seed: int, panels: int) -> Dict[str, Any]:
    """The seed's air and clear-sky temperatures, and each panel's glass under a day sun, warmer the higher it stands.

    At night the glass is left to the engine, which already puts it a few degrees under the air as the cold sky it
    mirrors does.
    """
    rng = np.random.default_rng([HEAT_SEED_STREAM, int(seed)])
    day = sun is not None and not sun.night
    air = float(rng.uniform(*(AIR_DAY_C if day or sun is None else AIR_NIGHT_C)))
    sky = air - float(rng.uniform(*SKY_BELOW_AIR_C))
    panel_c: List[float] = []
    if day:
        share = math.sin(math.radians(sun.elevation_deg)) / math.sin(math.radians(max_elevation_deg()))
        panel_c = [round(air + PANEL_RISE_C * share + float(rng.uniform(-PANEL_SPREAD_C, PANEL_SPREAD_C)), 1)
                   for _ in range(panels)]
    return {"air_c": round(air, 1), "sky_c": round(sky, 1), "panel_c": panel_c}


def reset(env: Any, ep: SolarEpisode) -> None:
    """Build the park for the seed, sink the environment's floor under it, start its movers, publish the fence and the
    terrain, and set its temperatures."""
    world = build_solar_map(seed=ep.seed, cli=env.CLIENT)
    # The floor stands at height 0, so every valley that dips under it would show its white tiles.
    floor = getattr(env, "PLANE_ID", None)
    if floor is not None:
        p.resetBasePositionAndOrientation(floor, [0.0, 0.0, FLOOR_DEPTH_M], [0.0, 0.0, 0.0, 1.0],
                                          physicsClientId=env.CLIENT)
    movers = build_solar_movers(world, seed=ep.seed, cli=env.CLIENT)
    glass = world.get("glass", ())
    ep.park = {"world": world, "movers": movers, **heat(getattr(env, "_sun", None), ep.seed, len(glass))}
    ep.fence = fence_line(world["asset_dir"])
    ep.terrain_uids = frozenset(int(uid) for uid in world["bodies"].get("terrain", ()))
    if THERMAL:
        for body, celsius in zip(glass, ep.park["panel_c"]):
            p.changeVisualShape(body, -1, temperature=celsius, physicsClientId=env.CLIENT)


def air_c(ep: SolarEpisode) -> float:
    """The air temperature of this seed, degrees Celsius."""
    return ep.park["air_c"]


def sky_c(ep: SolarEpisode) -> float:
    """The clear sky straight up as the thermal camera reads it this seed, degrees Celsius."""
    return ep.park["sky_c"]


def advance(env: Any, ep: SolarEpisode) -> None:
    """Move the park's movers to the step physics is about to run."""
    ep.park["movers"].advance(ep.step + 1)


def moving_bodies(ep: SolarEpisode) -> frozenset:
    """Every body the park moves itself, kept out of the clearance metric."""
    return ep.park["movers"].body_uids


def mover_kinds(ep: SolarEpisode) -> Dict[int, str]:
    """What each body the park moves is, by its body id: 'pickup' for every part of the truck, 'bird' for the bird's."""
    return {int(body): place["mover"] for place, body in ep.park["world"]["movers"]}


def asset_dir(ep: SolarEpisode) -> str:
    """The folder this patrol's park was built from, whose manifest is the site's survey."""
    return ep.park["world"]["asset_dir"]


def passable_outlines(ep: SolarEpisode) -> np.ndarray:
    """East, north and radius, world metres, of each standing piece the simulator lets a drone pass through but a
    real one would hit: a tree's crown."""
    return np.asarray(ep.park["world"].get("passable", ()), dtype=float).reshape(-1, 3)


def table_footprints(ep: SolarEpisode) -> np.ndarray:
    """The corners seen from above, world metres, of every piece of every panel table where this seed stands it:
    eight per piece."""
    return np.asarray(ep.park["world"].get("tables", ()), dtype=float).reshape(-1, 8, 2)
