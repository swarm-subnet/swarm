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

"""The test field built for a seed, and the light and air it is flown in.

The field is the company's real drone test field in Santa Coloma, Andorra, so it stands the same in every seed: the
map builder places every piece where the survey puts it. Each seed flies it by day under its own sun and sets the
air and the clear sky to the temperatures the thermal camera reads.

The site publishes the field's surveyed outline in the episode's fence slot, where Sentinel keeps its fence, and the
terrain and the ploughed field as the ground.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import numpy as np
import pybullet as p
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

from swarm.core.maps.testsite import builder

from ..solar_patrol import park as sentinel_park
from .episode import FieldEpisode

SEEDED_SUN = True
NIGHT_SHARE = 0.0                      # the real field is flown by day only
SKY_FROM_SUN = sentinel_park.SKY_FROM_SUN
DAYLIGHT = sentinel_park.DAYLIGHT
FLOOR_DEPTH_M = -1000.0                # the environment's flat floor goes here, under the valley's lowest ground

HEAT_SEED_STREAM = 0x7E57              # the field's own stream, so its temperatures never move another part's draws
# Andorra la Vella, Roc de Sant Pere at 1,075 m (1971 to 2000): mean daily highs from 6.9 C in January to 26.2 C in July.
AIR_DAY_C = (5.0, 26.0)
SKY_BELOW_AIR_C = sentinel_park.SKY_BELOW_AIR_C   # a clear mountain sky reads at least as cold as Sentinel's


def asset_dir() -> str:
    """The folder the field's assets are read from."""
    return builder.TESTSITE_ASSET_DIR


def survey(folder: str) -> Dict[str, Any]:
    """The manifest's site block: the field's outline and origin, and what else the survey lists around it."""
    return builder.testsite_manifest(folder)["site"]


def outline(folder: str) -> np.ndarray:
    """The field's surveyed edge, metres east and north in the map frame, anticlockwise."""
    ring = Polygon(np.asarray(survey(folder)["field_outline"], dtype=float))
    return np.array(orient(ring, 1.0).exterior.coords[:-1])


def heat(seed: int) -> Dict[str, float]:
    """The seed's air and clear-sky temperatures, degrees Celsius, by day."""
    rng = np.random.default_rng([HEAT_SEED_STREAM, int(seed)])
    air = float(rng.uniform(*AIR_DAY_C))
    sky = air - float(rng.uniform(*SKY_BELOW_AIR_C))
    return {"air_c": round(air, 1), "sky_c": round(sky, 1)}


def reset(env: Any, ep: FieldEpisode) -> None:
    """Build the field, sink the environment's floor under it, and publish its outline, its ground and its
    temperatures."""
    folder = asset_dir()
    world = builder.build_testsite_map(seed=ep.seed, cli=env.CLIENT, asset_dir=folder)
    floor = getattr(env, "PLANE_ID", None)
    if floor is not None:
        p.resetBasePositionAndOrientation(floor, [0.0, 0.0, FLOOR_DEPTH_M], [0.0, 0.0, 0.0, 1.0],
                                          physicsClientId=env.CLIENT)
    ep.park = {"world": world, **heat(ep.seed)}
    ep.fence = outline(folder)
    ep.terrain_uids = frozenset(int(uid) for uid in world["land"])


def advance(env: Any, ep: FieldEpisode) -> None:
    """Nothing on the field moves by itself."""


def moving_bodies(ep: FieldEpisode) -> frozenset:
    """Every body the field moves itself: none."""
    return frozenset()


def passable_outlines(ep: FieldEpisode) -> np.ndarray:
    """East, north and radius, world metres, of each tree crown near the field the survey lists, which the simulator
    lets a drone pass through and a real one would hit; none when the survey lists none."""
    return np.asarray(survey(ep.park["world"]["asset_dir"]).get("passable", ()), dtype=float).reshape(-1, 3)


def buildings(ep: FieldEpisode) -> np.ndarray:
    """Centre east, centre north, length, width and heading in degrees of each house the survey lists around the field,
    world metres; none when the survey lists none."""
    return np.asarray(survey(ep.park["world"]["asset_dir"]).get("buildings", ()), dtype=float).reshape(-1, 5)


def origin_lat_lon(ep: FieldEpisode) -> tuple:
    """Latitude and longitude, degrees, of the map frame's origin."""
    lat, lon = survey(ep.park["world"]["asset_dir"])["origin_lat_lon"]
    return float(lat), float(lon)


def has_assets(folder: str = "") -> bool:
    """Whether the field's assets are installed where the builder reads them."""
    folder = folder or asset_dir()
    return bool(folder) and os.path.isfile(os.path.join(folder, "manifest.json"))
