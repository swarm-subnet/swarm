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
clear sky and, by day, the panel glass to the temperatures the thermal camera reads. The farmers' truck reads warm
once it has driven (task 62): bonnet, grille and tyres hot while it drives, cooling towards the air once it parks.

The park also publishes the fence line, which never moves: the flight limit is drawn from it, and the site map
reads the survey rather than this seed's shifts.
"""

from __future__ import annotations

import math
import os
import tempfile
from functools import lru_cache
from typing import Any, Dict, List, Tuple

import numpy as np
import pybullet as p
from PIL import Image

from swarm.core.daylight import SunLight, max_elevation_deg
from swarm.core.maps.solar import farmers
from swarm.core.maps.solar.builder import build_solar_map, build_solar_movers, solar_fence, solar_manifest

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

TRUCK_HEAT_STREAM = 0x7AC4             # the truck's own stream, so its heat never moves another part's draws
TRUCK_HOT_C = (31.0, 50.0)             # bonnet and tyres of a vehicle that just drove (task 2)
# Towards the air once the engine is off, bonnet then tyres: a parked car's engine bay and tyres come back close to
# the air in 30 to 60 minutes.
TRUCK_COOL_S = (900.0, 1200.0)
TRUCK_EMISSIVITY = (0.9, 0.95)         # painted steel, rubber
# The truck body's own frame, metres, x forward and z up: the bonnet lies ahead of the windscreen, the grille between
# the bumper and the bonnet, and the heat centres on the engine under the bonnet, fading to its edges.
BONNET_FROM_X_M = 0.8
BONNET_FROM_Z_M = 0.55
GRILLE_FROM_X_M = 2.2
GRILLE_FROM_Z_M = 0.38
ENGINE_XY_M = (1.8, 0.0)
ENGINE_SPREAD_M = (0.55, 0.5)
BONNET_EDGE_SHARE = 0.6                # of the engine's heat at the bonnet's edges
GRILLE_SHARE = 0.8                     # the radiator behind the grille
HEAT_MAP_PX = 1024


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


def truck_heat(seed: int, drives: int) -> List[Tuple[float, float]]:
    """How hot each of the seed's drives leaves the truck's bonnet and tyres, degrees Celsius, the tyres never hotter
    than the bonnet."""
    rng = np.random.default_rng([TRUCK_HEAT_STREAM, int(seed)])
    hot = []
    for _ in range(drives):
        bonnet = float(rng.uniform(*TRUCK_HOT_C))
        hot.append((round(bonnet, 1), round(float(rng.uniform(TRUCK_HOT_C[0], bonnet)), 1)))
    return hot


@lru_cache(maxsize=2)
def truck_heat_map(obj_path: str) -> np.ndarray:
    """The share of the engine's heat at each texel of the truck body's texture, as the byte the engine reads: the
    bonnet strongest over the engine and fading to its edges, the grille steady, the rest of the body none.

    Each bonnet and grille triangle paints every texel it touches in its own texture space with the share of the point
    on the body there, so even a sliver thinner than a texel never reads as the cold body. Only element by element
    arithmetic is used, which rounds the same on every machine, so every validator loads the same map.
    """
    positions, uvs, faces = [], [], []
    with open(obj_path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("v "):
                positions.append([float(value) for value in line.split()[1:4]])
            elif line.startswith("vt "):
                uvs.append([float(value) for value in line.split()[1:3]])
            elif line.startswith("f "):
                corners = [token.split("/") for token in line.split()[1:]]
                if len(corners) != 3:
                    raise ValueError(f"truck body {obj_path} holds a face that is not a triangle")
                faces.append([[int(corner[0]) - 1, int(corner[1]) - 1] for corner in corners])
    index = np.asarray(faces)
    corner = np.asarray(positions)[index[:, :, 0]]
    texel = np.asarray(uvs)[index[:, :, 1]] * HEAT_MAP_PX
    texel[:, :, 1] = HEAT_MAP_PX - texel[:, :, 1]
    centre = (corner[:, 0] + corner[:, 1] + corner[:, 2]) / 3.0
    normal = np.cross(corner[:, 1] - corner[:, 0], corner[:, 2] - corner[:, 0])
    normal /= np.maximum(np.sqrt(normal[:, 0] * normal[:, 0] + normal[:, 1] * normal[:, 1] + normal[:, 2] * normal[:, 2]),
                         1e-12)[:, None]
    bonnet = (centre[:, 0] > BONNET_FROM_X_M) & (centre[:, 2] > BONNET_FROM_Z_M) & (normal[:, 2] > 0.5)
    grille = (centre[:, 0] > GRILLE_FROM_X_M) & (centre[:, 2] > GRILLE_FROM_Z_M) & (normal[:, 0] > 0.5) & ~bonnet
    heat = np.zeros((HEAT_MAP_PX, HEAT_MAP_PX), dtype=np.uint8)
    for face in np.flatnonzero(bonnet | grille):
        weights = _texel_samples(texel[face])
        if grille[face]:
            share = np.full(len(weights), GRILLE_SHARE)
        else:
            point = _blend(weights, corner[face])
            east = (point[:, 0] - ENGINE_XY_M[0]) / ENGINE_SPREAD_M[0]
            north = (point[:, 1] - ENGINE_XY_M[1]) / ENGINE_SPREAD_M[1]
            fall = 1.0 + east * east + north * north
            share = BONNET_EDGE_SHARE + (1.0 - BONNET_EDGE_SHARE) / (fall * fall)
        # The engine reads the texel under a point with its texture coordinates wrapped, so the same is done here.
        spot = np.floor(_blend(weights, texel[face])).astype(int) % HEAT_MAP_PX
        np.maximum.at(heat, (spot[:, 1], spot[:, 0]), np.round(share * 255.0).astype(np.uint8))
    heat.flags.writeable = False
    return heat


def _texel_samples(corners: np.ndarray) -> np.ndarray:
    """Barycentric weights of points that reach every texel a triangle in texture space touches: the centre of each
    texel inside it, and points along its edges no more than half a texel apart."""
    a, b, c = corners
    samples = []
    area = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
    if area != 0.0:
        low, high = np.floor(corners.min(axis=0)).astype(int), np.ceil(corners.max(axis=0)).astype(int)
        x, y = np.meshgrid(np.arange(low[0], high[0]) + 0.5, np.arange(low[1], high[1]) + 0.5)
        wb = ((x - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (y - a[1])) / area
        wc = ((b[0] - a[0]) * (y - a[1]) - (x - a[0]) * (b[1] - a[1])) / area
        inside = (wb >= 0.0) & (wc >= 0.0) & (wb + wc <= 1.0)
        samples.append(np.column_stack([1.0 - wb[inside] - wc[inside], wb[inside], wc[inside]]))
    for start in range(3):
        end = (start + 1) % 3
        along = np.linspace(0.0, 1.0, int(np.ceil(2.0 * np.hypot(*(corners[end] - corners[start])))) + 2)
        edge = np.zeros((len(along), 3))
        edge[:, start], edge[:, end] = 1.0 - along, along
        samples.append(edge)
    return np.vstack(samples)


def _blend(weights: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """The points barycentric weights pick out of a triangle's three corners, summed in a fixed order."""
    return weights[:, :1] * corners[0] + weights[:, 1:2] * corners[1] + weights[:, 2:] * corners[2]


def _load_truck_heat(cli: int, obj_path: str) -> int:
    """The truck body's heat map loaded into a client as a texture."""
    handle, path = tempfile.mkstemp(suffix=".png")
    try:
        os.close(handle)
        Image.fromarray(np.repeat(truck_heat_map(obj_path)[:, :, None], 3, axis=2)).save(path)
        return p.loadTexture(path, physicsClientId=cli)
    finally:
        os.remove(path)


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
    ep.park["truck"] = {"hot_c": truck_heat(ep.seed, farmers.CONFIG["most"]), "shown": None, "heat_map": -1}
    if THERMAL:
        for body, celsius in zip(glass, ep.park["panel_c"]):
            p.changeVisualShape(body, -1, temperature=celsius, physicsClientId=env.CLIENT)
        if "pickup_body" in movers.truck_parts:
            item = solar_manifest(world["asset_dir"])["items"]["pickup_body"]
            obj_path = os.path.join(world["asset_dir"], item["folder"], item["obj"])
            ep.park["truck"]["heat_map"] = _load_truck_heat(env.CLIENT, obj_path)
            warm_truck(env, ep, 0)


def truck_temperatures(ep: SolarEpisode, step: int) -> Tuple[float, float] | None:
    """The truck's bonnet and tyres at a step, degrees Celsius: as hot as its drive left them, cooling towards the air
    once it has parked; None while no truck has driven into view."""
    drive = ep.park["movers"].truck_drive(step)
    if drive is None:
        return None
    index, parked_s = drive
    air = air_c(ep)
    bonnet, tyres = (round(air + (hot - air) * math.exp(-parked_s / cool), 1)
                     for hot, cool in zip(ep.park["truck"]["hot_c"][index], TRUCK_COOL_S))
    return bonnet, tyres


def warm_truck(env: Any, ep: SolarEpisode, step: int) -> None:
    """Give the truck's bonnet, grille and tyres the heat they hold at a step; the rest of the body reads at the air
    and the glass stays with the engine's passive model."""
    truck = ep.park["truck"]
    celsius = truck_temperatures(ep, step) if truck["heat_map"] >= 0 else None
    if celsius is None or celsius == truck["shown"]:
        return
    truck["shown"] = celsius
    bonnet, tyres = celsius
    for item, body in ep.park["movers"].truck_parts.items():
        if item == "pickup_body":
            p.changeVisualShape(body, -1, thermalTextureUniqueId=truck["heat_map"], temperatureRange=[air_c(ep), bonnet],
                                emissivity=TRUCK_EMISSIVITY[0], physicsClientId=env.CLIENT)
        elif item.startswith("pickup_wheel"):
            p.changeVisualShape(body, -1, temperature=tyres, emissivity=TRUCK_EMISSIVITY[1], physicsClientId=env.CLIENT)


def air_c(ep: SolarEpisode) -> float:
    """The air temperature of this seed, degrees Celsius."""
    return ep.park["air_c"]


def sky_c(ep: SolarEpisode) -> float:
    """The clear sky straight up as the thermal camera reads it this seed, degrees Celsius."""
    return ep.park["sky_c"]


def advance(env: Any, ep: SolarEpisode) -> None:
    """Move the park's movers to the step physics is about to run, with the truck as warm as it is then."""
    ep.park["movers"].advance(ep.step + 1)
    warm_truck(env, ep, ep.step + 1)


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
