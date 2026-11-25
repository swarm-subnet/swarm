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

"""The field a seed flies, cut from one of the real Andorra areas, and the light and air it is flown in.

Every seed flies a field of its own. The seed's people count comes first (people module), then a field with room for
them, open or reaching into the trees, cut from one of the areas that holds its size for that kind of field (field_cut
and field_sizes modules), then that area is built around the field: drawn within VISUAL_REACH_M of the field's box, solid within SOLID_REACH_M of it, the
horizon whole. An area's survey grids and its tables of tree crowns, houses and keep-out cells are read once a
process; the cutter's view of an area is kept for the last few areas used.

The site publishes the field's outline in the episode's fence slot, where Sentinel keeps its fence, the area's
terrain tiles as the ground, the crown circles that touch the field and the houses round it from the cutter's tables,
and the cells where the dock may stand. Each seed flies it by day under its own sun, in the light the areas were
matched under (the family's area_light), or at night under its moon, and sets the air and the clear sky to the
temperatures the thermal camera reads, the air cooler at night and the higher the field lies. The park also carries
the area's kind of place and where its survey lets someone stand, which the animals and the people passing outside
read, the path cells round the field, and the field's own kind, which says whether the people on it may take cover
under the trees.

One seed in NIGHT_SHARE is flown at night. Its coin is drawn on the field's own stream, never on the sun's draws, so a
seed that stays a day seed keeps the sun, the air and everything else it holds with no night. At night the houses
round the field read a few degrees over the air, as heated rooms do, and its water warmer than the air, as a lake that
kept the day's heat does; by day both are left to the engine, as everything else is.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import pybullet as p

from swarm.core.daylight import own_night
from swarm.core.maps import andorra
from swarm.core.maps.andorra import builder, placements
from swarm.core.maps.andorra.survey import PERSON_KEY, Survey

from ..solar_patrol import park as sentinel_park
from . import field_cut, field_sizes, grass, people
from .episode import FieldEpisode

SEEDED_SUN = True
NIGHT_SHARE = 0.4                      # two seeds in five are flown at night
NIGHT_STREAM = 0x4E17                  # the night coin's own stream, so no day seed's sun moves
SKY_FROM_SUN = sentinel_park.SKY_FROM_SUN
DAYLIGHT = sentinel_park.DAYLIGHT
AREA_LIGHT = True
FLOOR_DEPTH_M = -1000.0                # the environment's flat floor goes here, under every area's lowest ground
VISUAL_REACH_M = 200.0                 # the cameras see this far past the field's box from inside it
SOLID_REACH_M = 30.0                   # the drone, the dock and the people never come this far past the field's box
AREAS_KEPT = 4                         # the cutter's view of an area is a few MB; a worker rarely meets one twice
PATH_KINDS = ("path", "river path")    # the survey's kinds of ground that people walk along
PATHS_REACH_M = 41.0                   # past the field's box: as far out as anyone passing it walks

HEAT_SEED_STREAM = 0x7E57              # the field's own stream, so its temperatures never move another part's draws
NIGHT_HEAT_STREAM = 0x4E18             # the night's houses and water, on a stream of their own
HOUSE_OVER_AIR_C = (2.0, 6.0)          # a lived-in house's walls and roof over the night air
HOUSE_SPREAD_C = 1.0                   # one house against another the same night
WATER_OVER_AIR_C = (2.0, 6.0)          # a lake over the night air
WATER_LOWEST_C = 0.5                   # open water never reads under freezing
# Andorra la Vella, Roc de Sant Pere at 1,075 m (1971 to 2000): mean daily highs from 6.9 C in January to 26.2 C in July.
# Mean daily lows run from -2.5 C in January to 11.4 C in July, daily means up to 18.8 C: a night flight lies between.
GROUND_FILE = "ground.json"            # an area's table of each ground picture's mean brightness
GROUND_NEAR_M = 30.0                   # the ground this far round the field's box sets the brightness it is read against
GROUND_DETAIL_C = 10.0                 # the ground is this much warmer a unit of brightness over that
GROUND_READ_TEXELS = 16.0              # read this wide for its warmth, 0.8 m of a 5 cm ground picture
TREE_SUN_C = 6.0                       # a full sun warms a leaf this much: it sheds heat as open ground cannot
ROCK_READ_TEXELS = 8.0                 # a stone's own picture is read this wide, so its grain is not heat either
# The engine's own numbers for a passive surface, kept where a body is given only one of its own.
ENGINE_DETAIL_C, ENGINE_SUN_C = 25.0, 22.0
AIR_DAY_C = (5.0, 26.0)
AIR_NIGHT_C = (-2.5, 15.0)
STATION_M = 1075.0
LAPSE_C_PER_M = 0.0065                 # the standard atmosphere's fall in air temperature with height
SKY_BELOW_AIR_C = sentinel_park.SKY_BELOW_AIR_C   # a clear mountain sky reads at least as cold as Sentinel's
GROUND_RAYS = 16
RAY_REACH_M = 1000.0


class AreaTables(NamedTuple):
    """What a field is cut against in one area: its survey grids and its crown, house and keep-out tables."""

    survey: Survey
    tables: placements.Tables


def package_root() -> Path:
    """The installed area package."""
    return andorra.andorra_root()


def asset_files(root: Optional[Union[str, Path]] = None) -> List[Path]:
    """Every file the site reads before it builds an area: the catalogue, and the manifest, survey and ground table of
    every area the size table lists as holding some field."""
    root = Path(root if root is not None else package_root())
    tables = (field_sizes.table(), field_sizes.wooded_table())
    names = sorted({name for table in tables for name, holds in table.items() if any(holds)})
    return [root / "areas.json", *(root / name / leaf for name in names for leaf in ("manifest.json", "spots.npz", GROUND_FILE))]


@lru_cache(maxsize=None)
def area_tables(root: str, name: str) -> AreaTables:
    """An area's survey and tables, read once a process; the survey must carry the cells a person may stand on."""
    folder = Path(root) / name
    survey = Survey(folder / "spots.npz")
    if survey.person is None:
        raise ValueError(f"{folder / 'spots.npz'} has no {PERSON_KEY} layer; install an area package that carries it")
    manifest = json.loads((folder / "manifest.json").read_text())
    return AreaTables(survey, placements.tables(folder, manifest, survey.shape, survey.z_shift(manifest["andorra"]["frame"])))


@lru_cache(maxsize=AREAS_KEPT)
def cut_area(root: str, name: str) -> field_cut.Area:
    """The cutter's view of an area: the cells a person may stand on as its open ground."""
    survey, tables = area_tables(root, name)
    return field_cut.Area(name, open_ground=survey.person, ground_cm=survey.ground_cm, keep_out=tables.keep_out,
                          crowns=tables.crowns, houses=tables.houses, dock=survey.dock, solid_top_cm=tables.solid_top_cm)


def draw(seed: int, root: Optional[Union[str, Path]] = None) -> field_cut.Field:
    """The seed's field: the people it asks for first, then a field of the size range with room for them, of the kind
    the seed draws, cut from an area that holds the size drawn for that kind, picked by the areas' weights."""
    root = str(root if root is not None else package_root())
    smallest, largest = field_cut.size_limits()
    room = max(smallest, people.min_area_m2(people.wanted(seed)))
    return field_cut.cut(seed, room, largest, field_sizes.table(), lambda name: cut_area(root, name),
                         field_sizes.wooded_table(), field_sizes.weights())


def around(outline: np.ndarray, reach_m: float) -> Tuple[float, float, float, float]:
    """The box (west, south, east, north) round an outline, widened by reach_m on every side."""
    (west, south), (east, north) = outline.min(axis=0), outline.max(axis=0)
    return float(west - reach_m), float(south - reach_m), float(east + reach_m), float(north + reach_m)


def elevation_m(tables: AreaTables, field: field_cut.Field) -> float:
    """How high the field lies above the sea: the survey's base and the median ground under its dock cells."""
    survey = tables.survey
    rows, cols = survey.cell(field.dock_spots[:, 0], field.dock_spots[:, 1])
    return survey.base_height_m + float(np.median(survey.ground_cm[rows, cols])) / 100.0


@lru_cache(maxsize=None)
def area_kinds(root: str) -> Dict[str, str]:
    """The kind of place of every area the package lists: town, lake, valley and the like."""
    return {row["slug"]: str(row["kind"]) for row in andorra.catalog(root)}


def open_at(survey: Survey) -> Callable[[Any, Any], np.ndarray]:
    """A function of east and north arrays, world metres, that says where the survey lets someone stand; nowhere off
    its grid."""
    rows_total, cols_total = survey.shape

    def standable(x: Any, y: Any) -> np.ndarray:
        """Whether each point lies on a cell of the person layer."""
        rows, cols = survey.cell(x, y)
        on_grid = (rows >= 0) & (rows < rows_total) & (cols >= 0) & (cols < cols_total)
        return on_grid & survey.person[np.clip(rows, 0, rows_total - 1), np.clip(cols, 0, cols_total - 1)]

    return standable


def paths_near(survey: Survey, outline: np.ndarray, reach_m: float = PATHS_REACH_M) -> np.ndarray:
    """East and north, world metres, of the centre of every path cell of the survey within reach_m of an outline's
    box; none when the survey names no such kind of ground."""
    names = survey.metadata.get("kinds") or ()
    wanted = [index for index, name in enumerate(names) if name in PATH_KINDS]
    if not wanted:
        return np.zeros((0, 2))
    west, south, east, north = around(outline, reach_m)
    rows, cols = np.nonzero(np.isin(survey.kind, wanted))
    x, y = survey.centre(rows, cols)
    near = (x >= west) & (x <= east) & (y >= south) & (y <= north)
    return np.column_stack([x[near], y[near]])


def night(seed: int) -> bool:
    """Whether the seed is flown at night, under its moon."""
    return own_night(seed, NIGHT_SHARE, NIGHT_STREAM)


def heat(seed: int, elevation: float = STATION_M) -> Dict[str, float]:
    """The seed's air and clear-sky temperatures, degrees Celsius, by day or at night as the seed is, for a field at a
    height above the sea."""
    rng = np.random.default_rng([HEAT_SEED_STREAM, int(seed)])
    air = float(rng.uniform(*(AIR_NIGHT_C if night(seed) else AIR_DAY_C))) - LAPSE_C_PER_M * (float(elevation) - STATION_M)
    sky = air - float(rng.uniform(*SKY_BELOW_AIR_C))
    return {"air_c": round(air, 1), "sky_c": round(sky, 1)}


def night_heat(seed: int, air_c: float, houses: int) -> Dict[str, Any]:
    """A night's water temperature and each house's, degrees Celsius, from the night air."""
    rng = np.random.default_rng([NIGHT_HEAT_STREAM, int(seed)])
    water = max(WATER_LOWEST_C, air_c + float(rng.uniform(*WATER_OVER_AIR_C)))
    rise = float(rng.uniform(*HOUSE_OVER_AIR_C))
    house = [round(air_c + rise + float(rng.uniform(-HOUSE_SPREAD_C, HOUSE_SPREAD_C)), 1) for _ in range(houses)]
    return {"water_c": round(water, 1), "house_c": house}


def warm_at_night(env: Any, ep: FieldEpisode, world: builder.LoadedArea) -> None:
    """On a night seed, set the thermal camera's temperature on every house and every water body of the area."""
    if not night(ep.seed) or not sentinel_park.THERMAL:
        return
    groups = world.groups or {}
    houses, water = sorted(groups.get("buildings", ())), sorted(groups.get("water", ()))
    ep.park.update(night_heat(ep.seed, ep.park["air_c"], len(houses)))
    for uid, celsius in zip(houses, ep.park["house_c"]):
        p.changeVisualShape(uid, -1, temperature=celsius, physicsClientId=env.CLIENT)
    for uid in water:
        p.changeVisualShape(uid, -1, temperature=ep.park["water_c"], physicsClientId=env.CLIENT)


@lru_cache(maxsize=AREAS_KEPT)
def ground_table(root: str, name: str) -> Dict[str, float]:
    """An area's mean linear brightness of each ground picture, by the terrain item that wears it."""
    return {item: float(value) for item, value in json.loads((Path(root) / name / GROUND_FILE).read_text())["luminance"].items()}


def ground_brightness(root: Union[str, Path], name: str, manifest: Dict[str, Any], box: Tuple[float, float, float, float]) -> Optional[float]:
    """The mean brightness of the ground pictures whose tiles reach into a box, each tile counted once in the
    manifest's order; None when no tile does."""
    table = ground_table(str(root), name)
    tiles = [place for place in manifest["placements"] if manifest["items"][place["item"]]["group"] == "terrain"]
    near = builder.overlaps(builder.footprints(manifest, tiles), box)
    values = [table[place["item"]] for place, keep in zip(tiles, near) if keep]
    return sum(values) / len(values) if values else None


def calm_ground(env: Any, ep: FieldEpisode, world: builder.LoadedArea, brightness: Optional[float]) -> None:
    """Tell the thermal camera how the area's ground and trees hold heat, on an engine that can be told: every ground
    tile is read against the one brightness of the ground near the field, a little under a metre wide, so neither a
    tile's edge nor the grain and the shadows of its photograph show as heat; a tree's leaves warm far less in the
    sun than bare ground does; and a stone's picture is read wider and weaker too. On an older engine all three are
    drawn by its fixed rule."""
    if brightness is None or not sentinel_park.THERMAL or not hasattr(p, "THERMAL_FIELD_PASSIVE"):
        return
    ep.park["ground_brightness"] = brightness
    for uid in sorted(world.terrain):
        p.changeVisualShape(uid, -1, thermalPassive=[GROUND_DETAIL_C, brightness, GROUND_READ_TEXELS, ENGINE_SUN_C],
                            physicsClientId=env.CLIENT)
    trees, rocks = (world.forests or {}).get("trees"), (world.forests or {}).get("rocks")
    if trees is not None:
        p.changeVisualShape(trees, -1, thermalPassive=[ENGINE_DETAIL_C, -1.0, 0.0, TREE_SUN_C], physicsClientId=env.CLIENT)
    if rocks is not None:
        p.changeVisualShape(rocks, -1, thermalPassive=[GROUND_DETAIL_C, -1.0, ROCK_READ_TEXELS, ENGINE_SUN_C],
                            physicsClientId=env.CLIENT)


def reset(env: Any, ep: FieldEpisode) -> None:
    """Cut the seed's field, build its area round it with the grass in hidden patches near the field, sink the
    environment's floor under it, and publish its outline, its ground, its kind of place, where someone may stand, its
    paths, its temperatures and its own kind; set how its ground and trees hold heat, and warm its houses and water on
    a night."""
    root = package_root()
    field = draw(ep.seed, root)
    world = builder.load_area(root / field.area, env.CLIENT, keep_xy=around(field.outline, VISUAL_REACH_M),
                              collide_xy=around(field.outline, SOLID_REACH_M),
                              tuft_xy=around(field.outline, grass.MARGIN_M), tuft_cell_m=grass.CELL_M)
    floor = getattr(env, "PLANE_ID", None)
    if floor is not None:
        p.resetBasePositionAndOrientation(floor, [0.0, 0.0, FLOOR_DEPTH_M], [0.0, 0.0, 0.0, 1.0],
                                          physicsClientId=env.CLIENT)
    tables = area_tables(str(root), field.area)
    ep.park = {"world": world, "field": field, "kind": area_kinds(str(root))[field.area],
               "open_at": open_at(tables.survey), "paths": paths_near(tables.survey, field.outline),
               "field_kind": field.kind, "grass": grass.lay(env.CLIENT, world.patches),
               **heat(ep.seed, elevation_m(tables, field))}
    ep.fence = np.array(field.outline, dtype=float)
    ep.terrain_uids = world.terrain
    calm_ground(env, ep, world, ground_brightness(root, field.area, world.manifest, around(field.outline, GROUND_NEAR_M)))
    warm_at_night(env, ep, world)


def advance(env: Any, ep: FieldEpisode) -> None:
    """Nothing on the field moves by itself."""


def moving_bodies(ep: FieldEpisode) -> frozenset:
    """Every body the field moves itself: none."""
    return frozenset()


def ground_at(cli: int, terrain_uids: frozenset, x: float, y: float) -> Optional[float]:
    """Terrain height under a point, past anything standing on it; None when no terrain lies there."""
    top = RAY_REACH_M
    for _ in range(GROUND_RAYS):
        hit = p.rayTest([x, y, top], [x, y, -RAY_REACH_M], physicsClientId=cli)[0]
        if hit[0] < 0:
            return None
        if hit[0] in terrain_uids:
            return float(hit[3][2])
        top = float(hit[3][2]) - 0.01
    return None


def field_of(ep: FieldEpisode) -> field_cut.Field:
    """The field this patrol flies."""
    return ep.park["field"]


def wooded(ep: FieldEpisode) -> bool:
    """Whether this patrol's field reaches into the trees; False for a patrol with no field cut."""
    return (ep.park or {}).get("field_kind") == field_cut.WOODED.kind


def passable_outlines(ep: FieldEpisode) -> np.ndarray:
    """East, north and radius, world metres, of every crown circle that reaches the field, merged past 64, which the
    simulator lets a drone pass through and a real one would hit; none when no tree does."""
    return np.asarray(field_of(ep).crowns, dtype=float).reshape(-1, 3)


def trees(ep: FieldEpisode) -> np.ndarray:
    """East, north and radius, world metres, of the crown circle of every tree that reaches the field, never merged
    as the site map's may be; none when no tree does."""
    field = field_of(ep)
    return np.asarray(field.crowns if field.trees is None else field.trees, dtype=float).reshape(-1, 3)


def buildings(ep: FieldEpisode) -> np.ndarray:
    """Centre east, centre north, length, width and heading in degrees of each house round the field, world metres;
    none when no house stands near it."""
    return np.asarray(field_of(ep).houses, dtype=float).reshape(-1, 5)


def dock_spots(ep: FieldEpisode) -> np.ndarray:
    """East and north, world metres, of the centre of every cell of the field where the dock may stand."""
    return np.asarray(field_of(ep).dock_spots, dtype=float).reshape(-1, 2)
