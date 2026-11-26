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

"""Test Field v2's site: each seed's field cut from an area, the area built round it, and what the patrol is told
about it; on the stand-in area package, and on a real package when SWARM_ANDORRA_PACKAGE names one."""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pybullet as p
import pytest

from swarm.challenge_families.solar_patrol import camera
from swarm.challenge_families.solar_patrol.contract import MAX_BUILDINGS, MAX_TABLES, SITE_MAP_SLICES
from swarm.challenge_families.test_field_v2 import field_cut, field_sizes, flight_limit, install, people, site
from swarm.challenge_families.test_field_v2 import task as field_task
from swarm.constants import SIM_DT
from swarm.core.maps.andorra.rendering import FAR_PLANE_M
from swarm.core.maps.andorra.survey import PERSON_KEY
from swarm.utils.env_factory import make_env_with_initial_obs
from validator.tests import area_stand_in, field_v2_stand_in
from validator.tests.test_field_v2_family import _SHIPPED

SEED = 9                               # asks for people, and the stand-in holds them
SHIPPED_ROWS = dict(field_sizes.ROWS)  # the size table as shipped, before any test swaps in the stand-in's


def _slice(vector: np.ndarray, name: str) -> np.ndarray:
    """One named part of a site map vector."""
    return vector[SITE_MAP_SLICES[name]]


def _blank(env, shot, *_night, **_options):
    """A black colour frame and an empty object map: no test here reads a picture."""
    return (np.zeros((shot.height, shot.width, 3), dtype=np.float32), np.full((shot.height, shot.width), -1, dtype=np.int32))


def _build(seed: int, *, verify_areas: bool = True):
    """Build the seed's patrol, bypassing archive verification only for a synthetic area."""
    with contextlib.redirect_stdout(io.StringIO()), pytest.MonkeyPatch.context() as patch:
        if not verify_areas:
            patch.setattr(install, "area_problems", lambda: [])
        return make_env_with_initial_obs(field_task.test_field_task(seed=seed, sim_dt=SIM_DT))


@pytest.fixture(scope="module")
def stand_in(tmp_path_factory):
    """The stand-in area as the only area, the people dressed in their everyday colours, and pictures left blank."""
    with pytest.MonkeyPatch.context() as patch:
        folder = tmp_path_factory.mktemp("stand_in")
        root = field_v2_stand_in.use_area(folder, patch)
        field_v2_stand_in.use_people(folder / "people", patch)
        patch.setattr(camera, "colour_frame", _blank)
        yield root
    site.area_tables.cache_clear()
    site.cut_area.cache_clear()


@pytest.fixture(scope="module")
def built(stand_in):
    """The seed's patrol built on the stand-in, with its first observation; closed after the module."""
    if not _SHIPPED:
        pytest.skip("the installed swarm-worlds has no M4TD or no intruder bodies")
    env, obs = _build(SEED, verify_areas=False)
    yield env, obs
    env.close()


def test_the_seeds_people_come_first_and_its_field_has_room_for_them(stand_in):
    """A field holds the people its seed asks for, whatever size it draws, and lies inside the size range."""
    smallest, largest = field_cut.size_limits()
    for seed in range(40):
        field = site.draw(seed)
        wanted = people.wanted(seed)
        assert smallest <= field.area_m2 <= largest
        assert people.capacity(field.area_m2) >= wanted and people.count(seed, field.area_m2) == wanted


def test_a_seed_builds_its_field_on_the_area(built, stand_in):
    """The patrol flies the field the cutter gave its seed: its outline anticlockwise as the fence, the area's terrain
    tiles as the ground, its crowns and houses, under the areas' far plane and with no white balance."""
    env, _obs = built
    ep, field = env._solar, site.draw(SEED)
    assert np.array_equal(ep.fence, field.outline) and people.area_m2(ep.fence) == field.area_m2
    x, y = ep.fence[:, 0], ep.fence[:, 1]
    assert float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)) > 0.0
    assert len(ep.terrain_uids) == len(area_stand_in.tile_centres())
    assert all(p.getBodyInfo(uid, physicsClientId=env.CLIENT) for uid in ep.terrain_uids)
    assert np.array_equal(site.passable_outlines(ep), field.crowns) and np.array_equal(site.buildings(ep), field.houses)
    assert np.array_equal(ep.crowns, field.crowns)
    assert ep.far_m == FAR_PLANE_M and camera.far_plane_m(ep) == FAR_PLANE_M and ep.white_balance is None


def test_the_park_names_the_kind_of_place_and_where_someone_may_stand(built, stand_in):
    """The park carries the area's kind and a function that follows the survey's person layer and is false off it."""
    env, _obs = built
    park = env._solar.park
    survey = site.area_tables(str(stand_in), area_stand_in.SLUG).survey
    assert park["kind"] == "valley"
    rows, cols = np.nonzero(survey.person)
    x, y = survey.centre(rows[:50], cols[:50])
    assert park["open_at"](x, y).all()
    rows, cols = np.nonzero(~survey.person)
    x, y = survey.centre(rows[:50], cols[:50])
    assert len(rows) and not park["open_at"](x, y).any()
    assert not park["open_at"](np.array([1.0e6, -1.0e6]), np.array([0.0, 1.0e6])).any()


def test_the_paths_round_a_field_are_the_surveys_path_cells_near_it():
    """The site lists the centre of every path cell within reach of the field's box, and none when the survey names
    no kinds of ground."""
    kind = np.zeros((40, 40), dtype=np.uint8)
    kind[10, 5:9], kind[30, 30], kind[20, 20] = 3, 10, 1
    centre = lambda row, col: (np.asarray(col) + 0.5 - 20.0, 20.0 - np.asarray(row) - 0.5)  # noqa: E731
    survey = SimpleNamespace(metadata={"kinds": ["excluded", "meadow", "dry grass", "path", "shore", "rock bench",
                                                 "clearing", "forest edge", "park", "square", "river path"]},
                             kind=kind, centre=centre)
    outline = np.array([[-18.0, 6.0], [-10.0, 6.0], [-10.0, 12.0], [-18.0, 12.0]])
    assert site.paths_near(survey, outline, reach_m=5.0).tolist() == [[-14.5, 9.5], [-13.5, 9.5], [-12.5, 9.5], [-11.5, 9.5]]
    assert len(site.paths_near(survey, outline, reach_m=60.0)) == 5
    assert site.paths_near(SimpleNamespace(metadata={}, kind=kind, centre=centre), outline).shape == (0, 2)


def test_the_site_map_holds_what_the_cutter_gave(built):
    """The first observation carries the outline, the crowns nearest the dock and the houses nearest the dock, all in
    metres from the dock."""
    env, obs = built
    ep, vector = env._solar, obs["site_map"]
    dock_xy = ep.dock_position[:2]
    count = int(_slice(vector, "fence_count")[0])
    assert count == len(ep.fence) == int(_slice(vector, "limit_count")[0])
    assert np.allclose(_slice(vector, "fence_xy").reshape(-1, 2)[:count], ep.fence - dock_xy, atol=1e-4)
    crowns = site.passable_outlines(ep)
    rows = int(_slice(vector, "table_count")[0])
    assert rows == min(len(crowns), MAX_TABLES)
    if rows:
        listed = _slice(vector, "tables").reshape(-1, 5)[:rows]
        nearest = crowns[np.argsort(np.hypot(*(crowns[:, :2] - dock_xy).T), kind="stable")][:rows]
        assert np.allclose(listed[:, :2], nearest[:, :2] - dock_xy, atol=1e-4)
        assert np.allclose(listed[:, 2], 2.0 * nearest[:, 2], atol=1e-4)
    houses = site.buildings(ep)
    assert int(_slice(vector, "building_count")[0]) == min(len(houses), MAX_BUILDINGS)


def test_the_dock_stands_on_one_of_the_fields_dock_cells(built):
    """The dock stands on the centre of a dock cell the cutter gave, inside the edge by its margin and off every
    crown."""
    env, _obs = built
    ep = env._solar
    spots = site.dock_spots(ep)
    assert np.min(np.hypot(*(spots - ep.dock_position[:2]).T)) < 1e-9
    assert flight_limit.edge_distance(ep, ep.dock_position[:2]) >= 5.0
    crowns = site.passable_outlines(ep)
    assert np.all(np.hypot(*(crowns[:, :2] - ep.dock_position[:2]).T) >= crowns[:, 2])


def test_the_seeds_people_stand_on_the_field(built):
    """Everyone the seed asks for stands on the field, inside its edge by the people's margin, on the area's ground."""
    env, _obs = built
    ep = env._solar
    persons = people.people(ep)
    assert len(persons) == people.wanted(SEED) == ep.outcome.threats > 0
    for person in persons:
        assert flight_limit.edge_distance(ep, np.asarray(person.xy)) >= people.PERSON_EDGE_M
        ground = site.ground_at(env.CLIENT, ep.terrain_uids, *person.xy)
        assert ground == pytest.approx(person.ground_z, abs=0.5)


def test_the_same_seed_builds_the_same_world_twice(built):
    """A second build of the seed stands the same field, dock, people, crowns and houses, and tells the model the same
    site map."""
    env, obs = built
    again, obs_again = _build(SEED, verify_areas=False)
    try:
        first, second = env._solar, again._solar
        assert np.array_equal(first.fence, second.fence) and np.array_equal(first.dock_position, second.dock_position)
        assert [(q.xy, q.pose, q.size) for q in people.people(first)] == [(q.xy, q.pose, q.size) for q in people.people(second)]
        assert np.array_equal(first.crowns, second.crowns) and np.array_equal(site.buildings(first), site.buildings(second))
        assert first.park["air_c"] == second.park["air_c"] and len(first.terrain_uids) == len(second.terrain_uids)
        assert np.array_equal(obs["site_map"], obs_again["site_map"])
    finally:
        again.close()


def test_the_areas_solid_tops_mark_its_stones_for_the_cutter(stand_in):
    """The cutter reads the top of the stand-in's solid stones, in the survey's centimetres, and nothing for the pebble
    that never collides."""
    survey, tables = site.area_tables(str(stand_in), area_stand_in.SLUG)
    tops = site.cut_area(str(stand_in), area_stand_in.SLUG).solid_top_cm
    assert np.array_equal(tops, tables.solid_top_cm)
    shift = survey.z_shift(area_stand_in.FRAME)
    for x, y in area_stand_in.STONES:
        assert tops[survey.cell(x, y)] == int(np.ceil((area_stand_in.ground_z(x) + 0.6 - shift) * 100.0))
    assert tops[survey.cell(*area_stand_in.PEBBLE)] < 0


def test_the_air_is_the_seeds_and_cooler_the_higher_the_field(stand_in):
    """A seed's air and sky are the same each time, in the station's range at its height, and 6.5 C cooler a km up."""
    low, high = site.heat(5), site.heat(5, site.STATION_M + 1000.0)
    assert low == site.heat(5) and site.AIR_DAY_C[0] <= low["air_c"] <= site.AIR_DAY_C[1]
    assert low["air_c"] - high["air_c"] == pytest.approx(6.5, abs=0.11)
    assert low["air_c"] - low["sky_c"] == pytest.approx(high["air_c"] - high["sky_c"], abs=0.11)


def test_a_night_is_two_seeds_in_five_colder_and_a_day_seeds_air_is_as_it_was():
    """About two fifths of seeds are nights, their air from the night's range, as much cooler a km up as by day; a day
    seed's air and sky are the draw the day range always gave it."""
    seeds = range(1000)
    nights = [s for s in seeds if site.night(s)]
    assert 0.36 < len(nights) / len(seeds) < 0.44 and site.AIR_NIGHT_C[1] < site.AIR_DAY_C[1]
    for s in nights[:50]:
        low, high = site.heat(s), site.heat(s, site.STATION_M + 1000.0)
        assert site.AIR_NIGHT_C[0] - 0.05 <= low["air_c"] <= site.AIR_NIGHT_C[1] + 0.05
        assert low["air_c"] - high["air_c"] == pytest.approx(6.5, abs=0.11)
    for s in [s for s in seeds if not site.night(s)][:50]:
        rng = np.random.default_rng([site.HEAT_SEED_STREAM, s])
        air = float(rng.uniform(*site.AIR_DAY_C))
        assert site.heat(s) == {"air_c": round(air, 1), "sky_c": round(air - float(rng.uniform(*site.SKY_BELOW_AIR_C)), 1)}


def test_at_night_the_houses_read_over_the_air_and_the_water_warmer_and_by_day_nothing_is_set(monkeypatch):
    """A night sets every house a few degrees over the air and the water warmer than the air, never under freezing;
    a day seed sets nothing, so its thermal frames stay the engine's own."""
    set_to = {}
    monkeypatch.setattr(site.p, "changeVisualShape", lambda uid, link, temperature, physicsClientId: set_to.update({uid: temperature}))
    monkeypatch.setattr(site.sentinel_park, "THERMAL", True)
    world = SimpleNamespace(groups={"buildings": frozenset({11, 12, 13}), "water": frozenset({21}), "terrain": frozenset({1})})
    day, night = next(s for s in range(50) if not site.night(s)), next(s for s in range(50) if site.night(s))
    ep = SimpleNamespace(seed=day, park={"air_c": 4.0})
    site.warm_at_night(SimpleNamespace(CLIENT=0), ep, world)
    assert not set_to and "water_c" not in ep.park
    ep = SimpleNamespace(seed=night, park={"air_c": -2.0})
    site.warm_at_night(SimpleNamespace(CLIENT=0), ep, world)
    assert set(set_to) == {11, 12, 13, 21} and set_to[21] == ep.park["water_c"] >= site.WATER_LOWEST_C
    low, high = site.HOUSE_OVER_AIR_C
    assert all(-2.0 + low - site.HOUSE_SPREAD_C - 0.05 <= set_to[uid] <= -2.0 + high + site.HOUSE_SPREAD_C + 0.05 for uid in (11, 12, 13))
    assert site.night_heat(night, -2.0, 3) == site.night_heat(night, -2.0, 3)


def test_the_ground_is_read_against_the_brightness_of_the_tiles_near_the_field(stand_in):
    """The brightness the thermal camera reads the ground against is the mean of the area's table over the tiles that
    reach into a box; a box off the area gives none."""
    manifest = json.loads((stand_in / area_stand_in.SLUG / "manifest.json").read_text())
    near = site.ground_brightness(stand_in, area_stand_in.SLUG, manifest, (-20.0, -20.0, 20.0, 20.0))
    assert near == pytest.approx(area_stand_in.GROUND_LUMINANCE)
    assert site.ground_brightness(stand_in, area_stand_in.SLUG, manifest, (900.0, 900.0, 950.0, 950.0)) is None


def test_the_grounds_brightness_counts_each_tile_in_the_box_once_and_no_other(tmp_path):
    """Three ground tiles 10 m wide stand 20 m apart, two wearing a dark picture and one a bright one, beside a tree:
    the brightness is the mean over exactly the tiles a box reaches, each standing tile counted once, never the tree."""
    tile = {"group": "terrain", "bounds_min": [-5.0, -5.0, 0.0], "bounds_max": [5.0, 5.0, 0.0]}
    manifest = {"items": {"dark": tile, "bright": tile, "tree": {"group": "trees"}},
                "placements": [{"item": item, "position": [x, 0.0, 0.0], "quaternion": [0.0, 0.0, 0.0, 1.0], "scale": [1.0, 1.0, 1.0]}
                               for item, x in (("dark", 0.0), ("bright", 20.0), ("tree", 20.0), ("dark", 40.0))]}
    (tmp_path / "area").mkdir()
    (tmp_path / "area" / site.GROUND_FILE).write_text(json.dumps({"luminance": {"dark": 0.1, "bright": 0.4}}))
    for box, mean in (((-2.0, -2.0, 2.0, 2.0), 0.1), ((-2.0, -2.0, 16.0, 2.0), 0.25), ((18.0, -2.0, 22.0, 2.0), 0.4),
                      ((-2.0, -2.0, 42.0, 2.0), 0.2), ((6.0, -2.0, 14.0, 2.0), None), ((-2.0, 6.0, 42.0, 9.0), None)):
        found = site.ground_brightness(tmp_path, "area", manifest, box)
        assert found is None if mean is None else found == pytest.approx(mean)


def test_a_seeds_area_is_picked_by_the_shipped_weights(monkeypatch):
    """The site hands the cutter the seed, the room its people need, both size tables and the shipped weights, so the
    area of every real seed is drawn by weight."""
    seen = []
    monkeypatch.setattr(field_cut, "cut", lambda *args: seen.append(args) or "the field")
    assert site.draw(5, root="nowhere") == "the field"
    seed, room, largest, table, _cut_area, wooded, weights = seen[0]
    assert (seed, table, wooded, weights) == (5, field_sizes.table(), field_sizes.wooded_table(), field_sizes.weights())
    assert field_cut.size_limits()[0] <= room <= largest == field_cut.size_limits()[1]


def test_the_ground_and_the_trees_are_told_how_they_hold_heat_only_where_the_engine_can_be(monkeypatch):
    """On an engine with the setting every ground tile is read against the field's own brightness, 16 texels wide at
    10 C a unit, under the engine's own sun, the trees warm by 6 C in the sun, and the stones' picture is read 8
    texels wide at 10 C a unit; with no brightness, no thermal camera or an older engine nothing is set."""
    calls = []
    monkeypatch.setattr(site.p, "changeVisualShape", lambda uid, link, **options: calls.append((uid, options)))
    monkeypatch.setattr(site.sentinel_park, "THERMAL", True)
    world = SimpleNamespace(terrain=frozenset({5, 3}), forests={"trees": 9, "rocks": 10})
    env, ep = SimpleNamespace(CLIENT=7), SimpleNamespace(park={})
    monkeypatch.delattr(site.p, "THERMAL_FIELD_PASSIVE", raising=False)
    site.calm_ground(env, ep, world, 0.2)
    assert calls == [] and "ground_brightness" not in ep.park
    monkeypatch.setattr(site.p, "THERMAL_FIELD_PASSIVE", 8, raising=False)
    site.calm_ground(env, ep, world, None)
    assert calls == []
    site.calm_ground(env, ep, world, 0.2)
    assert calls == [(3, {"thermalPassive": [10.0, 0.2, 16.0, 22.0], "physicsClientId": 7}),
                     (5, {"thermalPassive": [10.0, 0.2, 16.0, 22.0], "physicsClientId": 7}),
                     (9, {"thermalPassive": [25.0, -1.0, 0.0, 6.0], "physicsClientId": 7}),
                     (10, {"thermalPassive": [10.0, -1.0, 8.0, 22.0], "physicsClientId": 7})]
    assert ep.park["ground_brightness"] == 0.2
    calls.clear()
    monkeypatch.setattr(site.sentinel_park, "THERMAL", False)
    site.calm_ground(env, ep, SimpleNamespace(terrain=frozenset({3}), forests={}), 0.2)
    assert calls == []


def test_the_install_check_asks_for_the_area_package(stand_in, tmp_path, monkeypatch):
    """The install lists the catalogue and every area's manifest and survey, finds them on the stand-in, and names
    each one missing from an empty folder."""
    files = site.asset_files()
    assert files[0] == stand_in / "areas.json" and stand_in / area_stand_in.SLUG / "spots.npz" in files
    assert all(path.is_file() for path in files)
    assert {str(path) for path in files} <= set(install.asset_files())
    monkeypatch.setenv("SWARM_ANDORRA_ROOT", str(tmp_path))
    monkeypatch.setattr(install, "area_problems", lambda: [])
    found = install.problems()
    assert found and str(tmp_path / "areas.json") in found[0] and str(tmp_path / area_stand_in.SLUG / "manifest.json") in found[0]


def test_the_install_check_asks_that_every_area_is_the_one_the_worlds_release_lists(tmp_path, monkeypatch):
    """An alternate root must carry the release's mark for every area."""
    listed = {"01_a": "a" * 64, "02_b": "b" * 64}
    root = tmp_path / "alternate"
    for area in listed:
        (root / area).mkdir(parents=True)

    def verify(root):
        """Read the alternate root's marks and compare them with the fixed release list."""
        missing = [area for area, mark in listed.items()
                   if not (root / area / ".mark").is_file() or (root / area / ".mark").read_text() != mark]
        return [f"missing {', '.join(missing)}; run `python -m swarm_worlds.areas`"] if missing else []

    monkeypatch.setitem(sys.modules, "swarm_worlds.areas", SimpleNamespace(problems=verify))
    monkeypatch.setenv("SWARM_ANDORRA_ROOT", str(root))
    assert "missing 01_a, 02_b" in install.problems()[0]
    for area, mark in listed.items():
        (root / area / ".mark").write_text(mark)
    assert install.area_problems() == []


@pytest.mark.parametrize("fault", ["no_list", "broken_list", "no_archives", "bad_mark", "broken_import"])
def test_invalid_worlds_install_names_the_repair_command(tmp_path, monkeypatch, fault):
    """Expected import, archive-list and mark errors reach the normal preflight diagnostics."""
    lock, mark = tmp_path / "areas.lock.json", tmp_path / ".mark"
    if fault != "no_list":
        lock.write_text({"broken_list": "{", "no_archives": "{}"}.get(fault, '{"archives": {}}'))
    mark.write_bytes(b"\xff" if fault == "bad_mark" else b"ok")

    def verify(root):
        """Read the test archive list and mark as the worlds tool does."""
        json.loads(lock.read_text())["archives"]
        mark.read_text(encoding="utf-8")
        return []

    monkeypatch.setitem(sys.modules, "swarm_worlds.areas",
                        None if fault == "broken_import" else SimpleNamespace(problems=verify))
    found = install.problems()
    assert found[0].startswith("swarm-worlds ")
    assert "python -m swarm_worlds.areas" in found[0]


def test_an_area_without_the_person_layer_is_refused(stand_in, tmp_path, monkeypatch):
    """A package whose survey lacks the cells a person may stand on is refused by name, never cut on another layer."""
    root = field_v2_stand_in.use_area(tmp_path, monkeypatch)
    path = root / area_stand_in.SLUG / "spots.npz"
    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files if key != PERSON_KEY}
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match=PERSON_KEY):
        site.draw(1)
    site.area_tables.cache_clear()
    site.cut_area.cache_clear()


@pytest.fixture
def package():
    """A real area package named by SWARM_ANDORRA_PACKAGE, carrying the person layer; skipped without one."""
    configured = os.environ.get("SWARM_ANDORRA_PACKAGE")
    if not configured or not (Path(configured) / "areas.json").is_file():
        pytest.skip("set SWARM_ANDORRA_PACKAGE to a real area package to run this check")
    if not _SHIPPED:
        pytest.skip("the installed swarm-worlds has no M4TD or no intruder bodies")
    return Path(configured)


@pytest.mark.timeout(300)
def test_a_seed_builds_on_a_real_area(package, tmp_path, monkeypatch):
    """On the real package and its shipped size table, a seed's field lies in an area the table lists, the dock
    stands on one of its dock cells and everyone stands on the field."""
    monkeypatch.setenv("SWARM_ANDORRA_ROOT", str(package))
    monkeypatch.setattr(field_sizes, "ROWS", SHIPPED_ROWS)
    field_v2_stand_in.use_people(tmp_path / "people", monkeypatch)
    monkeypatch.setattr(camera, "colour_frame", _blank)
    env, obs = _build(SEED)
    try:
        ep = env._solar
        assert site.field_of(ep).area in field_sizes.table()
        assert np.min(np.hypot(*(site.dock_spots(ep) - ep.dock_position[:2]).T)) < 1e-9
        assert len(people.people(ep)) == people.wanted(SEED)
        for person in people.people(ep):
            assert flight_limit.edge_distance(ep, np.asarray(person.xy)) >= people.PERSON_EDGE_M
        assert int(obs["site_map"][0]) == len(ep.fence)
    finally:
        env.close()
