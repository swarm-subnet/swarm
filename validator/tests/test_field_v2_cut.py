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

"""The field cutter on a small stand-in area: every field it cuts keeps the rules, the same seed cuts the same
outline byte for byte, all 128 bits of a seed count, and a cut is fast enough for every reset."""
from __future__ import annotations

import hashlib
import time

import numpy as np
import pytest

from swarm.challenge_families.test_field_v2 import field_cut, field_sizes, people

SIDE = 200
NAME = "stand_in"
# Pinned outlines: the first 16 hex digits of the SHA-256 of each seed's outline bytes, the same on two machines.
PINNED = {
    0: "feaa0ed6922571a5",
    1: "42786555e889c6bf",
    2: "2fa1db8b57d02291",
    3: "e26844c005b760b3",
    4: "a46b55dda1369a03",
    5: "06c25676f896bbe8",
    6: "f8c1c08bfe8b061e",
    7: "803e91d4d8fd6eab",
    8: "effae6636d957840",
    9: "3a5da88b1a41d71f",
    10: "26830a036704eca2",
    11: "bf028324cfd6b5e3",
}
RISING_SEED = 3                        # a seed whose field on the rising meadow rises more than 6 m


def _stand_in(name: str = NAME) -> field_cut.Area:
    """A 200 m area: a meadow sloping gently east and south, a tree line along its north side and a clump of trees
    in it, a house, a pond and a road cutting off its east strip."""
    rows, cols = np.mgrid[0:SIDE, 0:SIDE]
    ground = (20000 + 6 * cols + 3 * rows).astype(np.uint16)
    keep_out = np.zeros((SIDE, SIDE), dtype=bool)
    keep_out[40:56, 60:80] = True              # a house
    keep_out[140:156, 40:60] = True            # a pond
    open_ground = ~field_cut.binary_dilation(keep_out, structure=field_cut.disk(2))
    open_ground[:, 150:157] = False            # a road, never open ground
    crowns = [[x, 168.0 - SIDE / 2, 4.0] for x in np.arange(-70.0, 71.0, 6.0)]
    crowns += [[x, y, 3.5] for x, y in ((0.0, 0.0), (5.0, 2.0), (2.0, -5.0))]
    crowns = np.array(crowns)
    x, y = cols + 0.5 - SIDE / 2, SIDE / 2 - rows - 0.5
    for cx, cy, r in crowns:
        open_ground &= (x - cx) ** 2 + (y - cy) ** 2 > (r + 1.5) ** 2
    houses = np.array([[-30.0, 52.0, 20.0, 16.0, 90.0]])
    return field_cut.Area(name, open_ground, ground, keep_out, crowns, houses)


@pytest.fixture(scope="module")
def area() -> field_cut.Area:
    """One stand-in area shared by the tests of this module."""
    return _stand_in()


def _table(*names: str) -> dict:
    """A size table in which the named areas hold every size up to 1.2 times today's field."""
    return {name: tuple(k < 7 for k in range(field_cut.SIZE_BINS)) for name in names}


def _cut(area: field_cut.Area, seed: int, min_m2: float = 1400.0, max_m2: float = 3300.0) -> field_cut.Field:
    """The seed's field on the stand-in area alone."""
    return field_cut.cut(seed, min_m2, max_m2, _table(area.name), lambda name: area)


def _digest(field: field_cut.Field) -> str:
    """A short fingerprint of a field's outline bytes."""
    return hashlib.sha256(np.ascontiguousarray(field.outline).tobytes()).hexdigest()[:16]


def _cells_in(outline: np.ndarray) -> np.ndarray:
    """Which stand-in cells have their centre inside an outline, by the cutter's own exact test."""
    ring = np.column_stack([SIDE // 2 - outline[:, 1], outline[:, 0] + SIDE // 2]).astype(np.int64)
    cells, top, left = field_cut.polygon_cells(ring, (SIDE, SIDE))
    full = np.zeros((SIDE, SIDE), dtype=bool)
    full[top:top + cells.shape[0], left:left + cells.shape[1]] = cells
    return full


def test_every_cut_field_keeps_the_rules(area):
    """Fields cut for many seeds are simple anticlockwise outlines of at most 64 corners, of the size asked, within
    95 m across, clear of the house and the pond, with room for the dock away from the edge and every crown."""
    for seed in range(12):
        field = _cut(area, seed)
        outline = field.outline
        assert 3 <= len(outline) <= field_cut.MAX_POINTS
        assert field_cut.shoelace(outline) > 0.0
        ring = [(int(SIDE // 2 - y), int(x + SIDE // 2)) for x, y in outline]
        assert field_cut.is_simple(ring)
        low, high = field_cut.bounds(field.target_m2, 1400.0, 3300.0)
        assert low <= field.area_m2 <= high
        assert field_cut.shoelace(outline) == pytest.approx(field.area_m2)
        gaps = outline[:, None, :] - outline[None, :, :]
        assert np.sqrt((gaps ** 2).sum(axis=2)).max() <= field_cut.MAX_ACROSS_M
        assert np.abs(outline).max() <= SIDE / 2 - field_cut.CORE_MARGIN_M
        assert not (_cells_in(outline) & area.keep_out).any()
        assert len(field.dock_spots) >= field_cut.MIN_DOCK_CELLS
        for spot in field.dock_spots:
            sides = np.roll(outline, -1, axis=0) - outline
            t = np.clip(((spot - outline) * sides).sum(axis=1) / (sides ** 2).sum(axis=1), 0.0, 1.0)
            assert np.hypot(*(outline + t[:, None] * sides - spot).T).min() >= field_cut.DOCK_INSIDE_M
            if len(field.crowns):
                assert (np.hypot(*(field.crowns[:, :2] - spot).T) - field.crowns[:, 2]).min() >= 3.4


def test_every_crown_that_reaches_the_field_is_listed(area):
    """Each tree circle that reaches into the field is one of the field's crowns, and none that stays out is."""
    for seed in range(6):
        field = _cut(area, seed)
        cells = _cells_in(field.outline)
        rows, cols = np.nonzero(cells)
        x, y = cols + 0.5 - SIDE / 2, SIDE / 2 - rows - 0.5
        listed = {tuple(np.round(c, 2)) for c in field.crowns}
        for cx, cy, r in area.crowns:
            reaches = ((x - cx) ** 2 + (y - cy) ** 2 <= r * r).any()
            if reaches:
                assert any(abs(lx - cx) < 0.011 and abs(ly - cy) < 0.011 and lr >= r for lx, ly, lr in listed)


def test_same_seed_cuts_the_same_outline_byte_for_byte(area):
    """Two cuts of a seed on fresh copies of the area give identical bytes, and the outlines of a dozen seeds match
    their pinned fingerprints."""
    fresh = _stand_in()
    for seed in PINNED:
        first, second = _cut(area, seed), _cut(fresh, seed)
        assert first.outline.tobytes() == second.outline.tobytes()
        assert first.crowns.tobytes() == second.crowns.tobytes()
        assert first.dock_spots.tobytes() == second.dock_spots.tobytes()
        assert _digest(first) == PINNED[seed]


def test_all_128_bits_of_a_seed_count(area):
    """Seeds that differ only in their top bits draw different sizes and cut different fields."""
    low = 0x1234
    fields = [_cut(area, low | (top << 100)) for top in (0, 1, 2 ** 27)]
    assert len({field.target_m2 for field in fields}) > 1
    assert len({field.outline.tobytes() for field in fields}) == 3


def test_a_seed_never_gets_a_smaller_field_than_it_asks_for(area):
    """With the smallest size raised, every field is at least that large; and an area table that holds no size of
    the range fails the cut instead of shrinking the field."""
    for seed in range(8):
        assert _cut(area, seed, min_m2=2600.0, max_m2=3000.0).area_m2 >= 2600.0
    with pytest.raises(field_cut.NoField):
        field_cut.cut(1, 3900.0, 4100.0, _table(area.name), lambda name: area)


def test_the_size_is_drawn_first_and_the_area_uniformly_among_those_that_hold_it():
    """Sizes fall uniformly in the range and each area holding the size comes first about equally often; an area
    that does not hold the size never comes up."""
    names = [f"area_{k}" for k in range(4)]
    table = {name: tuple(True for _ in range(field_cut.SIZE_BINS)) for name in names[:3]}
    table[names[3]] = tuple(k < 2 for k in range(field_cut.SIZE_BINS))
    firsts, sizes = {}, []
    for seed in range(600):
        target, order = field_cut.choose(seed, 2600.0, 4000.0, table)
        sizes.append(target)
        firsts[order[0]] = firsts.get(order[0], 0) + 1
        assert names[3] not in order and sorted(order) == names[:3]
    assert 2600 <= min(sizes) and max(sizes) <= 4000
    assert abs(np.mean(sizes) - 3300.0) < 60.0
    assert all(150 <= firsts[name] <= 250 for name in names[:3])


def test_a_weighted_order_draws_each_next_area_by_its_weight():
    """Every weighted order holds each name once; a name comes first in proportion to its weight, one with no weight
    as weight 1; the same stream gives the same order."""
    names, weights = ["heavy", "light", "plain"], {"heavy": 6, "light": 1}
    firsts, seconds = {}, {}
    for seed in range(4000):
        order = field_cut.weighted_order(np.random.default_rng([7, seed]), names, weights)
        assert sorted(order) == sorted(names)
        firsts[order[0]] = firsts.get(order[0], 0) + 1
        if order[0] == "heavy":
            seconds[order[1]] = seconds.get(order[1], 0) + 1
    assert abs(firsts["heavy"] / 4000 - 6 / 8) < 0.03 and abs(firsts["light"] / 4000 - 1 / 8) < 0.02
    assert abs(seconds["light"] / firsts["heavy"] - 0.5) < 0.04
    again = [field_cut.weighted_order(np.random.default_rng([7, 3]), names, weights) for _ in range(2)]
    assert again[0] == again[1]
    assert field_cut.weighted_order(np.random.default_rng(1), [], weights) == []


def test_weights_pick_the_first_area_and_leave_the_size_alone():
    """With weights the seed's size is the one it draws without them, an area that does not hold it still never comes
    up, and the area of weight 8 comes first about eight times as often as each of weight 1."""
    names = [f"area_{k}" for k in range(4)]
    table = {name: tuple(True for _ in range(field_cut.SIZE_BINS)) for name in names[:3]}
    table[names[3]] = tuple(k < 2 for k in range(field_cut.SIZE_BINS))
    firsts = {}
    for seed in range(1500):
        target, order = field_cut.choose(seed, 2600.0, 4000.0, table, weights={names[0]: 8, names[3]: 99})
        assert target == field_cut.choose(seed, 2600.0, 4000.0, table)[0]
        assert names[3] not in order and sorted(order) == names[:3]
        firsts[order[0]] = firsts.get(order[0], 0) + 1
    assert abs(firsts[names[0]] / 1500 - 0.8) < 0.04 and all(110 <= firsts[name] <= 190 for name in names[1:3])


def test_the_shipped_weights_spread_the_seeds_over_the_areas():
    """Over the sizes and kinds the seeds ask for, with the shipped tables and weights, no area comes first for more
    than one seed in 28 and at most five of the areas that hold any size come first for under one seed in a
    hundred; picked evenly, one area takes more than one seed in 23 and at least eight come under one in a hundred."""
    tables = {field_cut.OPEN.kind: field_sizes.table(), field_cut.WOODED.kind: field_sizes.wooded_table()}
    smallest, largest = field_cut.size_limits()
    usable = {name for table in tables.values() for name, holds in table.items() if any(holds)}

    def shares(weights):
        """Each area's share of 6,000 seeds' first picks, picked evenly or by weight."""
        firsts = dict.fromkeys(usable, 0)
        for seed in range(6000):
            room = max(smallest, people.min_area_m2(people.wanted(seed)))
            kind = field_cut.draw_kind(seed)
            names = field_cut.choose(seed, room, largest, tables[kind], field_cut.RECIPES[kind], weights)[1] if kind != field_cut.OPEN.kind else []
            names = names or field_cut.choose(seed, room, largest, tables[field_cut.OPEN.kind], field_cut.OPEN, weights)[1]
            firsts[names[0]] += 1
        return np.array(sorted(firsts.values())) / 6000.0

    even, weighted = shares(None), shares(field_sizes.weights())
    assert even[-1] > 1 / 23 and int((even < 0.01).sum()) >= 8
    assert weighted[-1] < 1 / 28 and int((weighted < 0.01).sum()) <= 5
    assert set(field_sizes.weights()) <= usable and all(1 <= weight <= 256 for weight in field_sizes.weights().values())


def test_size_bins_cover_the_range_without_gaps():
    """Every whole square metre from half to one and a half times today's field falls in exactly the bin whose
    edges hold it."""
    edges = field_cut.bin_edges()
    assert edges[0][0] == 1378 and edges[-1][1] == 4132
    for k, (smallest, largest) in enumerate(edges):
        assert field_cut.size_bin(smallest) == field_cut.size_bin(largest) == k
        if k:
            assert smallest == edges[k - 1][1] + 1


def test_an_area_holds_the_sizes_its_open_ground_fits():
    """The stand-in's wide meadow holds today's size; a small paddock of 30 by 40 m holds the smallest size but
    not today's."""
    meadow = _stand_in()
    assert field_cut.holds(meadow, field_cut.size_bin(2755), probes=40, needed=2)[0]
    small = _stand_in("paddock")
    small.open[:] = False
    small.open[80:110, 70:110] = True
    paddock = field_cut.Area("paddock", small.open, small.ground_cm, small.keep_out, small.crowns, small.houses)
    assert field_cut.holds(paddock, 0, probes=40, needed=2)[0]
    assert not field_cut.holds(paddock, field_cut.size_bin(2755), probes=20, needed=2)[0]


def test_a_size_table_marks_for_each_area_the_bins_it_holds():
    """The table built for the shipped rows has one mark per size bin for each area, the same marks an area's own
    probes give: the small paddock does not hold today's size."""
    small = _stand_in("paddock")
    small.open[:] = False
    small.open[80:110, 70:110] = True
    paddock = field_cut.Area("paddock", small.open, small.ground_cm, small.keep_out, small.crowns, small.houses)
    rows = field_cut.size_table([paddock], probes=20, needed=2)
    assert list(rows) == ["paddock"] and len(rows["paddock"]) == field_cut.SIZE_BINS
    today = field_cut.size_bin(2755)
    assert all(rows["paddock"][k] == field_cut.holds(paddock, k, 20, 2)[0] for k in (0, today))
    assert not rows["paddock"][today]


def test_the_shipped_size_table_marks_every_bin_of_every_area():
    """The shipped table has one mark per size bin for each of the 48 areas, and at least 40 of them hold some
    size."""
    rows = field_sizes.table()
    assert len(rows) == 48
    assert set("".join(field_sizes.ROWS.values())) <= {"1", "."}
    assert all(len(marks) == field_cut.SIZE_BINS for marks in rows.values())
    assert sum(any(marks) for marks in rows.values()) >= 40


def test_crowns_merge_into_covering_circles_never_dropped():
    """Over 64 circles are joined into at most 64, and every circle given lies inside one of them."""
    rng = np.random.default_rng(5)
    circles = np.column_stack([rng.integers(-4000, 4000, size=(150, 2)), rng.integers(150, 600, size=150)])
    merged = field_cut.merge_crowns(circles)
    assert len(merged) == field_cut.MAX_CROWNS
    for x, y, r in circles:
        gaps = np.sqrt((merged[:, 0] - x) ** 2.0 + (merged[:, 1] - y) ** 2.0) + r - merged[:, 2]
        assert gaps.min() <= 0.0


def test_too_many_crowns_are_merged_and_reject_a_field_only_when_they_cover_too_much(area):
    """A 72 m meadow with a clump of 100 saplings in its middle gives fields whose crowns are merged to 64 and still
    cover every sapling; a meadow under a dense stand of large trees gives none, for the crowns' share."""
    clump = np.array([[x, y, 0.6] for x in np.arange(-9.0, 10.0, 2.0) for y in np.arange(-9.0, 10.0, 2.0)])
    rows, cols = np.mgrid[0:SIDE, 0:SIDE]
    open_ground = (np.abs(cols + 0.5 - SIDE / 2) < 36) & (np.abs(SIDE / 2 - rows - 0.5) < 36)
    nothing = np.zeros((SIDE, SIDE), dtype=bool)
    clumped = field_cut.Area("clumped", open_ground, area.ground_cm, nothing, clump, area.houses)
    merged = 0
    for seed in range(6):
        field = field_cut.cut(seed, 3000.0, 3300.0, _table("clumped"), lambda name: clumped)
        assert len(field.crowns) <= field_cut.MAX_CROWNS
        if len(field.crowns) == field_cut.MAX_CROWNS:
            merged += 1
            ring_cm = np.rint(field.outline * 100.0).astype(np.int64)
            for x, y, r in clumped.crowns_cm[field_cut.touching(clumped.crowns_cm, ring_cm)] / 100.0:
                assert (np.hypot(field.crowns[:, 0] - x, field.crowns[:, 1] - y) + r <= field.crowns[:, 2] + 1e-9).any()
    assert merged > 0
    stand = np.array([[x, y, 3.0] for x in np.arange(-66.0, 67.0, 6.0) for y in np.arange(-66.0, 67.0, 6.0)])
    shaded = field_cut.Area("shaded", np.ones((SIDE, SIDE), dtype=bool), area.ground_cm, nothing, stand, area.houses)
    reasons = {}
    assert field_cut.cut_in(shaded, np.random.default_rng(3), 3000, 2700.0, 3300.0, tries=10, reasons=reasons) is None
    assert "crowns" in reasons


def test_the_plane_through_the_field_may_tilt_at_most_12_degrees():
    """A patch rising 22 cm a metre (12.4 degrees) is too steep, one rising 20 cm a metre (11.3 degrees) is not."""
    rows, cols = (v.ravel() for v in np.mgrid[0:20, 0:20])
    assert field_cut.plane_too_steep(rows, cols, 1000 + 22 * cols)
    assert not field_cut.plane_too_steep(rows, cols, 1000 + 20 * cols)


def test_steep_ground_rejects_a_field(area):
    """The same meadow tilted by 15 degrees gives no field: it rises over 8 m across any field of 2,000 m2."""
    rows, cols = np.mgrid[0:SIDE, 0:SIDE]
    tilted = field_cut.Area("tilted", area.open, (20000 + 27 * cols).astype(np.uint16), area.keep_out, area.crowns, area.houses)
    reasons = {}
    assert field_cut.cut_in(tilted, np.random.default_rng(1), 2000, 1800.0, 2200.0, tries=20, reasons=reasons) is None
    assert "relief" in reasons


def test_the_dock_stands_only_where_the_field_top_is_within_reach(area):
    """On ground rising more than 6 m across the field, dock spots lie only on its upper part, never more than 6 m
    below its highest ground."""
    rows, cols = np.mgrid[0:SIDE, 0:SIDE]
    ground = (20000 + 12 * cols).astype(np.uint16)
    rising = field_cut.Area("rising", area.open, ground, area.keep_out, area.crowns, area.houses)
    field = field_cut.cut(RISING_SEED, 2600.0, 3000.0, _table("rising"), lambda name: rising)
    cells = _cells_in(field.outline)
    top = int(ground[cells].max())
    assert top - int(ground[cells].min()) > field_cut.DOCK_RISE_CM
    spot_cols = (field.dock_spots[:, 0] + SIDE / 2 - 0.5).astype(int)
    spot_rows = (SIDE / 2 - field.dock_spots[:, 1] - 0.5).astype(int)
    assert (top - ground[spot_rows, spot_cols] <= field_cut.DOCK_RISE_CM).all()


def test_the_dock_stands_only_where_the_drone_can_fly_over_everything_standing_on_the_field(area):
    """With 4 m bushes all over the field the dock keeps to cells no more than 8 m below the tallest bush's top, which
    leaves out low cells the bare ground's 6 m rule would allow."""
    bushes = area.ground_cm + 400
    bushy = field_cut.Area("bushy", area.open, area.ground_cm, area.keep_out, area.crowns, area.houses,
                           solid_top_cm=bushes)
    for seed in range(4):
        field = field_cut.cut(seed, 2600.0, 3000.0, _table("bushy"), lambda name: bushy)
        cells = _cells_in(field.outline)
        tallest = int(bushes[cells].max())
        spot_cols = (field.dock_spots[:, 0] + SIDE / 2 - 0.5).astype(int)
        spot_rows = (SIDE / 2 - field.dock_spots[:, 1] - 0.5).astype(int)
        assert (tallest - area.ground_cm[spot_rows, spot_cols] <= field_cut.DOCK_SOLID_RISE_CM).all()
        assert tallest - int(area.ground_cm[cells].min()) > field_cut.DOCK_SOLID_RISE_CM


def test_every_shape_family_draws_an_anticlockwise_shape_round_its_centre():
    """Thousands of drawn shapes all have a positive area and hold their centre, and every family comes up."""
    rng = np.random.default_rng(17)
    families = set()
    for _ in range(4000):
        family, corners = field_cut.template(rng)
        families.add(family)
        assert field_cut.shoelace(corners) > 0.0
        assert field_cut.cells_in(corners * 10.0, -1, -1, 2, 2, (0, 0))[1, 1] or family == "ell"
    assert families == set(field_cut.FAMILIES)


def test_tracing_and_simplifying_a_staircase_gives_a_simple_anticlockwise_ring():
    """A square piece traces to its four corners anticlockwise on the map; a diagonal staircase simplifies to a
    straight side; two cells touching at a corner are joined before tracing."""
    piece = np.zeros((6, 6), dtype=bool)
    piece[1:4, 1:5] = True
    assert field_cut.trace(piece) == [(1, 1), (4, 1), (4, 5), (1, 5)]
    stairs = np.tril(np.ones((20, 20), dtype=bool))
    corners = field_cut.simplify(field_cut.trace(np.pad(stairs, 1)), (5, 4))
    assert len(corners) == 3 and field_cut.is_simple(corners)
    pinch = np.zeros((4, 4), dtype=bool)
    pinch[1, 1] = pinch[2, 2] = True
    assert field_cut.close_pinches(pinch)[1:3, 1:3].all()
    assert not field_cut.is_simple([(0, 0), (4, 4), (0, 4), (4, 0)])


def test_a_cut_is_fast_enough_for_every_reset(area):
    """A dozen cuts of the stand-in area take well under a second each."""
    start = time.perf_counter()
    for seed in range(100, 112):
        _cut(area, seed)
    assert (time.perf_counter() - start) / 12 < 0.5


# ---- fields that reach into the trees ------------------------------------------------------------------------------

def _woods(name: str = "woods") -> field_cut.Area:
    """A 200 m area: a meadow in the west half and a wood in the east half, a tree every 8 m, whose floor is not open
    ground."""
    rows, cols = np.mgrid[0:SIDE, 0:SIDE]
    ground = (20000 + 2 * cols + 2 * rows).astype(np.uint16)
    crowns = np.array([[cx, cy, 3.0] for cx in np.arange(4.0, 70.0, 8.0) for cy in np.arange(-60.0, 61.0, 8.0)])
    open_ground = cols + 0.5 - SIDE / 2 < 0.0
    nothing = np.zeros((SIDE, SIDE), dtype=bool)
    return field_cut.Area(name, open_ground, ground, nothing, crowns, np.zeros((0, 5)))


@pytest.fixture(scope="module")
def woods() -> field_cut.Area:
    """One meadow and wood shared by the tests of this module."""
    return _woods()


def _inside_count(field: field_cut.Field) -> int:
    """How many of a field's crown centres lie inside its outline."""
    crowns_cm = np.rint(field.crowns * 100.0).astype(np.int64)
    return int(field_cut.centres_inside(crowns_cm, np.rint(field.outline * 100.0).astype(np.int64)).sum())


def test_a_seed_asks_for_a_wooded_field_in_three_seeds_of_five():
    """Over 4,000 seeds the wooded share is 60 % within sampling error, and all 128 bits of a seed move the draw."""
    kinds = [field_cut.draw_kind(seed) for seed in range(4000)]
    assert set(kinds) == {"open", "wooded"}
    assert abs(kinds.count("wooded") / 4000 - 0.60) < 0.025
    assert len({field_cut.draw_kind(7 | (top << 100)) for top in range(40)}) == 2


def test_a_wooded_field_holds_four_trees_and_keeps_every_rule(woods):
    """Wooded fields cut from the meadow's edge hold at least four tree centres, keep the wooded recipe's rules when
    checked again, and are marked wooded; open fields there reach no tree."""
    table = _table(woods.name)
    for seed in range(8):
        field = field_cut.cut_kind(seed, 1400.0, 3300.0, table, lambda name: woods, field_cut.WOODED)
        assert field is not None and field.kind == "wooded"
        assert _inside_count(field) >= field_cut.WOODED.least_crowns
        ring = np.column_stack([SIDE // 2 - field.outline[:, 1], field.outline[:, 0] + SIDE // 2]).astype(np.int64)
        low, high = field_cut.bounds(field.target_m2, 1400.0, 3300.0)
        field_cut.check(woods, ring, low, high, field_cut.WOODED)
        open_field = field_cut.cut_kind(seed, 1400.0, 3300.0, table, lambda name: woods, field_cut.OPEN)
        assert open_field.kind == "open" and _inside_count(open_field) == 0


def test_an_open_seed_is_cut_as_it_always_was(area, woods):
    """With a wooded table that lists the wood, every seed that asks for an open field gets its pinned outline, and
    every seed that asks for a wooded field gets one there."""
    wooded_table = _table(woods.name)
    areas = {area.name: area, woods.name: woods}
    for seed in PINNED:
        field = field_cut.cut(seed, 1400.0, 3300.0, _table(area.name), areas.__getitem__, wooded_table)
        if field_cut.draw_kind(seed) == "open":
            assert field.kind == "open" and _digest(field) == PINNED[seed]
        else:
            assert field.kind == "wooded" and field.area == woods.name


def test_a_seed_that_finds_no_wooded_field_takes_its_open_one(area):
    """A seed that asks for a wooded field where no area yields one, or no area is listed, gets the very open field
    it would have had."""
    bare = field_cut.Area("bare", area.open, area.ground_cm, area.keep_out, np.zeros((0, 3)), area.houses)
    areas = {area.name: area, bare.name: bare}
    wooded_seeds = [seed for seed in PINNED if field_cut.draw_kind(seed) == "wooded"]
    assert wooded_seeds
    for seed in wooded_seeds:
        for wooded_table in ({}, _table(bare.name)):
            field = field_cut.cut(seed, 1400.0, 3300.0, _table(area.name), areas.__getitem__, wooded_table)
            assert field.kind == "open" and _digest(field) == PINNED[seed]


def test_an_area_holds_wooded_sizes_only_where_trees_stand_by_open_ground(area, woods):
    """The meadow and wood hold today's size as a wooded field; the same meadow without its trees holds none, and
    the wooded probes never move the open table's."""
    which = field_cut.size_bin(2755)
    assert field_cut.holds(woods, which, probes=40, needed=2, recipe=field_cut.WOODED)[0]
    bare = field_cut.Area("bare", woods.open, woods.ground_cm, woods.keep_out, woods.crowns[:3], woods.houses)
    assert not field_cut.holds(bare, which, probes=40, needed=2, recipe=field_cut.WOODED)[0]
    assert field_cut.holds(area, which, probes=40, needed=2) == field_cut.holds(_stand_in(), which, probes=40, needed=2)


def test_the_shipped_wooded_table_marks_every_bin_of_areas_the_open_table_knows():
    """The wooded table has one mark per size bin for areas of the package, and some areas hold wooded fields."""
    rows = field_sizes.wooded_table()
    assert set(rows) <= set(field_sizes.ROWS)
    assert set("".join(field_sizes.WOODED_ROWS.values())) <= {"1", "."}
    assert all(len(marks) == field_cut.SIZE_BINS for marks in rows.values())
    assert sum(any(marks) for marks in rows.values()) >= 10


def test_a_field_keeps_every_tree_that_reaches_it_unmerged(area, woods):
    """A field's trees are every crown circle that reaches it, as the area lists them: the same as its crowns when
    they are few, and still every one of them when the site map's crowns are merged."""
    for seed in range(4):
        field = _cut(area, seed)
        assert np.array_equal(field.trees, field.crowns)
    saplings = np.array([[x, y, 0.6] for x in np.arange(-9.0, 10.0, 2.0) for y in np.arange(-9.0, 10.0, 2.0)])
    rows, cols = np.mgrid[0:SIDE, 0:SIDE]
    meadow = (np.abs(cols + 0.5 - SIDE / 2) < 36) & (np.abs(SIDE / 2 - rows - 0.5) < 36)
    grove = field_cut.Area("grove", meadow, area.ground_cm, np.zeros((SIDE, SIDE), dtype=bool), saplings, area.houses)
    merged = 0
    for seed in range(6):
        field = field_cut.cut_kind(seed, 1400.0, 3300.0, _table("grove"), lambda name: grove, field_cut.OPEN)
        assert {(x, y) for x, y, _r in field.trees} <= {(x, y) for x, y, _r in saplings}
        assert len(field.crowns) == min(len(field.trees), field_cut.MAX_CROWNS)
        merged += len(field.trees) > field_cut.MAX_CROWNS
    assert merged
