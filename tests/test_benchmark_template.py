from __future__ import annotations

from collections import Counter

from swarm.constants import (
    BENCHMARK_FULL_SEED_COUNT,
    BENCHMARK_TEMPLATE,
    SCREENING_TEMPLATE,
)


VALID_TYPES = {1, 2, 3, 4, 5, 6}


def test_template_has_100_entries():
    assert len(BENCHMARK_TEMPLATE) == 100


def test_template_divides_benchmark_range_cleanly():
    assert BENCHMARK_FULL_SEED_COUNT % len(BENCHMARK_TEMPLATE) == 0


def test_template_covers_all_types_evenly():
    counts = Counter(slot["challenge_type"] for slot in BENCHMARK_TEMPLATE)
    assert set(counts) == VALID_TYPES
    for ct in VALID_TYPES:
        assert 16 <= counts[ct] <= 17


def test_moving_platform_rates_match_real_probabilities():
    moving = Counter()
    total = Counter()
    for slot in BENCHMARK_TEMPLATE:
        total[slot["challenge_type"]] += 1
        if slot["moving_platform"]:
            moving[slot["challenge_type"]] += 1
    # warehouse/forest never move; open is mostly moving; city/mountain/village ~quarter.
    assert moving.get(5, 0) == 0
    assert moving.get(6, 0) == 0
    assert moving[2] / total[2] > 0.7
    for ct in (1, 3, 4):
        assert 0.15 < moving[ct] / total[ct] < 0.35


def test_each_type_has_three_distance_bands():
    bands = {ct: set() for ct in VALID_TYPES}
    for slot in BENCHMARK_TEMPLATE:
        bands[slot["challenge_type"]].add(slot["distance_range"])
    for ct in VALID_TYPES:
        assert len(bands[ct]) == 3


def test_template_interleaved_first_six_cover_all_types():
    first_six = [slot["challenge_type"] for slot in BENCHMARK_TEMPLATE[:6]]
    assert set(first_six) == VALID_TYPES


def test_distance_and_height_ranges_are_valid():
    for i, slot in enumerate(BENCHMARK_TEMPLATE):
        lo, hi = slot["distance_range"]
        assert 0 < lo < hi, f"Slot {i}: bad distance_range {slot['distance_range']}"
        ghr = slot["goal_height_range"]
        if slot["challenge_type"] in (3, 4):
            assert ghr is None
        else:
            assert ghr is not None and ghr[1] > ghr[0]


def test_benchmark_template_distinct_from_screening():
    assert BENCHMARK_TEMPLATE is not SCREENING_TEMPLATE
    assert len(BENCHMARK_TEMPLATE) != len(SCREENING_TEMPLATE)
