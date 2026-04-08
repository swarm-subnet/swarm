from __future__ import annotations

import math
from collections import Counter

import pytest

from swarm.constants import (
    SCREENING_CHECKPOINT_SIZE,
    SCREENING_EARLY_FAIL_FACTORS,
    SCREENING_TEMPLATE,
    BENCHMARK_SCREENING_SEED_COUNT,
)


VALID_TYPES = {1, 2, 3, 4, 5, 6}
TYPE_NAMES = {1: "city", 2: "open", 3: "mountain", 4: "village", 5: "warehouse", 6: "forest"}


def test_template_has_50_entries():
    assert len(SCREENING_TEMPLATE) == SCREENING_CHECKPOINT_SIZE


def test_template_covers_all_types():
    types_in_template = {slot["challenge_type"] for slot in SCREENING_TEMPLATE}
    assert types_in_template == VALID_TYPES


def test_template_type_distribution():
    counts = Counter(slot["challenge_type"] for slot in SCREENING_TEMPLATE)
    assert counts[1] == 8   # city
    assert counts[2] == 8   # open
    assert counts[3] == 8   # mountain
    assert counts[4] == 9   # village
    assert counts[5] == 9   # warehouse
    assert counts[6] == 8   # forest
    assert sum(counts.values()) == 50


def test_template_moving_platform_counts():
    moving = Counter()
    for slot in SCREENING_TEMPLATE:
        if slot["moving_platform"]:
            moving[slot["challenge_type"]] += 1
    assert moving[1] == 2   # city
    assert moving[2] == 6   # open
    assert moving[3] == 2   # mountain
    assert moving[4] == 2   # village
    assert moving.get(5, 0) == 0  # warehouse
    assert moving.get(6, 0) == 0  # forest


def test_template_all_slots_have_required_keys():
    required = {"challenge_type", "distance_range", "goal_height_range", "moving_platform"}
    for i, slot in enumerate(SCREENING_TEMPLATE):
        assert required.issubset(slot.keys()), f"Slot {i} missing keys: {required - slot.keys()}"


def test_template_distance_ranges_are_valid():
    for i, slot in enumerate(SCREENING_TEMPLATE):
        lo, hi = slot["distance_range"]
        assert lo > 0, f"Slot {i}: distance_range lower must be positive"
        assert hi > lo, f"Slot {i}: distance_range upper must exceed lower"


def test_template_goal_height_ranges():
    for i, slot in enumerate(SCREENING_TEMPLATE):
        ct = slot["challenge_type"]
        ghr = slot["goal_height_range"]
        if ct in (3, 4):
            assert ghr is None, f"Slot {i}: mountain/village should have goal_height_range=None"
        else:
            assert ghr is not None, f"Slot {i}: type {ct} should have goal_height_range"
            lo, hi = ghr
            assert hi > lo, f"Slot {i}: goal_height_range upper must exceed lower"


def test_template_interleaved_ordering():
    first_6 = [slot["challenge_type"] for slot in SCREENING_TEMPLATE[:6]]
    assert len(set(first_6)) == 6, "First 6 slots should cover all 6 types (interleaved)"


def test_template_cycles_to_200():
    total = BENCHMARK_SCREENING_SEED_COUNT
    cycle = (SCREENING_TEMPLATE * ((total // len(SCREENING_TEMPLATE)) + 1))[:total]
    assert len(cycle) == 200

    counts = Counter(slot["challenge_type"] for slot in cycle)
    for ct in VALID_TYPES:
        assert counts[ct] >= 30, f"Type {ct} should appear >=30 times in 200 seeds"


def test_early_fail_factors_are_more_lenient():
    assert SCREENING_EARLY_FAIL_FACTORS[50] == 0.50
    assert SCREENING_EARLY_FAIL_FACTORS[100] == 0.70
    assert SCREENING_EARLY_FAIL_FACTORS[150] == 0.85
    for seeds, factor in sorted(SCREENING_EARLY_FAIL_FACTORS.items()):
        assert 0.0 < factor < 1.0
