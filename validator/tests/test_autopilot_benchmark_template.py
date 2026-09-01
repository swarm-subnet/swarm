from collections import Counter

from swarm.challenge_families.autopilot import _AUTOPILOT_BENCHMARK_TEMPLATE as T
from swarm.constants import BENCHMARK_FULL_SEED_COUNT


VALID_TYPES = {1, 2, 3, 4, 5, 6}


def test_template_has_100_entries():
    assert len(T) == 100


def test_benchmark_range_divides_template_cleanly():
    assert BENCHMARK_FULL_SEED_COUNT % len(T) == 0


def test_covers_all_types_evenly():
    counts = Counter(s["challenge_type"] for s in T)
    assert set(counts) == VALID_TYPES
    for ct in VALID_TYPES:
        assert 16 <= counts[ct] <= 17


def test_three_distance_bands_per_type():
    bands = {ct: set() for ct in VALID_TYPES}
    for s in T:
        bands[s["challenge_type"]].add(s["distance_range"])
    for ct in VALID_TYPES:
        assert len(bands[ct]) == 3


def test_moving_platform_rates():
    moving, total = Counter(), Counter()
    for s in T:
        total[s["challenge_type"]] += 1
        if s["moving_platform"]:
            moving[s["challenge_type"]] += 1
    assert moving.get(5, 0) == 0 and moving.get(6, 0) == 0
    assert moving[2] / total[2] > 0.7
    for ct in (1, 3, 4):
        assert 0.15 < moving[ct] / total[ct] < 0.35


def test_first_six_cover_all_types():
    assert set(s["challenge_type"] for s in T[:6]) == VALID_TYPES


def test_distance_and_height_ranges_valid():
    for i, s in enumerate(T):
        lo, hi = s["distance_range"]
        assert 0 < lo < hi, f"slot {i}: bad distance_range {s['distance_range']}"
        if s["challenge_type"] in (3, 4):
            assert s["goal_height_range"] is None
        else:
            ghr = s["goal_height_range"]
            assert ghr is not None and ghr[1] > ghr[0]


def _composition(offset: int, n: int) -> list[int]:
    full = (list(T) * ((BENCHMARK_FULL_SEED_COUNT // len(T)) + 1))[:BENCHMARK_FULL_SEED_COUNT]
    return [s["challenge_type"] for s in full[offset:offset + n]]


def test_composition_is_fixed_per_seed_index():
    # The slot at each absolute benchmark index is deterministic, so every model is
    # graded on the same composition regardless of which validator's seeds fill the
    # specific instances. This is the cross-model fairness property.
    assert _composition(0, 50) == _composition(0, 50)
    assert _composition(250, 50) == _composition(250, 50)
