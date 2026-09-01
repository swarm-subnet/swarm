from collections import Counter

from swarm.challenge_families.search_and_rescue import _SAR_BENCHMARK_TEMPLATE as T
from swarm.constants import BENCHMARK_FULL_SEED_COUNT, SAR_SCREENING_TEMPLATE


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


def test_distance_ranges_valid_and_platforms_static():
    for s in T:
        lo, hi = s["distance_range"]
        assert 0 < lo < hi
        assert s["moving_platform"] is False


def test_first_six_cover_all_types():
    assert set(s["challenge_type"] for s in T[:6]) == VALID_TYPES


def test_benchmark_ranges_start_at_screening_lo_and_widen_hi():
    screening = {s["challenge_type"]: s["distance_range"] for s in SAR_SCREENING_TEMPLATE}
    bands = {ct: [] for ct in VALID_TYPES}
    for s in T:
        bands[s["challenge_type"]].append(s["distance_range"])
    for ct in VALID_TYPES:
        screening_lo, screening_hi = screening[ct]
        assert min(lo for lo, _hi in bands[ct]) == screening_lo
        assert max(hi for _lo, hi in bands[ct]) > screening_hi


def test_benchmark_template_is_distinct_from_screening_template():
    assert len(T) != len(SAR_SCREENING_TEMPLATE)
