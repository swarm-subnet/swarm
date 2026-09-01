from swarm.challenge_families import get_challenge_family
from swarm.constants import (
    BENCHMARK_FULL_SEED_COUNT,
    INTERCEPTOR_MAX_START_DISTANCE_M,
    INTERCEPTOR_MIN_START_DISTANCE_M,
)


def test_benchmark_template_has_fixed_open_map_bands():
    family = get_challenge_family("cf_interceptor")
    template = family.benchmark_template()

    assert len(template) == 100
    assert BENCHMARK_FULL_SEED_COUNT % len(template) == 0
    assert all(s["challenge_type"] == 2 for s in template)

    bands = sorted({s["distance_range"] for s in template})
    assert len(bands) == 3
    for lo, hi in bands:
        assert INTERCEPTOR_MIN_START_DISTANCE_M <= lo < hi <= INTERCEPTOR_MAX_START_DISTANCE_M
    for left, right in zip(bands, bands[1:]):
        assert left[1] == right[0]


def test_screening_template_uses_actual_interceptor_gap_range():
    family = get_challenge_family("cf_interceptor")
    template = family.screening_template()

    assert len(template) == 8
    assert all(s["challenge_type"] == 2 for s in template)
    assert all(
        s["distance_range"] == (INTERCEPTOR_MIN_START_DISTANCE_M, INTERCEPTOR_MAX_START_DISTANCE_M)
        for s in template
    )
