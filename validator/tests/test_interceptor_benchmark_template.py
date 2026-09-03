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
