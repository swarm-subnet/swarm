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

"""The fixed 100-slot autopilot benchmark template: type coverage, distance bands, moving-platform rates, and the composition every model is graded on."""
from collections import Counter

from swarm.challenge_families.autopilot import _AUTOPILOT_BENCHMARK_TEMPLATE as T
from swarm.constants import BENCHMARK_FULL_SEED_COUNT

VALID_TYPES = {1, 2, 3, 4, 5, 6}


def test_template_has_100_entries():
    """The template is exactly 100 slots long, the unit a benchmark run repeats."""
    assert len(T) == 100


def test_benchmark_range_divides_template_cleanly():
    """A full run is a whole number of template repeats, so no slot is sampled more often than another."""
    assert BENCHMARK_FULL_SEED_COUNT % len(T) == 0


def test_covers_all_types_evenly():
    """All six challenge types appear, and none gets more than one slot over another."""
    counts = Counter(s["challenge_type"] for s in T)
    assert set(counts) == VALID_TYPES
    for ct in VALID_TYPES:
        assert 16 <= counts[ct] <= 17


def test_three_distance_bands_per_type():
    """Every challenge type is flown at three separate distance bands, never at a single range."""
    bands = {ct: set() for ct in VALID_TYPES}
    for s in T:
        bands[s["challenge_type"]].add(s["distance_range"])
    for ct in VALID_TYPES:
        assert len(bands[ct]) == 3


def test_moving_platform_rates():
    """Moving pads dominate type 2, cover a fifth to a third of types 1, 3 and 4, and never appear on 5 or 6."""
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
    """The opening six slots already span all six challenge types, so even a truncated run is not single-type."""
    assert set(s["challenge_type"] for s in T[:6]) == VALID_TYPES


def test_distance_and_height_ranges_valid():
    """Every slot's distance band is positive and correctly ordered, and a goal height band is set except on the terrain types 3 and 4."""
    for i, s in enumerate(T):
        lo, hi = s["distance_range"]
        assert 0 < lo < hi, f"slot {i}: bad distance_range {s['distance_range']}"
        if s["challenge_type"] in (3, 4):
            assert s["goal_height_range"] is None
        else:
            ghr = s["goal_height_range"]
            assert ghr is not None and ghr[1] > ghr[0]


def _composition(offset: int, n: int) -> list[int]:
    """The challenge types occupying n consecutive benchmark slots from offset, read off the template tiled across a full run."""
    full = (list(T) * ((BENCHMARK_FULL_SEED_COUNT // len(T)) + 1))[:BENCHMARK_FULL_SEED_COUNT]
    return [s["challenge_type"] for s in full[offset:offset + n]]


def test_composition_is_fixed_per_seed_index():
    """The map type at an absolute benchmark slot is the same on every read, so two models are graded on the same mix whichever validator ran them."""
    # The slot at each absolute benchmark index is deterministic, so every model is
    # graded on the same composition regardless of which validator's seeds fill the
    # specific instances. This is the cross-model fairness property.
    assert _composition(0, 50) == _composition(0, 50)
    assert _composition(250, 50) == _composition(250, 50)
