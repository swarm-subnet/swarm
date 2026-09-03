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

from swarm.challenge_families import build_benchmark_tasks


def _composition(family_id: str, seeds: list[int]) -> list[tuple[int, int]]:
    tasks = build_benchmark_tasks(
        sim_dt=0.02,
        seeds=seeds,
        family_id=family_id,
        offset=40,
        total_seed_count=800,
    )
    return [(task.challenge_type, task.num_drones) for task in tasks]


def test_benchmark_composition_is_fixed_per_absolute_index():
    seeds_a = list(range(1000, 1010))
    seeds_b = list(range(777000, 777010))

    for family_id in (
        "cf_autopilot",
        "cf_search_and_rescue",
        "cf_swarm_autopilot",
        "cf_swarm_sar",
        "cf_interceptor",
    ):
        assert _composition(family_id, seeds_a) == _composition(family_id, seeds_b)


def test_search_and_rescue_benchmark_challenge_sequence_is_template_driven():
    seeds_a = list(range(1000, 1010))
    seeds_b = list(range(777000, 777010))

    seq_a = [challenge_type for challenge_type, _n_drones in _composition("cf_search_and_rescue", seeds_a)]
    seq_b = [challenge_type for challenge_type, _n_drones in _composition("cf_search_and_rescue", seeds_b)]
    assert seq_a == seq_b
