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

from swarm.benchmark.engine_parts import dispatch, seeds
from swarm.domain_model import (
    BENCHMARK_GROUP_ORDER,
    CHALLENGE_TYPE_TO_BENCHMARK_GROUP,
)


def test_seed_group_inference_uses_domain_model_mappings():
    observed = {
        challenge_type: seeds._infer_bench_group(challenge_type, 123456)
        for challenge_type in CHALLENGE_TYPE_TO_BENCHMARK_GROUP
    }

    assert observed == dict(CHALLENGE_TYPE_TO_BENCHMARK_GROUP)


def test_dispatch_ram_estimates_cover_all_benchmark_groups():
    assert set(dispatch._GROUP_RAM_ESTIMATES_MB) == set(BENCHMARK_GROUP_ORDER)
