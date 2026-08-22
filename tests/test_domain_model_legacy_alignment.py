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
