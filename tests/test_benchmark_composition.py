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
