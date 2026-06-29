from swarm.validator.utils_parts import evaluation as ev
from swarm.constants import (
    BENCHMARK_SCREENING_SEED_COUNT,
    BENCHMARK_TEMPLATE,
    SCREENING_TEMPLATE,
)


def _capture(monkeypatch):
    seen = []

    def _fake_screening_task(*, sim_dt, seed, challenge_type, distance_range,
                             goal_height_range, moving_platform):
        slot = dict(challenge_type=challenge_type, distance_range=distance_range,
                    goal_height_range=goal_height_range, moving_platform=moving_platform)
        seen.append(slot)
        return ("task", seed)

    monkeypatch.setattr(ev, "screening_task", _fake_screening_task)
    return seen


def test_screening_range_uses_screening_template(monkeypatch):
    seen = _capture(monkeypatch)
    ev._task_for_seed_index(0, seed=111)
    assert seen[-1]["challenge_type"] == SCREENING_TEMPLATE[0]["challenge_type"]

    last = BENCHMARK_SCREENING_SEED_COUNT - 1
    ev._task_for_seed_index(last, seed=111)
    expected = SCREENING_TEMPLATE[last % len(SCREENING_TEMPLATE)]
    assert seen[-1]["challenge_type"] == expected["challenge_type"]
    assert seen[-1]["distance_range"] == expected["distance_range"]


def test_benchmark_boundary_switches_to_benchmark_template(monkeypatch):
    seen = _capture(monkeypatch)
    # exactly at the boundary -> first benchmark slot
    ev._task_for_seed_index(BENCHMARK_SCREENING_SEED_COUNT, seed=222)
    assert seen[-1]["challenge_type"] == BENCHMARK_TEMPLATE[0]["challenge_type"]
    assert seen[-1]["distance_range"] == BENCHMARK_TEMPLATE[0]["distance_range"]

    ev._task_for_seed_index(BENCHMARK_SCREENING_SEED_COUNT + 1, seed=222)
    assert seen[-1]["challenge_type"] == BENCHMARK_TEMPLATE[1]["challenge_type"]


def test_benchmark_index_wraps_modulo_template_length(monkeypatch):
    seen = _capture(monkeypatch)
    # one full benchmark-template cycle past the boundary maps back to slot 0
    abs_idx = BENCHMARK_SCREENING_SEED_COUNT + len(BENCHMARK_TEMPLATE)
    ev._task_for_seed_index(abs_idx, seed=333)
    assert seen[-1]["challenge_type"] == BENCHMARK_TEMPLATE[0]["challenge_type"]


def test_returns_none_when_task_build_fails(monkeypatch):
    def _boom(**_kwargs):
        raise RuntimeError("bad task")

    monkeypatch.setattr(ev, "screening_task", _boom)
    assert ev._task_for_seed_index(0, seed=1) is None
    assert ev._task_for_seed_index(BENCHMARK_SCREENING_SEED_COUNT, seed=1) is None
