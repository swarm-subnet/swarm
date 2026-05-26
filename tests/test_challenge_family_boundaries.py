from __future__ import annotations

from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]


def _read(rel_path: str) -> str:
    return (_ROOT / rel_path).read_text(encoding="utf-8")


def test_env_factory_uses_family_dispatch_instead_of_version_switching():
    source = _read("swarm/utils/env_factory.py")

    assert "runtime_profile_for_task" in source
    assert "normalize_version" not in source
    assert 'startswith("5.")' not in source


def test_validator_evaluation_routes_task_building_through_family_registry():
    source = _read("swarm/validator/utils_parts/evaluation.py")

    assert "build_random_task" in source
    assert "build_screening_tasks" in source
    assert "SAR_SCREENING_TEMPLATE" not in source
    assert "from swarm.validator.task_gen import random_task, screening_task" not in source


def test_rpc_evaluator_routes_scoring_through_family_registry():
    source = _read("swarm/validator/docker/docker_evaluator_parts/rpc.py")

    assert "evaluate_rollout" in source
    assert "from swarm.validator.reward import flight_reward" not in source
    assert "flight_reward(" not in source


def test_moving_drone_generic_runtime_no_longer_contains_sar_spawn_or_mission_logic():
    source = _read("swarm/core/moving_drone.py")

    assert "build_sar_world" not in source
    assert "SARSpawnError" not in source
    assert "SAR_NO_TOUCH_RADIUS" not in source
    assert "SAR_DWELL_SEC" not in source
    assert "self.family_runtime" in source


def test_benchmark_runtime_uses_family_dispatch_for_seed_task_building():
    seeds_source = _read("swarm/benchmark/engine_parts/seeds.py")
    workers_source = _read("swarm/benchmark/engine_parts/workers.py")

    assert "build_random_task" in seeds_source
    assert "from swarm.validator.task_gen import random_task" not in seeds_source
    assert "build_random_task" in workers_source
    assert "from swarm.validator.task_gen import random_task" not in workers_source


def test_autopilot_family_is_not_a_runtime_placeholder():
    source = _read("swarm/challenge_families/autopilot.py")

    assert "runtime_not_implemented:cf_autopilot" not in source
    assert "runtime_supported = True" in source
    assert "def normalize_rollout_metrics" in source
    assert "flight_reward(" not in source


def test_search_and_rescue_family_owns_its_normalization_logic():
    source = _read("swarm/challenge_families/search_and_rescue.py")

    assert "def normalize_rollout_metrics" in source
    assert "flight_reward(" not in source


def test_moving_drone_routes_training_reward_through_family_runtime():
    source = _read("swarm/core/moving_drone.py")

    assert "compute_training_reward(" in source
