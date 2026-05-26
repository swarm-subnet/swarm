from __future__ import annotations

from types import SimpleNamespace

import pytest

from swarm.challenge_families import (
    ChallengeFamilyEvaluation,
    benchmark_admission_policy_for_family,
    build_random_task,
    build_screening_tasks,
    evaluate_rollout,
    get_challenge_family,
    infer_task_family_id,
    list_registered_challenge_families,
    require_runtime_family,
    screening_policy_for_family,
    runtime_profile_for_task,
    runtime_family_for_task,
)
from swarm.protocol import MapTask
from swarm.validator.reward import flight_reward


def _sample_task(*, family_id: str = "cf_search_and_rescue") -> MapTask:
    return MapTask(
        map_seed=7,
        start=(0.0, 0.0, 1.0),
        goal=(3.0, 4.0, 2.0),
        sim_dt=0.02,
        horizon=60.0,
        challenge_type=1,
        family_id=family_id,
    )


def test_registered_challenge_families_include_runtime_ids():
    family_ids = list_registered_challenge_families()

    assert "cf_search_and_rescue" in family_ids
    assert "cf_autopilot" in family_ids


def test_runtime_family_for_task_uses_family_id_dispatch():
    task = _sample_task()
    family = runtime_family_for_task(task)

    assert family.family_id == "cf_search_and_rescue"
    assert family.runtime_supported is True


def test_runtime_profile_for_task_routes_family_bootstrap_and_metadata():
    sar_profile = runtime_profile_for_task(_sample_task())
    autopilot_profile = runtime_profile_for_task(_sample_task(family_id="cf_autopilot"))

    assert sar_profile.family_id == "cf_search_and_rescue"
    assert sar_profile.profile_name == "search_and_rescue"
    assert sar_profile.resource_class == "mission_search"
    assert sar_profile.env_bootstrap == {"sar_mode": True}
    assert sar_profile.image_key == "base"

    assert autopilot_profile.family_id == "cf_autopilot"
    assert autopilot_profile.profile_name == "autopilot_navigation"
    assert autopilot_profile.resource_class == "navigation"
    assert autopilot_profile.env_bootstrap == {"sar_mode": False}
    assert autopilot_profile.image_key == "base"


def test_family_screening_and_admission_policies_are_runtime_scoped():
    autopilot_screening = screening_policy_for_family("cf_autopilot")
    sar_screening = screening_policy_for_family("cf_search_and_rescue")
    autopilot_admission = benchmark_admission_policy_for_family("cf_autopilot")
    sar_admission = benchmark_admission_policy_for_family("cf_search_and_rescue")

    assert autopilot_screening["bootstrap_threshold"] == pytest.approx(0.01)
    assert autopilot_screening["min_improvement"] == pytest.approx(0.015)
    assert autopilot_screening["early_fail_checkpoints"] == {
        "50": 0.5,
        "100": 0.7,
        "150": 0.85,
    }
    assert sar_screening["bootstrap_threshold"] == pytest.approx(0.02)
    assert sar_screening["min_improvement"] == pytest.approx(0.02)
    assert sar_screening["early_fail_checkpoints"] == {
        "50": 0.45,
        "100": 0.65,
        "150": 0.8,
    }
    assert autopilot_admission["champion_min_improvement"] == pytest.approx(0.015)
    assert sar_admission["champion_min_improvement"] == pytest.approx(0.02)


def test_infer_task_family_id_uses_legacy_version_fallback():
    legacy_task = MapTask(
        map_seed=11,
        start=(0.0, 0.0, 1.0),
        goal=(1.0, 1.0, 1.0),
        sim_dt=0.02,
        horizon=20.0,
        challenge_type=2,
        family_id="",
        version="5.0.0",
    )

    assert infer_task_family_id(legacy_task) == "cf_search_and_rescue"


def test_require_runtime_family_accepts_autopilot():
    family = require_runtime_family("cf_autopilot")

    assert family.family_id == "cf_autopilot"
    assert family.runtime_supported is True


def test_build_random_task_routes_through_registered_family():
    task = build_random_task(
        sim_dt=0.02,
        seed=12345,
        family_id="cf_search_and_rescue",
    )

    assert task.family_id == "cf_search_and_rescue"


def test_autopilot_random_task_generation_is_deterministic():
    family = require_runtime_family("cf_autopilot")

    first = family.build_random_task(sim_dt=0.02, seed=777)
    second = family.build_random_task(sim_dt=0.02, seed=777)

    assert first == second
    assert first.family_id == "cf_autopilot"


def test_search_and_rescue_random_task_generation_is_deterministic():
    family = require_runtime_family("cf_search_and_rescue")

    first = family.build_random_task(sim_dt=0.02, seed=555)
    second = family.build_random_task(sim_dt=0.02, seed=555)

    assert first == second


def test_build_screening_tasks_route_through_registered_family():
    tasks = build_screening_tasks(
        sim_dt=0.02,
        seeds=[101, 102, 103],
        family_id="cf_search_and_rescue",
        total_seed_count=10,
    )

    assert len(tasks) == 3
    assert all(task.family_id == "cf_search_and_rescue" for task in tasks)


def test_autopilot_screening_tasks_follow_family_template():
    family = get_challenge_family("cf_autopilot")
    template = list(family.screening_template())

    tasks = family.build_screening_tasks(
        sim_dt=0.02,
        seeds=[501, 502, 503],
        offset=2,
        total_seed_count=12,
    )

    assert [task.challenge_type for task in tasks] == [
        template[2]["challenge_type"],
        template[3]["challenge_type"],
        template[4]["challenge_type"],
    ]
    assert all(task.family_id == "cf_autopilot" for task in tasks)


def test_search_and_rescue_screening_tasks_follow_family_template():
    family = get_challenge_family("cf_search_and_rescue")
    template = list(family.screening_template())

    tasks = family.build_screening_tasks(
        sim_dt=0.02,
        seeds=[401, 402, 403],
        offset=4,
        total_seed_count=50,
    )

    assert [task.challenge_type for task in tasks] == [
        template[4]["challenge_type"],
        template[5]["challenge_type"],
        template[6]["challenge_type"],
    ]
    assert all(task.family_id == "cf_search_and_rescue" for task in tasks)


def test_evaluate_rollout_returns_common_evaluation_schema():
    evaluation = evaluate_rollout(
        task=_sample_task(),
        success=True,
        t=4.0,
        horizon=60.0,
        min_clearance=1.5,
        collision=False,
        legitimate_model=True,
        failure_reason="NONE",
    )

    assert isinstance(evaluation, ChallengeFamilyEvaluation)
    assert evaluation.family_id == "cf_search_and_rescue"
    assert evaluation.success is True
    assert evaluation.failure_reason == "NONE"
    assert 0.0 <= evaluation.score <= 1.0
    assert evaluation.metrics["challenge_type"] == 1
    assert evaluation.metrics["environment_type"] == "city"
    assert evaluation.metrics["time_sec"] == pytest.approx(4.0)
    assert evaluation.metrics["horizon_sec"] == pytest.approx(60.0)
    assert evaluation.metrics["min_clearance"] == pytest.approx(1.5)
    assert evaluation.metrics["collision"] is False
    assert evaluation.normalized_metrics["final_score"] == pytest.approx(evaluation.score)
    assert evaluation.normalized_metrics["success_term"] == pytest.approx(1.0)
    assert 0.0 <= evaluation.normalized_metrics["time_term"] <= 1.0
    assert 0.0 <= evaluation.normalized_metrics["safety_term"] <= 1.0


def test_search_and_rescue_score_evaluation_is_reproducible():
    task = _sample_task()

    first = evaluate_rollout(
        task=task,
        success=False,
        t=9.5,
        horizon=60.0,
        min_clearance=0.7,
        collision=False,
        legitimate_model=True,
        failure_reason="INFEASIBLE",
    )
    second = evaluate_rollout(
        task=task,
        success=False,
        t=9.5,
        horizon=60.0,
        min_clearance=0.7,
        collision=False,
        legitimate_model=True,
        failure_reason="INFEASIBLE",
    )

    assert first == second
    assert first.normalized_metrics["participation_term"] == pytest.approx(0.01)


def test_autopilot_evaluation_matches_legacy_navigation_reward_curve():
    task = _sample_task(family_id="cf_autopilot")

    evaluation = evaluate_rollout(
        task=task,
        success=True,
        t=4.0,
        horizon=60.0,
        min_clearance=1.5,
        collision=False,
        legitimate_model=True,
        failure_reason="NONE",
    )

    expected = flight_reward(
        success=True,
        t=4.0,
        horizon=60.0,
        task=task,
        min_clearance=1.5,
        collision=False,
        legitimate_model=True,
        failure_reason="NONE",
        sar_mode=False,
    )
    assert evaluation.family_id == "cf_autopilot"
    assert evaluation.score == pytest.approx(expected)
    assert evaluation.metrics["environment_type"] == "city"
    assert evaluation.normalized_metrics["final_score"] == pytest.approx(expected)


def test_autopilot_score_monotonicity_prefers_success_and_faster_time():
    task = _sample_task(family_id="cf_autopilot")

    fast_success = evaluate_rollout(
        task=task,
        success=True,
        t=4.0,
        horizon=60.0,
        min_clearance=1.0,
        collision=False,
        legitimate_model=True,
        failure_reason="NONE",
    )
    slow_success = evaluate_rollout(
        task=task,
        success=True,
        t=24.0,
        horizon=60.0,
        min_clearance=1.0,
        collision=False,
        legitimate_model=True,
        failure_reason="NONE",
    )
    failed_run = evaluate_rollout(
        task=task,
        success=False,
        t=24.0,
        horizon=60.0,
        min_clearance=1.0,
        collision=False,
        legitimate_model=True,
        failure_reason="TIMEOUT",
    )

    assert fast_success.score > slow_success.score
    assert slow_success.score > failed_run.score


def test_family_runtime_training_reward_defaults_to_score_delta():
    family = require_runtime_family("cf_autopilot")
    evaluation = evaluate_rollout(
        task=_sample_task(family_id="cf_autopilot"),
        success=True,
        t=4.0,
        horizon=60.0,
        min_clearance=1.0,
        collision=False,
        legitimate_model=True,
        failure_reason="NONE",
    )

    reward = family.compute_training_reward(
        env=SimpleNamespace(),
        evaluation=evaluation,
        previous_score=0.25,
    )

    assert reward == pytest.approx(evaluation.score - 0.25)


def test_family_normalization_is_isolated_between_autopilot_and_search_and_rescue():
    autopilot = evaluate_rollout(
        task=_sample_task(family_id="cf_autopilot"),
        success=False,
        t=8.0,
        horizon=60.0,
        min_clearance=1.0,
        collision=False,
        legitimate_model=True,
        failure_reason="NONE",
    )
    search_and_rescue = evaluate_rollout(
        task=_sample_task(family_id="cf_search_and_rescue"),
        success=False,
        t=8.0,
        horizon=60.0,
        min_clearance=1.0,
        collision=False,
        legitimate_model=True,
        failure_reason="NONE",
    )

    assert autopilot.score == pytest.approx(0.01)
    assert autopilot.normalized_metrics["participation_term"] == pytest.approx(0.01)
    assert search_and_rescue.score == pytest.approx(0.0)
    assert search_and_rescue.normalized_metrics["participation_term"] == pytest.approx(0.0)


def test_autopilot_runtime_sets_collision_failure_reason():
    family = require_runtime_family("cf_autopilot")
    env = SimpleNamespace(
        _collision=True,
        _success=False,
        _failure_reason="NONE",
    )

    assert family.compute_terminated(env) is False
    assert env._failure_reason == "OBSTACLE_COLLISION"


def test_autopilot_runtime_sets_truncation_failure_reasons():
    family = require_runtime_family("cf_autopilot")

    tilt_env = SimpleNamespace(
        MAX_TILT_RAD=1.0,
        _time_alive=1.0,
        EP_LEN_SEC=60.0,
        _failure_reason="NONE",
    )
    timeout_env = SimpleNamespace(
        MAX_TILT_RAD=1.0,
        _time_alive=60.0,
        EP_LEN_SEC=60.0,
        _failure_reason="NONE",
    )

    assert family.compute_truncated(
        tilt_env,
        terminal_already=False,
        roll=1.2,
        pitch=0.0,
    ) is True
    assert tilt_env._failure_reason == "TILT"

    assert family.compute_truncated(
        timeout_env,
        terminal_already=False,
        roll=0.0,
        pitch=0.0,
    ) is True
    assert timeout_env._failure_reason == "TIMEOUT"
