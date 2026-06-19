"""Locks the existing single-drone behaviour against the swarm-autopilot work.

The 5-drone ``cf_swarm_autopilot`` family refactors ``reward.py`` (a shared
per-drone scorer) and ``task_gen.py`` (the deterministic pad draw). These
goldens fail loudly if ``cf_autopilot`` or ``cf_search_and_rescue`` scoring or
task generation drifts by a single value. Baseline lives in
``fixtures/swarm_single_drone_baseline.json``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from swarm.challenge_families.autopilot import AutopilotChallengeFamily
from swarm.protocol import MapTask
from swarm.validator import task_gen

_AP = AutopilotChallengeFamily()
_BASELINE = json.loads(
    (Path(__file__).parent / "fixtures" / "swarm_single_drone_baseline.json").read_text()
)


def _task(ct=1, start=(0.0, 0.0, 0.0), goal=(3.0, 4.0, 0.0), horizon=60.0) -> MapTask:
    return MapTask(
        map_seed=1, start=start, goal=goal, sim_dt=0.02,
        horizon=horizon, challenge_type=ct, family_id="cf_autopilot",
    )


_SCORING_CASES = {
    "illegit": dict(task=_task(), success=True, t=1.0, horizon=60.0,
                    min_clearance=1.0, collision=False, legitimate_model=False, failure_reason="NONE"),
    "eval_error": dict(task=_task(), success=False, t=1.0, horizon=60.0,
                       min_clearance=1.0, collision=False, legitimate_model=True, failure_reason="EVAL_ERROR"),
    "fail_participation": dict(task=_task(), success=False, t=2.0, horizon=60.0,
                               min_clearance=1.0, collision=False, legitimate_model=True, failure_reason="OBSTACLE_COLLISION"),
    "fail_t0": dict(task=_task(), success=False, t=0.0, horizon=60.0,
                    min_clearance=1.0, collision=False, legitimate_model=True, failure_reason="TIMEOUT"),
    "success_fast_safe": dict(task=_task(), success=True, t=0.1, horizon=60.0,
                              min_clearance=2.0, collision=False, legitimate_model=True, failure_reason="NONE"),
    "success_mid_clearance": dict(task=_task(), success=True, t=0.1, horizon=60.0,
                                  min_clearance=0.6, collision=False, legitimate_model=True, failure_reason="NONE"),
    "success_slow_time": dict(task=_task(start=(0.0, 0.0, 0.0), goal=(30.0, 0.0, 0.0)), success=True, t=40.0, horizon=60.0,
                              min_clearance=2.0, collision=False, legitimate_model=True, failure_reason="NONE"),
    "success_no_clearance": dict(task=_task(), success=True, t=0.1, horizon=60.0,
                                 min_clearance=None, collision=False, legitimate_model=True, failure_reason="NONE"),
    "success_with_collision": dict(task=_task(), success=True, t=1.0, horizon=60.0,
                                   min_clearance=2.0, collision=True, legitimate_model=True, failure_reason="NONE"),
}


@pytest.mark.parametrize("name", sorted(_SCORING_CASES))
def test_single_drone_scoring_unchanged(name):
    evaluation = _AP.evaluate_rollout(**_SCORING_CASES[name])
    expected = _BASELINE["scoring"][name]
    assert evaluation.score == expected["score"]
    assert evaluation.normalized_metrics == expected["normalized"]


def _taskgen_id(entry):
    return f"{entry['family_id']}-t{entry['challenge_type']}-s{entry['seed']}-{entry.get('via', 'type')}"


@pytest.mark.parametrize("entry", _BASELINE["taskgen"], ids=_taskgen_id)
def test_single_drone_task_gen_unchanged(entry):
    if entry.get("via") == "random_task":
        task = task_gen.random_task(sim_dt=0.02, seed=entry["seed"], family_id=entry["family_id"])
    else:
        task = task_gen.task_for_seed_and_type(
            sim_dt=0.02, seed=entry["seed"],
            challenge_type=entry["challenge_type"], family_id=entry["family_id"],
        )
    assert list(task.start) == entry["start"]
    assert list(task.goal) == entry["goal"]
    assert bool(task.moving_platform) == entry["moving_platform"]
    assert task.horizon == entry["horizon"]
    assert task.challenge_type == entry["challenge_type"]
