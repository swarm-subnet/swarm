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

from __future__ import annotations

from typing import Any, Optional

from swarm.domain_model import CHALLENGE_FAMILY_IDS

from .autopilot import AutopilotChallengeFamily
from .base import (
    ChallengeFamilyEvaluation,
    ChallengeFamilyRuntimeProfile,
    ChallengeFamilyRuntime,
    ChallengeFamilyRuntimeError,
)
from .interceptor import InterceptorChallengeFamily
from .office_interceptor import OfficeInterceptorChallengeFamily
from .search_and_rescue import SearchAndRescueChallengeFamily
from .swarm_autopilot import SwarmAutopilotChallengeFamily
from .swarm_sar import SwarmSarChallengeFamily


DEFAULT_RUNTIME_FAMILY_ID = "cf_autopilot"


_REGISTERED_FAMILIES: dict[str, ChallengeFamilyRuntime] = {
    "cf_search_and_rescue": SearchAndRescueChallengeFamily(),
    "cf_autopilot": AutopilotChallengeFamily(),
    "cf_swarm_autopilot": SwarmAutopilotChallengeFamily(),
    "cf_swarm_sar": SwarmSarChallengeFamily(),
    "cf_interceptor": InterceptorChallengeFamily(),
    "cf_interceptor_office": OfficeInterceptorChallengeFamily(),
}


def list_registered_challenge_families() -> tuple[str, ...]:
    return tuple(_REGISTERED_FAMILIES)


def get_challenge_family(family_id: str) -> ChallengeFamilyRuntime:
    family = _REGISTERED_FAMILIES.get(family_id)
    if family is None:
        raise ChallengeFamilyRuntimeError(f"unknown_challenge_family:{family_id}")
    return family


def require_runtime_family(family_id: str) -> ChallengeFamilyRuntime:
    family = get_challenge_family(family_id)
    if not family.runtime_supported:
        raise ChallengeFamilyRuntimeError(f"runtime_not_implemented:{family_id}")
    return family


def infer_task_family_id(task: Any) -> str:
    family_id = getattr(task, "family_id", None)
    if isinstance(family_id, str) and family_id:
        return family_id
    return DEFAULT_RUNTIME_FAMILY_ID


def runtime_family_for_task(task: Any) -> ChallengeFamilyRuntime:
    return require_runtime_family(infer_task_family_id(task))


def build_random_task(
    *,
    sim_dt: float,
    seed: Optional[int] = None,
    family_id: str = DEFAULT_RUNTIME_FAMILY_ID,
) -> Any:
    return require_runtime_family(family_id).build_random_task(sim_dt=sim_dt, seed=seed)


def build_screening_tasks(
    *,
    sim_dt: float,
    seeds: list[int],
    family_id: str = DEFAULT_RUNTIME_FAMILY_ID,
    offset: int = 0,
    total_seed_count: Optional[int] = None,
) -> list[Any]:
    return require_runtime_family(family_id).build_screening_tasks(
        sim_dt=sim_dt,
        seeds=seeds,
        offset=offset,
        total_seed_count=total_seed_count,
    )


def build_benchmark_tasks(
    *,
    sim_dt: float,
    seeds: list[int],
    family_id: str = DEFAULT_RUNTIME_FAMILY_ID,
    offset: int = 0,
    total_seed_count: Optional[int] = None,
) -> list[Any]:
    return require_runtime_family(family_id).build_benchmark_tasks(
        sim_dt=sim_dt,
        seeds=seeds,
        offset=offset,
        total_seed_count=total_seed_count,
    )


def screening_policy_for_family(family_id: str) -> dict[str, Any]:
    return require_runtime_family(family_id).screening_policy()


def benchmark_admission_policy_for_family(family_id: str) -> dict[str, Any]:
    return require_runtime_family(family_id).benchmark_admission_policy()


def env_kwargs_for_task(task: Any) -> dict[str, Any]:
    return runtime_family_for_task(task).env_kwargs_for_task(task)


def runtime_profile_for_task(task: Any) -> ChallengeFamilyRuntimeProfile:
    return runtime_family_for_task(task).runtime_profile(task)


def runtime_profile_for_tasks(tasks: list[Any]) -> ChallengeFamilyRuntimeProfile:
    if not tasks:
        raise ChallengeFamilyRuntimeError("runtime_profile_requires_tasks")
    profile = runtime_profile_for_task(tasks[0])
    for task in tasks[1:]:
        candidate = runtime_profile_for_task(task)
        if candidate != profile:
            raise ChallengeFamilyRuntimeError(
                "mixed_challenge_family_runtime_profiles"
            )
    return profile


def evaluate_rollout(
    *,
    task: Any,
    success: bool,
    t: float,
    horizon: float,
    min_clearance: Optional[float],
    collision: bool,
    failure_reason: str,
) -> ChallengeFamilyEvaluation:
    family = runtime_family_for_task(task)
    return family.evaluate_rollout(
        task=task,
        success=success,
        t=t,
        horizon=horizon,
        min_clearance=min_clearance,
        collision=collision,
        failure_reason=failure_reason,
    )


__all__ = [
    "AutopilotChallengeFamily",
    "CHALLENGE_FAMILY_IDS",
    "ChallengeFamilyEvaluation",
    "ChallengeFamilyRuntimeProfile",
    "ChallengeFamilyRuntime",
    "ChallengeFamilyRuntimeError",
    "DEFAULT_RUNTIME_FAMILY_ID",
    "InterceptorChallengeFamily",
    "OfficeInterceptorChallengeFamily",
    "SearchAndRescueChallengeFamily",
    "SwarmAutopilotChallengeFamily",
    "SwarmSarChallengeFamily",
    "build_random_task",
    "benchmark_admission_policy_for_family",
    "build_benchmark_tasks",
    "build_screening_tasks",
    "env_kwargs_for_task",
    "evaluate_rollout",
    "get_challenge_family",
    "infer_task_family_id",
    "list_registered_challenge_families",
    "require_runtime_family",
    "screening_policy_for_family",
    "runtime_profile_for_task",
    "runtime_profile_for_tasks",
    "runtime_family_for_task",
]
