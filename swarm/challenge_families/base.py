from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

from swarm.domain_model import (
    get_family_benchmark_admission_policy,
    get_family_screening_policy,
)


class ChallengeFamilyRuntimeError(ValueError):
    """Raised when a challenge family runtime cannot service a request."""


@dataclass(frozen=True)
class ChallengeFamilyEvaluation:
    family_id: str
    success: bool
    score: float
    failure_reason: str
    metrics: Dict[str, Any]
    normalized_metrics: Dict[str, float]


@dataclass(frozen=True)
class ChallengeFamilyRuntimeProfile:
    family_id: str
    profile_name: str = "default"
    resource_class: str = "standard"
    image_key: str = "base"
    env_bootstrap: Dict[str, Any] = field(default_factory=dict)
    docker_env: Dict[str, str] = field(default_factory=dict)
    docker_worker_cpus: Optional[str] = None
    docker_worker_memory: Optional[str] = None
    rpc_ping_timeout_sec: Optional[float] = None
    rpc_reset_timeout_sec: Optional[float] = None
    rpc_first_step_timeout_sec: Optional[float] = None
    rpc_step_timeout_sec: Optional[float] = None
    global_eval_base_sec: Optional[float] = None
    global_eval_per_seed_sec: Optional[float] = None
    global_eval_cap_sec: Optional[float] = None
    batch_timeout_multiplier: float = 1.0

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, payload: Dict[str, Any]) -> "ChallengeFamilyRuntimeProfile":
        return cls(
            family_id=str(payload.get("family_id", "")),
            profile_name=str(payload.get("profile_name", "default")),
            resource_class=str(payload.get("resource_class", "standard")),
            image_key=str(payload.get("image_key", "base")),
            env_bootstrap=dict(payload.get("env_bootstrap", {}) or {}),
            docker_env={str(k): str(v) for k, v in dict(payload.get("docker_env", {}) or {}).items()},
            docker_worker_cpus=(
                None if payload.get("docker_worker_cpus") in (None, "") else str(payload.get("docker_worker_cpus"))
            ),
            docker_worker_memory=(
                None if payload.get("docker_worker_memory") in (None, "") else str(payload.get("docker_worker_memory"))
            ),
            rpc_ping_timeout_sec=(
                None if payload.get("rpc_ping_timeout_sec") is None else float(payload.get("rpc_ping_timeout_sec"))
            ),
            rpc_reset_timeout_sec=(
                None if payload.get("rpc_reset_timeout_sec") is None else float(payload.get("rpc_reset_timeout_sec"))
            ),
            rpc_first_step_timeout_sec=(
                None
                if payload.get("rpc_first_step_timeout_sec") is None
                else float(payload.get("rpc_first_step_timeout_sec"))
            ),
            rpc_step_timeout_sec=(
                None if payload.get("rpc_step_timeout_sec") is None else float(payload.get("rpc_step_timeout_sec"))
            ),
            global_eval_base_sec=(
                None if payload.get("global_eval_base_sec") is None else float(payload.get("global_eval_base_sec"))
            ),
            global_eval_per_seed_sec=(
                None
                if payload.get("global_eval_per_seed_sec") is None
                else float(payload.get("global_eval_per_seed_sec"))
            ),
            global_eval_cap_sec=(
                None if payload.get("global_eval_cap_sec") is None else float(payload.get("global_eval_cap_sec"))
            ),
            batch_timeout_multiplier=float(payload.get("batch_timeout_multiplier", 1.0)),
        )


class ChallengeFamilyRuntime:
    family_id: str
    runtime_supported: bool = True

    def screening_policy(self) -> Dict[str, Any]:
        return get_family_screening_policy(self.family_id)

    def benchmark_admission_policy(self) -> Dict[str, Any]:
        return get_family_benchmark_admission_policy(self.family_id)

    def runtime_profile(self, task: Any) -> ChallengeFamilyRuntimeProfile:
        return ChallengeFamilyRuntimeProfile(
            family_id=self.family_id,
            profile_name=self.family_id,
            resource_class="standard",
            image_key="base",
            env_bootstrap=dict(self.env_kwargs_for_task(task)),
            docker_env={
                "SWARM_CHALLENGE_FAMILY_ID": self.family_id,
                "SWARM_RUNTIME_PROFILE": self.family_id,
                "SWARM_RUNTIME_RESOURCE_CLASS": "standard",
                "SWARM_RUNTIME_IMAGE_KEY": "base",
            },
        )

    def env_kwargs_for_task(self, task: Any) -> dict[str, Any]:
        _ = task
        return {}

    def state_clue_dim(self, task: Any) -> int:
        _ = task
        return 3

    def initialise_env_state(self, env: Any, *, requested_mode: bool = False) -> None:
        _ = env, requested_mode

    def reset_env_state(self, env: Any) -> None:
        _ = env

    def spawn_task_world(self, env: Any) -> None:
        _ = env

    def post_step_update(self, env: Any) -> None:
        _ = env

    def protected_body_uids(self, env: Any) -> set[int]:
        _ = env
        return set()

    def safety_patch(self, env: Any) -> Any | None:
        _ = env
        return None

    def compute_terminated(self, env: Any) -> bool:
        _ = env
        return False

    def compute_truncated(
        self,
        env: Any,
        *,
        terminal_already: bool,
        roll: float,
        pitch: float,
    ) -> bool:
        _ = terminal_already
        if abs(float(roll)) > float(env.MAX_TILT_RAD):
            return True
        if abs(float(pitch)) > float(env.MAX_TILT_RAD):
            return True
        return bool(env._time_alive >= env.EP_LEN_SEC)

    def build_info(self, env: Any) -> dict[str, Any]:
        _ = env
        return {}

    def clue_offset(self, env: Any, state_vec: Any) -> Any:
        return env.GOAL_POS - state_vec[0:3]

    def screening_template(self) -> tuple[dict[str, Any], ...]:
        return ()

    def build_random_task(self, *, sim_dt: float, seed: Optional[int]) -> Any:
        raise NotImplementedError

    def build_screening_tasks(
        self,
        *,
        sim_dt: float,
        seeds: list[int],
        offset: int = 0,
        total_seed_count: Optional[int] = None,
    ) -> list[Any]:
        raise NotImplementedError

    def evaluate_rollout(
        self,
        *,
        task: Any,
        success: bool,
        t: float,
        horizon: float,
        min_clearance: Optional[float],
        collision: bool,
        legitimate_model: bool,
        failure_reason: str,
    ) -> ChallengeFamilyEvaluation:
        metrics = self.build_rollout_metrics(
            task=task,
            success=success,
            t=t,
            horizon=horizon,
            min_clearance=min_clearance,
            collision=collision,
            legitimate_model=legitimate_model,
            failure_reason=failure_reason,
        )
        normalized_metrics = self.normalize_rollout_metrics(task=task, metrics=metrics)
        score = float(normalized_metrics.get("final_score", 0.0))
        return ChallengeFamilyEvaluation(
            family_id=self.family_id,
            success=bool(success),
            score=score,
            failure_reason=str(failure_reason),
            metrics=metrics,
            normalized_metrics=normalized_metrics,
        )

    def build_rollout_metrics(
        self,
        *,
        task: Any,
        success: bool,
        t: float,
        horizon: float,
        min_clearance: Optional[float],
        collision: bool,
        legitimate_model: bool,
        failure_reason: str,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def normalize_rollout_metrics(
        self,
        *,
        task: Any,
        metrics: Dict[str, Any],
    ) -> Dict[str, float]:
        raise NotImplementedError

    def compute_training_reward(
        self,
        *,
        env: Any,
        evaluation: ChallengeFamilyEvaluation,
        previous_score: float,
    ) -> float:
        _ = env
        return float(evaluation.score - previous_score)
