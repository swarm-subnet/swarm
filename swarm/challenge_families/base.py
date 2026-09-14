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

"""Base class every challenge family runtime subclasses, plus the helpers that lay out its seed templates."""

from __future__ import annotations

import inspect
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

from swarm.constants import BENCHMARK_FULL_SEED_COUNT, SWARM_MAX_DRONES, SWARM_MIN_DRONES
from swarm.domain_model import (
    get_family_benchmark_admission_policy,
    get_family_screening_policy,
    get_policy_interface_contract,
    get_supported_interface_versions,
)


def _supports_keyword_arg(callable_obj: Any, keyword: str) -> bool:
    """Return True when the callable accepts that parameter, and True as well when its signature cannot be read."""
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return True
    return keyword in signature.parameters


def banded_pool(
    challenge_type: int,
    distance: tuple[float, float],
    *,
    n_slots: int,
    n_bands: int,
    moving_prob: float,
    goal_height_range: Optional[tuple[float, float]] = None,
) -> list[dict[str, Any]]:
    """Lay out n_slots entries cycling through n_bands equal-width distance bands, the leading moving_prob share on moving platforms."""
    lo, hi = distance
    width = (hi - lo) / n_bands
    n_moving = round(n_slots * moving_prob)
    pool: list[dict[str, Any]] = []
    for i in range(n_slots):
        band = i % n_bands
        pool.append(dict(
            challenge_type=challenge_type,
            distance_range=(round(lo + band * width, 1), round(lo + (band + 1) * width, 1)),
            goal_height_range=goal_height_range,
            moving_platform=(i < n_moving),
        ))
    return pool


def interleave(pools: list[list[dict[str, Any]]], expected: int) -> tuple[dict[str, Any], ...]:
    """Round-robin the pools into a single slot tuple, raising RuntimeError when the total is not the expected count."""
    slots: list[dict[str, Any]] = []
    for i in range(max(len(p) for p in pools)):
        for pool in pools:
            if i < len(pool):
                slots.append(pool[i])
    if len(slots) != expected:
        raise RuntimeError(f"Template must have {expected} entries, got {len(slots)}")
    return tuple(slots)


def without_challenge_type(
    template: tuple[dict[str, Any], ...],
    challenge_type: int,
) -> tuple[dict[str, Any], ...]:
    """Drop every template slot that targets the given challenge type."""
    return tuple(
        slot for slot in template
        if int(slot.get("challenge_type", -1)) != int(challenge_type)
    )


def with_drone_counts(slots: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
    """Copy each slot with an n_drones value cycling from SWARM_MIN_DRONES up to SWARM_MAX_DRONES."""
    span = SWARM_MAX_DRONES - SWARM_MIN_DRONES + 1
    return tuple(dict(slot, n_drones=SWARM_MIN_DRONES + (i % span)) for i, slot in enumerate(slots))


class ChallengeFamilyRuntimeError(ValueError):
    """Raised when a challenge family runtime cannot service a request."""


@dataclass(frozen=True)
class ChallengeFamilyEvaluation:
    """One scored rollout: the outcome flag, the final score, the failure reason and both metric dicts."""

    family_id: str
    success: bool
    score: float
    failure_reason: str
    metrics: Dict[str, Any]
    normalized_metrics: Dict[str, float]


@dataclass(frozen=True)
class ChallengeFamilyRuntimeProfile:
    """The container image, worker environment and RPC timeout budget a family needs to fly one evaluation."""

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
        """Flatten the profile into a plain mapping, the form sent across a process or RPC boundary."""
        return asdict(self)

    @classmethod
    def from_mapping(cls, payload: Dict[str, Any]) -> "ChallengeFamilyRuntimeProfile":
        """Rebuild a profile from such a mapping, coercing each field and reading an empty string as unset."""
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
    """Hooks the environment and the evaluator call per family: world spawning, episode state, termination and scoring."""

    family_id: str
    runtime_supported: bool = True
    # Opt in to the seeded daylight sun (swarm.core.daylight); off keeps today's light.
    seeded_sun: bool = False
    # Depth camera backend: "tiny" is TinyRenderer, "raycast" the Embree ray caster in the same wheel.
    render_backend: str = "tiny"
    # Drone physics mode, a gym_pybullet_drones Physics value: "pyb" is rotor thrust only,
    # "pyb_gnd_drag_dw" adds the URDF's air drag, ground effect and downwash.
    physics_mode: str = "pyb"

    def screening_policy(self) -> Dict[str, Any]:
        """Registry thresholds that decide when a model survives screening: improvement floors and early-fail checkpoints."""
        return get_family_screening_policy(self.family_id)

    def benchmark_admission_policy(self) -> Dict[str, Any]:
        """Registry rules that decide whether a screened model earns a place in the full benchmark run."""
        return get_family_benchmark_admission_policy(self.family_id)

    def runtime_profile(self, task: Any) -> ChallengeFamilyRuntimeProfile:
        """Default container settings: the standard resource class, the base image, and the family id stamped into the worker env."""
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
        """Extra constructor arguments the environment needs for this family, empty where no special mode applies."""
        _ = task
        return {}

    def sky_colors(self, task: Any) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
        """Horizon and zenith colours painted where a colour camera sees nothing, or None
        for the plain white background every family renders today."""
        _ = task
        return None

    def observation_interface_version(self, task: Any) -> str:
        """First entry in the registry's supported version list, the contract a policy is held to by default."""
        _ = task
        return get_supported_interface_versions(self.family_id)[0]

    def observation_assembly(self, task: Any) -> dict[str, Any]:
        """Channel layout from the registry contract: which sensors are stacked, in what order, for the policy input."""
        contract = get_policy_interface_contract(
            self.family_id,
            self.observation_interface_version(task),
        )
        return contract["observation_assembly"]

    def state_clue_dim(self, task: Any) -> int:
        """How many trailing floats of the state vector carry the goal or search clue, three for a plain flight."""
        _ = task
        return 3

    def initialise_env_state(self, env: Any, *, requested_mode: bool = False) -> None:
        """Attach the family's own attributes to a freshly constructed env; a plain flight needs none."""
        _ = env, requested_mode

    def reset_env_state(self, env: Any) -> None:
        """Clear the family's per-episode attributes on reset, before the world is spawned again."""
        _ = env

    def spawn_task_world(self, env: Any) -> None:
        """Build this family's world on reset: the map itself, then its own bodies. Nothing by default."""
        _ = env

    def post_step_update(self, env: Any) -> None:
        """Refresh family-owned bookkeeping after the control step's physics has run, such as dwell timers."""
        _ = env

    def advance_world(self, env: Any) -> None:
        """Per control step, before physics: advance family-owned world entities
        (e.g. the interceptor target drone). No-op for families without one."""
        _ = env

    def apply_world_physics(self, env: Any) -> None:
        """Per PyBullet substep, before stepSimulation: apply forces to family-owned
        bodies (e.g. the interceptor target's rotors) or disturbance forces on the
        miner drones (e.g. the office VPS drift). No-op by default."""
        _ = env

    def protected_body_uids(self, env: Any) -> set[int]:
        """PyBullet uids excluded from the fatal-collision check, the obstacle cull and the clearance metric."""
        _ = env
        return set()

    def safety_patch(self, env: Any) -> Any | None:
        """A region on one support body whose contacts the clearance metric ignores, or None when the family declares none."""
        _ = env
        return None

    def compute_terminated(self, env: Any) -> bool:
        """Family-specific end of episode on top of the shared success and collision checks; never fires by default."""
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
        """Cut the episode short once roll or pitch exceeds MAX_TILT_RAD, or the flight reaches EP_LEN_SEC."""
        _ = terminal_already
        if abs(float(roll)) > float(env.MAX_TILT_RAD):
            return True
        if abs(float(pitch)) > float(env.MAX_TILT_RAD):
            return True
        return bool(env._time_alive >= env.EP_LEN_SEC)

    def build_info(self, env: Any) -> dict[str, Any]:
        """Extra fields merged into the env's per-step info dict, empty where the family logs nothing of its own."""
        _ = env
        return {}

    def screening_template(self) -> tuple[dict[str, Any], ...]:
        """Per-slot settings the screening seeds are cut from, empty until a family supplies its own."""
        return ()

    def benchmark_template(self) -> tuple[dict[str, Any], ...]:
        """Per-slot settings the benchmark seeds are cut from, empty where every seed is drawn at random instead."""
        return ()

    def build_random_task(self, *, sim_dt: float, seed: Optional[int]) -> Any:
        """Draw one freely sampled task for the given seed; unimplemented here, every family defines it."""
        raise NotImplementedError

    def _build_template_tasks(
        self,
        template: tuple[dict[str, Any], ...],
        *,
        sim_dt: float,
        seeds: list[int],
        offset: int,
        total_seed_count: Optional[int],
    ) -> list[Any]:
        """Repeat the template to cover total_seed_count, take the window starting at offset, and emit one task per seed."""
        from swarm.validator import task_gen as legacy_task_gen

        template = list(template)
        template_length = total_seed_count if total_seed_count is not None else len(seeds)
        full_template = (template * ((template_length // len(template)) + 1))[:template_length]
        template_slice = full_template[offset:offset + len(seeds)]

        tasks = []
        for seed, slot in zip(seeds, template_slice):
            kwargs = {
                "sim_dt": sim_dt,
                "seed": seed,
                "challenge_type": slot["challenge_type"],
                "distance_range": slot["distance_range"],
            }
            if _supports_keyword_arg(legacy_task_gen.screening_task, "family_id"):
                kwargs["family_id"] = self.family_id
            if slot.get("goal_height_range") is not None and _supports_keyword_arg(
                legacy_task_gen.screening_task, "goal_height_range"
            ):
                kwargs["goal_height_range"] = slot["goal_height_range"]
            if "moving_platform" in slot and _supports_keyword_arg(
                legacy_task_gen.screening_task, "moving_platform"
            ):
                kwargs["moving_platform"] = slot["moving_platform"]
            if slot.get("n_drones") is not None and _supports_keyword_arg(
                legacy_task_gen.screening_task, "n_drones"
            ):
                kwargs["n_drones"] = slot["n_drones"]
            tasks.append(legacy_task_gen.screening_task(**kwargs))
        return tasks

    def build_screening_tasks(
        self,
        *,
        sim_dt: float,
        seeds: list[int],
        offset: int = 0,
        total_seed_count: Optional[int] = None,
    ) -> list[Any]:
        """Cut the screening seeds from the family template, raising NotImplementedError when it declares none."""
        template = self.screening_template()
        if not template:
            raise NotImplementedError
        return self._build_template_tasks(
            template, sim_dt=sim_dt, seeds=seeds, offset=offset, total_seed_count=total_seed_count,
        )

    def build_benchmark_tasks(
        self,
        *,
        sim_dt: float,
        seeds: list[int],
        offset: int = 0,
        total_seed_count: Optional[int] = None,
    ) -> list[Any]:
        """Families without a benchmark template fall back to random per-seed tasks."""
        template = self.benchmark_template()
        if not template:
            return [self.build_random_task(sim_dt=sim_dt, seed=seed) for seed in seeds]
        return self._build_template_tasks(
            template,
            sim_dt=sim_dt,
            seeds=seeds,
            offset=offset,
            total_seed_count=total_seed_count if total_seed_count is not None else BENCHMARK_FULL_SEED_COUNT,
        )

    def evaluate_rollout(
        self,
        *,
        task: Any,
        success: bool,
        t: float,
        horizon: float,
        min_clearance: Optional[float],
        collision: bool,
        failure_reason: str,
    ) -> ChallengeFamilyEvaluation:
        """Score a finished flight: collect the raw metrics, normalize them, and package final_score with both dicts."""
        metrics = self.build_rollout_metrics(
            task=task,
            success=success,
            t=t,
            horizon=horizon,
            min_clearance=min_clearance,
            collision=collision,
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
        failure_reason: str,
    ) -> Dict[str, Any]:
        """Raw measurements the family records for one flight, before any weighting; unimplemented here."""
        raise NotImplementedError

    def normalize_rollout_metrics(
        self,
        *,
        task: Any,
        metrics: Dict[str, Any],
    ) -> Dict[str, float]:
        """Turn raw measurements into the family's weighted terms and the final_score they sum to; unimplemented here."""
        raise NotImplementedError

    def compute_training_reward(
        self,
        *,
        env: Any,
        evaluation: ChallengeFamilyEvaluation,
        previous_score: float,
    ) -> float:
        """Per-step reward as the gain in score since the last step, so a flat episode pays nothing."""
        _ = env
        return float(evaluation.score - previous_score)
