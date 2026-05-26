from __future__ import annotations

import inspect
from typing import Any, Optional

import numpy as np
import pybullet as p

from swarm.core.env_builder.sar_tagging import build_and_tag_map
from swarm.domain_model import CHALLENGE_TYPE_TO_ENVIRONMENT_TYPE
from swarm.protocol import FailureReason, SCHEMA_VERSION
from swarm.validator.reward import (
    PARTICIPATION_REWARD,
    _calculate_safety_term,
    _calculate_target_time,
    _clamp,
    calculate_time_term,
)

from .base import ChallengeFamilyRuntime, ChallengeFamilyRuntimeProfile


AUTOPILOT_GOAL_REACH_RADIUS_M = 1.0
_AUTOPILOT_SCREENING_TEMPLATE: tuple[dict[str, Any], ...] = (
    {"challenge_type": 1, "distance_range": (16.0, 26.0)},
    {"challenge_type": 2, "distance_range": (16.0, 24.0)},
    {"challenge_type": 3, "distance_range": (28.0, 52.0)},
    {"challenge_type": 4, "distance_range": (24.0, 42.0)},
    {"challenge_type": 5, "distance_range": (10.0, 20.0)},
    {"challenge_type": 6, "distance_range": (16.0, 28.0)},
)


def _supports_keyword_arg(callable_obj: Any, keyword: str) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return True
    return keyword in signature.parameters


def _navigation_distance_to_goal(env: Any) -> float:
    state = env._getDroneStateVector(0)
    return float(np.linalg.norm(state[0:3] - env.GOAL_POS))


class AutopilotChallengeFamily(ChallengeFamilyRuntime):
    family_id = "cf_autopilot"
    runtime_supported = True

    def runtime_profile(self, task: Any) -> ChallengeFamilyRuntimeProfile:
        _ = task
        return ChallengeFamilyRuntimeProfile(
            family_id=self.family_id,
            profile_name="autopilot_navigation",
            resource_class="navigation",
            image_key="base",
            env_bootstrap={"sar_mode": False},
            docker_env={
                "SWARM_CHALLENGE_FAMILY_ID": self.family_id,
                "SWARM_RUNTIME_PROFILE": "autopilot_navigation",
                "SWARM_RUNTIME_RESOURCE_CLASS": "navigation",
                "SWARM_RUNTIME_IMAGE_KEY": "base",
                "SWARM_RUNTIME_ENV_BOOTSTRAP": "sar_mode=false",
            },
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
    ) -> dict[str, Any]:
        challenge_type = int(getattr(task, "challenge_type", -1))
        target_time = _calculate_target_time(task) if task is not None else None
        return {
            "challenge_type": challenge_type,
            "environment_type": CHALLENGE_TYPE_TO_ENVIRONMENT_TYPE.get(
                challenge_type,
                "unknown",
            ),
            "time_sec": float(t),
            "horizon_sec": float(horizon),
            "target_time_sec": None if target_time is None else float(target_time),
            "min_clearance": None if min_clearance is None else float(min_clearance),
            "collision": bool(collision),
            "legitimate_model": bool(legitimate_model),
            "failure_reason": str(failure_reason),
            "success": bool(success),
        }

    def normalize_rollout_metrics(
        self,
        *,
        task: Any,
        metrics: dict[str, Any],
    ) -> dict[str, float]:
        horizon = float(metrics["horizon_sec"])
        if horizon <= 0.0:
            raise ValueError("'horizon' must be positive")

        success = bool(metrics["success"])
        collision = bool(metrics["collision"])
        legitimate_model = bool(metrics["legitimate_model"])
        failure_reason = str(metrics["failure_reason"])
        t = float(metrics["time_sec"])

        if not legitimate_model or failure_reason == FailureReason.EVAL_ERROR.value:
            return {
                "success_term": 0.0,
                "time_term": 0.0,
                "safety_term": 0.0,
                "participation_term": 0.0,
                "final_score": 0.0,
            }

        if not success:
            participation_term = PARTICIPATION_REWARD if t > 0.0 else 0.0
            return {
                "success_term": 0.0,
                "time_term": 0.0,
                "safety_term": 0.0,
                "participation_term": participation_term,
                "final_score": participation_term,
            }

        if collision:
            participation_term = PARTICIPATION_REWARD if t > 0.0 else 0.0
            return {
                "success_term": 0.0,
                "time_term": 0.0,
                "safety_term": 0.0,
                "participation_term": participation_term,
                "final_score": participation_term,
            }

        success_term = 1.0
        target_time = _calculate_target_time(task) if task is not None else None
        time_term = calculate_time_term(t=t, horizon=horizon, target_time=target_time)
        challenge_type = int(metrics["challenge_type"])
        min_clearance = metrics["min_clearance"]
        if min_clearance is not None:
            safety_term = _calculate_safety_term(
                float(min_clearance),
                collision=False,
                challenge_type=challenge_type,
            )
        else:
            safety_term = 1.0

        final_score = _clamp((0.45 * success_term) + (0.45 * time_term) + (0.10 * safety_term))
        return {
            "success_term": float(success_term),
            "time_term": float(time_term),
            "safety_term": float(safety_term),
            "participation_term": 0.0,
            "final_score": float(final_score),
        }

    def env_kwargs_for_task(self, task: Any) -> dict[str, Any]:
        _ = task
        return {"sar_mode": False}

    def initialise_env_state(self, env: Any, *, requested_mode: bool = False) -> None:
        _ = requested_mode
        env.sar_mode = False
        self.reset_env_state(env)

    def reset_env_state(self, env: Any) -> None:
        env._autopilot_goal_reach_radius_m = AUTOPILOT_GOAL_REACH_RADIUS_M
        env._autopilot_min_goal_distance = float(
            np.linalg.norm(np.asarray(env.task.start, dtype=float) - env.GOAL_POS)
        )
        env._autopilot_world_tags = {}

    def screening_template(self) -> tuple[dict[str, Any], ...]:
        return _AUTOPILOT_SCREENING_TEMPLATE

    def build_random_task(self, *, sim_dt: float, seed: Optional[int]) -> Any:
        from swarm.validator import task_gen as legacy_task_gen

        kwargs = {"sim_dt": sim_dt, "seed": seed}
        if _supports_keyword_arg(legacy_task_gen.random_task, "family_id"):
            kwargs["family_id"] = self.family_id
        return legacy_task_gen.random_task(**kwargs)

    def build_screening_tasks(
        self,
        *,
        sim_dt: float,
        seeds: list[int],
        offset: int = 0,
        total_seed_count: Optional[int] = None,
    ) -> list[Any]:
        from swarm.validator import task_gen as legacy_task_gen

        template = list(self.screening_template())
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
            tasks.append(legacy_task_gen.screening_task(**kwargs))
        return tasks

    def spawn_task_world(self, env: Any) -> None:
        env.task.start = env._original_start
        env.task.goal = env._original_goal
        cli = getattr(env, "CLIENT", 0)

        tagger = build_and_tag_map(
            cli=cli,
            seed=env.task.map_seed,
            challenge_type=env.task.challenge_type,
            start=env.task.start,
            goal=env.task.goal,
            sar_mode=False,
        )
        env._autopilot_world_tags = dict(tagger.body_tags)

        start_xyz = np.array(env.task.start, dtype=float)
        start_quat = p.getQuaternionFromEuler([0.0, 0.0, 0.0])
        p.resetBasePositionAndOrientation(
            env.DRONE_IDS[0],
            start_xyz.tolist(),
            start_quat,
            physicsClientId=cli,
        )

        plane_id = getattr(env, "PLANE_ID", None)
        if plane_id is not None:
            if int(getattr(env.task, "challenge_type", 0)) == 2:
                p.resetBasePositionAndOrientation(
                    plane_id,
                    [0.0, 0.0, -1000.0],
                    [0.0, 0.0, 0.0, 1.0],
                    physicsClientId=cli,
                )
            p.changeVisualShape(
                plane_id,
                -1,
                rgbaColor=[0, 0, 0, 0],
                physicsClientId=cli,
            )

        env._build_cull_targets()

    def post_step_update(self, env: Any) -> None:
        if env._collision:
            return

        distance_to_goal = _navigation_distance_to_goal(env)
        if distance_to_goal < env._autopilot_min_goal_distance:
            env._autopilot_min_goal_distance = distance_to_goal

        if distance_to_goal <= env._autopilot_goal_reach_radius_m and not env._success:
            env._success = True
            env._t_to_goal = env._time_alive
            env._failure_reason = FailureReason.NONE.value

    def compute_terminated(self, env: Any) -> bool:
        if env._collision and not env._success:
            if env._failure_reason == FailureReason.NONE.value:
                env._failure_reason = FailureReason.OBSTACLE_COLLISION.value
        return False

    def compute_truncated(
        self,
        env: Any,
        *,
        terminal_already: bool,
        roll: float,
        pitch: float,
    ) -> bool:
        if abs(float(roll)) > float(env.MAX_TILT_RAD):
            if not terminal_already:
                env._failure_reason = FailureReason.TILT.value
            return True
        if abs(float(pitch)) > float(env.MAX_TILT_RAD):
            if not terminal_already:
                env._failure_reason = FailureReason.TILT.value
            return True
        if env._time_alive >= env.EP_LEN_SEC:
            if not terminal_already:
                env._failure_reason = FailureReason.TIMEOUT.value
            return True
        return False

    def build_info(self, env: Any) -> dict[str, Any]:
        return {
            "autopilot_goal_reach_radius_m": float(env._autopilot_goal_reach_radius_m),
            "autopilot_min_goal_distance": float(env._autopilot_min_goal_distance),
            "schema_version": SCHEMA_VERSION,
            "task_version": str(getattr(env.task, "version", "")),
        }
