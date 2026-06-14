from __future__ import annotations

import inspect
import math
from typing import Any, Optional

import numpy as np
import pybullet as p

from swarm.constants import (
    SAR_CONFIRM_HORIZ_RADIUS,
    SAR_CONFIRM_SPEED_MAX,
    SAR_DWELL_SEC,
    SAR_HOVER_BAND,
    SAR_HYSTERESIS_GRACE,
    SAR_NO_TOUCH_RADIUS,
    SAR_SCREENING_TEMPLATE,
    SAR_SEARCH_RADIUS,
    SPEED_LIMIT,
)
from swarm.core.env_builder.sar_world import build_sar_world
from swarm.core.env_builder.spawn_pipeline import SARSpawnError
from swarm.domain_model import CHALLENGE_TYPE_TO_ENVIRONMENT_TYPE
from swarm.protocol import FailureReason, SCHEMA_VERSION
from swarm.validator.reward import (
    PARTICIPATION_REASONS,
    PARTICIPATION_REWARD,
    _calculate_safety_term,
    _calculate_sar_target_time,
    _clamp,
    calculate_time_term,
)

from .base import ChallengeFamilyRuntime, ChallengeFamilyRuntimeProfile


def _supports_keyword_arg(callable_obj: Any, keyword: str) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return True
    return keyword in signature.parameters


def _sar_drone_state(env: Any) -> tuple[Any, Any]:
    state = env._getDroneStateVector(0)
    return state[0:3], state[10:13]


class SearchAndRescueChallengeFamily(ChallengeFamilyRuntime):
    family_id = "cf_search_and_rescue"
    runtime_supported = True

    def runtime_profile(self, task: Any) -> ChallengeFamilyRuntimeProfile:
        _ = task
        return ChallengeFamilyRuntimeProfile(
            family_id=self.family_id,
            profile_name="search_and_rescue",
            resource_class="mission_search",
            image_key="base",
            env_bootstrap={"sar_mode": True},
            docker_env={
                "SWARM_CHALLENGE_FAMILY_ID": self.family_id,
                "SWARM_RUNTIME_PROFILE": "search_and_rescue",
                "SWARM_RUNTIME_RESOURCE_CLASS": "mission_search",
                "SWARM_RUNTIME_IMAGE_KEY": "base",
                "SWARM_RUNTIME_ENV_BOOTSTRAP": "sar_mode=true",
            },
            batch_timeout_multiplier=1.1,
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
        target_time = _calculate_sar_target_time(task) if task is not None else None
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
            participation_term = (
                PARTICIPATION_REWARD if failure_reason in PARTICIPATION_REASONS else 0.0
            )
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
        target_time = _calculate_sar_target_time(task) if task is not None else None
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
        return {"sar_mode": True}

    def state_clue_dim(self, task: Any) -> int:
        _ = task
        return 2

    def initialise_env_state(self, env: Any, *, requested_mode: bool = False) -> None:
        _ = requested_mode
        env.sar_mode = True
        env.sar_world = None
        self.reset_env_state(env)
        env._sar_spawn_attempts = 0
        env._search_area_center = env.GOAL_POS.copy()

    def reset_env_state(self, env: Any) -> None:
        env._sar_predicate_active = False
        env._sar_dwell_time = 0.0
        env._sar_spawn_failed = False
        env._sar_max_dwell_observed = 0.0
        env._sar_min_horizontal_distance = float("inf")
        env._sar_min_sphere_distance = float("inf")
        env.sar_world = None
        env._search_area_center = env.GOAL_POS.copy()

    def screening_template(self) -> tuple[dict[str, Any], ...]:
        return tuple(SAR_SCREENING_TEMPLATE)

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

    def protected_body_uids(self, env: Any) -> set[int]:
        return set(env.sar_world.victim_uids) if env.sar_world is not None else set()

    def safety_patch(self, env: Any) -> Any | None:
        if env.sar_world is None:
            return None
        return env.sar_world.safety_patch

    def spawn_task_world(self, env: Any) -> None:
        env.task.start = env._original_start
        env.task.goal = env._original_goal
        cli = getattr(env, "CLIENT", 0)

        try:
            env.sar_world = build_sar_world(
                cli=cli,
                seed=env.task.map_seed,
                challenge_type=env.task.challenge_type,
                start=env.task.start,
                goal=env.task.goal,
            )
        except SARSpawnError as exc:
            try:
                import bittensor as _bt

                _bt.logging.warning(
                    f"SAR spawn pipeline exhausted attempts for seed "
                    f"{env.task.map_seed} challenge_type "
                    f"{env.task.challenge_type}: {exc}"
                )
            except Exception:
                pass
            env.sar_world = None
            env._sar_spawn_failed = True
            env._failure_reason = FailureReason.SPAWN_FAILURE.value

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

        if env.sar_world is not None:
            sc = env.sar_world.search_centre or (0.0, 0.0)
            env._search_area_center = np.array(
                [float(sc[0]), float(sc[1]), 0.0],
                dtype=float,
            )
            try:
                env.task.search_centre = (float(sc[0]), float(sc[1]))
            except Exception:
                pass
        else:
            env._search_area_center = env.GOAL_POS.copy()

        env._build_cull_targets()

    def check_predicate(self, env: Any) -> bool:
        if env.sar_world is None:
            return False

        pos, vel = _sar_drone_state(env)
        vc = np.asarray(env.sar_world.victim_centre, dtype=float)
        _, v_max = env.sar_world.victim_aabb
        victim_top_z = float(v_max[2])

        horiz = float(np.linalg.norm(pos[0:2] - vc[0:2]))
        height_above = float(pos[2] - victim_top_z)
        speed = float(np.linalg.norm(vel))
        dist_3d = float(np.linalg.norm(pos - vc))

        if horiz < env._sar_min_horizontal_distance:
            env._sar_min_horizontal_distance = horiz
        if dist_3d < env._sar_min_sphere_distance:
            env._sar_min_sphere_distance = dist_3d

        grace = SAR_HYSTERESIS_GRACE if env._sar_predicate_active else 0.0
        horiz_max = SAR_CONFIRM_HORIZ_RADIUS + grace
        speed_max = SAR_CONFIRM_SPEED_MAX + grace
        band_lo = SAR_HOVER_BAND[0] - grace
        band_hi = SAR_HOVER_BAND[1] + grace

        if dist_3d < SAR_NO_TOUCH_RADIUS:
            return False
        if horiz > horiz_max:
            return False
        if speed > speed_max:
            return False
        if not (band_lo <= height_above <= band_hi):
            return False
        return True

    def post_step_update(self, env: Any) -> None:
        if env._sar_spawn_failed:
            if env._failure_reason == FailureReason.NONE.value:
                env._failure_reason = FailureReason.SPAWN_FAILURE.value
            return

        active = self.check_predicate(env)
        if active:
            env._sar_predicate_active = True
            env._sar_dwell_time += env._sim_dt
            if env._sar_dwell_time > env._sar_max_dwell_observed:
                env._sar_max_dwell_observed = env._sar_dwell_time
            if env._sar_dwell_time >= SAR_DWELL_SEC and not env._success:
                env._success = True
                env._t_to_goal = env._time_alive
                env._failure_reason = FailureReason.NONE.value
        else:
            env._sar_predicate_active = False
            env._sar_dwell_time = 0.0

    def compute_terminated(self, env: Any) -> bool:
        if env._sar_spawn_failed:
            return True
        if env.sar_world is not None:
            pos, _ = _sar_drone_state(env)
            vc = np.asarray(env.sar_world.victim_centre, dtype=float)
            if float(np.linalg.norm(pos - vc)) < SAR_NO_TOUCH_RADIUS:
                if not env._success:
                    env._failure_reason = FailureReason.NO_TOUCH_SPHERE.value
                return True

        if env._collision and not env._success:
            if env._failure_reason == FailureReason.NONE.value:
                env._failure_reason = FailureReason.OBSTACLE_COLLISION.value
        return False

    def infeasible(self, env: Any) -> bool:
        if env.sar_world is None:
            return False
        time_left = env.EP_LEN_SEC - env._time_alive
        if time_left <= 0:
            return False
        sc = env.sar_world.search_centre
        if sc is None:
            return False
        pos, _ = _sar_drone_state(env)
        dist_to_centre = float(np.linalg.norm(pos[0:2] - np.asarray(sc, dtype=float)))
        nearest_reachable = max(0.0, dist_to_centre - SAR_SEARCH_RADIUS)
        dwell_remaining = max(0.0, SAR_DWELL_SEC - env._sar_dwell_time)
        min_time_required = nearest_reachable / max(SPEED_LIMIT, 1e-6) + dwell_remaining
        return bool(time_left < min_time_required)

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

        if not terminal_already and self.infeasible(env):
            env._failure_reason = FailureReason.INFEASIBLE.value
            return True

        return False

    def build_info(self, env: Any) -> dict[str, Any]:
        return {
            "sar_min_horizontal_distance": float(env._sar_min_horizontal_distance),
            "sar_min_sphere_distance": float(env._sar_min_sphere_distance),
            "sar_max_dwell": float(env._sar_max_dwell_observed),
            "sar_spawn_attempts": int(env._sar_spawn_attempts),
            "t_to_confirm": (
                float(env._t_to_goal)
                if env._success and env._t_to_goal is not None
                else None
            ),
            "schema_version": SCHEMA_VERSION,
            "task_version": str(getattr(env.task, "version", "")),
        }

    def clue_offset(self, env: Any, state_vec: Any) -> Any:
        return env._search_area_center - state_vec[0:3]

    def legacy_sar_drone_state(self, env: Any) -> tuple[Any, Any]:
        return _sar_drone_state(env)

    def legacy_sar_check_predicate(self, env: Any) -> bool:
        return self.check_predicate(env)

    def legacy_sar_step_update(self, env: Any) -> None:
        self.post_step_update(env)

    def legacy_sar_infeasible(self, env: Any) -> bool:
        return self.infeasible(env)
