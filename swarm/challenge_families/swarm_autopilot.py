from __future__ import annotations

import math
import random
from typing import Any

import numpy as np
import pybullet as p

from swarm.constants import (
    SEARCH_AREA_NOISE_Z,
    SEARCH_RADIUS_MIN,
    START_PLATFORM_TAKEOFF_BUFFER,
    SWARM_SEARCH_RADIUS,
)
from swarm.core.env_builder.platform_placement import build_autopilot_world
from swarm.core.env_builder.sar_tagging import build_and_tag_map
from swarm.protocol import SCHEMA_VERSION
from swarm.validator.reward import _calculate_swarm_target_time, _score_single_drone

from .autopilot import AutopilotChallengeFamily
from .base import ChallengeFamilyRuntimeProfile


class SwarmAutopilotChallengeFamily(AutopilotChallengeFamily):
    """Five drones flown by one centralized policy to per-drone goal pads.

    Reuses the single-drone autopilot scoring per drone and reports the
    arithmetic mean. The simulator owns the per-drone resolve/freeze/termination
    mechanics; this family places the 5 pads and aggregates the score.
    """

    family_id = "cf_swarm_autopilot"
    runtime_supported = True

    def runtime_profile(self, task: Any) -> ChallengeFamilyRuntimeProfile:
        _ = task
        return ChallengeFamilyRuntimeProfile(
            family_id=self.family_id,
            profile_name="swarm_autopilot",
            resource_class="swarm_navigation",
            image_key="base",
            env_bootstrap={"sar_mode": False},
            docker_env={
                "SWARM_CHALLENGE_FAMILY_ID": self.family_id,
                "SWARM_RUNTIME_PROFILE": "swarm_autopilot",
                "SWARM_RUNTIME_RESOURCE_CLASS": "swarm_navigation",
                "SWARM_RUNTIME_IMAGE_KEY": "base",
                "SWARM_RUNTIME_ENV_BOOTSTRAP": "sar_mode=false",
            },
            batch_timeout_multiplier=1.3,
        )

    def env_kwargs_for_task(self, task: Any) -> dict[str, Any]:
        _ = task
        return {"sar_mode": False}

    def compute_terminated(self, env: Any) -> bool:
        # The simulator ends the episode when every drone is resolved (frozen).
        _ = env
        return False

    def build_info(self, env: Any) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "task_version": str(getattr(env.task, "version", "")),
            "swarm_adjusted_starts": [list(s) for s in getattr(env, "_swarm_adjusted_starts", ())],
            "swarm_adjusted_goals": [list(g) for g in getattr(env, "_swarm_adjusted_goals", ())],
        }

    def spawn_task_world(self, env: Any) -> None:
        cli = getattr(env, "CLIENT", 0)
        task = env.task
        starts = [tuple(float(c) for c in s) for s in task.starts]
        goals = [tuple(float(c) for c in g) for g in task.goals]
        n = len(starts)

        static_base = p.getNumBodies(physicsClientId=cli)
        tagger = build_and_tag_map(
            cli=cli,
            seed=task.map_seed,
            challenge_type=task.challenge_type,
            start=starts[0],
            goal=goals[0],
            safe_zone_radius=0.0,
            sar_mode=False,
        )

        all_start_uids: list = []
        all_end_uids: list = []
        per_drone_end: list = []
        adj_starts: list = []
        adj_goals: list = []

        for i in range(n):
            (
                end_uids,
                start_uids,
                start_surface_z,
                goal_surface_z,
                adj_start,
                adj_goal,
            ) = build_autopilot_world(
                tagger,
                cli,
                task.map_seed + i * 101,
                starts[i],
                goals[i],
                int(task.challenge_type),
                False,
                static_base,
            )

            s = list(adj_start) if adj_start is not None else list(starts[i])
            g = list(adj_goal) if adj_goal is not None else list(goals[i])
            if start_surface_z is not None:
                s = [s[0], s[1], start_surface_z + START_PLATFORM_TAKEOFF_BUFFER]
            if goal_surface_z is not None:
                g = [g[0], g[1], goal_surface_z]

            adj_starts.append(tuple(s))
            adj_goals.append(tuple(g))
            all_start_uids.extend(start_uids)
            all_end_uids.extend(end_uids)
            per_drone_end.append(frozenset(end_uids))

        env._swarm_platform_groups = list(per_drone_end)
        env._swarm_uid_to_group = {
            int(uid): gi for gi, grp in enumerate(per_drone_end) for uid in grp
        }
        env._end_platform_uids = list(all_end_uids)
        env._start_platform_uids = list(all_start_uids)
        env._platform_uids = frozenset(all_end_uids) | frozenset(all_start_uids)
        env._autopilot_world_tags = dict(tagger.body_tags)
        env._swarm_adjusted_starts = tuple(adj_starts)
        env._swarm_adjusted_goals = tuple(adj_goals)

        env.GOAL_POSES = np.asarray(adj_goals, dtype=float)
        env.GOAL_POS = env.GOAL_POSES[0]

        # One shared, bigger noisy search clue centred on the platform field
        # (same mechanism as single-drone autopilot, larger radius).
        centroid = env.GOAL_POSES.mean(axis=0)
        seed = int(task.map_seed)
        radius = random.Random((seed + 888888) & 0xFFFFFFFF).uniform(SEARCH_RADIUS_MIN, SWARM_SEARCH_RADIUS)
        noise_rng = np.random.RandomState(seed)
        noise_xy = noise_rng.uniform(-radius, radius, size=2)
        noise_z = noise_rng.uniform(-SEARCH_AREA_NOISE_Z, SEARCH_AREA_NOISE_Z)
        center = centroid.copy()
        center[0] += noise_xy[0]
        center[1] += noise_xy[1]
        center[2] = max(0.0, center[2] + noise_z)
        env._search_area_center = center

        # Shared single-use platform pool: a drone lands on any unclaimed platform.
        env._swarm_claimed = set()

        quat = p.getQuaternionFromEuler([0.0, 0.0, 0.0])
        for i in range(n):
            p.resetBasePositionAndOrientation(
                env.DRONE_IDS[i], list(adj_starts[i]), quat, physicsClientId=cli,
            )

        plane_id = getattr(env, "PLANE_ID", None)
        if plane_id is not None:
            if int(getattr(task, "challenge_type", 0)) == 2:
                p.resetBasePositionAndOrientation(
                    plane_id, [0.0, 0.0, -1000.0], [0.0, 0.0, 0.0, 1.0], physicsClientId=cli,
                )
            p.changeVisualShape(plane_id, -1, rgbaColor=[0, 0, 0, 0], physicsClientId=cli)

        env._build_cull_targets()

    def protected_body_uids(self, env: Any) -> set[int]:
        return set(getattr(env, "_platform_uids", frozenset()))

    def score_swarm(self, task: Any, info: dict[str, Any]) -> dict[str, Any]:
        """Aggregate per-drone outcomes into the mean autopilot score."""
        succ = info["per_drone_success"]
        t2g = info["per_drone_t_to_goal"]
        clr = info["per_drone_min_clearance"]
        col = info["per_drone_collision"]
        fr = info["per_drone_failure_reason"]
        raw_starts = info.get("swarm_adjusted_starts") or task.starts
        raw_goals = info.get("swarm_adjusted_goals") or task.goals
        starts = [tuple(float(c) for c in s) for s in raw_starts]
        goals = [tuple(float(c) for c in g) for g in raw_goals]
        n = len(succ)

        per_drone_score: list = []
        for i in range(n):
            nearest = min(
                goals, key=lambda g: math.hypot(g[0] - starts[i][0], g[1] - starts[i][1])
            )
            target_time = _calculate_swarm_target_time(starts[i], nearest, n - 1)
            t = float(t2g[i]) if (succ[i] and t2g[i] is not None) else float(task.horizon)
            per_drone_score.append(
                _score_single_drone(
                    success=bool(succ[i]),
                    t=t,
                    horizon=float(task.horizon),
                    target_time=target_time,
                    min_clearance=clr[i],
                    collision=bool(col[i]),
                    challenge_type=int(task.challenge_type),
                    legitimate_model=True,
                    failure_reason=str(fr[i]),
                )
            )

        final = sum(per_drone_score) / len(per_drone_score) if per_drone_score else 0.0
        return {"final_score": final, "per_drone_final_score": per_drone_score}
