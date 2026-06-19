from __future__ import annotations

import random
from typing import Any

import numpy as np
import pybullet as p

from swarm.constants import (
    SAR_CONFIRM_HORIZ_RADIUS,
    SAR_CONFIRM_SPEED_MAX,
    SAR_DWELL_SEC,
    SAR_HOVER_BAND,
    SAR_HYSTERESIS_GRACE,
    SAR_NO_TOUCH_RADIUS,
    START_PLATFORM_TAKEOFF_BUFFER,
    SWARM_SAR_SEARCH_RADIUS,
)
from swarm.core.env_builder.body_tagger import BodyTagger
from swarm.core.env_builder.platform import build_start_platform, surface_z_at
from swarm.core.env_builder.sar_world import build_sar_world
from swarm.core.env_builder.search_clue import sample_search_centre
from swarm.core.env_builder.spawn_pipeline import SARSpawnError
from swarm.protocol import FailureReason, SCHEMA_VERSION
from swarm.validator.reward import (
    PARTICIPATION_REASONS,
    PARTICIPATION_REWARD,
    _calculate_safety_term,
    _calculate_swarm_sar_target_time,
    _clamp,
    calculate_time_term,
)

from .base import ChallengeFamilyRuntimeProfile
from .search_and_rescue import SearchAndRescueChallengeFamily


class SwarmSarChallengeFamily(SearchAndRescueChallengeFamily):
    """A swarm of 2-8 drones flown by one centralized policy, all searching for
    one shared victim over a much bigger search area. The team succeeds the
    moment any drone hover-confirms the victim; a crash zeroes the team safety
    term and entering the victim's no-touch sphere fails the whole mission.
    """

    family_id = "cf_swarm_sar"
    runtime_supported = True

    def runtime_profile(self, task: Any) -> ChallengeFamilyRuntimeProfile:
        _ = task
        return ChallengeFamilyRuntimeProfile(
            family_id=self.family_id,
            profile_name="swarm_search_and_rescue",
            resource_class="mission_search",
            image_key="base",
            env_bootstrap={"sar_mode": True},
            docker_env={
                "SWARM_CHALLENGE_FAMILY_ID": self.family_id,
                "SWARM_RUNTIME_PROFILE": "swarm_search_and_rescue",
                "SWARM_RUNTIME_RESOURCE_CLASS": "mission_search",
                "SWARM_RUNTIME_IMAGE_KEY": "base",
                "SWARM_RUNTIME_ENV_BOOTSTRAP": "sar_mode=true",
            },
            batch_timeout_multiplier=1.3,
        )

    def reset_env_state(self, env: Any) -> None:
        super().reset_env_state(env)
        n = int(getattr(env, "NUM_DRONES", 1))
        env._d_sar_predicate_active = [False] * n
        env._d_sar_dwell_time = [0.0] * n
        env._d_sar_min_horizontal = [float("inf")] * n
        env._d_sar_min_sphere = [float("inf")] * n
        env._sar_team_success = False
        env._sar_team_t = None
        env._sar_mission_failed = False
        env._sar_team_failure_reason = FailureReason.TIMEOUT.value

    def spawn_task_world(self, env: Any) -> None:
        cli = getattr(env, "CLIENT", 0)
        task = env.task
        starts = [tuple(float(c) for c in s) for s in task.starts]
        goals = [tuple(float(c) for c in g) for g in task.goals]
        n = len(starts)

        try:
            env.sar_world = build_sar_world(
                cli=cli,
                seed=task.map_seed,
                challenge_type=task.challenge_type,
                start=starts[0],
                goal=goals[0],
            )
        except SARSpawnError as exc:
            try:
                import bittensor as _bt

                _bt.logging.warning(
                    f"Swarm SAR spawn pipeline exhausted attempts for seed "
                    f"{task.map_seed} challenge_type {task.challenge_type}: {exc}"
                )
            except Exception:
                pass
            env.sar_world = None
            env._sar_spawn_failed = True
            env._failure_reason = FailureReason.SPAWN_FAILURE.value
            env._sar_team_failure_reason = FailureReason.SPAWN_FAILURE.value

        all_pad_uids: list = []
        adj_starts: list = []
        if env.sar_world is not None:
            tagger = BodyTagger(cli)
            for s in starts:
                sx, sy = float(s[0]), float(s[1])
                surf = surface_z_at(cli, sx, sy)
                pad_uids, pad_top = build_start_platform(
                    tagger, cli, sx, sy, surf, int(task.challenge_type),
                )
                all_pad_uids.extend(pad_uids)
                adj_starts.append((sx, sy, pad_top + START_PLATFORM_TAKEOFF_BUFFER))
        else:
            adj_starts = [tuple(float(c) for c in s) for s in starts]

        env._platform_uids = frozenset(all_pad_uids)
        env._swarm_adjusted_starts = tuple(adj_starts)

        quat = p.getQuaternionFromEuler([0.0, 0.0, 0.0])
        for i in range(n):
            p.resetBasePositionAndOrientation(
                env.DRONE_IDS[i], list(adj_starts[i]), quat, physicsClientId=cli,
            )

        if env.sar_world is not None:
            vc = np.asarray(env.sar_world.victim_centre, dtype=float)
            env.GOAL_POSES = np.tile(vc, (n, 1))
            env.GOAL_POS = vc.copy()
            # One shared, much bigger search clue centred on the victim.
            rng = random.Random((int(task.map_seed) ^ 0x53415253) & 0xFFFFFFFF)
            sc = sample_search_centre(
                rng, (float(vc[0]), float(vc[1])), radius=SWARM_SAR_SEARCH_RADIUS,
            )
            env.sar_world.search_centre = (float(sc[0]), float(sc[1]))
            env._search_area_center = np.array([float(sc[0]), float(sc[1]), 0.0], dtype=float)
            try:
                task.search_centre = (float(sc[0]), float(sc[1]))
            except Exception:
                pass
        else:
            base = np.asarray(adj_starts[0], dtype=float) if adj_starts else np.zeros(3, dtype=float)
            env.GOAL_POSES = np.tile(base, (n, 1))
            env.GOAL_POS = base.copy()
            env._search_area_center = base.copy()

        plane_id = getattr(env, "PLANE_ID", None)
        if plane_id is not None:
            if int(getattr(task, "challenge_type", 0)) == 2:
                p.resetBasePositionAndOrientation(
                    plane_id, [0.0, 0.0, -1000.0], [0.0, 0.0, 0.0, 1.0], physicsClientId=cli,
                )
            p.changeVisualShape(plane_id, -1, rgbaColor=[0, 0, 0, 0], physicsClientId=cli)

        env._build_cull_targets()

    def update_sar_dwell_multi(self, env: Any, nth_drone: int) -> None:
        """Per-drone SAR dwell predicate against the single shared victim.

        The team succeeds the first time any drone holds the hover-confirm
        predicate for SAR_DWELL_SEC; entering the no-touch sphere fails the
        whole mission.
        """
        if env.sar_world is None:
            return

        i = int(nth_drone)
        state = env._getDroneStateVector(i)
        pos = state[0:3]
        vel = state[10:13]
        vc = np.asarray(env.sar_world.victim_centre, dtype=float)
        _, v_max = env.sar_world.victim_aabb
        victim_top_z = float(v_max[2])

        horiz = float(np.linalg.norm(pos[0:2] - vc[0:2]))
        height_above = float(pos[2] - victim_top_z)
        speed = float(np.linalg.norm(vel))
        dist_3d = float(np.linalg.norm(pos - vc))

        if horiz < env._d_sar_min_horizontal[i]:
            env._d_sar_min_horizontal[i] = horiz
        if dist_3d < env._d_sar_min_sphere[i]:
            env._d_sar_min_sphere[i] = dist_3d

        if dist_3d < SAR_NO_TOUCH_RADIUS:
            # Any no-touch breach fails the whole mission, order-independent.
            env._sar_mission_failed = True
            env._sar_team_failure_reason = FailureReason.NO_TOUCH_SPHERE.value
            env._d_sar_predicate_active[i] = False
            env._d_sar_dwell_time[i] = 0.0
            return

        grace = SAR_HYSTERESIS_GRACE if env._d_sar_predicate_active[i] else 0.0
        horiz_max = SAR_CONFIRM_HORIZ_RADIUS + grace
        speed_max = SAR_CONFIRM_SPEED_MAX + grace
        band_lo = SAR_HOVER_BAND[0] - grace
        band_hi = SAR_HOVER_BAND[1] + grace

        active = (
            horiz <= horiz_max
            and speed <= speed_max
            and band_lo <= height_above <= band_hi
        )
        if active:
            env._d_sar_predicate_active[i] = True
            env._d_sar_dwell_time[i] += env._sim_dt
            if (
                env._d_sar_dwell_time[i] >= SAR_DWELL_SEC
                and not env._sar_team_success
                and not env._sar_mission_failed
            ):
                env._sar_team_success = True
                env._sar_team_t = env._time_alive
                env._d_success[i] = True
                env._d_t_to_goal[i] = env._time_alive
                env._d_failure_reason[i] = FailureReason.NONE.value
        else:
            env._d_sar_predicate_active[i] = False
            env._d_sar_dwell_time[i] = 0.0

    def compute_terminated(self, env: Any) -> bool:
        if getattr(env, "_sar_spawn_failed", False):
            return True
        if getattr(env, "_sar_mission_failed", False):
            return True
        if getattr(env, "_sar_team_success", False):
            return True
        return False

    def build_info(self, env: Any) -> dict[str, Any]:
        team_success = bool(
            getattr(env, "_sar_team_success", False)
            and not getattr(env, "_sar_mission_failed", False)
        )
        team_t = getattr(env, "_sar_team_t", None)
        return {
            "success": team_success,
            "sar_team_success": bool(getattr(env, "_sar_team_success", False)),
            "sar_team_t": (float(team_t) if team_t is not None else None),
            "sar_mission_failed": bool(getattr(env, "_sar_mission_failed", False)),
            "sar_team_failure_reason": str(
                getattr(env, "_sar_team_failure_reason", FailureReason.TIMEOUT.value)
            ),
            "swarm_adjusted_starts": [list(s) for s in getattr(env, "_swarm_adjusted_starts", ())],
            "sar_per_drone_min_horizontal": [
                float(x) for x in getattr(env, "_d_sar_min_horizontal", [])
            ],
            "sar_per_drone_min_sphere": [
                float(x) for x in getattr(env, "_d_sar_min_sphere", [])
            ],
            "schema_version": SCHEMA_VERSION,
            "task_version": str(getattr(env.task, "version", "")),
        }

    def score_swarm(self, task: Any, info: dict[str, Any]) -> dict[str, Any]:
        """Team-find SAR score: a shared success/time, with the safety term
        forced to zero if any drone crashed and averaged over the swarm otherwise."""
        col = info["per_drone_collision"]
        clr = info["per_drone_min_clearance"]
        n = len(col)
        team_success = bool(info.get("sar_team_success")) and not bool(info.get("sar_mission_failed"))

        if not team_success:
            reason = str(info.get("sar_team_failure_reason") or FailureReason.TIMEOUT.value)
            if reason == FailureReason.TIMEOUT.value:
                per_fr = info.get("per_drone_failure_reason", [])
                crashed = [
                    r for r in per_fr
                    if r and r not in (FailureReason.NONE.value, FailureReason.TIMEOUT.value)
                ]
                if per_fr and len(crashed) == len(per_fr):
                    reason = str(crashed[0])
            part = PARTICIPATION_REWARD if reason in PARTICIPATION_REASONS else 0.0
            return {
                "final_score": part,
                "per_drone_final_score": [part] * n,
                "success": False,
                "failure_reason": reason,
            }

        raw_starts = info.get("swarm_adjusted_starts") or task.starts
        starts = [tuple(float(c) for c in s) for s in raw_starts]
        team_t = info.get("sar_team_t")
        team_t = float(team_t) if team_t is not None else float(task.horizon)
        target_time = _calculate_swarm_sar_target_time(
            starts, getattr(task, "search_centre", None), n, SWARM_SAR_SEARCH_RADIUS,
        )
        time_term = calculate_time_term(t=team_t, horizon=float(task.horizon), target_time=target_time)

        if any(bool(c) for c in col):
            team_safety = 0.0
        else:
            ct = int(task.challenge_type)
            safeties = [
                _calculate_safety_term(float(c), collision=False, challenge_type=ct) for c in clr
            ]
            team_safety = sum(safeties) / len(safeties) if safeties else 1.0

        final = _clamp((0.45 * 1.0) + (0.45 * time_term) + (0.10 * team_safety))
        return {
            "final_score": final,
            "per_drone_final_score": [final] * n,
            "success": True,
            "failure_reason": FailureReason.NONE.value,
        }
