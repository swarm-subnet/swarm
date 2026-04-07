"""Training helpers for Swarm drone policies.

This module keeps the validator-facing observation contract unchanged:
the actor should still consume only ``{"depth", "state"}``.

For training, we expose extra simulator labels through ``info["privileged"]`` so
you can build:
- privileged teachers
- asymmetric critics
- perception auxiliary losses
- DAgger relabeling pipelines
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional

import gymnasium as gym
import numpy as np
import pybullet as p

# `swarm.protocol` may import bittensor, which pulls ansible defaults that
# expect a writable temp directory. Keep training-time imports self-contained.
_ANSIBLE_LOCAL_TEMP = Path(os.environ.get("ANSIBLE_LOCAL_TEMP", "/tmp/ansible"))
_ANSIBLE_LOCAL_TEMP.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("ANSIBLE_LOCAL_TEMP", str(_ANSIBLE_LOCAL_TEMP))

from swarm.protocol import MapTask
from swarm.constants import SIM_DT as VALIDATOR_SIM_DT
from swarm.utils.env_factory import make_env
from swarm.validator.task_gen import task_for_seed_and_type

# Match the validator/benchmark runtime by default.
DEFAULT_SIM_DT = VALIDATOR_SIM_DT
VALIDATOR_CTRL_FREQ = int(round(1.0 / VALIDATOR_SIM_DT))
FIXED_ACTION_DIM = 5
FIXED_ACTION_HISTORY_SLOTS = VALIDATOR_CTRL_FREQ // 2
FIXED_STATE_DIM = 12 + FIXED_ACTION_HISTORY_SLOTS * FIXED_ACTION_DIM + 1 + 3
CHALLENGE_TYPES = (1, 2, 3, 4, 5, 6)


def _assert_validator_sim_dt(sim_dt: float) -> None:
    if not np.isclose(float(sim_dt), float(VALIDATOR_SIM_DT)):
        raise ValueError(
            "The training pipeline is pinned to the validator timestep. "
            f"Expected sim_dt={VALIDATOR_SIM_DT}, got {sim_dt}."
        )


@dataclass(frozen=True)
class CurriculumStage:
    """A simple curriculum stage definition for task sampling."""

    name: str
    challenge_types: tuple[int, ...]
    moving_platform: Optional[bool] = None


CURRICULUM_STAGES: tuple[CurriculumStage, ...] = (
    CurriculumStage("city_static", (1,), moving_platform=False),
    CurriculumStage("city_dynamic", (1,), moving_platform=True),
    CurriculumStage("open_static", (2,), moving_platform=False),
    CurriculumStage("open_dynamic", (2,), moving_platform=True),
    CurriculumStage("mountain_static", (3,), moving_platform=False),
    CurriculumStage("mountain_dynamic", (3,), moving_platform=True),
    CurriculumStage("village_static", (4,), moving_platform=False),
    CurriculumStage("village_dynamic", (4,), moving_platform=True),
    CurriculumStage("warehouse_static", (5,), moving_platform=False),
    CurriculumStage("forest_static", (6,), moving_platform=False),
)


def build_task(
    seed: int,
    challenge_type: int,
    *,
    sim_dt: float = DEFAULT_SIM_DT,
    moving_platform: Optional[bool] = None,
) -> MapTask:
    """Build a single task with an explicit map type."""

    _assert_validator_sim_dt(sim_dt)
    return task_for_seed_and_type(
        sim_dt=sim_dt,
        seed=seed,
        challenge_type=challenge_type,
        moving_platform=moving_platform,
    )


def serialize_task(task: MapTask) -> dict[str, Any]:
    """Convert a task dataclass into a JSON-friendly payload."""

    payload = asdict(task)
    payload["start"] = list(payload["start"])
    payload["goal"] = list(payload["goal"])
    return payload


def task_from_payload(payload: Mapping[str, Any]) -> MapTask:
    """Rebuild a ``MapTask`` from a manifest payload."""

    sim_dt = float(payload["sim_dt"])
    _assert_validator_sim_dt(sim_dt)
    return MapTask(
        map_seed=int(payload["map_seed"]),
        start=tuple(float(v) for v in payload["start"]),
        goal=tuple(float(v) for v in payload["goal"]),
        sim_dt=sim_dt,
        horizon=float(payload["horizon"]),
        challenge_type=int(payload["challenge_type"]),
        search_radius=float(payload.get("search_radius", 10.0)),
        moving_platform=bool(payload.get("moving_platform", False)),
        version=str(payload.get("version", "1")),
    )


def iter_curriculum_tasks(
    *,
    seed_start: int,
    num_tasks: int,
    challenge_types: Iterable[int],
    sim_dt: float = DEFAULT_SIM_DT,
    moving_platform: Optional[bool] = None,
) -> Iterator[MapTask]:
    """Yield a deterministic, type-balanced stream of tasks."""

    types = tuple(int(t) for t in challenge_types)
    if not types:
        raise ValueError("challenge_types must contain at least one type")

    for idx in range(num_tasks):
        yield build_task(
            seed=seed_start + idx,
            challenge_type=types[idx % len(types)],
            sim_dt=sim_dt,
            moving_platform=moving_platform,
        )


def iter_stage_tasks(
    stage: CurriculumStage,
    *,
    seed_start: int,
    num_tasks: int,
    sim_dt: float = DEFAULT_SIM_DT,
) -> Iterator[MapTask]:
    """Yield tasks for a predefined curriculum stage."""

    yield from iter_curriculum_tasks(
        seed_start=seed_start,
        num_tasks=num_tasks,
        challenge_types=stage.challenge_types,
        sim_dt=sim_dt,
        moving_platform=stage.moving_platform,
    )


class PrivilegedTrainingWrapper(gym.Wrapper):
    """Attach privileged simulator labels to ``info`` for training only."""

    _CRUISE_ALTITUDE_OFFSETS_BY_TYPE: dict[int, tuple[float, ...]] = {
        1: (1.5, 3.0, 5.0, 7.0, 9.0),
        2: (1.0, 2.0, 3.0, 4.5, 6.0),
        3: (2.0, 4.0, 6.0, 8.0, 10.0),
        4: (1.5, 3.0, 5.0, 7.0, 9.0),
        5: (0.75, 1.5, 2.5, 3.5, 4.5),
        6: (1.5, 2.5, 4.0, 5.5, 7.0),
    }

    def __init__(self, env: gym.Env, *, raycast_stride_steps: int = 1):
        super().__init__(env)
        self.raycast_stride_steps = max(1, int(raycast_stride_steps))
        self._cached_heavy_fields: dict[str, Any] | None = None
        self._cached_heavy_step: int = -1

    def _segment_blocked(self, start: np.ndarray, end: np.ndarray) -> tuple[bool, np.ndarray | None]:
        env = self.env
        start = np.asarray(start, dtype=np.float32).reshape(3)
        end = np.asarray(end, dtype=np.float32).reshape(3)
        if hasattr(env, "_platform_path_blocked"):
            blocked, hit = env._platform_path_blocked(start, end)
            if hit is None:
                return bool(blocked), None
            return bool(blocked), np.asarray(hit, dtype=np.float32).reshape(3)

        cli = getattr(env, "CLIENT", 0)
        result = p.rayTest(start.tolist(), end.tolist(), physicsClientId=cli)
        if not result:
            return False, None
        body_uid = result[0][0]
        if body_uid == -1:
            return False, None
        return True, np.asarray(result[0][3], dtype=np.float32).reshape(3)

    def _plan_cruise_route(
        self,
        *,
        drone_pos: np.ndarray,
        platform_pos: np.ndarray,
        challenge_type: int,
        moving_platform: bool,
    ) -> dict[str, Any]:
        hover_height = 1.25 if moving_platform else 1.0
        hover_pos = np.asarray(platform_pos, dtype=np.float32).copy()
        hover_pos[2] += hover_height

        challenge_offsets = self._CRUISE_ALTITUDE_OFFSETS_BY_TYPE.get(
            int(challenge_type),
            (1.5, 3.0, 5.0, 7.0, 9.0),
        )
        base_altitude = max(float(drone_pos[2]), float(hover_pos[2]))
        candidate_zs = sorted(
            {
                round(base_altitude, 3),
                round(float(hover_pos[2]), 3),
                *[round(base_altitude + offset, 3) for offset in challenge_offsets],
            }
        )

        best_candidate: dict[str, Any] | None = None
        for cruise_z in candidate_zs:
            cruise_pos = np.array([drone_pos[0], drone_pos[1], cruise_z], dtype=np.float32)
            transit_pos = np.array([platform_pos[0], platform_pos[1], cruise_z], dtype=np.float32)
            blocked_up, hit_up = self._segment_blocked(drone_pos, cruise_pos)
            blocked_horiz, hit_horiz = self._segment_blocked(cruise_pos, transit_pos)
            blocked_down, hit_down = self._segment_blocked(transit_pos, hover_pos)
            blocked_count = int(blocked_up) + int(blocked_horiz) + int(blocked_down)
            cost = blocked_count * 1000.0 + abs(cruise_z - hover_pos[2]) + 0.05 * abs(cruise_z - drone_pos[2])
            candidate = {
                "cruise_z": float(cruise_z),
                "cruise_position": cruise_pos,
                "transit_position": transit_pos,
                "hover_position": hover_pos,
                "vertical_blocked": bool(blocked_up),
                "horizontal_blocked": bool(blocked_horiz),
                "descent_blocked": bool(blocked_down),
                "vertical_hit_point": hit_up,
                "horizontal_hit_point": hit_horiz,
                "descent_hit_point": hit_down,
                "blocked_count": blocked_count,
                "cost": float(cost),
            }
            if best_candidate is None or candidate["cost"] < best_candidate["cost"]:
                best_candidate = candidate
            if blocked_count == 0:
                break

        direct_clear, direct_hit = self._segment_blocked(drone_pos, hover_pos)
        planner = dict(best_candidate or {})
        planner["hover_height_m"] = hover_height
        planner["direct_to_hover_blocked"] = bool(direct_clear)
        planner["direct_to_hover_clear"] = not bool(direct_clear)
        planner["direct_hit_point"] = direct_hit
        return planner

    def _build_privileged_info(self, obs: dict) -> dict:
        state = np.asarray(obs["state"], dtype=np.float32)
        drone_pos = state[0:3]
        drone_rpy = state[3:6]
        drone_vel = state[6:9]

        goal_pos = np.asarray(getattr(self.env, "GOAL_POS"), dtype=np.float32).reshape(3)
        platform_pos = np.asarray(
            getattr(self.env, "_current_platform_pos", goal_pos),
            dtype=np.float32,
        ).reshape(3)
        platform_velocity = np.asarray(
            getattr(self.env, "_platform_velocity", np.zeros(3, dtype=np.float32)),
            dtype=np.float32,
        ).reshape(3)
        search_area_center = np.asarray(
            getattr(self.env, "_search_area_center", goal_pos),
            dtype=np.float32,
        ).reshape(3)
        adjusted_start = np.asarray(self.env.task.start, dtype=np.float32).reshape(3)
        adjusted_goal = np.asarray(self.env.task.goal, dtype=np.float32).reshape(3)
        challenge_type = int(getattr(self.env.task, "challenge_type", 0))
        moving_platform = bool(getattr(self.env, "_moving", False))
        rel_goal = goal_pos - drone_pos
        rel_platform = platform_pos - drone_pos
        rel_search_center = search_area_center - drone_pos
        step_counter = int(getattr(self.env, "step_counter", 0))
        refresh_heavy = (
            self._cached_heavy_fields is None
            or self._cached_heavy_step < 0
            or (step_counter - self._cached_heavy_step) >= self.raycast_stride_steps
        )
        if refresh_heavy:
            planner = self._plan_cruise_route(
                drone_pos=drone_pos,
                platform_pos=platform_pos,
                challenge_type=challenge_type,
                moving_platform=moving_platform,
            )
            goal_los_blocked, goal_los_hit = self._segment_blocked(drone_pos, goal_pos)
            platform_los_blocked, platform_los_hit = self._segment_blocked(drone_pos, platform_pos)
            search_los_blocked, search_los_hit = self._segment_blocked(drone_pos, search_area_center)
            self._cached_heavy_fields = {
                "planner": planner,
                "line_of_sight_to_goal": not bool(goal_los_blocked),
                "line_of_sight_to_platform": not bool(platform_los_blocked),
                "line_of_sight_to_search_center": not bool(search_los_blocked),
                "goal_los_hit_point": goal_los_hit,
                "platform_los_hit_point": platform_los_hit,
                "search_los_hit_point": search_los_hit,
            }
            self._cached_heavy_step = step_counter
        heavy = dict(self._cached_heavy_fields or {})

        privileged = {
            "map_seed": int(getattr(self.env.task, "map_seed", 0)),
            "challenge_type": challenge_type,
            "moving_platform": moving_platform,
            "horizon_sec": float(getattr(self.env.task, "horizon", 0.0)),
            "adjusted_start": adjusted_start,
            "adjusted_goal": adjusted_goal,
            "drone_position": drone_pos.astype(np.float32),
            "drone_rpy": drone_rpy.astype(np.float32),
            "drone_velocity": drone_vel.astype(np.float32),
            "goal_position": goal_pos,
            "platform_position": platform_pos,
            "platform_velocity": platform_velocity,
            "search_area_center": search_area_center,
            "relative_goal": rel_goal,
            "relative_platform": rel_platform,
            "relative_search_center": rel_search_center,
            "distance_to_goal": float(np.linalg.norm(rel_goal)),
            "distance_to_platform": float(np.linalg.norm(rel_platform)),
            "xy_distance_to_platform": float(np.linalg.norm(rel_platform[:2])),
            "z_error_to_platform": float(rel_platform[2]),
            "distance_to_search_center": float(np.linalg.norm(rel_search_center)),
            "line_of_sight_to_goal": bool(heavy.get("line_of_sight_to_goal", False)),
            "line_of_sight_to_platform": bool(heavy.get("line_of_sight_to_platform", False)),
            "line_of_sight_to_search_center": bool(heavy.get("line_of_sight_to_search_center", False)),
            "goal_los_hit_point": heavy.get("goal_los_hit_point"),
            "platform_los_hit_point": heavy.get("platform_los_hit_point"),
            "search_los_hit_point": heavy.get("search_los_hit_point"),
            "planner": heavy.get("planner", {}),
            "time_alive": float(getattr(self.env, "_time_alive", 0.0)),
            "landing_stable_time": float(getattr(self.env, "_landing_stable_time", 0.0)),
            "platform_contact": bool(getattr(self.env, "_platform_contact", False)),
            "ever_platform_contact": bool(getattr(self.env, "_ever_platform_contact", False)),
            "platform_contact_steps": int(getattr(self.env, "_platform_contact_steps", 0)),
            "success": bool(getattr(self.env, "_success", False)),
            "collision": bool(getattr(self.env, "_collision", False)),
        }

        teacher_state = np.concatenate(
            [
                privileged["relative_goal"],
                privileged["relative_platform"],
                privileged["platform_velocity"],
                privileged["relative_search_center"],
                np.array(
                    [
                        float(privileged["challenge_type"]),
                        float(privileged["moving_platform"]),
                        privileged["time_alive"],
                    ],
                    dtype=np.float32,
                ),
            ]
        ).astype(np.float32)
        privileged["teacher_state"] = teacher_state
        return privileged

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._cached_heavy_fields = None
        self._cached_heavy_step = -1
        info = dict(info)
        info["privileged"] = self._build_privileged_info(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["privileged"] = self._build_privileged_info(obs)
        return obs, reward, terminated, truncated, info


def make_training_env(
    task: MapTask,
    *,
    gui: bool = False,
    privileged: bool = True,
    raycast_stride_steps: int = 1,
):
    """Create an env suitable for imitation learning and RL training."""

    _assert_validator_sim_dt(float(task.sim_dt))
    env = make_env(task, gui=gui)
    if privileged:
        env = PrivilegedTrainingWrapper(env, raycast_stride_steps=raycast_stride_steps)
    return env


__all__ = [
    "CHALLENGE_TYPES",
    "CURRICULUM_STAGES",
    "CurriculumStage",
    "DEFAULT_SIM_DT",
    "FIXED_ACTION_DIM",
    "FIXED_ACTION_HISTORY_SLOTS",
    "FIXED_STATE_DIM",
    "PrivilegedTrainingWrapper",
    "VALIDATOR_CTRL_FREQ",
    "VALIDATOR_SIM_DT",
    "build_task",
    "iter_curriculum_tasks",
    "iter_stage_tasks",
    "make_training_env",
    "serialize_task",
    "task_from_payload",
]
