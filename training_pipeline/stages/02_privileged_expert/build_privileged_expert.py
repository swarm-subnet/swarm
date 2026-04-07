"""Build and evaluate the privileged expert teacher.

This stage is intentionally manifest-driven. It never resamples tasks outside
the Stage-01 curriculum artifacts.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import json
import multiprocessing as mp
import os
from pathlib import Path
import statistics
import sys
import threading
import time
import traceback
from typing import Any, Iterable

import numpy as np
from tqdm.auto import tqdm

DEFAULT_MODEL_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]
for root in (DEFAULT_MODEL_ROOT, REPO_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from training_env import make_training_env
from training_env import task_from_payload
from training_lib.common import ensure_dir, load_json, save_json, seed_everything
from training_lib.experts import (
    PrivilegedExpertConfig,
    ExpertIdentity,
    build_specialist_policy,
    evaluate_quality_gate,
    load_quality_gate,
    make_expert_policy,
    map_category_label,
    save_expert_config,
)
from progressive_curriculum import (
    PROGRESSIVE_STATIC_STAGES,
    ExpertCurriculumStage,
    curriculum_stage_payload,
    iter_curriculum_stage_tasks,
)


PROFILE_DEFAULTS = {
    # Keep manifest/task-list evaluation aligned with the real validator
    # runtime by default. Screening-style truncation is still available via
    # explicit --max-steps, but it should not be the built-in default.
    "debug": {"split": "train", "episodes_per_stage": 1, "max_steps": None},
    "validation": {"split": "val", "episodes_per_stage": 2, "max_steps": None},
    "full": {"split": "all", "episodes_per_stage": None, "max_steps": None},
}

CHALLENGE_TYPE_LABELS = {
    1: "city",
    2: "open",
    3: "mountain",
    4: "village",
    5: "warehouse",
    6: "forest",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curriculum-manifest", type=Path, default=DEFAULT_MODEL_ROOT / "artifacts" / "01_env_and_curriculum" / "curriculum_manifest.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_MODEL_ROOT / "artifacts" / "02_privileged_expert")
    parser.add_argument("--expert-registry", type=Path, default=None, help="Optional specialist registry for category-specific expert config selection.")
    parser.add_argument("--map-category", type=str, default=None, help="Required with --expert-registry.")
    parser.add_argument("--profile", type=str, choices=tuple(PROFILE_DEFAULTS.keys()), default="debug")
    parser.add_argument("--task-source", type=str, choices=("manifest", "custom_radius", "task_list"), default="manifest")
    parser.add_argument("--task-list-path", type=Path, default=None, help="JSON file with explicit task rows discovered for stage-02 curricula.")
    parser.add_argument("--split", type=str, choices=("train", "val", "test", "all"), default=None)
    parser.add_argument("--episodes-per-stage", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--stage-name", action="append", default=None, help="Repeatable stage filter.")
    parser.add_argument("--challenge-type", action="append", type=int, default=None, help="Repeatable challenge-type filter.")
    parser.add_argument("--map-seed", action="append", type=int, default=None, help="Repeatable exact map-seed filter.")
    parser.add_argument("--max-episodes-total", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of worker processes for episode evaluation. Use small values like 2-4.")
    parser.add_argument("--disable-depth-obstacle-checks", action="store_true", help="Disable depth-based obstacle avoidance and rely on privileged route hints.")
    parser.add_argument("--privileged-raycast-stride", type=int, default=1, help="Refresh expensive privileged ray-cast fields every N steps.")
    parser.add_argument("--progress-every", type=int, default=10, help="Print progress every N completed episodes.")
    parser.add_argument("--no-tqdm", action="store_true", help="Disable tqdm progress bars.")

    parser.add_argument("--cruise-speed-m-s", type=float, default=1.8)
    parser.add_argument("--climb-speed-m-s", type=float, default=1.0)
    parser.add_argument("--align-speed-m-s", type=float, default=0.6)
    parser.add_argument("--descend-speed-m-s", type=float, default=0.35)
    parser.add_argument("--touchdown-speed-m-s", type=float, default=0.18)
    parser.add_argument("--recovery-speed-m-s", type=float, default=0.7)
    parser.add_argument("--obstacle-distance-m", type=float, default=1.8)
    parser.add_argument("--static-hover-height-m", type=float, default=1.0)
    parser.add_argument("--moving-hover-height-m", type=float, default=1.25)
    parser.add_argument("--static-descend-xy-radius-m", type=float, default=0.8)
    parser.add_argument("--moving-descend-xy-radius-m", type=float, default=1.2)
    parser.add_argument("--static-touchdown-xy-radius-m", type=float, default=0.3)
    parser.add_argument("--moving-touchdown-xy-radius-m", type=float, default=1.0)
    parser.add_argument("--hover-z-tolerance-m", type=float, default=0.35)
    parser.add_argument("--static-rel-xy-speed-gate-m-s", type=float, default=0.35)
    parser.add_argument("--moving-rel-xy-speed-gate-m-s", type=float, default=0.85)
    parser.add_argument("--static-vertical-speed-gate-m-s", type=float, default=0.25)
    parser.add_argument("--moving-intercept-gain-sec-per-m", type=float, default=0.10)
    parser.add_argument("--stall-timeout-sec", type=float, default=2.5)
    parser.add_argument("--recovery-lateral-m", type=float, default=1.0)
    parser.add_argument("--recovery-climb-extra-m", type=float, default=1.0)

    parser.add_argument("--curriculum-stage", type=str, default=None, help="Named progressive curriculum stage. See progressive_curriculum.py.")
    parser.add_argument("--curriculum-seed-start", type=int, default=20000)
    parser.add_argument("--curriculum-num-episodes", type=int, default=8)
    parser.add_argument("--radius-min-m", type=float, default=None)
    parser.add_argument("--radius-max-m", type=float, default=None)
    parser.add_argument("--curriculum-moving-platform", action="store_true")
    parser.add_argument("--teacher-id", type=str, default="expert_shared")
    parser.add_argument("--teacher-version", type=str, default="v0")
    return parser.parse_args()


def resolve_eval_settings(args: argparse.Namespace) -> dict[str, Any]:
    defaults = PROFILE_DEFAULTS[args.profile]
    return {
        "profile": args.profile,
        "split": args.split or defaults["split"],
        "episodes_per_stage": args.episodes_per_stage if args.episodes_per_stage is not None else defaults["episodes_per_stage"],
        "max_steps": args.max_steps if args.max_steps is not None else defaults["max_steps"],
    }


def iter_manifest_tasks(
    manifest: dict[str, Any],
    *,
    split: str,
    episodes_per_stage: int | None,
    stage_names: set[str] | None,
    challenge_types: set[int] | None,
    max_episodes_total: int | None,
) -> Iterable[tuple[str, str, dict[str, Any]]]:
    emitted = 0
    split_names = ("train", "val", "test") if split == "all" else (split,)
    for stage in manifest["stages"]:
        stage_name = stage["stage_name"]
        if stage_names is not None and stage_name not in stage_names:
            continue
        for split_name in split_names:
            payloads = list(stage["splits"][split_name])
            if episodes_per_stage is not None:
                payloads = payloads[:episodes_per_stage]
            for payload in payloads:
                if challenge_types is not None and int(payload["challenge_type"]) not in challenge_types:
                    continue
                yield stage_name, split_name, payload
                emitted += 1
                if max_episodes_total is not None and emitted >= max_episodes_total:
                    return


def iter_task_list_tasks(task_list_payload: dict[str, Any]) -> Iterable[tuple[str, str, dict[str, Any]]]:
    for row in task_list_payload.get("tasks", []):
        yield (
            str(row.get("stage_name", "task_list")),
            str(row.get("split", "curriculum")),
            dict(row["task"]),
        )


def build_config(args: argparse.Namespace) -> PrivilegedExpertConfig:
    return PrivilegedExpertConfig(
        cruise_speed_m_s=args.cruise_speed_m_s,
        climb_speed_m_s=args.climb_speed_m_s,
        align_speed_m_s=args.align_speed_m_s,
        descend_speed_m_s=args.descend_speed_m_s,
        touchdown_speed_m_s=args.touchdown_speed_m_s,
        recovery_speed_m_s=args.recovery_speed_m_s,
        obstacle_distance_m=args.obstacle_distance_m,
        use_depth_obstacle_checks=not args.disable_depth_obstacle_checks,
        static_hover_height_m=args.static_hover_height_m,
        moving_hover_height_m=args.moving_hover_height_m,
        static_descend_xy_radius_m=args.static_descend_xy_radius_m,
        moving_descend_xy_radius_m=args.moving_descend_xy_radius_m,
        static_touchdown_xy_radius_m=args.static_touchdown_xy_radius_m,
        moving_touchdown_xy_radius_m=args.moving_touchdown_xy_radius_m,
        hover_z_tolerance_m=args.hover_z_tolerance_m,
        static_rel_xy_speed_gate_m_s=args.static_rel_xy_speed_gate_m_s,
        moving_rel_xy_speed_gate_m_s=args.moving_rel_xy_speed_gate_m_s,
        static_vertical_speed_gate_m_s=args.static_vertical_speed_gate_m_s,
        moving_intercept_gain_sec_per_m=args.moving_intercept_gain_sec_per_m,
        stall_timeout_sec=args.stall_timeout_sec,
        recovery_lateral_m=args.recovery_lateral_m,
        recovery_climb_extra_m=args.recovery_climb_extra_m,
    )


def filter_tasks(
    tasks: list[tuple[str, str, dict[str, Any]]],
    *,
    map_seeds: set[int] | None,
) -> list[tuple[str, str, dict[str, Any]]]:
    if not map_seeds:
        return list(tasks)
    return [row for row in tasks if int(row[2]["map_seed"]) in map_seeds]


def install_monitor_eof_suppression() -> None:
    original_hook = threading.excepthook

    def _hook(args: threading.ExceptHookArgs) -> None:
        if args.exc_type is EOFError:
            extracted = traceback.extract_tb(args.exc_traceback)
            if any(frame.name == "_monitor" and frame.filename.endswith("logging/handlers.py") for frame in extracted):
                return
        original_hook(args)

    threading.excepthook = _hook


def resolve_curriculum_stage(args: argparse.Namespace) -> ExpertCurriculumStage:
    if args.curriculum_stage:
        matches = [stage for stage in PROGRESSIVE_STATIC_STAGES if stage.name == args.curriculum_stage]
        if not matches:
            known = ", ".join(stage.name for stage in PROGRESSIVE_STATIC_STAGES)
            raise ValueError(f"Unknown curriculum stage {args.curriculum_stage!r}. Known: {known}")
        stage = matches[0]
        return ExpertCurriculumStage(
            name=stage.name,
            challenge_type=stage.challenge_type,
            radius_min_m=args.radius_min_m if args.radius_min_m is not None else stage.radius_min_m,
            radius_max_m=args.radius_max_m if args.radius_max_m is not None else stage.radius_max_m,
            moving_platform=args.curriculum_moving_platform if args.curriculum_moving_platform else stage.moving_platform,
            num_episodes=args.curriculum_num_episodes or stage.num_episodes,
            seed_start=args.curriculum_seed_start,
            goal_z_min_m=stage.goal_z_min_m,
            goal_z_max_m=stage.goal_z_max_m,
            goal_z_relative_to_start=stage.goal_z_relative_to_start,
        )

    challenge_types = list(args.challenge_type or [])
    if len(challenge_types) != 1:
        raise ValueError("custom_radius mode requires exactly one --challenge-type value")
    if args.radius_min_m is None or args.radius_max_m is None:
        raise ValueError("custom_radius mode requires --radius-min-m and --radius-max-m")
    return ExpertCurriculumStage(
        name=f"type_{challenge_types[0]}_r{args.radius_min_m:g}_{args.radius_max_m:g}_{'moving' if args.curriculum_moving_platform else 'static'}",
        challenge_type=int(challenge_types[0]),
        radius_min_m=float(args.radius_min_m),
        radius_max_m=float(args.radius_max_m),
        moving_platform=bool(args.curriculum_moving_platform),
        num_episodes=int(args.curriculum_num_episodes),
        seed_start=int(args.curriculum_seed_start),
    )


def median_or_zero(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True))
        handle.write("\n")


def challenge_type_label(challenge_type: int) -> str:
    return CHALLENGE_TYPE_LABELS.get(int(challenge_type), f"type_{challenge_type}")


def map_category_label(challenge_type: int, moving_platform: bool) -> str:
    motion = "dynamic" if bool(moving_platform) else "static"
    return f"{challenge_type_label(int(challenge_type))}_{motion}"


def infer_failure_bucket(row: dict[str, Any]) -> str:
    if row["success"]:
        return "success"
    if row["collision"]:
        if row["moving_platform"] and row.get("ever_platform_contact", False):
            return "contact_collision"
        return "collision"
    if row["termination_reason"] == "tilt_limit":
        return "tilt_limit"
    if row["termination_reason"] in {"timeout", "max_steps"}:
        if row["moving_platform"]:
            if row.get("ever_platform_contact", False):
                return "contact_timeout"
            if row["min_xy_distance_to_platform_m"] <= 0.30:
                return "missed_contact_timeout"
            if row["min_distance_to_platform_m"] <= 2.0:
                return "near_goal_timeout"
            if row["progress_ratio"] < 0.25:
                return "low_progress_timeout"
            return "far_timeout"
        if row["min_distance_to_platform_m"] <= 1.0 and row["max_landing_stable_time_sec"] > 0.0:
            return "unstable_touchdown"
        if row["min_distance_to_platform_m"] <= 2.0:
            return "near_goal_timeout"
        if row["progress_ratio"] < 0.25:
            return "low_progress_timeout"
        return "far_timeout"
    return "other"


def format_episode_summary(row: dict[str, Any], *, episode_number: int, total_episodes: int) -> str:
    kind = challenge_type_label(int(row["challenge_type"]))
    motion = "moving" if bool(row["moving_platform"]) else "static"
    result = "SUCCESS" if bool(row["success"]) else "FAIL"
    bucket = infer_failure_bucket(row)
    return (
        f"[episode {episode_number}/{total_episodes}] "
        f"seed={row['seed']} "
        f"type={kind} "
        f"motion={motion} "
        f"result={result} "
        f"reason={row['termination_reason']} "
        f"bucket={bucket} "
        f"steps={row['steps']} "
        f"score={row['score']:.3f} "
        f"final_dist={row['final_distance_to_platform_m']:.2f}m "
        f"min_xy={row['min_xy_distance_to_platform_m']:.2f}m "
        f"min_dist={row['min_distance_to_platform_m']:.2f}m "
        f"progress={row['progress_toward_platform_m']:.2f}m "
        f"stable={row['max_landing_stable_time_sec']:.2f}s"
    )


def run_episode(
    *,
    env,
    task,
    policy,
    episode_index: int,
    stage_name: str,
    split_name: str,
    seed: int,
    max_steps: int | None,
    use_tqdm: bool,
) -> dict[str, Any]:
    observation, info = env.reset(seed=seed)
    policy.reset()

    privileged = dict(info["privileged"])
    initial_distance = float(privileged["distance_to_platform"])
    initial_xy_distance = float(privileged["xy_distance_to_platform"])
    initial_goal_distance = float(privileged["distance_to_goal"])
    initial_pos = np.asarray(observation["state"][0:3], dtype=np.float32).copy()
    prev_pos = initial_pos.copy()
    prev_action: np.ndarray | None = None

    total_reward = 0.0
    path_length = 0.0
    min_distance = initial_distance
    min_xy_distance = initial_xy_distance
    mode_counts: Counter[str] = Counter()
    mode_switches = 0
    last_mode: str | None = None
    action_jerk_sum = 0.0
    action_jerk_max = 0.0
    line_of_sight_steps = 0
    max_landing_stable = float(info.get("landing_stable_time", 0.0))
    ever_platform_contact = bool(info.get("platform_contact", False) or info.get("ever_platform_contact", False))
    max_platform_contact_steps = int(info.get("platform_contact_steps", 0))
    steps = 0
    terminated = False
    truncated = False

    expected_steps = int(max(1, round(float(task.horizon) / float(task.sim_dt))))
    if max_steps is not None:
        expected_steps = min(expected_steps, int(max_steps))

    step_bar = None
    if use_tqdm:
        step_bar = tqdm(
            total=expected_steps,
            desc=f"episode {episode_index + 1} seed={seed}",
            position=1,
            leave=False,
            dynamic_ncols=True,
        )

    while not (terminated or truncated):
        if max_steps is not None and steps >= max_steps:
            break

        action = np.asarray(policy.act(observation, info), dtype=np.float32)
        metadata = dict(policy.get_last_metadata())
        mode = str(metadata.get("expert_mode", "unknown"))
        mode_counts[mode] += 1
        if last_mode is not None and mode != last_mode:
            mode_switches += 1
        last_mode = mode

        if prev_action is not None:
            jerk = float(np.linalg.norm(action - prev_action))
            action_jerk_sum += jerk
            action_jerk_max = max(action_jerk_max, jerk)
        prev_action = action.copy()

        expected_shape = tuple(getattr(env.action_space, "shape", ()))
        if expected_shape and action.shape != expected_shape:
            action = action.reshape(expected_shape)

        next_observation, reward, terminated, truncated, next_info = env.step(action)
        total_reward += float(reward)

        privileged = dict(next_info["privileged"])
        current_pos = np.asarray(next_observation["state"][0:3], dtype=np.float32)
        path_length += float(np.linalg.norm(current_pos - prev_pos))
        prev_pos = current_pos

        min_distance = min(min_distance, float(privileged["distance_to_platform"]))
        min_xy_distance = min(min_xy_distance, float(privileged["xy_distance_to_platform"]))
        line_of_sight_steps += int(bool(privileged["line_of_sight_to_platform"]))
        max_landing_stable = max(max_landing_stable, float(next_info.get("landing_stable_time", 0.0)))
        ever_platform_contact = ever_platform_contact or bool(
            next_info.get("platform_contact", False) or next_info.get("ever_platform_contact", False)
        )
        max_platform_contact_steps = max(max_platform_contact_steps, int(next_info.get("platform_contact_steps", 0)))

        observation = next_observation
        info = next_info
        steps += 1
        if step_bar is not None:
            step_bar.update(1)
            if steps == 1 or (steps % 10 == 0) or terminated or truncated:
                step_bar.set_postfix(
                    mode=mode,
                    dist=f"{float(privileged['distance_to_platform']):.2f}",
                    xy=f"{float(privileged['xy_distance_to_platform']):.2f}",
                    stable=f"{max_landing_stable:.2f}",
                )

    if step_bar is not None:
        step_bar.close()

    final_privileged = dict(info["privileged"])
    final_metadata = dict(policy.get_last_metadata())
    final_state = np.asarray(observation["state"], dtype=np.float32).reshape(-1)
    final_pos = final_state[0:3]
    final_rpy = final_state[3:6]
    final_vel = final_state[6:9]
    platform_vel = np.asarray(final_privileged["platform_velocity"], dtype=np.float32).reshape(3)
    rel_vxy = float(np.linalg.norm(final_vel[:2] - platform_vel[:2]))
    final_distance = float(final_privileged["distance_to_platform"])
    final_xy_distance = float(final_privileged["xy_distance_to_platform"])
    final_z_error = float(final_privileged["z_error_to_platform"])
    progress_m = initial_distance - final_distance
    progress_ratio = progress_m / max(initial_distance, 1e-6)
    path_efficiency = initial_distance / max(path_length, initial_distance, 1e-6)

    if bool(info.get("success", False)):
        termination_reason = "success"
    elif bool(info.get("collision", False)):
        termination_reason = "collision"
    elif max_steps is not None and steps >= max_steps and not (terminated or truncated):
        termination_reason = "max_steps"
    elif truncated:
        env_max_tilt = float(getattr(getattr(env, "unwrapped", env), "MAX_TILT_RAD", np.inf))
        if abs(float(final_rpy[0])) > env_max_tilt or abs(float(final_rpy[1])) > env_max_tilt:
            termination_reason = "tilt_limit"
        else:
            termination_reason = "timeout"
    else:
        termination_reason = "other"

    return {
        "episode_index": episode_index,
        "stage_name": stage_name,
        "split": split_name,
        "seed": int(seed),
        "challenge_type": int(final_privileged["challenge_type"]),
        "moving_platform": bool(final_privileged["moving_platform"]),
        "map_category": map_category_label(
            challenge_type=int(final_privileged["challenge_type"]),
            moving_platform=bool(final_privileged["moving_platform"]),
        ),
        "teacher_id": str(final_metadata.get("teacher_id", "expert_shared")),
        "teacher_version": str(final_metadata.get("teacher_version", "v0")),
        "success": bool(info.get("success", False)),
        "collision": bool(info.get("collision", False)),
        "ever_platform_contact": bool(ever_platform_contact),
        "termination_reason": termination_reason,
        "steps": int(steps),
        "total_reward": float(total_reward),
        "score": float(info.get("score", 0.0)),
        "initial_distance_to_platform_m": float(initial_distance),
        "initial_xy_distance_to_platform_m": float(initial_xy_distance),
        "initial_distance_to_goal_m": float(initial_goal_distance),
        "final_distance_to_platform_m": float(final_distance),
        "final_xy_distance_to_platform_m": float(final_xy_distance),
        "final_z_error_to_platform_m": float(final_z_error),
        "final_roll_rad": float(final_rpy[0]),
        "final_pitch_rad": float(final_rpy[1]),
        "min_distance_to_platform_m": float(min_distance),
        "min_xy_distance_to_platform_m": float(min_xy_distance),
        "progress_toward_platform_m": float(progress_m),
        "progress_ratio": float(progress_ratio),
        "path_length_m": float(path_length),
        "path_efficiency": float(path_efficiency),
        "line_of_sight_fraction": float(line_of_sight_steps / max(steps, 1)),
        "max_landing_stable_time_sec": float(max_landing_stable),
        "max_platform_contact_time_sec": float(max_platform_contact_steps) * float(task.sim_dt),
        "final_relative_xy_speed_m_s": float(rel_vxy),
        "final_vertical_speed_m_s": float(abs(final_vel[2])),
        "mean_action_jerk": float(action_jerk_sum / max(steps - 1, 1)),
        "max_action_jerk": float(action_jerk_max),
        "dominant_mode": mode_counts.most_common(1)[0][0] if mode_counts else "unknown",
        "mode_switches": int(mode_switches),
        "mode_counts": dict(sorted(mode_counts.items())),
    }


def run_episode_worker(
    *,
    payload: dict[str, Any],
    config_payload: dict[str, Any],
    episode_index: int,
    stage_name: str,
    split_name: str,
    max_steps: int | None,
    raycast_stride_steps: int,
) -> dict[str, Any]:
    install_monitor_eof_suppression()
    task = task_from_payload(payload)
    env = make_training_env(
        task,
        gui=False,
        privileged=True,
        raycast_stride_steps=raycast_stride_steps,
    )
    policy = make_expert_policy(PrivilegedExpertConfig(**config_payload))
    try:
        return run_episode(
            env=env,
            task=task,
            policy=policy,
            episode_index=episode_index,
            stage_name=stage_name,
            split_name=split_name,
            seed=task.map_seed,
            max_steps=max_steps,
            use_tqdm=False,
        )
    finally:
        env.close()


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    success_rate = mean_or_zero([float(row["success"]) for row in rows])
    collision_rate = mean_or_zero([float(row["collision"]) for row in rows])
    timeout_rate = mean_or_zero([float(row["termination_reason"] in {"timeout", "max_steps"}) for row in rows])
    mean_score = mean_or_zero([float(row["score"]) for row in rows])
    mean_progress_ratio = mean_or_zero([float(row["progress_ratio"]) for row in rows])
    selection_score = (
        0.55 * success_rate
        + 0.20 * mean_score
        + 0.15 * float(np.clip(mean_progress_ratio, 0.0, 1.0))
        + 0.05 * (1.0 - collision_rate)
        + 0.05 * (1.0 - timeout_rate)
    )
    success_rows = [row for row in rows if row["success"]]
    return {
        "num_episodes": len(rows),
        "success_rate": success_rate,
        "collision_rate": collision_rate,
        "timeout_rate": timeout_rate,
        "mean_score": mean_score,
        "median_score": median_or_zero([float(row["score"]) for row in rows]),
        "mean_final_distance_to_platform_m": mean_or_zero([float(row["final_distance_to_platform_m"]) for row in rows]),
        "median_final_distance_to_platform_m": median_or_zero([float(row["final_distance_to_platform_m"]) for row in rows]),
        "mean_progress_toward_platform_m": mean_or_zero([float(row["progress_toward_platform_m"]) for row in rows]),
        "mean_progress_ratio": mean_progress_ratio,
        "mean_path_efficiency": mean_or_zero([float(row["path_efficiency"]) for row in rows]),
        "mean_steps": mean_or_zero([float(row["steps"]) for row in rows]),
        "mean_steps_to_success": mean_or_zero([float(row["steps"]) for row in success_rows]),
        "mean_landing_stable_time_sec": mean_or_zero([float(row["max_landing_stable_time_sec"]) for row in rows]),
        "mean_action_jerk": mean_or_zero([float(row["mean_action_jerk"]) for row in rows]),
        "aggregate_selection_score": float(selection_score),
    }


def build_failure_buckets(rows: list[dict[str, Any]]) -> dict[str, Any]:
    overall = Counter()
    by_stage: dict[str, Counter[str]] = defaultdict(Counter)

    for row in rows:
        bucket = infer_failure_bucket(row)
        overall[bucket] += 1
        by_stage[row["stage_name"]][bucket] += 1

    return {
        "overall": dict(sorted(overall.items())),
        "by_stage": {stage: dict(sorted(counter.items())) for stage, counter in sorted(by_stage.items())},
    }


def maybe_save_comparison(
    *,
    output_dir: Path,
    baseline_summary: dict[str, Any] | None,
    candidate_summary: dict[str, Any],
) -> None:
    if baseline_summary is None:
        return
    comparison = {
        "baseline": baseline_summary["aggregate_metrics"],
        "candidate": candidate_summary["aggregate_metrics"],
        "deltas": {
            "success_rate": candidate_summary["aggregate_metrics"]["success_rate"] - baseline_summary["aggregate_metrics"]["success_rate"],
            "collision_rate": candidate_summary["aggregate_metrics"]["collision_rate"] - baseline_summary["aggregate_metrics"]["collision_rate"],
            "mean_score": candidate_summary["aggregate_metrics"]["mean_score"] - baseline_summary["aggregate_metrics"]["mean_score"],
            "mean_final_distance_to_platform_m": candidate_summary["aggregate_metrics"]["mean_final_distance_to_platform_m"] - baseline_summary["aggregate_metrics"]["mean_final_distance_to_platform_m"],
            "aggregate_selection_score": candidate_summary["aggregate_metrics"]["aggregate_selection_score"] - baseline_summary["aggregate_metrics"]["aggregate_selection_score"],
        },
    }
    save_json(output_dir / "expert_eval_comparison.json", comparison)


def build_map_category_report(by_map_category: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    report_rows: list[dict[str, Any]] = []
    for map_category, metrics in sorted(by_map_category.items()):
        report_rows.append(
            {
                "map_category": map_category,
                "num_episodes": int(metrics["num_episodes"]),
                "success_rate": float(metrics["success_rate"]),
                "mean_score": float(metrics["mean_score"]),
                "mean_final_distance_to_platform_m": float(metrics["mean_final_distance_to_platform_m"]),
                "mean_steps": float(metrics["mean_steps"]),
            }
        )
    return report_rows


def print_map_category_report(report_rows: list[dict[str, Any]]) -> None:
    if not report_rows:
        return
    print("Per-map-category report:")
    for row in report_rows:
        print(
            f"  {row['map_category']}: "
            f"episodes={row['num_episodes']} "
            f"success_rate={row['success_rate']:.3f} "
            f"mean_score={row['mean_score']:.3f} "
            f"mean_final_dist={row['mean_final_distance_to_platform_m']:.2f}m "
            f"mean_steps={row['mean_steps']:.1f}"
        )


def main() -> None:
    install_monitor_eof_suppression()
    args = parse_args()
    eval_settings = resolve_eval_settings(args)
    seed_everything(args.random_seed)
    ensure_dir(args.output_dir)
    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")
    if args.num_workers > 1 and args.gui:
        raise ValueError("--num-workers > 1 is not compatible with --gui")

    specialist_spec = None
    if args.expert_registry is not None:
        if not args.map_category:
            raise ValueError("--expert-registry requires --map-category")
        policy, specialist_spec = build_specialist_policy(
            registry_path=args.expert_registry,
            map_category=args.map_category,
        )
        config = policy.config
        expert_identity = asdict(policy.identity)
    else:
        config = build_config(args)
        policy = make_expert_policy(
            config,
            identity=ExpertIdentity(
                teacher_id=str(args.teacher_id),
                teacher_version=str(args.teacher_version),
                map_category=str(args.map_category or "global"),
            ),
        )
        expert_identity = asdict(policy.identity)
    save_expert_config(args.output_dir / "expert_config.json", config)
    save_json(args.output_dir / "expert_config_used.json", asdict(config))
    save_json(args.output_dir / "expert_config_expanded.json", asdict(config))
    save_json(args.output_dir / "expert_identity.json", expert_identity)
    wall_start = time.perf_counter()
    started_at_utc = datetime.now(timezone.utc).isoformat()
    stage_filter = set(args.stage_name) if args.stage_name else None
    challenge_filter = set(args.challenge_type) if args.challenge_type else None
    map_seed_filter = set(args.map_seed) if args.map_seed else None

    per_episode_rows: list[dict[str, Any]] = []
    partial_jsonl_path = args.output_dir / "expert_eval_per_episode.partial.jsonl"
    if partial_jsonl_path.exists():
        partial_jsonl_path.unlink()
    episode_index = 0
    source_payload: dict[str, Any]
    if args.task_source == "manifest":
        manifest = load_json(args.curriculum_manifest)
        tasks = list(
            iter_manifest_tasks(
                manifest,
                split=eval_settings["split"],
                episodes_per_stage=eval_settings["episodes_per_stage"],
                stage_names=stage_filter,
                challenge_types=challenge_filter,
                max_episodes_total=args.max_episodes_total,
            )
        )
        source_payload = {
            "task_source": "manifest",
            "curriculum_manifest": str(args.curriculum_manifest),
        }
    elif args.task_source == "custom_radius":
        curriculum_stage = resolve_curriculum_stage(args)
        tasks = list(iter_curriculum_stage_tasks(curriculum_stage))
        source_payload = {
            "task_source": "custom_radius",
            "curriculum_stage": curriculum_stage_payload(curriculum_stage),
        }
    else:
        if args.task_list_path is None:
            raise ValueError("--task-source task_list requires --task-list-path")
        task_list_payload = load_json(args.task_list_path)
        tasks = list(iter_task_list_tasks(task_list_payload))
        source_payload = {
            "task_source": "task_list",
            "task_list_path": str(args.task_list_path),
            "task_list_name": task_list_payload.get("name"),
            "selection_config": task_list_payload.get("selection_config"),
        }
    tasks = filter_tasks(tasks, map_seeds=map_seed_filter)
    if not tasks:
        raise ValueError("No manifest tasks matched the requested filters.")

    use_tqdm = (not args.no_tqdm) and sys.stderr.isatty()
    parallel_workers = min(int(args.num_workers), len(tasks))
    use_parallel = parallel_workers > 1
    episodes_bar = None
    if use_tqdm:
        episodes_bar = tqdm(
            total=len(tasks),
            desc="episodes",
            position=0,
            leave=True,
            dynamic_ncols=True,
        )
    if use_parallel:
        config_payload = asdict(config)
        max_workers = min(parallel_workers, max(1, min(4, os.cpu_count() or parallel_workers)))
        futures = {}
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context("spawn")) as executor:
            for episode_index, (stage_name, split_name, payload) in enumerate(tasks):
                futures[
                    executor.submit(
                        run_episode_worker,
                        payload=payload,
                        config_payload=config_payload,
                        episode_index=episode_index,
                        stage_name=stage_name,
                        split_name=split_name,
                        max_steps=eval_settings["max_steps"],
                        raycast_stride_steps=args.privileged_raycast_stride,
                    )
                ] = episode_index
            completed_rows = 0
            for future in as_completed(futures):
                row = future.result()
                per_episode_rows.append(row)
                append_jsonl(partial_jsonl_path, row)
                episode_summary = format_episode_summary(
                    row,
                    episode_number=int(row["episode_index"]) + 1,
                    total_episodes=len(tasks),
                )
                if use_tqdm:
                    tqdm.write(episode_summary)
                else:
                    print(episode_summary)
                completed_rows += 1
                if episodes_bar is not None:
                    success_count = sum(int(item["success"]) for item in per_episode_rows)
                    collision_count = sum(int(item["collision"]) for item in per_episode_rows)
                    episodes_bar.update(1)
                    episodes_bar.set_postfix(
                        success=success_count,
                        collisions=collision_count,
                        last_seed=row["seed"],
                        last_reason=row["termination_reason"],
                        workers=max_workers,
                    )
                if args.progress_every > 0 and (completed_rows % args.progress_every == 0 or completed_rows == len(tasks)):
                    success_count = sum(int(item["success"]) for item in per_episode_rows)
                    collision_count = sum(int(item["collision"]) for item in per_episode_rows)
                    print(
                        f"[{completed_rows}/{len(tasks)}] "
                        f"successes={success_count} "
                        f"collisions={collision_count} "
                        f"latest_seed={row['seed']} "
                        f"latest_reason={row['termination_reason']}"
                    )
        per_episode_rows.sort(key=lambda row: int(row["episode_index"]))
    else:
        for episode_index, (stage_name, split_name, payload) in enumerate(tasks):
            task = task_from_payload(payload)
            env = make_training_env(
                task,
                gui=args.gui,
                privileged=True,
                raycast_stride_steps=args.privileged_raycast_stride,
            )
            try:
                row = run_episode(
                    env=env,
                    task=task,
                    policy=policy,
                    episode_index=episode_index,
                    stage_name=stage_name,
                    split_name=split_name,
                    seed=task.map_seed,
                    max_steps=eval_settings["max_steps"],
                    use_tqdm=use_tqdm,
                )
            finally:
                env.close()
            per_episode_rows.append(row)
            append_jsonl(partial_jsonl_path, row)
            episode_summary = format_episode_summary(
                row,
                episode_number=episode_index + 1,
                total_episodes=len(tasks),
            )
            if use_tqdm:
                tqdm.write(episode_summary)
            else:
                print(episode_summary)
            if episodes_bar is not None:
                success_count = sum(int(item["success"]) for item in per_episode_rows)
                collision_count = sum(int(item["collision"]) for item in per_episode_rows)
                episodes_bar.update(1)
                episodes_bar.set_postfix(
                    success=success_count,
                    collisions=collision_count,
                    last_seed=row["seed"],
                    last_reason=row["termination_reason"],
                )
            if args.progress_every > 0 and ((episode_index + 1) % args.progress_every == 0 or (episode_index + 1) == len(tasks)):
                success_count = sum(int(item["success"]) for item in per_episode_rows)
                collision_count = sum(int(item["collision"]) for item in per_episode_rows)
                print(
                    f"[{episode_index + 1}/{len(tasks)}] "
                    f"successes={success_count} "
                    f"collisions={collision_count} "
                    f"latest_seed={row['seed']} "
                    f"latest_reason={row['termination_reason']}"
                )

    if episodes_bar is not None:
        episodes_bar.close()

    aggregate_metrics = summarize_group(per_episode_rows)
    by_stage = {
        stage_name: summarize_group([row for row in per_episode_rows if row["stage_name"] == stage_name])
        for stage_name in sorted({row["stage_name"] for row in per_episode_rows})
    }
    by_challenge_type = {
        str(challenge_type): summarize_group([row for row in per_episode_rows if row["challenge_type"] == challenge_type])
        for challenge_type in sorted({row["challenge_type"] for row in per_episode_rows})
    }
    by_motion = {
        "static": summarize_group([row for row in per_episode_rows if not row["moving_platform"]]),
        "moving": summarize_group([row for row in per_episode_rows if row["moving_platform"]]),
    }
    by_map_category = {
        map_category: summarize_group([row for row in per_episode_rows if row["map_category"] == map_category])
        for map_category in sorted({row["map_category"] for row in per_episode_rows})
    }
    wall_time_sec = float(time.perf_counter() - wall_start)
    completed_at_utc = datetime.now(timezone.utc).isoformat()
    runtime = {
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "wall_time_sec": wall_time_sec,
        "mean_wall_time_per_episode_sec": wall_time_sec / max(1, len(per_episode_rows)),
        "episodes_per_hour": len(per_episode_rows) * 3600.0 / max(wall_time_sec, 1e-6),
    }
    map_category_report = build_map_category_report(by_map_category)

    summary = {
        "expert": "strong",
        "teacher": expert_identity,
        "profile": eval_settings["profile"],
        "task_source": source_payload,
        "filters": {
            "split": eval_settings["split"],
            "episodes_per_stage": eval_settings["episodes_per_stage"],
            "max_steps": eval_settings["max_steps"],
            "stage_names": sorted(stage_filter) if stage_filter else None,
            "challenge_types": sorted(challenge_filter) if challenge_filter else None,
            "map_seeds": sorted(map_seed_filter) if map_seed_filter else None,
            "max_episodes_total": args.max_episodes_total,
            "num_workers": max(1, parallel_workers),
        },
        "num_episodes": len(per_episode_rows),
        "aggregate_metrics": aggregate_metrics,
        "by_stage": by_stage,
        "by_challenge_type": by_challenge_type,
        "by_motion": by_motion,
        "by_map_category": by_map_category,
        "runtime": runtime,
    }

    failure_buckets = build_failure_buckets(per_episode_rows)

    save_json(args.output_dir / "expert_eval_summary.json", summary)
    save_json(args.output_dir / "expert_eval_per_episode.json", per_episode_rows)
    save_json(args.output_dir / "expert_failure_buckets.json", failure_buckets)
    save_json(args.output_dir / "expert_eval_by_map_category.json", map_category_report)
    if specialist_spec is not None and specialist_spec.quality_gate_path is not None:
        registry_root = args.expert_registry.resolve().parent
        gate_path = (registry_root / specialist_spec.quality_gate_path).resolve()
        quality_gate = load_quality_gate(gate_path)
        gate_report = evaluate_quality_gate(
            summary=summary,
            map_category=specialist_spec.map_category,
            quality_gate=quality_gate,
            teacher_id=specialist_spec.teacher_id,
            teacher_version=specialist_spec.teacher_version,
        )
        save_json(args.output_dir / "expert_quality_gate_report.json", gate_report)

    print(f"Evaluated {len(per_episode_rows)} episodes using the strong privileged expert")
    print(f"Summary saved to {args.output_dir / 'expert_eval_summary.json'}")
    print(f"Per-map-category report saved to {args.output_dir / 'expert_eval_by_map_category.json'}")
    print_map_category_report(map_category_report)


if __name__ == "__main__":
    main()
