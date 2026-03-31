"""Discover stage-02 task lists for progressively easier privileged-expert curricula.

The stage-01 manifest remains the benchmark source of truth. This script is
training-only: it lets stage 02 define genuinely easy curriculum tiers by
filtering tasks after world construction, using runtime-adjusted signals such
as start-to-platform height gap and line of sight.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np

DEFAULT_MODEL_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]
for root in (DEFAULT_MODEL_ROOT, REPO_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from training_env import make_training_env, serialize_task
from training_lib.common import ensure_dir, save_json
from progressive_curriculum import build_custom_radius_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--name", type=str, default="open_r1_5_static_lowdrop")
    parser.add_argument("--challenge-type", type=int, default=2)
    parser.add_argument("--radius-min-m", type=float, default=1.0)
    parser.add_argument("--radius-max-m", type=float, default=5.0)
    parser.add_argument("--moving-platform", action="store_true")
    parser.add_argument("--goal-z-min-m", type=float, default=-0.5)
    parser.add_argument("--goal-z-max-m", type=float, default=0.5)
    parser.add_argument("--goal-z-relative-to-start", action="store_true", default=True)
    parser.add_argument("--seed-start", type=int, default=20000)
    parser.add_argument("--scan-count", type=int, default=100)
    parser.add_argument("--max-accepted", type=int, default=8)
    parser.add_argument("--max-height-gap-m", type=float, default=2.0)
    parser.add_argument("--max-distance-m", type=float, default=6.0)
    parser.add_argument("--require-los", action="store_true", default=True)
    return parser.parse_args()


def build_selection_row(
    *,
    stage_name: str,
    split: str,
    task_payload: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    return {
        "stage_name": stage_name,
        "split": split,
        "task": task_payload,
        "runtime": runtime,
    }


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_path.parent)

    selected: list[dict[str, Any]] = []
    scanned: list[dict[str, Any]] = []

    for offset in range(args.scan_count):
        seed = args.seed_start + offset
        task = build_custom_radius_task(
            seed=seed,
            challenge_type=args.challenge_type,
            radius_min_m=args.radius_min_m,
            radius_max_m=args.radius_max_m,
            moving_platform=args.moving_platform,
            goal_z_min_m=args.goal_z_min_m,
            goal_z_max_m=args.goal_z_max_m,
            goal_z_relative_to_start=args.goal_z_relative_to_start,
        )
        env = make_training_env(task, gui=False, privileged=True)
        try:
            observation, info = env.reset(seed=seed)
            privileged = dict(info["privileged"])
            start_pos = np.asarray(observation["state"][0:3], dtype=np.float32)
            platform_pos = np.asarray(privileged["platform_position"], dtype=np.float32)
            runtime = {
                "seed": int(seed),
                "distance_to_platform_m": float(privileged["distance_to_platform"]),
                "xy_distance_to_platform_m": float(privileged["xy_distance_to_platform"]),
                "height_gap_m": float(start_pos[2] - platform_pos[2]),
                "line_of_sight_to_platform": bool(privileged["line_of_sight_to_platform"]),
                "adjusted_start": start_pos.astype(float).tolist(),
                "platform_position": platform_pos.astype(float).tolist(),
            }
            scanned.append(runtime)

            accept = True
            if abs(runtime["height_gap_m"]) > args.max_height_gap_m:
                accept = False
            if runtime["distance_to_platform_m"] > args.max_distance_m:
                accept = False
            if args.require_los and not runtime["line_of_sight_to_platform"]:
                accept = False

            if accept:
                selected.append(
                    build_selection_row(
                        stage_name=args.name,
                        split="curriculum",
                        task_payload=serialize_task(task),
                        runtime=runtime,
                    )
                )
                if len(selected) >= args.max_accepted:
                    break
        finally:
            env.close()

    payload = {
        "name": args.name,
        "task_source": "discovered_task_list",
        "selection_config": {
            "challenge_type": int(args.challenge_type),
            "radius_min_m": float(args.radius_min_m),
            "radius_max_m": float(args.radius_max_m),
            "moving_platform": bool(args.moving_platform),
            "goal_z_min_m": float(args.goal_z_min_m),
            "goal_z_max_m": float(args.goal_z_max_m),
            "goal_z_relative_to_start": bool(args.goal_z_relative_to_start),
            "seed_start": int(args.seed_start),
            "scan_count": int(args.scan_count),
            "max_accepted": int(args.max_accepted),
            "max_height_gap_m": float(args.max_height_gap_m),
            "max_distance_m": float(args.max_distance_m),
            "require_los": bool(args.require_los),
        },
        "num_scanned": len(scanned),
        "num_selected": len(selected),
        "tasks": selected,
        "scanned_preview": scanned[: min(len(scanned), 32)],
    }
    save_json(args.output_path, payload)
    print(f"Selected {len(selected)} tasks from {len(scanned)} scanned seeds")
    print(f"Saved task list to {args.output_path}")


if __name__ == "__main__":
    main()
