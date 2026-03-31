"""Generate a reproducible validator-like task list for stage-02 evaluation.

This mirrors the validator sampling path:
- sample 32-bit seeds uniformly
- call ``random_task(sim_dt=0.02, seed=seed)`` for each seed

The real validator uses ``SystemRandom`` per epoch. For reproducible local
evaluation we default to a deterministic RNG seed while keeping the downstream
task generation path identical to validator evaluation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys
from typing import Any

DEFAULT_MODEL_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]
for root in (DEFAULT_MODEL_ROOT, REPO_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from training_env import DEFAULT_SIM_DT, serialize_task
from training_lib.common import ensure_dir, save_json
from swarm.validator.task_gen import random_task

_MAX_SEED = 2**32 - 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--name", type=str, default="validator_like_200")
    parser.add_argument("--num-tasks", type=int, default=200)
    parser.add_argument("--seed-mode", type=str, choices=("deterministic", "system"), default="deterministic")
    parser.add_argument("--random-seed", type=int, default=20260330)
    return parser.parse_args()


def build_validator_like_task_list_payload(
    *,
    name: str,
    num_tasks: int,
    seed_mode: str,
    random_seed: int,
) -> dict[str, Any]:
    rng: random.Random
    if seed_mode == "system":
        rng = random.SystemRandom()
    else:
        rng = random.Random(random_seed)

    rows: list[dict[str, Any]] = []
    challenge_hist: dict[str, int] = {}
    moving_hist = {"static": 0, "moving": 0}
    sampled_seeds: list[int] = []

    for _ in range(num_tasks):
        seed = int(rng.randint(0, _MAX_SEED))
        sampled_seeds.append(seed)
        task = random_task(sim_dt=DEFAULT_SIM_DT, seed=seed)
        payload = serialize_task(task)
        challenge_hist[str(int(task.challenge_type))] = challenge_hist.get(str(int(task.challenge_type)), 0) + 1
        moving_hist["moving" if bool(task.moving_platform) else "static"] += 1
        rows.append(
            {
                "stage_name": "validator_like",
                "split": "screening_like",
                "task": payload,
                "runtime": {
                    "sampled_seed": seed,
                    "challenge_type": int(task.challenge_type),
                    "moving_platform": bool(task.moving_platform),
                },
            }
        )

    return {
        "name": name,
        "task_source": "validator_like_task_list",
        "selection_config": {
            "num_tasks": int(num_tasks),
            "seed_mode": str(seed_mode),
            "random_seed": int(random_seed) if seed_mode == "deterministic" else None,
            "sim_dt": float(DEFAULT_SIM_DT),
            "sampling_path": "validator.random_task",
        },
        "num_selected": len(rows),
        "challenge_type_histogram": dict(sorted(challenge_hist.items())),
        "motion_histogram": moving_hist,
        "sampled_seeds": sampled_seeds,
        "tasks": rows,
    }


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_path.parent)
    payload = build_validator_like_task_list_payload(
        name=args.name,
        num_tasks=args.num_tasks,
        seed_mode=args.seed_mode,
        random_seed=args.random_seed,
    )
    save_json(args.output_path, payload)
    print(f"Generated {payload['num_selected']} validator-like tasks")
    print(f"Challenge histogram: {payload['challenge_type_histogram']}")
    print(f"Motion histogram: {payload['motion_histogram']}")
    print(f"Saved task list to {args.output_path}")


if __name__ == "__main__":
    main()
