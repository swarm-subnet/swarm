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

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List

from swarm.constants import SIM_DT
from swarm.validator.docker.docker_evaluator import DockerSecureEvaluator
from swarm.validator.task_gen import task_for_seed_and_type


def summarize_idempotency_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not runs:
        raise ValueError("At least one run is required for idempotency summary.")

    unique_scores = sorted({float(run["score"]) for run in runs})
    unique_sim_times = sorted({float(run["time_sec"]) for run in runs})
    unique_wall_times = sorted({float(run["wall_time_sec"]) for run in runs})
    unique_success_values = sorted({bool(run["success"]) for run in runs})

    return {
        "runs": runs,
        "unique_scores": unique_scores,
        "unique_sim_times": unique_sim_times,
        "unique_wall_times": unique_wall_times,
        "unique_success_values": unique_success_values,
        "idempotent_score": len(unique_scores) == 1,
        "idempotent_sim_time": len(unique_sim_times) == 1,
        "idempotent_success": len(unique_success_values) == 1,
        "strict_idempotent": (
            len(unique_scores) == 1
            and len(unique_sim_times) == 1
            and len(unique_success_values) == 1
        ),
    }


def run_idempotency(
    *,
    model_path,
    uid: int,
    seed: int,
    challenge_type: int,
    runs: int,
    worker_id: int = 0,
) -> Dict[str, Any]:
    if runs <= 0:
        raise ValueError("runs must be positive")

    evaluator = DockerSecureEvaluator()
    if not evaluator._base_ready:
        raise RuntimeError("Docker evaluator base image is not ready.")

    run_rows: List[Dict[str, Any]] = []
    for run_index in range(1, runs + 1):
        task = task_for_seed_and_type(
            SIM_DT,
            seed=seed,
            challenge_type=challenge_type,
        )
        wall_start = time.perf_counter()
        results = asyncio.run(
            evaluator.evaluate_seeds_batch(
                tasks=[task],
                uid=uid,
                model_path=model_path,
                worker_id=worker_id,
            )
        )
        wall_elapsed = time.perf_counter() - wall_start
        if len(results) != 1:
            raise RuntimeError(
                f"Expected one ValidationResult per idempotency run, got {len(results)}"
            )
        result = results[0]
        run_rows.append(
            {
                "run": run_index,
                "success": bool(getattr(result, "success")),
                "time_sec": float(getattr(result, "time_sec")),
                "wall_time_sec": float(wall_elapsed),
                "score": float(getattr(result, "score")),
            }
        )

    summary = summarize_idempotency_runs(run_rows)
    summary.update(
        {
            "uid": int(uid),
            "seed": int(seed),
            "challenge_type": int(challenge_type),
            "runs_requested": int(runs),
        }
    )
    return summary
