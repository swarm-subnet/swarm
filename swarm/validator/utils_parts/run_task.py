"""Single-task worker for the new-flow validator.

The new ``forward`` loop hands one task at a time to ``run_task``. The
function downloads the model, runs the requested phase, and submits
the result. Cancellation flows through ``cancel_flag`` (set by the SSE
listener); the streaming evaluator checks it at chunk boundaries.

Backend recomputes the authoritative score from ``SeedScore`` rows, so
the ``score`` we submit is a sanity-check averaged from the seeds the
validator just ran. Per-type scores are likewise local; backend
recomputes per-type means by aggregating across the relevant tasks.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, Optional

import bittensor as bt
import numpy as np

from swarm.constants import MODEL_DIR
from swarm.utils.hash import sha256sum

from .evaluation import _run_full_benchmark, _run_screening
from .model_fetch import _ensure_models_from_backend


async def run_task(
    self,
    task: Dict[str, Any],
    *,
    cancel_flag: asyncio.Event,
    wake_flag: asyncio.Event,
) -> None:
    uid = int(task.get("uid", -1))
    phase = str(task.get("phase", ""))
    task_id = task.get("task_id")
    seeds_from = int(task.get("seeds_from", 0))
    seeds_to = task.get("seeds_to")
    seeds_to = int(seeds_to) if seeds_to else None
    batch_id = task.get("batch_id")
    epoch = int(
        task.get("epoch_number")
        or self.seed_manager.epoch_number
        or 0
    )
    model_hash = str(task.get("model_hash", ""))
    github_url = str(task.get("github_url", ""))

    if uid < 0 or not phase or task_id is None:
        bt.logging.warning(f"run_task: malformed task payload {task}")
        return

    paths = await _ensure_models_from_backend(
        self,
        [{"uid": uid, "model_hash": model_hash, "github_url": github_url}],
    )
    entry = paths.get(uid)
    if entry is None:
        bt.logging.warning(f"run_task: failed to fetch model for UID {uid}")
        return
    model_path: Path = entry[0]
    if not model_path.exists() or sha256sum(model_path) != model_hash:
        bt.logging.warning(f"run_task: model hash mismatch for UID {uid}")
        return

    if phase == "SCREENING":
        result = await _run_screening(
            self,
            uid,
            model_path,
            reeval=False,
            task_id=task_id,
            seeds_from=seeds_from,
            seeds_to=seeds_to,
            cancel_flag=cancel_flag,
            batch_id=batch_id,
        )
        avg, all_scores, per_type_raw, cancel_reason, early_failed = result
        per_type_avgs = _per_type_means(per_type_raw)
        seeds_evaluated = seeds_from + len(all_scores)
        sanity_score = float(avg) if all_scores else 0.0
    elif phase in ("BENCHMARK", "REEVAL"):
        result = await _run_full_benchmark(
            self,
            uid,
            model_path,
            reeval=(phase == "REEVAL"),
            task_id=task_id,
            cancel_flag=cancel_flag,
            seeds_from=seeds_from,
            seeds_to=seeds_to,
            batch_id=batch_id,
        )
        avg, per_type_avgs, all_scores, _per_type_raw, cancel_reason = result
        seeds_evaluated = seeds_from + len(all_scores)
        sanity_score = float(avg) if all_scores else 0.0
        early_failed = False
    else:
        bt.logging.warning(f"run_task: unsupported phase {phase}")
        return

    if cancel_reason and "cancel_flag" in cancel_reason:
        bt.logging.info(
            f"run_task: aborted by SSE for UID {uid} ({cancel_reason})"
        )
        return
    if cancel_reason and not early_failed:
        bt.logging.warning(
            f"run_task: phase ended with cancel_reason={cancel_reason}; "
            "letting backend decide via late-submission gate"
        )

    submission = await self.backend_api.submit_task_result(
        task_id=int(task_id),
        score=sanity_score,
        per_type_scores=per_type_avgs,
        seeds_evaluated=int(seeds_evaluated),
        early_failed=bool(early_failed),
        epoch_number=int(epoch),
    )
    reason = submission.get("reason") or ""
    if reason.startswith("seed_gap_at_index_"):
        # Backend says we're missing a seed; next /next-task will hand
        # back a task with the correct seeds_from. Nothing to do here.
        bt.logging.warning(
            f"run_task: backend reports seed gap for UID {uid}: {reason}"
        )


def _per_type_means(per_type_raw: Dict[str, list]) -> Dict[str, float]:
    return {
        name: float(np.mean(values))
        for name, values in per_type_raw.items()
        if values
    }
