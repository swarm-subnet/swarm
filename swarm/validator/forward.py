# ---------------------------------------------------------------
# Forward loop for the Swarm validator neuron – Policy API (v2)
# ---------------------------------------------------------------
from __future__ import annotations

import asyncio
import importlib
import traceback
from dataclasses import asdict
from typing import Dict, List

import bittensor as bt
import numpy as np

from swarm.protocol import (
    MapTask,
    PolicySynapse,
    PolicyRef,
    ValidationResult,
)
from swarm.utils.uids import get_random_uids
from swarm.utils.hash import sha256sum
from swarm.utils.chunking import iter_chunks
from swarm.validator.loader import temp_venv
from swarm.utils.env_factory import make_env

from .task_gen import random_task
from .reward    import flight_reward

from swarm.constants import (
    SIM_DT,
    HORIZON_SEC,
    SAMPLE_K,
    QUERY_TIMEOUT,
    FORWARD_SLEEP_SEC,
)

# ----------------------------------------------------------------
# Miner handshake helpers
# ----------------------------------------------------------------
async def _get_pilots(self, task: MapTask) -> Dict[int, object]:
    """
    Wrapper around self._get_pilots for code‑reuse inside older neuron
    scaffolding (if you keep using this standalone module).
    """
    uids = get_random_uids(self, k=SAMPLE_K)
    return await self._get_pilots(uids, task)        # calls method in __init__


# ----------------------------------------------------------------
# Episode simulation
# ----------------------------------------------------------------
def _run_episode(task: MapTask, uid: int, pilot: object) -> ValidationResult:
    env = make_env(task, gui=False, raw_rpm=True, randomise=True)
    obs = env.reset()
    pilot.reset(task)

    t_sim   = 0.0
    energy  = 0.0
    success = False

    while t_sim < task.horizon:
        rpm = pilot.act(obs, t_sim)
        obs, _, done, info = env.step(rpm[None, :])
        t_sim += SIM_DT
        energy += np.abs(rpm).sum() * SIM_DT
        if done:
            success = info.get("success", False)
            break

    score = flight_reward(success, t_sim, energy, task.horizon)
    return ValidationResult(uid, success, t_sim, energy, score)


# ----------------------------------------------------------------
# Weight update – identical to previous implementation
# ----------------------------------------------------------------
def _apply_weight_update(self, results: List[ValidationResult]) -> None:
    if not results:
        bt.logging.warning("No validation results – skipping weight update.")
        return

    uids_np   = np.fromiter((r.uid   for r in results), dtype=np.int64)
    scores_np = np.fromiter((r.score for r in results), dtype=np.float32)

    self.update_scores(scores_np, uids_np)
    bt.logging.info(f"Updated scores for {len(results)} miners.")


# ----------------------------------------------------------------
# Public entry point called from neurons/validator.py
# ----------------------------------------------------------------
async def forward(self) -> None:
    """
    New closed‑loop validator iteration based on PolicyRef/Pilot.
    """
    try:
        self.forward_count = getattr(self, "forward_count", 0) + 1
        bt.logging.info(f"[Forward #{self.forward_count}] start")

        # -------- 1) Build task --------------------------------
        task: MapTask = random_task(sim_dt=SIM_DT, horizon=HORIZON_SEC)

        # -------- 2) Collect pilots ----------------------------
        pilots = await _get_pilots(self, task)
        bt.logging.info(f"Loaded {len(pilots)} pilots.")

        # -------- 3) Score pilots ------------------------------
        results: List[ValidationResult] = []
        for uid, pilot in pilots.items():
            try:
                results.append(_run_episode(task, uid, pilot))
            except Exception as e:
                bt.logging.warning(f"Episode failed for miner {uid}: {e}")
                traceback.print_exc()

        # Telemetry
        if results:
            best = max(r.score for r in results)
            avg  = np.mean([r.score for r in results])
            bt.logging.info(f"Scores: best={best:.3f} avg={avg:.3f}")
        else:
            bt.logging.warning("No successful episodes this round.")

        # -------- 4) Weight update -----------------------------
        _apply_weight_update(self, results)

    except Exception as err:
        bt.logging.error(f"Validator forward error: {err}")

    await asyncio.sleep(FORWARD_SLEEP_SEC)
