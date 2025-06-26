# swarm/validator/forward.py
# ---------------------------------------------------------------
# Forward loop for the Swarm validator neuron.
# ---------------------------------------------------------------
from __future__ import annotations

import asyncio
import time
from typing import Dict, List
import traceback
import numpy as np
import bittensor as bt

from swarm.protocol import (
    MapTask, FlightPlan, ValidationResult, FlightPlanSynapse,
)
from swarm.utils.uids import get_random_uids

from .task_gen import random_task
from .replay   import replay_once
from .reward   import flight_reward

from swarm.constants import (SIM_DT,      # 50 Hz physics step sent to miners
    HORIZON_SEC,      # max simulated flight time
    SAMPLE_K,       # miners sampled per forward
    QUERY_TIMEOUT,      # dendrite timeout (s)
    FORWARD_SLEEP_SEC,       # pause between forwards
    EMA_ALPHA)      # weights EMA coefficient


# ────────── Internal helpers (use self from outer scope) ────────
async def _query_miners(self, task: MapTask) -> dict[int, FlightPlan]:
    """
    Broadcast the MapTask to a sample of miners and collect FlightPlans.
    Uses the unified FlightPlanSynapse for both directions:
        • Validator → Miner:  task fields set, plan fields empty
        • Miner     → Validator:  plan fields set (task fields optional)
    """
    # 1. Choose a random sample of miners (uids → axons)
    uids: list[int] = get_random_uids(self, k=SAMPLE_K)
    axons           = [self.metagraph.axons[uid] for uid in uids]
    print(f"Querying {len(axons)} miners: {uids}")

    # 2. Build the outbound synapse *from the task*
    syn = FlightPlanSynapse.from_task(task)
    syn.version = self.version                # propagate protocol version

    # 3. Send the query and gather replies
    replies: list[FlightPlanSynapse] = await self.dendrite(
        axons=axons,
        synapse=syn,
        deserialize=True,
        timeout=QUERY_TIMEOUT,
    )

    # 4. Extract FlightPlans (skip miners that returned nothing/invalid)
    plans: dict[int, FlightPlan] = {}
    for uid, rep in zip(uids, replies):
        try:
            plan = rep.plan
            plans[uid] = plan
        except Exception as e:
            print(f"[ERROR] Failed to parse plan from miner {uid}: {type(e).__name__} — {e}")
            traceback.print_exc()
    return plans


def _score_plan(task: MapTask, uid: int, plan: FlightPlan | None) -> ValidationResult:
    """
    Re-simulate miner’s trajectory and compute reward components.
    If a miner returned an empty / invalid plan we assign score == 0.
    """
    # ── Treat “no plan” or empty-command list as an automatic failure ──
    if plan is None or not plan.commands:
        return ValidationResult(
            uid      = uid,
            success  = False,
            time_sec = task.horizon,   # full-horizon time-penalty
            energy   = 0.0,
            score    = 0.0,
        )

    # ── Normal scoring path ─────────────────────────────────────────────
    success, t_sim, energy = replay_once(task, plan)
    score = flight_reward(success, t_sim, energy, task.horizon)
    return ValidationResult(uid, success, t_sim, energy, score)


def _apply_weight_update(self, results: List[ValidationResult]) -> None:
    """
    Push miners’ scores on-chain using bittensor’s modern helper methods.

    Assumes your validator class implements:
      • self.update_scores(rewards: np.ndarray, uids: np.ndarray)
      • self.set_weights()              # no arguments
    """
    if not results:
        bt.logging.warning("No validation results – skipping weight update.")
        return

    # Align UIDs and scores
    uids_np    = np.array([r.uid   for r in results], dtype=np.int64)
    scores_np  = np.array([r.score for r in results], dtype=np.float32)

    # Update the scores cache and push weights on-chain
    self.update_scores(scores_np, uids_np)
    self.set_weights()      # cached weights are sent on-chain
    bt.logging.info(f"Updated weights for {len(uids_np)} miners.")


# ────────── Public API: called from neurons/validator.py ────────
async def forward(self) -> None:
    """
    One full validator iteration:
      1. build deterministic MapTask
      2. broadcast ➜ collect FlightPlans
      3. replay & score
      4. update on‑chain weights (EMA)
      5. brief sleep
    """
    try:
        # -------- bookkeeping -------------------------------
        if not hasattr(self, "forward_count"):
            self.forward_count = 0
        self.forward_count += 1

        bt.logging.info(f"[Forward #{self.forward_count}] start")

        # -------- 1) build task ------------------------------
        task: MapTask = random_task(sim_dt=SIM_DT, horizon=HORIZON_SEC)
        print(f"Querying miners")
        # -------- 2) query miners ----------------------------
        plans: Dict[int, FlightPlan] = await _query_miners(self, task)

        # -------- 3) replay & score --------------------------
        print(f"Received {len(plans)} FlightPlans from miners.")
        results: List[ValidationResult] = [
            _score_plan(task, uid, plan) for uid, plan in plans.items()
        ]

        # quick telemetry
        if results:
            best = max(r.score for r in results)
            avg  = sum(r.score for r in results) / len(results)
            bt.logging.info(
                f"Scored {len(results)} miners | best={best:.3f} avg={avg:.3f}"
            )
        else:
            bt.logging.warning("No valid FlightPlans returned by miners.")

        # -------- 4) weight update ---------------------------
        _apply_weight_update(self, results)

    except Exception as err:
        bt.logging.error(f"Validator forward error: {err}")

    # -------- 5) sleep --------------------------------------
    await asyncio.sleep(FORWARD_SLEEP_SEC)
