# swarm/validator/forward.py
# ---------------------------------------------------------------
# Forward loop for the Swarm validator neuron.
# ---------------------------------------------------------------
from __future__ import annotations

import asyncio
import time
from typing import Dict, List
import traceback

import bittensor as bt

from swarm.protocol import (
    MapTask, FlightPlan, ValidationResult, FlightPlanSynapse,
)
from swarm.utils.uids import get_random_uids
from swarm.utils.weight_utils import update_ema_weights

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
    print(f"Synapse: {syn}")

    # 3. Send the query and gather replies
    replies: list[FlightPlanSynapse] = await self.dendrite(
        axons=axons,
        synapse=syn,
        deserialize=True,
        timeout=QUERY_TIMEOUT,
    )
    print(f"Replies: {replies}")

    # 4. Extract FlightPlans (skip miners that returned nothing/invalid)
    plans: dict[int, FlightPlan] = {}
    for uid, rep in zip(uids, replies):
        try:
            plan = rep.plan
            if plan is None:
                raise ValueError("empty plan in reply")
            plans[uid] = plan
        except Exception as e:
            print(f"[ERROR] Failed to parse plan from miner {uid}: {type(e).__name__} — {e}")
            traceback.print_exc()
    return plans


def _score_plan(task: MapTask, uid: int, plan: FlightPlan) -> ValidationResult:
    """
    Re‑simulate miner’s trajectory and compute reward components.
    """
    success, t_sim, energy = replay_once(task, plan)
    score                  = flight_reward(success, t_sim, energy, task.horizon)
    return ValidationResult(
        uid       = uid,
        success   = success,
        time_sec  = t_sim,
        energy    = energy,
        score     = score,
    )


def _apply_weight_update(self, results: List[ValidationResult]) -> None:
    """
    Exponential‑moving‑average (EMA) weight update then push on‑chain.
    """
    if not results:
        return

    if not hasattr(self, "prev_weights"):
        self.prev_weights = {}

    new_scores = {r.uid: r.score for r in results}
    self.prev_weights = update_ema_weights(self.prev_weights, new_scores, alpha=EMA_ALPHA)
    self.set_weights(self.prev_weights)


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
        print(f"plans:", plans)
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
        _apply_weight_update(results)

    except Exception as err:
        bt.logging.error(f"Validator forward error: {err}")

    # -------- 5) sleep --------------------------------------
    await asyncio.sleep(FORWARD_SLEEP_SEC)
