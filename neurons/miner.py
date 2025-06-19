# swarm/miner/miner.py
# ---------------------------------------------------------------------
# Swarm Miner – receives MapTask ➜ returns FlightPlan.
# ---------------------------------------------------------------------
from __future__ import annotations
import math, time, typing
from typing import List, Tuple

import bittensor as bt
from loguru import logger

from swarm.base.miner import BaseSwarmMiner
from swarm.protocol import (
    MapTask, FlightPlan, RPMCmd,
    TaskFeedbackSynapse, SetOperatorEndpointSynapse,
)
from swarm.miner.stats import MinerStats
from swarm.utils.config import config                    # black‑/priority‑cfg
from swarm.utils.logging import ColoredLogger

# ---------------------------------------------------------------------#
SAFE_ASCENT_Z   = 3.0          # m   –  climb to at least this height
ASCENT_RATE     = 1.0          # m/s –  vertical speed for ascend/descend
CRUISE_SPEED    = 3.0          # m/s –  horizontal speed
RPM_HOVER       = 2400         # rpm –  approx hover
RPM_ASCEND      = 2800         # rpm –  all motors slight thrust ↑
RPM_DESCEND     = 2200         # rpm –  gentle drop
RPM_CRUISE_FWD  = (2600, 2600, 3000, 3000)  # pitch forward ≈ go straight
# ---------------------------------------------------------------------#


# ────────────────────────────────────────────────────────────────────
# Simple deterministic “strategy” (will be improved later)
# ────────────────────────────────────────────────────────────────────
def _leg_times(start: Tuple[float, float, float],
               goal : Tuple[float, float, float],
               safe_z: float) -> Tuple[float, float, float]:
    """Return (ascend_t, cruise_t, descend_t) in seconds."""
    dz_up   = max(0.0, safe_z - start[2])
    dz_down = max(0.0, safe_z - goal[2])
    ascend_t   = dz_up   / ASCENT_RATE
    descend_t  = dz_down / ASCENT_RATE
    dx = goal[0] - start[0]
    dy = goal[1] - start[1]
    horizontal_dist = math.hypot(dx, dy)
    cruise_t  = horizontal_dist / CRUISE_SPEED
    return ascend_t, cruise_t, descend_t


def flying_strategy(task: MapTask) -> List[RPMCmd]:
    """
    Craft a *very* naive list of RPMCmd so the replay can, in theory, reach
    the goal: climb → cruise forward → descend.
    """
    start  = task.start
    goal   = task.goal
    safe_z = max(SAFE_ASCENT_Z, start[2], goal[2])

    t_up, t_cruise, t_down = _leg_times(start, goal, safe_z)

    cmds: List[RPMCmd] = []

    # 1️⃣ Ascend
    cmds.append(RPMCmd(t=0.0, rpm=(RPM_ASCEND,)*4))
    if t_up > 0:
        cmds.append(RPMCmd(t=t_up, rpm=(RPM_HOVER,)*4))

    # 2️⃣ Cruise – pitch slightly forward
    if t_cruise > 0:
        cmds.append(RPMCmd(t=t_up,          rpm=RPM_CRUISE_FWD))
        cmds.append(RPMCmd(t=t_up+t_cruise, rpm=(RPM_HOVER,)*4))

    # 3️⃣ Descend
    if t_down > 0:
        cmds.append(RPMCmd(t=t_up+t_cruise,       rpm=(RPM_DESCEND,)*4))
        cmds.append(RPMCmd(t=t_up+t_cruise+t_down, rpm=(RPM_HOVER,)*4))

    # Final safety hover until horizon expires
    cmds.append(RPMCmd(t=task.horizon, rpm=(RPM_HOVER,)*4))
    return cmds


# ────────────────────────────────────────────────────────────────────
class Miner(BaseSwarmMiner):
    """
    Concrete Swarm miner:
      • implements `solve(task)`  → returns FlightPlan
      • handles optional feedback / endpoint synapses
      • shares blacklist / priority across endpoints
    """

    def __init__(self, config: bt.Config | None = None):
        super().__init__(config=config)
        self.miner_stats = MinerStats()
        self.load_state()

    # -------- core logic ------------------------------------------------
    def solve(self, task: MapTask) -> FlightPlan:
        """Produce a FlightPlan for the given MapTask."""
        ColoredLogger.info(
            f"🛩  Solving MapTask seed={task.map_seed} "
            f"start={task.start} goal={task.goal}", ColoredLogger.CYAN
        )
        cmds = flying_strategy(task)
        plan = FlightPlan(commands=cmds, sha256="")    # __post_init__ computes hash
        self._log_plan(plan)
        return plan

    # -------- feedback endpoint ----------------------------------------
    async def forward_feedback(self, syn: TaskFeedbackSynapse) -> TaskFeedbackSynapse:
        """
        Show validator feedback & update local stats.
        """
        self.miner_stats.log_feedback(syn.score, syn.execution_time)
        syn.print_in_terminal(miner_stats=self.miner_stats)
        return syn

    # -------- operator endpoint echo -----------------------------------
    async def forward_set_organic_endpoint(
        self, syn: SetOperatorEndpointSynapse
    ) -> SetOperatorEndpointSynapse:
        syn.endpoint = config.operator_endpoint
        return syn

    # -------- util ------------------------------------------------------
    def _log_plan(self, plan: FlightPlan) -> None:
        ColoredLogger.info("FlightPlan (first 4 cmds):", ColoredLogger.GRAY)
        for c in plan.commands[:4]:
            logger.info(f"  t={c.t:.2f}s  rpm={c.rpm}")

    # -------- shared blacklist / priority ------------------------------
    async def _common_blacklist(
        self,
        syn: typing.Union[bt.Synapse, TaskFeedbackSynapse, SetOperatorEndpointSynapse],
    ) -> typing.Tuple[bool, str]:
        if syn.dendrite is None or syn.dendrite.hotkey is None:
            return True, "Missing dendrite / hotkey"

        hk = syn.dendrite.hotkey
        if hk not in self.metagraph.hotkeys:
            return True, "Unknown caller"

        uid = self.metagraph.hotkeys.index(hk)

        # ensure caller is validator if requested
        if config.blacklist.force_validator_permit and not self.metagraph.validator_permit[uid]:
            return True, "Caller is not validator"

        if self.metagraph.S[uid] < config.blacklist.minimum_stake_requirement:
            return True, "Insufficient stake"

        return False, "ok"

    async def blacklist(self, syn) -> typing.Tuple[bool, str]:
        return await self._common_blacklist(syn)

    async def blacklist_feedback(self, syn) -> typing.Tuple[bool, str]:
        return await self._common_blacklist(syn)

    async def blacklist_set_organic_endpoint(self, syn) -> typing.Tuple[bool, str]:
        return await self._common_blacklist(syn)

    async def _common_priority(
        self,
        syn: typing.Union[bt.Synapse, TaskFeedbackSynapse, SetOperatorEndpointSynapse],
    ) -> float:
        hk = getattr(syn.dendrite, "hotkey", None)
        if hk and hk in self.metagraph.hotkeys:
            uid = self.metagraph.hotkeys.index(hk)
            return float(self.metagraph.S[uid])
        return 0.0

    async def priority(self, syn) -> float:
        return await self._common_priority(syn)

    async def priority_feedback(self, syn) -> float:
        return await self._common_priority(syn)

    async def priority_set_organic_endpoint(self, syn) -> float:
        return await self._common_priority(syn)


# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Miner entry‑point – keep it simple
    with Miner() as miner:
        bt.logging.success(f"🚀 Swarm miner online  hotkey={miner.wallet.hotkey}")
        while True:
            time.sleep(5)
