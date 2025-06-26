# neurons/miner.py
# -------------------------------------------------------------------------
#  Swarm 
#
#  Miner node: receives a flight‑navigation MapTask from the validator,
#  runs a deterministic open‑loop planning policy (`flying_strategy`),
#  and returns a FlightPlan to the caller.
#
#  The class keeps the generic blacklist / priority logic supplied by
#  BaseMinerNeuron so that stake‑ and validator‑based filtering continues
#  to work exactly as on every other subnet.
# -------------------------------------------------------------------------

import time
import typing
from typing import Tuple

import bittensor as bt

# ── Swarm‑specific --------------------------------------------------------
from swarm.base.miner import BaseMinerNeuron                      
from swarm.protocol import (                                      
    MapTaskSynapse,
    FlightPlanSynapse,
    FlightPlan,
)
from swarm.core.flying_strategy import flying_strategy               

# Optional coloured logging – fall back gracefully if unavailable
try:
    from swarm.utils.logging import ColoredLogger
except Exception:                                                 
    class _Stub:
        RED = GREEN = YELLOW = BLUE = GRAY = ""
        @staticmethod
        def info(msg, *a, **kw):  bt.logging.info(msg)
        @staticmethod
        def success(msg, *a, **kw): bt.logging.info(msg)
        @staticmethod
        def warning(msg, *a, **kw): bt.logging.warning(msg)
        @staticmethod
        def error(msg, *a, **kw): bt.logging.error(msg)
    ColoredLogger = _Stub()                                       

# =========================================================================
#  Miner implementation
# =========================================================================
class Miner(BaseMinerNeuron):
    # ------------------------------------------------------------------
    # Life‑cycle
    # ------------------------------------------------------------------
    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)     
        self.load_state()                      

        ColoredLogger.success("Swarm Miner initialised.", ColoredLogger.GREEN)

    # ------------------------------------------------------------------
    # Main RPC endpoint – the **only one** required for this subnet
    # ------------------------------------------------------------------
    async def forward(self, synapse: MapTaskSynapse) -> FlightPlanSynapse:
        """
        1.  Unpack MapTask sent by the validator.
        2.  Run open‑loop planning policy (no GUI inside the miner).
        3.  Pack the FlightPlan back into a synapse and return.
        """
        try:
            validator = getattr(synapse.dendrite, "hotkey", "<?>")
            ColoredLogger.info(f"[forward] Request from {validator}", ColoredLogger.YELLOW)

            task      = synapse.task
            bt.logging.info("Generating FlightPlan …")

            # ---- deterministic planning policy ---------------------
            cmds = flying_strategy(task, gui=False)
            plan = FlightPlan(commands=cmds, sha256="")       # hash auto‑computed

            ColoredLogger.success("FlightPlan ready.", ColoredLogger.GREEN)
            return FlightPlanSynapse.from_plan(plan)

        except Exception as err:
            # A failure must *never* crash the miner – reply with a stub plan.
            bt.logging.error(f"Miner forward error: {err}")

            empty_plan = FlightPlan(commands=[], sha256="")
            return FlightPlanSynapse.from_plan(empty_plan)

    # ------------------------------------------------------------------
    #  Black‑list logic (unchanged except for type names)
    # ------------------------------------------------------------------
    async def blacklist(self, synapse: MapTaskSynapse) -> Tuple[bool, str]:
        return await self._common_blacklist(synapse)

    async def _common_blacklist(
        self, synapse: typing.Union[MapTaskSynapse, FlightPlanSynapse]
    ) -> Tuple[bool, str]:
        """
        Reject calls from unknown / under‑staked callers or, optionally,
        from non‑validator hotkeys.
        """
        #Temporary override for testing
        return True
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning("Request without dendrite/hotkey.")
            return True, "Missing dendrite or hotkey"

        hotkey = synapse.dendrite.hotkey

        # 1) unknown hotkey?
        if (
            not self.config.blacklist.allow_non_registered
            and hotkey not in self.metagraph.hotkeys
        ):
            return True, f"Unrecognised hotkey: {hotkey}"

        uid = self.metagraph.hotkeys.index(hotkey)

        # 2) validator permit enforcement
        if self.config.blacklist.force_validator_permit and not self.metagraph.validator_permit[uid]:
            return True, f"Hotkey {hotkey} lacks validator permit"

        # 3) minimum stake check
        stake      = self.metagraph.S[uid]
        min_stake  = self.config.blacklist.minimum_stake_requirement
        if stake < min_stake:
            return True, f"Stake {stake:.2f} < required {min_stake:.2f}"

        return False, "OK"

    # ------------------------------------------------------------------
    #  Priority logic 
    # ------------------------------------------------------------------
    async def priority(self, synapse: MapTaskSynapse) -> float:
        return await self._common_priority(synapse)

    async def _common_priority(
        self, synapse: typing.Union[MapTaskSynapse, FlightPlanSynapse]
    ) -> float:
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            return 0.0

        hotkey = synapse.dendrite.hotkey
        if hotkey not in self.metagraph.hotkeys:
            return 0.0

        uid = self.metagraph.hotkeys.index(hotkey)
        return float(self.metagraph.S[uid]) 
if __name__ == "__main__":
    """
    Miner_entrypoint
    """
    with Miner() as miner:
        while True:
            time.sleep(5)
