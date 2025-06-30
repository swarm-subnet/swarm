# neurons/miner.py
# -------------------------------------------------------------------------
#  Swarm Miner
# -------------------------------------------------------------------------
#  Receives a flight‑navigation *task* embedded in a ``FlightPlanSynapse``
#  (task‑fields set, plan‑fields empty), runs a deterministic open‑loop
#  planning policy (``flying_strategy``) and returns a ``FlightPlanSynapse``
#  containing the resulting plan (and echoing the task for stateless
#  validation).
#
#  Generic blacklist / priority logic from ``BaseMinerNeuron`` is preserved.
# -------------------------------------------------------------------------

import time
import typing
from typing import Tuple

import bittensor as bt

# ── Swarm‑specific --------------------------------------------------------
from swarm.base.miner import BaseMinerNeuron
from swarm.protocol import (
    FlightPlanSynapse,
    FlightPlan,
)
from swarm.core.flying_strategy import flying_strategy

# Optional coloured logging – fall back gracefully if unavailable
try:
    from swarm.utils.logging import ColoredLogger
except Exception:  # pragma: no cover – colour module optional

    class _Stub:  # noqa: D401 – simple stub
        RED = GREEN = YELLOW = BLUE = GRAY = ""

        @staticmethod
        def info(msg, *a, **kw):
            bt.logging.info(msg)

        @staticmethod
        def success(msg, *a, **kw):
            bt.logging.info(msg)

        @staticmethod
        def warning(msg, *a, **kw):
            bt.logging.warning(msg)

        @staticmethod
        def error(msg, *a, **kw):
            bt.logging.error(msg)

    ColoredLogger = _Stub()  # type: ignore[misc]


# =========================================================================
#  Miner implementation
# =========================================================================
class Miner(BaseMinerNeuron):
    # ------------------------------------------------------------------
    # Life‑cycle
    # ------------------------------------------------------------------
    def __init__(self, config=None):
        super().__init__(config=config)
        self.load_state()

        ColoredLogger.success("Swarm Miner initialised.", ColoredLogger.GREEN)

    # ------------------------------------------------------------------
    # Main RPC endpoint – the **only one** required for this subnet
    # ------------------------------------------------------------------
    async def forward(self, synapse: FlightPlanSynapse) -> FlightPlanSynapse:  # noqa: D401
        """Plan generator called by validators.

        Parameters
        ----------
        synapse : FlightPlanSynapse
            Inbound synapse from the validator.  Its *task* fields are set;
            plan fields are empty.

        Returns
        -------
        FlightPlanSynapse
            Outbound synapse carrying the generated plan (plus a copy of the
            task so the validator can match the result without extra state).
        """
        try:
            # ---------------- log origin ---------------------------------
            validator = getattr(synapse.dendrite, "hotkey", "<?>")  # type: ignore[attr-defined]
            ColoredLogger.info(f"[forward] Request from {validator}", ColoredLogger.YELLOW)

            # ---------------- extract task -------------------------------
            task = synapse.task
            if task is None:
                raise ValueError("Inbound synapse missing MapTask fields")

            bt.logging.info("Generating FlightPlan …")

            # ------------- deterministic planning ------------------------
            cmds = flying_strategy(task, gui=False)
            plan = FlightPlan(commands=cmds)  # sha256 auto‑computed

            ColoredLogger.success("FlightPlan ready.", ColoredLogger.GREEN)

            # ------------- wrap & return ---------------------------------
            reply = FlightPlanSynapse.from_plan(plan, task=task)
            return reply

        except Exception as err:  # pragma: no cover – defensive path
            # A failure must *never* crash the miner – reply with a stub plan.
            bt.logging.error(f"Miner forward error: {err}")

            empty_plan = FlightPlan(commands=[])
            return FlightPlanSynapse.from_plan(empty_plan)

    # ------------------------------------------------------------------
    #  Black‑list logic (unchanged except for type names)
    # ------------------------------------------------------------------
    async def blacklist(self, synapse: FlightPlanSynapse) -> Tuple[bool, str]:
        return await self._common_blacklist(synapse)

    async def _common_blacklist(self, synapse: FlightPlanSynapse) -> Tuple[bool, str]:
        """Reject calls from unknown / under-staked callers – except the owner validator."""

        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning("Request without dendrite/hotkey.")
            return True, "Missing dendrite or hotkey"

        hotkey = synapse.dendrite.hotkey

        # 1) unknown hotkey?
        if (
            not self.config.blacklist.allow_non_registered            # type: ignore[attr-defined]
            and hotkey not in self.metagraph.hotkeys
        ):
            return True, f"Unrecognised hotkey: {hotkey}"

        uid = self.metagraph.hotkeys.index(hotkey)

        # 2) validator-permit enforcement
        if (
            self.config.blacklist.force_validator_permit              # type: ignore[attr-defined]
            and not self.metagraph.validator_permit[uid]
        ):
            return True, f"Hotkey {hotkey} lacks validator permit"

        # 3) minimum stake check
        stake = self.metagraph.S[uid]
        min_stake = self.config.blacklist.minimum_stake_requirement   # type: ignore[attr-defined]
        if stake < min_stake:
            return True, f"Stake {stake:.2f} < required {min_stake:.2f}"

        return False, "OK"

    # ------------------------------------------------------------------
    #  Priority logic
    # ------------------------------------------------------------------
    async def priority(self, synapse: FlightPlanSynapse) -> float:
        return await self._common_priority(synapse)

    async def _common_priority(self, synapse: FlightPlanSynapse) -> float:
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            return 0.0

        hotkey = synapse.dendrite.hotkey
        if hotkey not in self.metagraph.hotkeys:
            return 0.0

        uid = self.metagraph.hotkeys.index(hotkey)
        return float(self.metagraph.S[uid])


# -------------------------------------------------------------------------
#  Stand‑alone entrypoint
# -------------------------------------------------------------------------
if __name__ == "__main__":
    with Miner() as miner:
        while True:
            time.sleep(5)
