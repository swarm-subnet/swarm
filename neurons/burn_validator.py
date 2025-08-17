# ╭──────────────────────────────────────────────────────────────────────╮
# neurons/validator.py                                                   #
# ALWAYS sets full weight to UID 0               #
# ╰──────────────────────────────────────────────────────────────────────╯
from __future__ import annotations

import time
from typing import List

import bittensor as bt
from swarm.base.validator import BaseValidatorNeuron
from swarm.constants import SAMPLE_K


class Validator(BaseValidatorNeuron):
    """
    Minimal validator: on every epoch head, zeroes all miners’ weights
    except UID 0, which receives 100 % of the emission.
    """

    # ───────────────────────── initialization ───────────────────────── #
    def __init__(self, config=None):
        super().__init__(config=config)

    # ╭──────────────────────────── main loop ──────────────────────────╮
    async def forward(self) -> None:
        """
        Validator that burns all miners’ weights

        Implementation:
        • Fetch current miner UIDs
        • Build a weight vector with 1.0 for UID 0, 0.0 elsewhere
        • Update scores in‑memory and broadcast on‑chain (unless --no-epoch)
        """
        time.sleep(300)
        miner_uids: List[int] = list(range(0, SAMPLE_K))
        weights = [1.0 if uid == 0 else 0.0 for uid in miner_uids]

        # Store scores locally so they can be inspected via RPC
        self.update_scores(weights, miner_uids)

        # Push weights to the chain unless user passed --no-epoch
        if not self.config.no_epoch:
            self.set_weights()

        bt.logging.success(
            f"🟢 Weights broadcast: {sum(weights):.1f} total, "
            f"{weights.count(1.0)} UID(s) at 1.0 (UID 0 only)"
        )
        


# ╭────────────────── production keep‑alive (optional) ──────────────────╮
if __name__ == "__main__":

    with Validator() as validator:
        while True:
            bt.logging.info(f"Validator running... {time.time()}")
            time.sleep(5)
