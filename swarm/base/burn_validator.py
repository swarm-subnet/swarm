from __future__ import annotations

import time
from typing import List

import bittensor as bt
from swarm.base.validator import BaseValidatorNeuron


class Validator(BaseValidatorNeuron):

    def __init__(self, config=None):
        super().__init__(config=config)

    async def forward(self) -> None:
        time.sleep(300)
        miner_uids: List[int] = list(range(self.metagraph.n))
        weights = [1.0 if uid == 0 else 0.0 for uid in miner_uids]

        self.update_scores(weights, miner_uids)

        self.set_weights()

        bt.logging.success(
            f"🟢 Weights broadcast: {sum(weights):.1f} total, "
            f"{weights.count(1.0)} UID(s) at 1.0 (UID 0 only)"
        )


if __name__ == "__main__":

    with Validator() as validator:
        while True:
            bt.logging.info(f"Validator running... {time.time()}")
            time.sleep(5)
