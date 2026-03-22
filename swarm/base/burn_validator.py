from __future__ import annotations

import os
import sys
import time
from typing import List

import bittensor as bt
from swarm.base.validator import BaseValidatorNeuron


HEARTBEAT_SEC = int(os.getenv("BURN_VALIDATOR_HEARTBEAT_SEC", "5"))
STALL_TIMEOUT_SEC = int(os.getenv("BURN_VALIDATOR_STALL_TIMEOUT_SEC", "900"))


def _restart_self(reason: str) -> None:
    bt.logging.error(f"{reason}; restarting burn validator process")
    os.execv(sys.executable, [sys.executable] + sys.argv)


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
        last_step = validator.step
        last_progress_at = time.monotonic()
        while True:
            now = time.monotonic()
            thread_alive = validator.thread.is_alive() if validator.thread else False

            if validator.step != last_step:
                last_step = validator.step
                last_progress_at = now

            stalled_for = now - last_progress_at
            if not thread_alive:
                _restart_self("validator worker thread died")
            if stalled_for > STALL_TIMEOUT_SEC:
                _restart_self(
                    f"validator worker stalled for {stalled_for:.1f}s at step {validator.step}"
                )

            bt.logging.info(
                "Validator running... "
                f"{time.time()} step={validator.step} "
                f"worker_alive={thread_alive} stalled_for={stalled_for:.1f}s"
            )
            time.sleep(HEARTBEAT_SEC)
