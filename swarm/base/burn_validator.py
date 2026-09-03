# The MIT License (MIT)
# Copyright © 2026 Swarm

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

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
