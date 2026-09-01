#!/usr/bin/env python3
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

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from validator.scripts.weight_history import (
    build_subtensor,
    get_last_update_vector,
    get_subnet_epoch_state,
    get_uid_for_hotkey,
)


def has_set_weights_current_epoch(
    netuid: int,
    hotkey: str,
    network: str = "finney",
    chain_endpoint: str = "",
    block: int | None = None,
) -> bool:
    """
    Return True if the validator hotkey has submitted weights in the current epoch.

    If ``block`` is provided, "current epoch" means the epoch containing that block.
    """
    target = chain_endpoint or network
    subtensor = build_subtensor(target)

    try:
        current_block = subtensor.get_current_block() if block is None else int(block)
        validator_uid = get_uid_for_hotkey(
            subtensor, netuid=netuid, hotkey=hotkey, block=current_block
        )
        if validator_uid is None:
            return False

        _, blocks_since = get_subnet_epoch_state(
            subtensor, netuid=netuid, block=current_block
        )
        epoch_start = current_block - blocks_since

        last_updates = get_last_update_vector(
            subtensor, netuid=netuid, block=current_block
        )
        if validator_uid >= len(last_updates):
            return False

        last_update_block = int(last_updates[validator_uid])
        return epoch_start <= last_update_block <= current_block
    finally:
        close_fn = getattr(subtensor, "close", None)
        if callable(close_fn):
            close_fn()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether a validator has set weights in the current epoch."
    )
    parser.add_argument("--netuid", type=int, required=True, help="Subnet netuid.")
    parser.add_argument("--hotkey", required=True, help="Validator hotkey ss58.")
    parser.add_argument(
        "--network",
        default="finney",
        help="Bittensor network name or websocket endpoint. Default: finney.",
    )
    parser.add_argument(
        "--chain-endpoint",
        default="",
        help="Explicit websocket endpoint. Overrides --network.",
    )
    parser.add_argument(
        "--block",
        type=int,
        default=None,
        help="Optional block to evaluate instead of the live chain tip.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = has_set_weights_current_epoch(
        netuid=args.netuid,
        hotkey=args.hotkey,
        network=args.network,
        chain_endpoint=args.chain_endpoint,
        block=args.block,
    )
    print(str(result).lower())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
