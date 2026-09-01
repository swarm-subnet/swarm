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

from ._shared import *
from .state import load_model_hash_tracker


def _detect_new_models(
    self, model_paths: Dict[int, Tuple[Path, str]]
) -> Dict[int, Tuple[Path, str, str]]:
    """Detect models that have changed (new hash) since last check."""
    tracker = load_model_hash_tracker()
    new_models = {}

    for uid, (path, github_url) in model_paths.items():
        try:
            current_hash = sha256sum(path)
            uid_str = str(uid)
            old_hash = tracker.get(uid_str)

            if old_hash != current_hash:
                if old_hash:
                    bt.logging.info(
                        f"🔄 Model changed for UID {uid}: "
                        f"{old_hash[:16]}... → {current_hash[:16]}..."
                    )
                else:
                    bt.logging.info(
                        f"🆕 New model for UID {uid}: {current_hash[:16]}..."
                    )
                new_models[uid] = (path, current_hash, github_url)

        except Exception as e:
            bt.logging.warning(f"Failed to check model hash for UID {uid}: {e}")

    return new_models


def _get_validator_stake(self) -> float:
    """Get this validator's stake from metagraph."""
    try:
        my_uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        stake = float(self.metagraph.S[my_uid])
        return stake
    except Exception as e:
        bt.logging.warning(f"Failed to get validator stake: {e}")
        return 0.0


def _get_miner_coldkey(self, uid: int) -> str:
    """Get miner's coldkey from metagraph."""
    try:
        return self.metagraph.coldkeys[uid]
    except Exception:
        return ""
