# neurons/miner.py
# -------------------------------------------------------------------------
#  Swarm Miner (SDK v2.0.0 – Policy API)
# -------------------------------------------------------------------------
#  Implements the handshake described in Phase 2:
#
#  1) Every inbound request → send a PolicyRef (points at our model).
#  2) If the validator replies with `need_blob=True` → stream the model
#     back in fixed‑size chunks (PolicyChunk messages).
# -------------------------------------------------------------------------

import time
from dataclasses import asdict
from pathlib import Path
from typing import Tuple
import base64
import bittensor as bt

# ── Swarm core ────────────────────────────────────────────────────────────
from swarm.base.miner import BaseMinerNeuron
from swarm.protocol import PolicySynapse, PolicyRef, PolicyChunk
from swarm.utils.hash import sha256sum
from swarm.utils.chunking import iter_chunks

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
    #  **Adjust these constants for your own model**
    # ------------------------------------------------------------------
    POLICY_PATH = Path("model/ppo_policy.zip")
    ENTRYPOINT  = ""             # not used by SB3 but kept for future proofing
    FRAMEWORK   = "sb3-ppo"

    # ------------------------------------------------------------------
    #  Life cycle
    # ------------------------------------------------------------------
    def __init__(self, config=None):
        super().__init__(config=config)
        self.load_state()

        if not self.POLICY_PATH.exists():
            raise FileNotFoundError(f"Model not found: {self.POLICY_PATH}")

        self._sha256 = sha256sum(self.POLICY_PATH)
        self._size   = self.POLICY_PATH.stat().st_size

        ColoredLogger.success("Swarm Miner initialised.", ColoredLogger.GREEN)

    # ------------------------------------------------------------------
    #  Main RPC endpoint
    # ------------------------------------------------------------------
    async def forward(self, synapse: PolicySynapse) -> PolicySynapse:
        """
        • need_blob absent / False → return PolicyRef  
        • need_blob True          → return a single base‑64 chunk
        """
        try:
            vk = getattr(synapse.dendrite, "hotkey", "<??>")
            ColoredLogger.info(f"[forward] from {vk}", ColoredLogger.YELLOW)

            # ── validator wants the model binary ──────────────────────
            if synapse.need_blob:
                ColoredLogger.info("Sending full model blob …", ColoredLogger.BLUE)

                raw_bytes = self.POLICY_PATH.read_bytes()
                b64_str   = base64.b64encode(raw_bytes).decode("ascii")

                return PolicySynapse.from_chunk(
                    PolicyChunk(sha256=self._sha256, data=b64_str)
                )

            # ── first handshake: send manifest ────────────────────────
            ref = PolicyRef(
                sha256     = self._sha256,
                entrypoint = self.ENTRYPOINT,
                framework  = self.FRAMEWORK,
                size_bytes = self._size,
            )
            ColoredLogger.success("Sent PolicyRef.", ColoredLogger.GREEN)
            return PolicySynapse.from_ref(ref)

        except Exception as e:
            bt.logging.error(f"Miner forward error: {e}")
            return PolicySynapse()          # fail‑safe empty reply

        
    # ------------------------------------------------------------------
    #  Black‑list logic (unchanged except for type names)
    # ------------------------------------------------------------------
    async def blacklist(self, synapse: PolicySynapse) -> Tuple[bool, str]:
        return await self._common_blacklist(synapse)

    async def _common_blacklist(self, synapse: PolicySynapse) -> Tuple[bool, str]:
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning("Request without dendrite/hotkey.")
            return True, "Missing dendrite or hotkey"

        hotkey = synapse.dendrite.hotkey

        if hotkey =="5DPY75H4H8QxWpWJ91LYhfGCjLjygDtsXG63J7TmgZ4ixykp": #whitelist test validator
            return False, "OK"  # whitelisted test validator

        # 1 – unknown hotkey?
        if (
            not self.config.blacklist.allow_non_registered            # type: ignore[attr-defined]
            and hotkey not in self.metagraph.hotkeys
        ):
            return True, f"Unrecognised hotkey: {hotkey}"

        uid = self.metagraph.hotkeys.index(hotkey)

        # 2 – validator permit enforcement
        if (
            self.config.blacklist.force_validator_permit              # type: ignore[attr-defined]
            and not self.metagraph.validator_permit[uid]
        ):
            return True, f"Hotkey {hotkey} lacks validator permit"

        # 3 – minimum stake check
        stake = self.metagraph.S[uid]
        min_stake = self.config.blacklist.minimum_stake_requirement   # type: ignore[attr-defined]
        if stake < min_stake:
            return True, f"Stake {stake:.2f} < required {min_stake:.2f}"

        return False, "OK"

    # ------------------------------------------------------------------
    #  Priority logic
    # ------------------------------------------------------------------
    async def priority(self, synapse: PolicySynapse) -> float:
        return await self._common_priority(synapse)

    async def _common_priority(self, synapse: PolicySynapse) -> float:
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            return 0.0

        hotkey = synapse.dendrite.hotkey
        if hotkey not in self.metagraph.hotkeys:
            return 0.0

        uid = self.metagraph.hotkeys.index(hotkey)
        return float(self.metagraph.S[uid])


# -------------------------------------------------------------------------
#  Stand‑alone entry‑point
# -------------------------------------------------------------------------
if __name__ == "__main__":
    with Miner() as miner:
        while True:
            time.sleep(5)
