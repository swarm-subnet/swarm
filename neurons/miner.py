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
from pathlib import Path
from typing import Tuple
import base64
import bittensor as bt

# ── Swarm core ────────────────────────────────────────────────────────────
from swarm.base.miner import BaseMinerNeuron
from swarm.protocol import PolicySynapse, PolicyRef, PolicyChunk
from swarm.utils.hash import sha256sum

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
    #  Whitelisted validators - only these can access the miner
    # ------------------------------------------------------------------
    WHITELISTED_VALIDATORS = {
        "5FTr8ZAQCGieBGdqXvGxHcAuzcEscEyvUQmnRZ8PJnEsn124": "RoundTable21",
        "5FKk6ucEKuKzLspVYSv9fVHonumxMJ33MdHqbVjZi2NUs124": "Rizzo", 
        "5FF6pxRem43f7wCisfXevqYVURZtxxnC4kYTx4dnNAWqi9vg": "Owner",
        "5CsvRJXuR955WojnGMdok1hbhffZyB4N5ocrv82f3p5A2zVp": "tao5",
        "5CUwbDbxCm3A4uk3rC69gQuphyG1CZaWBZRjFQTnvvMMPGun": "Yuma",
        "5EhiBKjj56jE1a6rLPP14TtrzxiwgfG8qk7nuZprkbYKH87C": "OTF",
        "5DPY75H4H8QxWpWJ91LYhfGCjLjygDtsXG63J7TmgZ4ixykp": "TestValidator"  # Keep existing test validator
    }

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

        # Check if validator is whitelisted
        if hotkey in self.WHITELISTED_VALIDATORS:
            validator_name = self.WHITELISTED_VALIDATORS[hotkey]
            ColoredLogger.success(f"Allowing whitelisted validator: {validator_name} ({hotkey})", ColoredLogger.GREEN)
            return False, f"Whitelisted validator: {validator_name}"

        # If not whitelisted, deny access
        ColoredLogger.warning(f"Denying non-whitelisted validator: {hotkey}", ColoredLogger.RED)
        return True, f"Validator {hotkey} not in whitelist"

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