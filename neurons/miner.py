# neurons/miner.py
# -------------------------------------------------------------------------
#  Swarm Miner (SDK v2.0.0 – Policy API)
# -------------------------------------------------------------------------
#  Implements the handshake described in Phase 2:
#
#  1) Every inbound request → send a PolicyRef (points at our wheel).
#  2) If the validator replies with `need_blob=True` → stream the wheel
#     back in fixed‑size chunks (PolicyChunk messages).
# -------------------------------------------------------------------------

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Tuple

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
    #  **Adjust these constants for your own wheel**
    # ------------------------------------------------------------------
    WHEEL_PATH = Path("dist/miner_submission-0.1-py3-none-any.whl")
    ENTRYPOINT = "pilot:Pilot"
    FRAMEWORK = "torch1.13"  # pick one of the allowed tags

    # ------------------------------------------------------------------
    # Life‑cycle
    # ------------------------------------------------------------------
    def __init__(self, config=None):
        super().__init__(config=config)
        self.load_state()

        # Pre‑compute metadata once at boot
        if not self.WHEEL_PATH.exists():
            raise FileNotFoundError(
                f"Wheel not found at {self.WHEEL_PATH}. "
                "Build it first with `python -m build --wheel`."
            )
        self._sha256 = sha256sum(self.WHEEL_PATH)
        self._size = self.WHEEL_PATH.stat().st_size

        ColoredLogger.success("Swarm Miner initialised.", ColoredLogger.GREEN)

    # ------------------------------------------------------------------
    # Main RPC endpoint (single required method)
    # ------------------------------------------------------------------
    async def forward(self, synapse: PolicySynapse) -> PolicySynapse:  # noqa: D401
        """Handshake endpoint called by validators."""
        try:
            validator = getattr(synapse.dendrite, "hotkey", "<?>")  # type: ignore[attr-defined]
            ColoredLogger.info(f"[forward] Request from {validator}", ColoredLogger.YELLOW)

            # ------------------------------------------------------------
            # 2‑step handshake: validator → miner
            #   • first call:  need_blob absent / False  → send ref
            #   • second call: need_blob True           → stream wheel
            # ------------------------------------------------------------
            if synapse.need_blob:
                # --------------------------------------------------------
                # Step 2 – stream wheel chunks until EOF
                # --------------------------------------------------------
                ColoredLogger.info("Validator requested blob; streaming …", ColoredLogger.BLUE)

                for data in iter_chunks(self.WHEEL_PATH):
                    chunk_msg = PolicySynapse(
                        chunk=asdict(PolicyChunk(sha256=self._sha256, data=data))
                    )
                    await synapse.dendrite.send(chunk_msg)  # type: ignore[attr-defined]

                ColoredLogger.success("Finished streaming wheel.", ColoredLogger.GREEN)
                # Nothing more to return for this call
                return PolicySynapse()

            # ------------------------------------------------------------
            # Step 1 – send PolicyRef
            # ------------------------------------------------------------
            ref = PolicyRef(
                sha256=self._sha256,
                entrypoint=self.ENTRYPOINT,
                framework=self.FRAMEWORK,
                size_bytes=self._size,
            )
            ColoredLogger.success("Sent PolicyRef.", ColoredLogger.GREEN)
            return PolicySynapse(ref=asdict(ref))

        except Exception as err:  # pragma: no cover – defensive path
            bt.logging.error(f"Miner forward error: {err}")
            #  Return an *empty* synapse so the validator can handle failure gracefully
            return PolicySynapse()

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
