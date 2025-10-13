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
from typing import Tuple, Optional
import base64
import tempfile
import shutil
from zipfile import ZipFile
import bittensor as bt

# ── Swarm core ────────────────────────────────────────────────────────────
from swarm.base.miner import BaseMinerNeuron
from swarm.protocol import PolicySynapse, PolicyRef, PolicyChunk
from swarm.utils.hash import sha256sum

from bittensor_wallet import Keypair                        
from bittensor.core.errors import NotVerifiedException      

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
        "5EbgPJdzg1daqm9DcXJ98hGUQUKU84uffumUJzEt6Cva835H": "TestValidator"  # Keep existing test validator
    }

    # ------------------------------------------------------------------
    #  Life cycle
    # ------------------------------------------------------------------
    def __init__(self, config=None):
        super().__init__(config=config)
        self.load_state()

        if not self.POLICY_PATH.exists():
            raise FileNotFoundError(f"Model not found: {self.POLICY_PATH}")

        self._training_code_sha256: Optional[str] = None
        self._serving_path = self._prepare_model_with_training_code()

        self._sha256 = sha256sum(self._serving_path)
        self._size   = self._serving_path.stat().st_size
        self.axon.verify_fns[PolicySynapse.__name__] = self._verify_validator_request
        ColoredLogger.success("Swarm Miner initialised.", ColoredLogger.GREEN)

    def _prepare_model_with_training_code(self) -> Path:
        """Package training code into model.zip if available"""
        training_code_path = self.POLICY_PATH.parent / "training_code.zip"

        if not training_code_path.exists():
            ColoredLogger.info("No training_code.zip found - serving model without training code", ColoredLogger.YELLOW)
            return self.POLICY_PATH

        try:
            self._training_code_sha256 = sha256sum(training_code_path)

            with ZipFile(self.POLICY_PATH, 'r') as model_zip:
                if 'training_code.zip' in model_zip.namelist():
                    ColoredLogger.info("Model already contains training code", ColoredLogger.GREEN)
                    return self.POLICY_PATH

            tmpdir = Path(tempfile.mkdtemp(prefix="miner_model_"))
            packaged_model = tmpdir / "model_with_training.zip"

            with ZipFile(packaged_model, 'w') as out_zip:
                with ZipFile(self.POLICY_PATH, 'r') as model_zip:
                    for item in model_zip.namelist():
                        data = model_zip.read(item)
                        out_zip.writestr(item, data)

                out_zip.write(training_code_path, arcname='training_code.zip')

            ColoredLogger.success(f"Packaged training code (SHA: {self._training_code_sha256[:16]}...)", ColoredLogger.GREEN)
            return packaged_model

        except Exception as e:
            ColoredLogger.error(f"Failed to package training code: {e}", ColoredLogger.RED)
            return self.POLICY_PATH


    async def _verify_validator_request(self, synapse: PolicySynapse) -> None:
        """
        Rejects any RPC that is not cryptographically proven to come from
        one of the whitelisted validator hotkeys.

        Signature *must* be present and valid.  If anything is missing or
        incorrect we raise `NotVerifiedException`, which the Axon middleware
        converts into a 401 reply.
        """
        # ----------  basic sanity checks  ----------
        if synapse.dendrite is None:
            raise NotVerifiedException("Missing dendrite terminal in request")

        hotkey    = synapse.dendrite.hotkey
        signature = synapse.dendrite.signature
        nonce     = synapse.dendrite.nonce
        uuid      = synapse.dendrite.uuid
        body_hash = synapse.computed_body_hash

        # 1 — is the sender even on our allow‑list?
        if hotkey not in self.WHITELISTED_VALIDATORS:
            raise NotVerifiedException(f"{hotkey} is not a whitelisted validator")

        # 2 — signature header is mandatory
        if not signature:
            raise NotVerifiedException("Request carries no signature header")

        # 3 — run all the standard Bittensor checks (nonce window, replay,
        #     timeout, signature, …).  This *does not* insist on a signature,
        #     so we still do step 4 afterwards.
        message = (
            f"nonce: {nonce}. "
            f"hotkey {hotkey}. "
            f"self hotkey {self.wallet.hotkey.ss58_address}. "
            f"uuid {uuid}. "
            f"body hash {body_hash} "
        )
        ColoredLogger.info(
            f"Verifying message: {message}",
            ColoredLogger.YELLOW,
        )

        await self.axon.default_verify(synapse)

        # 5 — all good ➜ let the middleware continue
        ColoredLogger.success(
            f"Verified call from {self.WHITELISTED_VALIDATORS[hotkey]} ({hotkey})",
            ColoredLogger.GREEN,
        )

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

                raw_bytes = self._serving_path.read_bytes()
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
                training_code_sha256 = self._training_code_sha256,
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
    
        # 1 — now we can safely trust synapse.dendrite.hotkey
        hotkey = synapse.dendrite.hotkey

        if hotkey in self.WHITELISTED_VALIDATORS:
            name = self.WHITELISTED_VALIDATORS[hotkey]
            ColoredLogger.success(f"Synapse from {name} with ({hotkey} arrived, starting verification.")
            return False, f"whitelisted: {name}"

        ColoredLogger.warning(f"Denying non‑whitelisted validator {hotkey}")
        return True, f"{hotkey} not in whitelist"

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