# neurons/miner.py
# -------------------------------------------------------------------------
#  Swarm Miner (SDK v2.3 – GitHub-hosted models)
# -------------------------------------------------------------------------
#  Implements the handshake:
#
#  1) Validator sends an empty PolicySynapse.
#  2) Miner replies with a PolicyRef containing the model hash and
#     the public GitHub repository URL where submission.zip is hosted.
#  3) Validator downloads the ZIP directly from GitHub.
# -------------------------------------------------------------------------

import os
import time
from pathlib import Path
from typing import Tuple

import bittensor as bt

from swarm.base.miner import BaseMinerNeuron
from swarm.protocol import PolicySynapse, PolicyRef
from swarm.utils.hash import sha256sum
from swarm.utils.github import validate_github_url

from bittensor.core.errors import NotVerifiedException

try:
    from swarm.utils.logging import ColoredLogger
except Exception:

    class _Stub:
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
    POLICY_PATH = Path(__file__).parent.parent / "Submission" / "submission.zip"
    ENTRYPOINT = ""
    FRAMEWORK = "sb3-ppo"

    WHITELISTED_VALIDATORS = {
        "5FTr8ZAQCGieBGdqXvGxHcAuzcEscEyvUQmnRZ8PJnEsn124": "RoundTable21",
        "5FKk6ucEKuKzLspVYSv9fVHonumxMJ33MdHqbVjZi2NUs124": "Rizzo",
        "5FF6pxRem43f7wCisfXevqYVURZtxxnC4kYTx4dnNAWqi9vg": "Owner",
        "5CsvRJXuR955WojnGMdok1hbhffZyB4N5ocrv82f3p5A2zVp": "tao5",
        "5CUwbDbxCm3A4uk3rC69gQuphyG1CZaWBZRjFQTnvvMMPGun": "Yuma",
        "5EhiBKjj56jE1a6rLPP14TtrzxiwgfG8qk7nuZprkbYKH87C": "OTF",
        "5FCvTkZK44fcs1iHsyUce8ZgJQD8351QJiVA8YvvuA6YcP2v": "New vali",
        "5FBqnTwnCq6yeVeXTnVGHbiRR6zh6ZGbKVRBusu4zSCu4WUw": "New vali2",
        "5EbgPJdzg1daqm9DcXJ98hGUQUKU84uffumUJzEt6Cva835H": "TestValidator",
    }

    def __init__(self, config=None):
        super().__init__(config=config)
        self.load_state()

        if not self.POLICY_PATH.exists():
            raise FileNotFoundError(f"Model not found: {self.POLICY_PATH}")

        self._sha256 = sha256sum(self.POLICY_PATH)
        self._size = self.POLICY_PATH.stat().st_size

        raw_url = os.environ.get("GITHUB_URL", "").strip()
        self._github_url = validate_github_url(raw_url)
        if not self._github_url:
            raise ValueError(
                "GITHUB_URL environment variable is required. "
                "Set it to your public GitHub repo, e.g. "
                "GITHUB_URL=https://github.com/yourname/your-model"
            )

        self.axon.verify_fns[PolicySynapse.__name__] = self._verify_validator_request
        ColoredLogger.success(
            f"Swarm Miner initialised (github={self._github_url}).",
            ColoredLogger.GREEN,
        )

    async def _verify_validator_request(self, synapse: PolicySynapse) -> None:
        if synapse.dendrite is None:
            raise NotVerifiedException("Missing dendrite terminal in request")

        hotkey = synapse.dendrite.hotkey
        signature = synapse.dendrite.signature
        nonce = synapse.dendrite.nonce
        uuid = synapse.dendrite.uuid
        body_hash = synapse.computed_body_hash

        if hotkey not in self.WHITELISTED_VALIDATORS:
            raise NotVerifiedException(f"{hotkey} is not a whitelisted validator")

        if not signature:
            raise NotVerifiedException("Request carries no signature header")

        message = (
            f"nonce: {nonce}. "
            f"hotkey {hotkey}. "
            f"self hotkey {self.wallet.hotkey.ss58_address}. "
            f"uuid {uuid}. "
            f"body hash {body_hash} "
        )
        ColoredLogger.info(f"Verifying message: {message}", ColoredLogger.YELLOW)

        await self.axon.default_verify(synapse)

        ColoredLogger.success(
            f"Verified call from {self.WHITELISTED_VALIDATORS[hotkey]} ({hotkey})",
            ColoredLogger.GREEN,
        )

    async def forward(self, synapse: PolicySynapse) -> PolicySynapse:
        try:
            vk = getattr(synapse.dendrite, "hotkey", "<??>")
            ColoredLogger.info(f"[forward] from {vk}", ColoredLogger.YELLOW)

            ref = PolicyRef(
                sha256=self._sha256,
                entrypoint=self.ENTRYPOINT,
                framework=self.FRAMEWORK,
                size_bytes=self._size,
                github_url=self._github_url,
            )
            ColoredLogger.success("Sent PolicyRef.", ColoredLogger.GREEN)
            return PolicySynapse.from_ref(ref)

        except Exception as e:
            bt.logging.error(f"Miner forward error: {e}")
            return PolicySynapse()

    async def blacklist(self, synapse: PolicySynapse) -> Tuple[bool, str]:
        return await self._common_blacklist(synapse)

    async def _common_blacklist(self, synapse: PolicySynapse) -> Tuple[bool, str]:
        hotkey = synapse.dendrite.hotkey

        if hotkey in self.WHITELISTED_VALIDATORS:
            name = self.WHITELISTED_VALIDATORS[hotkey]
            ColoredLogger.success(
                f"Synapse from {name} with ({hotkey} arrived, starting verification."
            )
            return False, f"whitelisted: {name}"

        ColoredLogger.warning(f"Denying non-whitelisted validator {hotkey}")
        return True, f"{hotkey} not in whitelist"

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
#  Stand-alone entry-point
# -------------------------------------------------------------------------
if __name__ == "__main__":
    with Miner() as miner:
        while True:
            time.sleep(5)
