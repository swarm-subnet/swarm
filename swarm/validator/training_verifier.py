import asyncio
from pathlib import Path
from typing import List

import bittensor as bt
import numpy as np

from swarm.constants import (
    VERIFICATION_TOP_N,
    VERIFICATION_PASS_THRESHOLD,
    VERIFICATION_ENFORCEMENT_ENABLED,
    MODEL_DIR,
)
from swarm.validator.docker.docker_trainer import DockerTrainer
from swarm.validator.verification_state import VerificationState
from swarm.validator.forward import load_victory_history, calculate_score_metrics


class TrainingVerifier:

    def __init__(self, validator):
        self.validator = validator
        self.verification_state = VerificationState()
        self.running = False

    async def start_verification_loop(self):
        """Start continuous training verification loop"""
        if self.running:
            bt.logging.warning("Training verification loop already running")
            return

        self.running = True
        bt.logging.info("Starting continuous training verification loop")

        while self.running:
            try:
                await self._verification_cycle()
            except Exception as e:
                bt.logging.error(f"Verification cycle error: {e}")

            await asyncio.sleep(60)

    def stop_verification_loop(self):
        """Stop the background verification loop"""
        self.running = False
        bt.logging.info("Stopping training verification loop")

    async def _verification_cycle(self):
        """Single verification cycle"""
        self.verification_state.cleanup_expired()

        top_miners = self._get_top_miners()
        if not top_miners:
            return

        for uid in top_miners:
            status = self.verification_state.get_status(uid)
            if status == "verified":
                continue

            bt.logging.info(f"Starting verification for UID {uid}")
            self.verification_state.set_pending(uid)

            await self._verify_miner(uid)

    def _get_top_miners(self) -> List[int]:
        """Get top N miners by average score"""
        try:
            history = load_victory_history()
            if not hasattr(self.validator, 'metagraph'):
                bt.logging.warning("Validator has no metagraph")
                return []

            all_uids = np.array(range(len(self.validator.metagraph.n)))
            score_metrics = calculate_score_metrics(history, all_uids)

            if not score_metrics:
                return []

            sorted_metrics = sorted(score_metrics, key=lambda x: (-x[1], -x[2], x[0]))
            top_uids = [uid for uid, _, _ in sorted_metrics[:VERIFICATION_TOP_N]]

            return top_uids

        except Exception as e:
            bt.logging.error(f"Failed to get top miners: {e}")
            return []

    async def _verify_miner(self, uid: int):
        """Run complete verification for a miner"""
        try:
            model_path = MODEL_DIR / f"UID_{uid}.zip"
            if not model_path.exists():
                self.verification_state.set_failed(uid, "Model file not found")
                return

            result = await DockerTrainer.verify_training_full(model_path, uid)

            if not result:
                self.verification_state.set_failed(uid, "Verification failed to complete")
                return

            if result.get("error"):
                self.verification_state.set_failed(uid, result["error"])
                return

            final_score = result.get("final_score", 0.0)
            passed = result.get("passed", False)

            bt.logging.info(f"UID {uid} verification score: {final_score:.2f}, passed: {passed}")

            if passed and final_score >= VERIFICATION_PASS_THRESHOLD:
                self.verification_state.set_verified(uid, final_score, result)
            else:
                self.verification_state.set_failed(uid, f"Score {final_score:.2f} below threshold or failed checks")

        except Exception as e:
            bt.logging.error(f"Verification failed for UID {uid}: {e}")
            self.verification_state.set_failed(uid, f"Verification error: {e}")

    def should_enforce_verification(self) -> bool:
        """Check if verification enforcement is enabled"""
        return VERIFICATION_ENFORCEMENT_ENABLED

    def get_verification_status_display(self, uid: int) -> str:
        """Get human-readable verification status for a UID"""
        status = self.verification_state.get_status(uid)
        if status == "verified":
            return "✓ VERIFIED"
        elif status == "pending":
            return "⏳ PENDING"
        elif status == "failed":
            return "✗ FAILED"
        else:
            return "? UNKNOWN"

    def is_miner_verified(self, uid: int) -> bool:
        """Check if a miner is verified"""
        return self.verification_state.get_status(uid) == "verified"
