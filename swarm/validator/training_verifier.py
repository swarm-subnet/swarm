import asyncio
import json
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from zipfile import ZipFile

import bittensor as bt
import numpy as np
import torch

from swarm.constants import (
    TRAINING_LOOP_INTERVAL_SEC,
    VERIFICATION_TOP_N,
    VERIFICATION_PASS_THRESHOLD,
    VERIFICATION_CORRELATION_THRESHOLD,
    VERIFICATION_MEAN_DIFF_THRESHOLD,
    VERIFICATION_TEST_TASKS,
    MODEL_DIR,
    SIM_DT,
    HORIZON_SEC,
)
from swarm.validator.code_safety import CodeSafetyValidator
from swarm.validator.docker.docker_trainer import DockerTrainer
from swarm.validator.docker.docker_evaluator import DockerSecureEvaluator
from swarm.validator.verification_state import VerificationState
from swarm.validator.task_gen import random_task
from swarm.validator.forward import load_victory_history, calculate_score_metrics


class TrainingVerifier:

    def __init__(self, validator):
        self.validator = validator
        self.verification_state = VerificationState()
        self.docker_evaluator = DockerSecureEvaluator()
        self.running = False

    async def start_verification_loop(self):
        """Start background training verification loop"""
        if self.running:
            bt.logging.warning("Training verification loop already running")
            return

        self.running = True
        bt.logging.info("Starting training verification loop (6-hour interval)")

        while self.running:
            try:
                await self._verification_cycle()
            except Exception as e:
                bt.logging.error(f"Verification cycle error: {e}")

            await asyncio.sleep(TRAINING_LOOP_INTERVAL_SEC)

    def stop_verification_loop(self):
        """Stop the background verification loop"""
        self.running = False
        bt.logging.info("Stopping training verification loop")

    async def _verification_cycle(self):
        """Single verification cycle"""
        bt.logging.info("=== Training Verification Cycle Start ===")

        self.verification_state.cleanup_expired()

        top_miners = self._get_top_miners()
        if not top_miners:
            bt.logging.debug("No top miners to verify")
            return

        bt.logging.info(f"Top {len(top_miners)} miners for verification: {top_miners}")

        for uid in top_miners:
            status = self.verification_state.get_status(uid)
            if status == "verified":
                bt.logging.debug(f"UID {uid} already verified and cached")
                continue

            bt.logging.info(f"Starting verification for UID {uid}")
            self.verification_state.set_pending(uid)

            await self._verify_miner(uid)

        bt.logging.info("=== Training Verification Cycle Complete ===")

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

            training_code_path = await self._extract_training_code(model_path, uid)
            if not training_code_path:
                self.verification_state.set_failed(uid, "No training code provided")
                return

            scores = {}
            details = {}

            is_safe, safety_score, findings = CodeSafetyValidator.validate_training_code(training_code_path)
            scores['code_safety'] = safety_score
            details['code_safety'] = {'is_safe': is_safe, 'findings': findings}
            bt.logging.info(f"UID {uid} code safety: {safety_score:.2f} (safe: {is_safe})")

            if not is_safe:
                self.verification_state.set_failed(uid, f"Code safety failed: {findings}")
                return

            training_score, trained_model_path = await self._run_training(training_code_path, uid)
            scores['training_execution'] = training_score
            details['training_execution'] = {'completed': training_score > 0, 'model_path': str(trained_model_path) if trained_model_path else None}
            bt.logging.info(f"UID {uid} training execution: {training_score:.2f}")

            if training_score == 0 or not trained_model_path:
                self.verification_state.set_failed(uid, "Training execution failed")
                return

            arch_score = self._compare_architectures(model_path, trained_model_path)
            scores['architecture_match'] = arch_score
            details['architecture_match'] = {'score': arch_score}
            bt.logging.info(f"UID {uid} architecture match: {arch_score:.2f}")

            perf_score, perf_details = await self._compare_performance(model_path, trained_model_path, uid)
            scores['performance_match'] = perf_score
            details['performance_match'] = perf_details
            bt.logging.info(f"UID {uid} performance match: {perf_score:.2f}")

            final_score = (
                0.20 * scores['code_safety'] +
                0.10 * scores['training_execution'] +
                0.20 * scores['architecture_match'] +
                0.50 * scores['performance_match']
            )

            bt.logging.info(f"UID {uid} final verification score: {final_score:.2f}")

            if final_score >= VERIFICATION_PASS_THRESHOLD:
                self.verification_state.set_verified(uid, final_score, details)
            else:
                self.verification_state.set_failed(uid, f"Score {final_score:.2f} below threshold {VERIFICATION_PASS_THRESHOLD}")

        except Exception as e:
            bt.logging.error(f"Verification failed for UID {uid}: {e}")
            self.verification_state.set_failed(uid, f"Verification error: {e}")

    async def _extract_training_code(self, model_path: Path, uid: int) -> Optional[Path]:
        """Extract training_code.zip from model.zip"""
        try:
            with ZipFile(model_path, 'r') as zf:
                if 'training_code.zip' not in zf.namelist():
                    bt.logging.warning(f"UID {uid} model missing training_code.zip")
                    return None

                tmpdir = Path(tempfile.mkdtemp(prefix=f"training_code_{uid}_"))
                training_code_path = tmpdir / "training_code.zip"

                with zf.open('training_code.zip') as src:
                    with open(training_code_path, 'wb') as dst:
                        dst.write(src.read())

                return training_code_path

        except Exception as e:
            bt.logging.error(f"Failed to extract training code for UID {uid}: {e}")
            return None

    async def _run_training(self, training_code_path: Path, uid: int) -> Tuple[float, Optional[Path]]:
        """Run training and return score + model path"""
        try:
            output_dir = Path(tempfile.mkdtemp(prefix=f"trained_output_{uid}_"))
            seed = int(time.time() * 1000) % (2**31)
            timesteps = 50000

            trained_model_path = await DockerTrainer.run_training(
                training_code_path=training_code_path,
                output_dir=output_dir,
                seed=seed,
                timesteps=timesteps,
                uid=uid,
            )

            if trained_model_path and trained_model_path.exists():
                return 1.0, trained_model_path
            else:
                return 0.0, None

        except Exception as e:
            bt.logging.error(f"Training execution failed for UID {uid}: {e}")
            return 0.0, None

    def _compare_architectures(self, uploaded_model: Path, trained_model: Path) -> float:
        """Compare model architectures"""
        try:
            uploaded_policy = self._load_policy_weights(uploaded_model)
            trained_policy = self._load_policy_weights(trained_model)

            if uploaded_policy is None or trained_policy is None:
                return 0.0

            uploaded_layers = list(uploaded_policy.keys())
            trained_layers = list(trained_policy.keys())

            if uploaded_layers != trained_layers:
                bt.logging.warning(f"Layer mismatch: {len(uploaded_layers)} vs {len(trained_layers)}")
                return 0.5

            total_params_uploaded = sum(p.numel() for p in uploaded_policy.values())
            total_params_trained = sum(p.numel() for p in trained_policy.values())

            if total_params_uploaded != total_params_trained:
                bt.logging.warning(f"Parameter count mismatch: {total_params_uploaded} vs {total_params_trained}")
                return 0.5

            shape_matches = sum(1 for k in uploaded_layers if uploaded_policy[k].shape == trained_policy[k].shape)
            shape_score = shape_matches / len(uploaded_layers)

            return shape_score

        except Exception as e:
            bt.logging.error(f"Architecture comparison failed: {e}")
            return 0.0

    def _load_policy_weights(self, model_path: Path) -> Optional[Dict]:
        """Load policy.pth from model.zip"""
        try:
            with ZipFile(model_path, 'r') as zf:
                if 'policy.pth' not in zf.namelist():
                    return None

                with zf.open('policy.pth') as f:
                    state_dict = torch.load(f, map_location='cpu')
                    return state_dict

        except Exception as e:
            bt.logging.error(f"Failed to load policy weights: {e}")
            return None

    async def _compare_performance(
        self,
        uploaded_model: Path,
        trained_model: Path,
        uid: int
    ) -> Tuple[float, Dict]:
        """Compare performance on test tasks"""
        try:
            test_seeds = [42 + i * 1000 for i in range(VERIFICATION_TEST_TASKS)]
            uploaded_scores = []
            trained_scores = []

            for seed in test_seeds:
                task = random_task(sim_dt=SIM_DT, horizon=HORIZON_SEC, seed=seed)

                uploaded_result = await self.docker_evaluator.evaluate_model(task, uid, uploaded_model)
                uploaded_scores.append(uploaded_result.score)

                trained_result = await self.docker_evaluator.evaluate_model(task, uid, trained_model)
                trained_scores.append(trained_result.score)

            if len(uploaded_scores) == 0 or len(trained_scores) == 0:
                return 0.0, {'error': 'No evaluation results'}

            uploaded_arr = np.array(uploaded_scores, dtype=np.float64)
            trained_arr = np.array(trained_scores, dtype=np.float64)

            correlation = np.corrcoef(uploaded_arr, trained_arr)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0

            mean_diff = np.abs(uploaded_arr - trained_arr).mean()

            correlation_pass = correlation >= VERIFICATION_CORRELATION_THRESHOLD
            mean_diff_pass = mean_diff <= VERIFICATION_MEAN_DIFF_THRESHOLD

            if correlation_pass and mean_diff_pass:
                score = 1.0
            elif correlation_pass or mean_diff_pass:
                score = 0.5
            else:
                score = 0.0

            details = {
                'correlation': float(correlation),
                'mean_difference': float(mean_diff),
                'uploaded_scores': uploaded_scores,
                'trained_scores': trained_scores,
                'correlation_pass': correlation_pass,
                'mean_diff_pass': mean_diff_pass,
            }

            return score, details

        except Exception as e:
            bt.logging.error(f"Performance comparison failed: {e}")
            return 0.0, {'error': str(e)}
