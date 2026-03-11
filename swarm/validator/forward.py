# ---------------------------------------------------------------
#  Swarm validator – forward loop (Policy API v2)
# ---------------------------------------------------------------
from __future__ import annotations

import asyncio
import traceback

import bittensor as bt
import numpy as np

from swarm.constants import (
    FORWARD_SLEEP_SEC,
    MAP_CACHE_PREBUILD_ALL_AT_START,
    MODEL_DIR,
    SAMPLE_K,
)
from swarm.core.env_builder import cleanup_old_epoch_cache, set_map_cache_epoch
from swarm.utils.hash import sha256sum
from swarm.utils.uids import get_random_uids

from .backend_api import BackendApiClient
from .docker.docker_evaluator import DockerSecureEvaluator
from .seed_manager import BenchmarkSeedManager
from .utils import (
    NORMAL_MODEL_QUEUE_PROCESS_LIMIT,
    _apply_backend_weights_to_scores,
    _detect_new_models,
    _ensure_models,
    _get_processable_queue_keys,
    _get_validator_stake,
    _process_normal_queue_item,
    _refresh_normal_model_queue,
    _run_full_benchmark,
    _run_map_cache_prebuild_all_once,
    _run_map_cache_warmup_step,
    _run_screening,
    _submit_score_with_ack,
    load_normal_model_queue,
    save_normal_model_queue,
    set_cached_score,
)


async def forward(self) -> None:
    """
    Benchmark-style validator forward.

    Flow:
    1. Sync with backend → get weights + reeval queue
    2. Apply weights from backend (set on-chain)
    3. Process re-eval queue (if any)
    4. Detect new/changed models
    5. For each new model: screening → full benchmark → submit
    """
    try:
        self.forward_count = getattr(self, "forward_count", 0) + 1
        bt.logging.info(f"[Forward #{self.forward_count}] start")

        # ──────────────────────────────────────────────────────────────
        # STEP 0: Initialize components if needed
        # ──────────────────────────────────────────────────────────────
        if not hasattr(self, 'seed_manager'):
            self.seed_manager = BenchmarkSeedManager()
            set_map_cache_epoch(self.seed_manager.epoch_number)

        if not hasattr(self, 'backend_api'):
            try:
                self.backend_api = BackendApiClient(wallet=self.wallet)
            except ValueError as e:
                bt.logging.error(f"Backend API initialization failed: {e}")
                bt.logging.error("Set SWARM_BACKEND_API_URL environment variable")
                await asyncio.sleep(FORWARD_SLEEP_SEC)
                return

        if not hasattr(self, 'docker_evaluator') or not DockerSecureEvaluator._base_ready:
            bt.logging.error("Docker evaluator not ready")
            await asyncio.sleep(FORWARD_SLEEP_SEC)
            return

        validator_hotkey = self.wallet.hotkey.ss58_address
        validator_stake = _get_validator_stake(self)

        # ──────────────────────────────────────────────────────────────
        # STEP 0.25: Publish any unpublished old epoch seeds
        # ──────────────────────────────────────────────────────────────
        for pub in self.seed_manager.get_pending_publications():
            ep = pub.get("epoch_number")
            try:
                await self.backend_api.publish_epoch_seeds(
                    epoch_number=ep,
                    seeds=pub.get("seeds", []),
                    started_at=pub.get("started_at", ""),
                    ended_at=pub.get("ended_at", ""),
                    benchmark_version=pub.get("benchmark_version"),
                )
                self.seed_manager.mark_epoch_published(ep)
                bt.logging.info(f"Published epoch {ep} seeds to backend")
            except Exception as e:
                bt.logging.warning(f"Failed to publish epoch {ep} seeds: {e}")

        # ──────────────────────────────────────────────────────────────
        # STEP 0.5: Epoch transition detection + map-cache warmup
        # ──────────────────────────────────────────────────────────────
        if self.seed_manager.check_epoch_transition():
            old_epoch = self.seed_manager.advance_to_new_epoch()
            set_map_cache_epoch(self.seed_manager.epoch_number)
            bt.logging.info(
                f"Epoch transition: {old_epoch} -> {self.seed_manager.epoch_number}"
            )

            for pub in self.seed_manager.get_pending_publications():
                ep = pub.get("epoch_number")
                try:
                    await self.backend_api.publish_epoch_seeds(
                        epoch_number=ep,
                        seeds=pub.get("seeds", []),
                        started_at=pub.get("started_at", ""),
                        ended_at=pub.get("ended_at", ""),
                        benchmark_version=pub.get("benchmark_version"),
                    )
                    self.seed_manager.mark_epoch_published(ep)
                    bt.logging.info(f"Published epoch {ep} seeds to backend")
                except Exception as e:
                    bt.logging.warning(f"Failed to publish epoch {ep} seeds: {e}")

            cleanup_old_epoch_cache(keep_epoch=self.seed_manager.epoch_number)

        if MAP_CACHE_PREBUILD_ALL_AT_START:
            await _run_map_cache_prebuild_all_once(self)
        else:
            await _run_map_cache_warmup_step(self)

        # ──────────────────────────────────────────────────────────────
        # STEP 1: Sync with backend
        # ──────────────────────────────────────────────────────────────
        bt.logging.info("📡 Syncing with backend...")
        sync_data = await self.backend_api.sync()

        if sync_data.get("fallback"):
            bt.logging.warning("Backend unavailable — burning 100% emissions")

        self._current_top = sync_data.get("current_top", {})
        reeval_queue = sync_data.get("reeval_queue", [])

        # ──────────────────────────────────────────────────────────────
        # STEP 2: Apply weights from backend
        # ──────────────────────────────────────────────────────────────
        backend_weights = {} if sync_data.get("fallback") else sync_data.get("weights", {})
        _apply_backend_weights_to_scores(self, backend_weights)
        nonzero_uids = int(np.count_nonzero(self.scores))
        bt.logging.info(
            f"⚖️ Applied backend weights to local scores: {nonzero_uids} non-zero UID(s)"
        )

        # ──────────────────────────────────────────────────────────────
        # STEP 3: Process re-eval queue
        # ──────────────────────────────────────────────────────────────
        if sync_data.get("fallback") and reeval_queue:
            bt.logging.warning(
                f"Backend unavailable: skipping {len(reeval_queue)} cached re-eval item(s)"
            )
        else:
            for reeval_item in reeval_queue:
                uid = reeval_item.get("uid")
                reason = reeval_item.get("reason", "unknown")
                bt.logging.info(f"🔄 Re-evaluation requested for UID {uid}: {reason}")

                model_path = MODEL_DIR / f"UID_{uid}.zip"
                if not model_path.exists():
                    bt.logging.warning(f"Model not found for re-eval UID {uid}")
                    continue

                model_hash = sha256sum(model_path)

                screening_score, screening_scores = await _run_screening(
                    self, uid, model_path
                )
                full_score, per_type_scores, full_scores = await _run_full_benchmark(
                    self, uid, model_path
                )

                combined_scores = screening_scores + full_scores
                all_seeds_count = len(combined_scores)
                total_score = (
                    float(np.median(combined_scores)) if combined_scores else 0.0
                )

                recorded, terminal, ack_reason = await _submit_score_with_ack(
                    self,
                    uid=uid,
                    validator_hotkey=validator_hotkey,
                    validator_stake=validator_stake,
                    model_hash=model_hash,
                    total_score=total_score,
                    per_type_scores=per_type_scores,
                    seeds_evaluated=all_seeds_count,
                    epoch_number=self.seed_manager.epoch_number,
                )

                if not recorded:
                    if terminal:
                        bt.logging.error(
                            f"Re-eval score rejected permanently for UID {uid}: "
                            f"{ack_reason}"
                        )
                    else:
                        bt.logging.warning(
                            f"Re-eval score submit failed for UID {uid}, "
                            f"will retry next sync: {ack_reason}"
                        )
                    continue

                set_cached_score(model_hash, self.seed_manager.epoch_number, {
                    "uid": uid,
                    "total_score": total_score,
                    "screening_score": screening_score,
                    "full_score": full_score,
                    "per_type_scores": per_type_scores,
                    "seeds_evaluated": all_seeds_count
                })

                bt.logging.info(
                    f"Re-eval complete for UID {uid}: score={total_score:.4f}"
                )

        # ──────────────────────────────────────────────────────────────
        # STEP 4: Discovery refresh (normal-model queue producer)
        # ──────────────────────────────────────────────────────────────
        uids = get_random_uids(self, k=SAMPLE_K)
        bt.logging.info(f"Checking {len(uids)} miners for model updates...")

        model_paths = await _ensure_models(self, uids)
        if not model_paths:
            bt.logging.info("No models found this cycle")
            new_models = {}
        else:
            new_models = _detect_new_models(self, model_paths)
            if new_models:
                bt.logging.info(
                    f"🆕 Found {len(new_models)} new/changed models for queue"
                )
            else:
                bt.logging.info("No new/changed models detected")

        queue = (
            _refresh_normal_model_queue(new_models)
            if new_models
            else load_normal_model_queue()
        )

        # ──────────────────────────────────────────────────────────────
        # STEP 5: Queue worker (normal-model pipeline consumer)
        # ──────────────────────────────────────────────────────────────
        processable_keys = _get_processable_queue_keys(
            queue, NORMAL_MODEL_QUEUE_PROCESS_LIMIT
        )
        if not processable_keys:
            bt.logging.info("No normal-model queue items ready this cycle")
        else:
            bt.logging.info(f"📦 Processing {len(processable_keys)} queued model(s)")
            for queue_key in processable_keys:
                await _process_normal_queue_item(
                    self,
                    queue=queue,
                    key=queue_key,
                    validator_hotkey=validator_hotkey,
                    validator_stake=validator_stake,
                )

        items = queue.get("items", {})
        completed_keys = [
            key for key, item in items.items()
            if item.get("status") in ("completed", "terminal_rejected")
        ]
        for key in completed_keys:
            items.pop(key, None)

        queue["items"] = items
        save_normal_model_queue(queue)

        self.docker_evaluator.cleanup()
        bt.logging.info(f"[Forward #{self.forward_count}] complete")

    except Exception as e:
        bt.logging.error(f"Validator forward error: {e}")
        bt.logging.error(traceback.format_exc())
